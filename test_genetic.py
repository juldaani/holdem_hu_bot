#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:08:15 2019

@author: juho
"""



import numpy as np
from numba import jit, njit, prange
import copy
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras


sys.path.append('/home/juho/dev_folder/')
from holdem_hu_bot.agents import RndAgent, CallAgent, Agent, generateRndActions
from holdem_hu_bot.game_data_container import GameDataContainer
from holdem_hu_bot.common_stuff import suitsOnehotLut, ranksOnehotLut, encodeCardsOnehotNb
from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute, GameState




def getMasks(gameStates, playerNumber):
    playerNum = playerNumber
    
    actingPlayerNum = np.argmax(gameStates.players[:,6].reshape((-1,2)),1)
    actingPlayerMask = actingPlayerNum == playerNum
    gameEndMask = gameStates.controlVariables[:,1] == 1
    gameFailedMask = gameStates.controlVariables[:,1] == -999
    allMask = actingPlayerMask & ~gameEndMask & ~gameFailedMask
    
    return allMask, actingPlayerMask, gameEndMask, gameFailedMask


def scaler(features, winLen):
    features[:,:,winLen:] -= 0.5    # cards
    
    features[:,:5,:winLen] -= 500    
    features[:,:5,:winLen] /= 1000
    features[:,-2,:winLen] -= 2
    features[:,-2,:winLen] /= 4
    
    return features


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def computeFeaturesNb(boardsData, playersData, winLen, ranksOnehotLut, suitsOnehotLut):
    eventFeats = np.zeros((6,winLen))
    
    smallBlind = boardsData[-1,1] # Small blinds amounts are used for normalization
    actingPlayerIdx = playersData[-1,14]
    nonActingPlayerIdx = np.abs(actingPlayerIdx-1)
    
    # Pots, stacks etc. money stuffs
    pots = boardsData[:,0]
    bets = playersData[:,3] + playersData[:,11]
    pots = pots + bets
    stacks = np.asfortranarray(playersData[:,np.array([2,10])])
    
    # Normalized pots and stacks
    potsNormalized = pots / smallBlind
    stacksNormalized = stacks / smallBlind
    
    ownStacksNormalized = stacksNormalized[:, actingPlayerIdx]
    opponentStacksNormalized = stacksNormalized[:, nonActingPlayerIdx]
    
    # Betting round
    visibleCardsMask = boardsData[:,3:8].astype(np.bool_)
    bettingRound = np.sum(visibleCardsMask[:,2:].astype(np.int64),1)
    
    # Put features into array
    idx = min(len(potsNormalized), winLen)
    eventFeats[0, -idx:] = ownStacksNormalized[-idx:]
    eventFeats[1, -idx:-1] = np.abs(np.diff(ownStacksNormalized))[-(idx-1):]
    eventFeats[2, -idx:] = opponentStacksNormalized[-idx:]
    eventFeats[3, -idx:-1] = np.abs(np.diff(opponentStacksNormalized))[-(idx-1):]
    eventFeats[4, -idx:] = potsNormalized[-idx:]
    eventFeats[5, -idx:] = (bettingRound+1)[-idx:]    # Add 1 so we make difference to default value 0

    # Encode cards one hot
    boardCards = boardsData[-1,8:].reshape((1,-1))
#    boardCards[boardCards == -999] = 0  # Assign zero if failure code because otherwise 
        # 'encodeCardsOnehot' will fail
    visibleCardsMask = boardsData[-1,3:8].astype(np.bool_).reshape((1,-1))
    boardcardSuitsOnehot, boardcardRanksOnehot = encodeCardsOnehotNb(boardCards, visibleCardsMask, 
                                                                     ranksOnehotLut, suitsOnehotLut)
    holecards = playersData[-1:,np.array([0,1,8,9])]
    holecards = holecards.reshape((2,2))[actingPlayerIdx]
#    holecards[holecards == -999] = 0  # Assign zero if failure code because otherwise 
        # 'encodeCardsOnehot' will fail
    holecardSuitsOnehot, holecardRanksOnehot = encodeCardsOnehotNb(holecards.reshape(1,-1), 
                                                                   np.ones(holecards.shape, dtype=np.bool_), 
                                                                   ranksOnehotLut, suitsOnehotLut)

    return eventFeats, boardcardSuitsOnehot[0], boardcardRanksOnehot[0], holecardSuitsOnehot[0], \
        holecardRanksOnehot[0]
        

#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
@jit(nopython=True, fastmath=True, nogil=True)
def computeFeaturesWrapperNb(boardsData, playersData, gameDataIndexes, idxIdx, winLen, mask):
    maskIndexes = np.nonzero(mask)[0] + 1
    features = np.zeros((len(maskIndexes), 7, winLen+17), dtype=np.float32)

    stIdx, endIdx = np.zeros(len(maskIndexes), dtype=np.uint64), np.zeros(len(maskIndexes), np.uint64)
    for k in range(len(stIdx)):
        iMask = maskIndexes[k]
        stIdx[k], endIdx[k] = idxIdx[iMask-1], idxIdx[iMask]
        
#    for i in prange(len(stIdx)):
    for i in range(len(stIdx)):
        prevIdx, curIdx = stIdx[i], endIdx[i] 
        curGameDataIdx = gameDataIndexes[prevIdx:curIdx]

        eventFeats, boardSuits, boardRanks, holeSuits, holeRanks \
            = computeFeaturesNb(boardsData[curGameDataIdx], playersData[curGameDataIdx], 
                                winLen, ranksOnehotLut, suitsOnehotLut)
        
        features[i,:eventFeats.shape[0],:eventFeats.shape[1]] = eventFeats
        
        features[i,:2,eventFeats.shape[1]:eventFeats.shape[1]+4] = holeSuits
        features[i,2:,eventFeats.shape[1]:eventFeats.shape[1]+4] = boardSuits
        features[i,:2,eventFeats.shape[1]+4:] = holeRanks
        features[i,2:,eventFeats.shape[1]+4:] = boardRanks
    
    return features


@jit(nopython=True, fastmath=True, nogil=True)
def modelOutputsToActions(modelOutputs, pots, availableActions):
    actions = np.zeros((len(modelOutputs),2), dtype=np.int64)-1
    
    # fold, call, min raise, raise pot*[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], all-in
    amounts = np.zeros(modelOutputs.shape[1], dtype=np.int64)
    for i in range(len(modelOutputs)):
        callAmount = availableActions[i,0]
        minRaiseAmount = availableActions[i,1]
        allInAmount = availableActions[i,2]
        raiseAmounts = pots[i] * np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        
        amounts[:] = 0
        amounts[0] = 1  # Fold is always 1
        amounts[1] = callAmount
        amounts[2] = minRaiseAmount
        amounts[3:-1] = raiseAmounts
        amounts[-1] = allInAmount
    
        maxIdx = np.argsort(modelOutputs[i])[-1]
        amnt = amounts[maxIdx]

        if(maxIdx == 0):    # Fold
            actions[i,0] = amnt
        else:
            # Check that amount is not out of bounds
            if((amnt < callAmount) or (amnt > callAmount and amnt < minRaiseAmount) or (amnt > allInAmount)):
                nearestIdx = np.argmax(availableActions[i] - amnt)
                amnt = availableActions[i,nearestIdx]
            
            actions[i,1] = amnt
    
    return actions


def getWinAmounts(gameContainer, initStacks):
    data, _ = gameContainer.getData()
    lastIndexes, _ = gameContainer.getLastIndexes()
    finalStacks = data['playersData'][lastIndexes][:,[2,10]]
    
    return finalStacks - initStacks


def getOptimizedWinAmounts(gameDataContainer, initStacks, RND_AGENT_IDX, AI_AGENT_IDX, N_RND_PLAYS_PER_HAND):
    winAmountsAi = getWinAmounts(gameDataContainer, initStacks)[:,AI_AGENT_IDX]
    
    # On each row there is win amounts for ai player for a certain hand (same hand is played multiple times)
    idxToOrig = np.arange(len(winAmountsAi)).reshape((-1,N_RND_PLAYS_PER_HAND))
    winAmountsAi = winAmountsAi.reshape((-1,N_RND_PLAYS_PER_HAND))
    
    # For each hand pick the smallest win amount
    minColumn = np.argsort(winAmountsAi)[:,0]
    winAmountsAi = winAmountsAi[np.arange(len(winAmountsAi)),minColumn]
    idxToOrig = idxToOrig[np.arange(len(winAmountsAi)),minColumn]
    
    # Pick hands the ai is still winning and reduce the win amounts of the hands
    positiveWinAmountsMask = winAmountsAi > 0
    idxToOptimize = idxToOrig[positiveWinAmountsMask]
#    winAmountsToOptimize = winAmountsAi[positiveWinAmountsMask]

    # If win amounts for ai player are negative for all games
    if(np.sum(positiveWinAmountsMask) == 0):
        return winAmountsAi
    
    # Pick game states to be optimized
    gameData, indexes = gameDataContainer.getData()
    players, boards, availableActions, controlVariables = [], [], [], []
    for i,idx in enumerate(idxToOptimize):
        gameDataIdx = np.array(indexes[idx])
        
        playersData = gameData['playersData'][gameDataIdx]
        boardsData = gameData['boardsData'][gameDataIdx]
        controlVarsData = gameData['controlVariablesData'][gameDataIdx]
        availActionsData = gameData['availableActionsData'][gameDataIdx]
        
        actingPlayerIdx = playersData[:-1,14]   # Exclude last index
    
        rndAgentActingIdx = np.nonzero(actingPlayerIdx == RND_AGENT_IDX)[0]
        rndAgentActingIdx = rndAgentActingIdx[np.random.randint(len(rndAgentActingIdx))]
#        rndAgentActingIdx = rndAgentActingIdx[0]
    
        players.append(GameDataContainer.unflattenPlayersData(playersData[rndAgentActingIdx].reshape((1,-1))))
        boards.append(boardsData[rndAgentActingIdx])
        availableActions.append(availActionsData[rndAgentActingIdx])
        controlVariables.append(controlVarsData[rndAgentActingIdx])
    
    gameStatesOptimized = GameState(np.row_stack(boards), np.row_stack(players), np.row_stack(controlVariables), 
                                    np.row_stack(availableActions))
    
    # Execute fold action for the game states
    foldActions = np.zeros((len(idxToOptimize),2), dtype=np.int)-1
    foldActions[:,0] = 1
    gameStatesOptimized = executeActions(gameStatesOptimized, foldActions)
    
    # Compute optimized win amounts
    finalStacks = np.column_stack((gameStatesOptimized.players[::2,2], gameStatesOptimized.players[1::2,2]))
    optimizedWinAmounts = winAmountsAi.copy()
    optimizedWinAmounts[positiveWinAmountsMask] = (finalStacks - initStacks[idxToOptimize])[:,AI_AGENT_IDX]
    optimizedWinAmounts = optimizedWinAmounts / smallBlindsForGames[::N_RND_PLAYS_PER_HAND]

    return optimizedWinAmounts


def playGames(gameDataContainer, gameStates, aiModel, RND_AGENT_IDX, AI_AGENT_IDX):
    
    while(1):    
        gameDataContainer.addData(gameStates, mockActions)
        gameDataIndexes, gameNums, idxIdx, gameData = gameDataContainer.getAllIndexes()
        
        # Rnd agent actions
        maskRndAgent, _, _, _, = getMasks(gameStates, RND_AGENT_IDX)
        actionsRndAgent = generateRndActions(gameStates.availableActions[maskRndAgent], foldProb=0.0, 
                                             allInRaiseProb=0.1)
        
        # Ai agent actions
        maskAiAgent, _, _, _, = getMasks(gameStates, AI_AGENT_IDX)
        features = computeFeaturesWrapperNb(gameData['boardsData'], gameData['playersData'], gameDataIndexes,
                                            idxIdx, WIN_LEN, maskAiAgent)
        potsAiAgent = features[:, 4, WIN_LEN-1].copy()
        features = scaler(features, WIN_LEN)
        features = features.reshape((len(features), features.shape[1]*features.shape[2]))
    
        assert np.sum(maskAiAgent) == len(features)     # Just test that the data matches
        
        # Calculate outputs
        modelOutput = aiModel(tf.convert_to_tensor(features, dtype=tf.float32)).numpy()
    
        # Convert model outputs into actions
        smallBlinds = gameStates.boards[maskAiAgent,1]
        potsAiAgent = (potsAiAgent * smallBlinds).astype(np.int)
        availableActions = gameStates.availableActions[maskAiAgent]
        actionsAiAgent = modelOutputsToActions(modelOutput, potsAiAgent, availableActions)
        
        # Put actions from ai and rnd agent together
        actionsToExecute[:] = -999
        actionsToExecute[maskRndAgent] = actionsRndAgent
        actionsToExecute[maskAiAgent] = actionsAiAgent
        
        # Feed actions into game engine
        gameStates = executeActions(gameStates, actionsToExecute)
    
        # Termination criteria
        nValidGames = np.sum(gameStates.controlVariables[:,1]==0)
#        print(nValidGames)
        if(nValidGames == 0):
            break
        
    # Save also the last game state
    gameDataContainer.addData(gameStates, mockActions)

    return gameDataContainer


# %%

if __name__ == "__main__":
    
    # %%
    
    # Initialize agent
    
    #SEED = 123
    
    POPULATION_SIZE = 100
    RATIO_BEST_INDIVIDUALS = 0.10
    MUTATION_SIGMA = 1.0e-4
    
    N_HANDS_FOR_EVAL = 100000
    N_RND_PLAYS_PER_HAND = 1
    
    RND_AGENT_IDX = 0
    AI_AGENT_IDX = np.abs(RND_AGENT_IDX-1)
    WIN_LEN = 2
    
    
    # Init ai-models
    models = []
    for i in range(POPULATION_SIZE):
        m = keras.Sequential([
            keras.layers.Dense(50, activation='relu'),
    #        keras.layers.Dense(100, activation='relu'),
            # output: fold, call, min raise, raise pot*[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], all-in
    #        keras.layers.Dense(10, activation='softmax')])
    #        keras.layers.Dense(10, activation='sigmoid')])
            keras.layers.Dense(10, activation='relu')])
        models.append(m)
    
    
    
    ## Disable gpu
    #import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #
    #if tf.test.gpu_device_name():
    #    print('GPU found')
    #else:
    #    print("No GPU found")
    
    
    # %%
    
    
    # Create game data for evaluation
    initGameStates, initStacks = initRandomGames(N_HANDS_FOR_EVAL)
    
    initGameStates.availableActions = np.repeat(initGameStates.availableActions, N_RND_PLAYS_PER_HAND, axis=0)
    initGameStates.boards = np.repeat(initGameStates.boards, N_RND_PLAYS_PER_HAND, axis=0)
    initGameStates.controlVariables = np.repeat(initGameStates.controlVariables, N_RND_PLAYS_PER_HAND, axis=0)
    initGameStates.players = GameDataContainer.unflattenPlayersData(np.repeat(
            GameDataContainer.flattenPlayersData(initGameStates.players), N_RND_PLAYS_PER_HAND, axis=0))
    initGameStates.validMask = np.repeat(initGameStates.validMask, N_RND_PLAYS_PER_HAND, axis=0)
    initGameStates.validMaskPlayers = np.repeat(initGameStates.validMaskPlayers, N_RND_PLAYS_PER_HAND, axis=0)
    initStacks = np.repeat(initStacks, N_RND_PLAYS_PER_HAND, axis=0)
    smallBlindsForGames = initGameStates.boards[:,1]
    
    mockActions = np.zeros((len(initGameStates.availableActions),2), dtype=np.int64) - 999
    actionsToExecute = np.zeros((len(initGameStates.availableActions),2), dtype=np.int64) - 999
    
    
    
    bestIndexes = []
    populationFitness, bestFitness = [], []
    for k in range(200000):
        
        states, stacks = initRandomGames(int(N_HANDS_FOR_EVAL*0.10))
        smallBlinds = states.boards[:,1]
        rndIdx = np.random.choice(N_HANDS_FOR_EVAL, size=len(stacks), replace=0)
        
        smallBlindsForGames[rndIdx] = smallBlinds
        initStacks[rndIdx] = stacks
        
        initGameStates.availableActions[rndIdx] = states.availableActions
        initGameStates.boards[rndIdx] = states.boards
        initGameStates.controlVariables[rndIdx] = states.controlVariables
        rndIdx2 = np.repeat(rndIdx*2, 2)
        rndIdx2[1::2] = rndIdx*2+1
        initGameStates.players[rndIdx2] = states.players
        
        
        # Play games
        modelWinAmounts = []
        for i in range(len(models)):
    #        print(i)
            
            gameStates = copy.deepcopy(initGameStates)
            gameDataContainer = GameDataContainer(N_HANDS_FOR_EVAL*N_RND_PLAYS_PER_HAND)
            
            gameDataContainer = playGames(gameDataContainer, gameStates, models[i], RND_AGENT_IDX, AI_AGENT_IDX)
    #        optimizedWinAmounts = getOptimizedWinAmounts(gameDataContainers[i], initStacks, RND_AGENT_IDX, 
    #                                                     AI_AGENT_IDX, N_RND_PLAYS_PER_HAND)
    #        modelWinAmounts.append(optimizedWinAmounts)
            winAmnts = getWinAmounts(gameDataContainer, initStacks)[:,AI_AGENT_IDX] / smallBlindsForGames
            modelWinAmounts.append(winAmnts)
        
        modelFitness = [np.mean(amounts) for amounts in modelWinAmounts]
    #    modelFitness = [np.mean(amounts)/np.std(amounts) for amounts in modelWinAmounts]
        
        #plt.hist(modelWinAmountsAvg, bins=30)
        print(k, np.mean(modelFitness), np.max(modelFitness))
    #    populationFitness.append(np.mean(modelFitness))
    #    bestFitness.append(np.max(modelFitness))
        
        sorter = np.argsort(modelFitness)
        bestIdx = sorter[-int(len(sorter)*RATIO_BEST_INDIVIDUALS):]
    
        # Save data
        [tf.keras.models.save_model(model, 'data/models/'+str(i)) for i,model in enumerate(models)]
        np.save('data/'+str(k)+'_win_amounts', modelWinAmounts)
    #    m = tf.keras.models.load_model('aa.aa')    # This is how to load, just a note
    
        
        # Put the best individual without mutation to the next generation
    #    nextGeneration = [None for i in range(POPULATION_SIZE)]
    #    nextGeneration[bestIdx[-1]] = models[bestIdx[-1]]
        nextGeneration = []
        nextGeneration.append(models[bestIdx[-1]])
    #    nextGeneration = [models[idx] for idx in bestIdx]
        
        # Mutate
        for i in range(POPULATION_SIZE-len(nextGeneration)):
            idx = bestIdx[np.random.randint(len(bestIdx))]
                
            model = copy.deepcopy(models[idx])
            weights = model.get_weights()
            weightsUpdated = [w + np.random.normal(scale=MUTATION_SIGMA, size=w.shape) for w in weights]
            model.set_weights(weightsUpdated)
            
            nextGeneration.append(model)
        
        models = nextGeneration
            
    
    #plt.plot(populationFitness[0:])
    #plt.plot(bestFitness[0:])
    
    
    # %%
    
