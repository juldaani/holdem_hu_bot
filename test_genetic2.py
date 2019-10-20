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

import multiprocessing as mp


import torch
torch.set_num_threads(1)
import torch.nn as nn
#import torch.nn.functional as F


#import tensorflow as tf
#tf.enable_eager_execution()
#from tensorflow import keras


sys.path.append('/home/juho/dev_folder/')
from holdem_hu_bot.agents import generateRndActions
from holdem_hu_bot.game_data_container import GameDataContainer
from holdem_hu_bot.common_stuff import suitsOnehotLut, ranksOnehotLut
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


@jit(nopython=True, fastmath=True)
def computeFeaturesNb(boardsData, playersData, winLen, validMask, features):
    for i in range(len(features)):
        if(validMask[i] == False):
            continue
        
        curFeatures = features[i]
        
        # Shift one
        curFeaturesView = curFeatures.reshape(-1)
        curFeaturesView[:-1] = curFeaturesView[1:]
        
        curBoard = boardsData[i]
        curPlayers = playersData[i*2:i*2+2].reshape(-1)
        
        smallBlind = curBoard[1]   # Small blinds amounts are used for normalization
        actingPlayerIdx = curPlayers[14]
        nonActingPlayerIdx = np.abs(actingPlayerIdx-1)
        
        # Pots, stacks etc. money stuffs
        pot = curBoard[0]
        bets = curPlayers[3] + curPlayers[11]
        pot += bets
        stacks = curPlayers[np.array([2,10])]
        
        # Normalized pots and stacks
        potNormalized = pot / smallBlind
        stacksNormalized = stacks / smallBlind
        ownStackNormalized = stacksNormalized[actingPlayerIdx]
        opponentStackNormalized = stacksNormalized[nonActingPlayerIdx]
        
        # Betting round
        bettingRound = np.sum(curBoard[3:8])
        
        curFeatures[0,winLen-1] = ownStackNormalized
        curFeatures[1,winLen-1] = np.abs(curFeatures[0,winLen-2] - ownStackNormalized)
        curFeatures[2,winLen-1] = opponentStackNormalized
        curFeatures[3,winLen-1] = np.abs(curFeatures[2,winLen-2] - opponentStackNormalized)
        curFeatures[4,winLen-1] = potNormalized
        curFeatures[5,winLen-1] = bettingRound
        
        # Encode cards one hot
        boardCards = curBoard[8:]
        visibleCardsMask = curBoard[3:8].astype(np.bool_)
        for k in range(len(boardCards)):
            if(visibleCardsMask[k] == False):
                continue
            rankOneHot = ranksOnehotLut[boardCards[k]]
            suitOneHot = suitsOnehotLut[boardCards[k]]
            curFeatures[k,winLen:winLen+4] = suitOneHot
            curFeatures[k,winLen+4:] = rankOneHot
        
        holecards = curPlayers[np.array([0,1,8,9])].reshape((2,2))[actingPlayerIdx]
        for k in range(len(holecards)):
            rankOneHot = ranksOnehotLut[holecards[k]]
            suitOneHot = suitsOnehotLut[holecards[k]]
            curFeatures[(-2+k),winLen:winLen+4] = suitOneHot
            curFeatures[(-2+k),winLen+4:] = rankOneHot
            
    return features


@jit(nopython=True, fastmath=True)
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


def getWinAmounts(gameStates, initStacks):
    finalStacks = np.column_stack((gameStates.players[::2,2],gameStates.players[1::2,2]))

    return finalStacks - initStacks


def getWinAmountsForModels(gameStates, initStacks, agentIdx):
    modelWinAmounts = []
    for states in gameStates:
        smallBlinds = states.boards[:,1]
        winAmounts = getWinAmounts(states, initStacks)
        assert np.sum(winAmounts) == 0  # Check that games are zero-sum
        winAmounts = winAmounts[:,agentIdx] / smallBlinds
        modelWinAmounts.append(winAmounts)
        
    return modelWinAmounts


def optimizeWinAmounts(modelWinAmounts):
    for i in range(len(modelWinAmounts)):
        winAmounts = modelWinAmounts[i]
        m = winAmounts > 0
        winAmounts[m] = 3 + np.random.rand(np.sum(m)) * (winAmounts[m]-3)
        
    return modelWinAmounts
    

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


def playGames(gameStates, aiModel, WIN_LEN, RND_AGENT_IDX, AI_AGENT_IDX):
    features = np.zeros((len(gameStates.boards), 7, WIN_LEN+17))
    actionsToExecute = np.zeros((len(gameStates.boards),2), dtype=np.int64) - 999
    
    while(1):
        # Rnd agent actions
        maskRndAgent, _, _, _, = getMasks(gameStates, RND_AGENT_IDX)
#        actionsRndAgent = generateRndActions(gameStates.availableActions[maskRndAgent], foldProb=0.0, 
#                                             allInRaiseProb=0.1)
        actionsRndAgent = np.zeros((np.sum(maskRndAgent),2))-1
        actionsRndAgent[:,1] = gameStates.availableActions[maskRndAgent,0]   # Only call
        
        # Ai agent actions
        maskAiAgent, _, _, _, = getMasks(gameStates, AI_AGENT_IDX)
        
        features = computeFeaturesNb(gameStates.boards, gameStates.players, WIN_LEN, maskAiAgent, features)
        featuresScaled = scaler(features[maskAiAgent], WIN_LEN)
        featuresScaled = featuresScaled.reshape((len(featuresScaled), 
                                                 featuresScaled.shape[1]*featuresScaled.shape[2]))
    
        # Calculate outputs
#        modelOutput = aiModel(tf.convert_to_tensor(featuresScaled, dtype=tf.float32)).numpy()
#        modelOutput = aiModel.predict(featuresScaled)
        with torch.no_grad():
            featuresScaled = torch.from_numpy(featuresScaled).float()
            modelOutput = aiModel(featuresScaled).numpy()
        if(len(modelOutput) == 0):
            modelOutput = np.zeros((0,10), dtype=np.float)
    
        # Convert model outputs into actions
        smallBlinds = gameStates.boards[maskAiAgent,1]
        potsAiAgent = features[:, 4, WIN_LEN-1][maskAiAgent]
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
        
    # Check that all games were succesful
    assert np.all(~(gameStates.controlVariables[:,1] == -999))
        
    return gameStates


def playGamesWrapper(gameStates, aiModel, WIN_LEN, RND_AGENT_IDX, AI_AGENT_IDX, idx):
    return (playGames(gameStates, aiModel, WIN_LEN, RND_AGENT_IDX, AI_AGENT_IDX), idx)


def playGamesParallel(initGameStates, models, nCores, winLen, rndAgentIdx, aiAgentIdx):
    pool = mp.Pool(nCores)
    result_objects = [pool.apply_async(playGamesWrapper, args=(copy.deepcopy(initGameStates), models[i],
                                                  winLen, rndAgentIdx, aiAgentIdx, i)) 
                                                            for i in range(len(models))]
    results = [r.get() for r in result_objects]

    # Sort results because it is not quaranteed that apply_async returns them in correct order
    orderNum = np.array([res[1] for res in results])
    sorter = np.argsort(orderNum)
    finalGameStates = np.array([res[0] for res in results])

    return finalGameStates[sorter]


class AiModel(nn.Module):
    def __init__(self, winLen):
        super(AiModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(7*(winLen+17), 250),
            nn.ReLU(),
            nn.Linear(250, 10))
        
        # Get references to weights and biases. These are used when mutating the model.
        self.weights, self.biases = [], []
        for layer in self.layers:
            # Hack. Throws an AttributeError if there is no weights associated for the layer, e.q., nn.Relu
            try:
                self.weights.append(layer.weight)
                self.biases.append(layer.bias)
            except AttributeError:
                pass
        
    def forward(self, x):
        x = self.layers(x)
        return x

#        def mutate(self, sigma):
#            for i in range(len(self.weights)):
#                w = self.weights[i].data.numpy()
#                b = self.biases[i].data.numpy()
#                w += np.random.normal(scale=sigma, size=w.shape)
#                b += np.random.normal(scale=sigma, size=b.shape)

    def mutateWeights(self, sigma, ratio=1.0):
        for i in range(len(self.weights)):
            w = self.weights[i].data.numpy().reshape(-1)
            rndIdx = np.random.choice(len(w), size=max(1,int(ratio*len(w))), replace=0)
            w[rndIdx] += np.random.normal(scale=sigma, size=len(rndIdx))
#                w = self.weights[i].data.numpy()
#                w += np.random.normal(scale=sigma, size=w.shape)

    def mutateBiases(self, sigma, ratio=1.0):
        for i in range(len(self.biases)):
            b = self.biases[i].data.numpy().reshape(-1)
            rndIdx = np.random.choice(len(b), size=max(1,int(ratio*len(b))), replace=0)
            b[rndIdx] += np.random.normal(scale=sigma, size=len(rndIdx))
#                b = self.biases[i].data.numpy()
#                b += np.random.normal(scale=sigma, size=b.shape)
            
    def mutate(self, sigma, ratio=1.0):
        self.mutateWeights(sigma, ratio=ratio)
        self.mutateBiases(sigma, ratio=ratio)


#if __name__ == "__main__":
    
    # %%
    
    # Initialize agent
    
    #SEED = 123
    
    POPULATION_SIZE = 200
    RATIO_BEST_INDIVIDUALS = 0.10
    MUTATION_SIGMA = 1.0e-2
    MUTATION_RATIO = 1.0
    
    N_HANDS_FOR_EVAL = 25000
    N_HANDS_FOR_RE_EVAL = 100000
    N_RND_PLAYS_PER_HAND = 1
    
    RND_AGENT_IDX = 0
    AI_AGENT_IDX = np.abs(RND_AGENT_IDX-1)
    WIN_LEN = 2
    
    N_CORES = 6
    
    device = torch.device('cpu')
    
    

    models = []
    for i in range(POPULATION_SIZE):
#        m = keras.Sequential()
#        m.add(keras.layers.Dense(50, activation='relu', input_dim=7*(WIN_LEN+17)))
#        m.add(keras.layers.Dense(10, activation='relu'))
        
        models.append(AiModel(WIN_LEN).to(device))
    models = np.array(models)

#    # Disable gpu
#    import os
#    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#    if tf.test.gpu_device_name():
#        print('GPU found')
#    else:
#        print("No GPU found")
        


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
#    smallBlindsForGames = initGameStates.boards[:,1]
    
    populationFitness, bestFitness = [], []
    
    
    # %%
    
    for k in range(50):
        
        
        states, stacks = initRandomGames(int(N_HANDS_FOR_EVAL*0.25))
#        smallBlinds = states.boards[:,1]
        rndIdx = np.random.choice(N_HANDS_FOR_EVAL, size=len(stacks), replace=0)
        
#        smallBlindsForGames[rndIdx] = smallBlinds
        initStacks[rndIdx] = stacks
        
        initGameStates.availableActions[rndIdx] = states.availableActions
        initGameStates.boards[rndIdx] = states.boards
        initGameStates.controlVariables[rndIdx] = states.controlVariables
        rndIdx2 = np.repeat(rndIdx*2, 2)
        rndIdx2[1::2] = rndIdx*2+1
        initGameStates.players[rndIdx2] = states.players
        
        
        
        # Play games
        finalGameStates = playGamesParallel(initGameStates, models, N_CORES, WIN_LEN, RND_AGENT_IDX, 
                                            AI_AGENT_IDX)
        assert len(finalGameStates) == POPULATION_SIZE
        
        modelWinAmounts = getWinAmountsForModels(finalGameStates, initStacks, AI_AGENT_IDX)
#        modelWinAmounts = optimizeWinAmounts(modelWinAmounts)
        
        modelFitness = [np.mean(amounts) for amounts in modelWinAmounts]
#        modelFitness = [np.mean(amounts)/np.std(amounts) for amounts in modelWinAmounts]
#        modelFitness = [(np.mean(amounts)) + (np.sum(~(np.isclose(amounts,-1) | \
#                            np.isclose(amounts,-2)))/len(amounts)) for amounts in modelWinAmounts]
        
        sorter = np.argsort(modelFitness)
        bestIdx = sorter[-int(len(sorter)*RATIO_BEST_INDIVIDUALS):]
    
    
    
        replayGameStates, replayStacks = initRandomGames(N_HANDS_FOR_RE_EVAL)
        replayFinalGameStates = playGamesParallel(replayGameStates, models[bestIdx], N_CORES, WIN_LEN, 
                                                  RND_AGENT_IDX, AI_AGENT_IDX)
        assert len(finalGameStates) == POPULATION_SIZE
        replayModelWinAmounts = getWinAmountsForModels(replayFinalGameStates, replayStacks, AI_AGENT_IDX)
#        replayModelWinAmounts = optimizeWinAmounts(replayModelWinAmounts)
#        replayModelFitness = [(np.mean(np.concatenate((amounts,amounts2)))) + \
#                              (np.sum(~(np.isclose(np.concatenate((amounts,amounts2)),-1) | \
#                                        np.isclose(np.concatenate((amounts,amounts2)),-2)))/ \
#                                            len(np.concatenate((amounts,amounts2))))
        replayModelFitness = [np.mean(np.concatenate((amounts,amounts2))) \
#        replayModelFitness = [np.mean(np.concatenate((amounts,amounts2)))/ \
#                np.std(np.concatenate((amounts,amounts2))) \
            for amounts,amounts2 in zip(replayModelWinAmounts,modelWinAmounts)]
#        print(np.argsort(replayModelFitness))
        bestIdx = bestIdx[np.argsort(replayModelFitness)]

        [np.mean(np.concatenate((amounts,amounts2))) \
             for amounts,amounts2 in zip(replayModelWinAmounts,modelWinAmounts)]
        
        populationFitness.append(np.mean(modelFitness))
        bestFitness.append(np.max(replayModelFitness))
        
        print('................................')
        print(k, np.mean(modelFitness), np.max(replayModelFitness))
#        print(k, np.mean([np.mean(amounts) for amounts in modelWinAmounts]), 
#              np.max([np.mean(np.concatenate((amounts,amounts2))) \
#                      for amounts,amounts2 in zip(replayModelWinAmounts,modelWinAmounts)]))
#        print(np.argsort(replayModelFitness))
        
    
        # Save data
#        [tf.keras.models.save_model(model, 'data/models/'+str(i)) for i,model in enumerate(models)]
#        np.save('data/'+str(k)+'_win_amounts', modelWinAmounts)
    #    m = tf.keras.models.load_model('aa.aa')    # This is how to load, just a note
    
        # Put the best individual without mutation to the next generation
        nextGeneration = []
        nextGeneration = [models[idx] for idx in bestIdx[-3:]]
        
        # Mutate
        for i in range(POPULATION_SIZE-len(nextGeneration)):
            idx = bestIdx[np.random.randint(len(bestIdx))]
            
            model = copy.deepcopy(models[idx])
            model.mutate(MUTATION_SIGMA, ratio=MUTATION_RATIO)
            
#            weights = model.get_weights()
#            weightsUpdated = [w + np.random.normal(scale=MUTATION_SIGMA, size=w.shape) for w in weights]
#            model.set_weights(weightsUpdated)
            
            nextGeneration.append(model)
        
        models = np.array(nextGeneration)
            
    
    n = 5
    plt.plot(populationFitness[n:])
    plt.plot(bestFitness[n:])
    
    
    # %%
    
