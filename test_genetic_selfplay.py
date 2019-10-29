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

sys.path.append('/home/juho/dev_folder/')
from holdem_hu_bot.agents import generateRndActions
from holdem_hu_bot.game_data_container import GameDataContainer
from holdem_hu_bot.common_stuff import suitsOnehotLut, ranksOnehotLut
from texas_hu_engine.wrappers import executeActions, createActionsToExecute, GameState
from texas_hu_engine.engine_numba import initGame



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
 
    
def playGames(gameStates, models, WIN_LEN):
    features = np.zeros((len(gameStates.boards), 7, WIN_LEN+17))
    actionsToExecute = np.zeros((len(gameStates.boards),2), dtype=np.int64) - 999
    
    while(1):
        
        actionsForAgents, masksForAgents = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
        for idxAgent, model in enumerate(models):
            maskAgent, _, _, _, = getMasks(gameStates, idxAgent)
            
            features = computeFeaturesNb(gameStates.boards, gameStates.players, WIN_LEN, maskAgent, features)
            featuresScaled = scaler(features[maskAgent], WIN_LEN)
            featuresScaled = featuresScaled.reshape((len(featuresScaled), 
                                                     featuresScaled.shape[1]*featuresScaled.shape[2]))
        
            # Calculate model outputs
            with torch.no_grad():
                featuresScaled = torch.from_numpy(featuresScaled).float()
                modelOutput = model(featuresScaled).numpy()
            if(len(modelOutput) == 0):
                modelOutput = np.zeros((0,10), dtype=np.float)
        
            # Convert model outputs into actions
            smallBlinds = gameStates.boards[maskAgent,1]
            pots = features[:, 4, WIN_LEN-1][maskAgent]
            pots = (pots * smallBlinds).astype(np.int)
            availableActions = gameStates.availableActions[maskAgent]
            agentActions = modelOutputsToActions(modelOutput, pots, availableActions)
            
            actionsForAgents[idxAgent] = agentActions
            masksForAgents[idxAgent] = maskAgent
        
        # Put actions from models together
        actionsToExecute[:] = -999
        actionsToExecute[masksForAgents[0]] = actionsForAgents[0]
        actionsToExecute[masksForAgents[1]] = actionsForAgents[1]
        
        # Feed actions into the game engine
        gameStates = executeActions(gameStates, actionsToExecute)
    
        # Termination criteria
        nValidGames = np.sum(gameStates.controlVariables[:,1]==0)
#        print(nValidGames)
        if(nValidGames == 0):
            break
        
    # Check that all games were succesful
    assert np.all(~(gameStates.controlVariables[:,1] == -999))
    
    return gameStates

    
def playGamesWrapper(gameStates, models, WIN_LEN, idx):
    return (playGames(gameStates, models, WIN_LEN), idx)


def playGamesParallel(initGameStates, modelsPopulation, modelOpponent, nCores, winLen):
    pool = mp.Pool(nCores)
    result_objects = [pool.apply_async(playGamesWrapper, args=(copy.deepcopy(initGameStates), 
                                                               (modelsPopulation[i], modelOpponent), winLen, i)) 
                                                            for i in range(len(modelsPopulation))]
    results = [r.get() for r in result_objects]

    # Sort results because it is not quaranteed that apply_async returns them in correct order
    orderNum = np.array([res[1] for res in results])
    sorter = np.argsort(orderNum)
    finalGameStates = np.array([res[0] for res in results])
    
    pool.close()

    return finalGameStates[sorter]


def initRandomGames(nGames, seed=-1):
    boards, players, controlVariables, availableActions, initStacks = initGamesWrapper(nGames, seed=seed)
    
    return GameState(boards, players, controlVariables, availableActions), initStacks


#@jit(nopython=True, fastmath=True)
def initGamesWrapper(nGames, seed=-1):
    if(seed != -1):
        np.random.seed(seed)
    
    boardsArr = np.zeros((nGames*2, 13), dtype=np.int32)
    playersArr = np.zeros((nGames*2*2, 8), dtype=np.int32)
    controlVariablesArr = np.zeros((nGames*2, 3), dtype=np.int16)
    availableActionsArr = np.zeros((nGames*2, 3), dtype=np.int64)
    initStacksArr = np.zeros((nGames*2, 2), dtype=np.int64)
    
    for i in range(nGames):
        tmpCards = np.random.choice(52, size=9, replace=0)
        boardCards = tmpCards[:5]
        holeCards = np.zeros((2,2), dtype=np.int64)
        holeCards[0,:] = tmpCards[5:7]
        holeCards[1,:] = tmpCards[7:]
        smallBlindPlayerIdx = np.random.randint(0,high=2)
        smallBlindAmount = np.random.randint(1,high=100)
        initStacks = np.array([np.random.randint(smallBlindAmount*2, high=smallBlindAmount*400 + \
                                                 np.random.randint(smallBlindAmount)), 
                               np.random.randint(smallBlindAmount*2, high=smallBlindAmount*400 + \
                                                 np.random.randint(smallBlindAmount))])
        
        board, players, controlVariables, availableActions = initGame(boardCards, smallBlindPlayerIdx, 
                                                                      smallBlindAmount, initStacks.copy(), 
                                                                      holeCards)
        boardsArr[i,:] = board
        playersArr[i*2:i*2+2,:] = players
        controlVariablesArr[i,:] = controlVariables
        availableActionsArr[i,:] = availableActions
        initStacksArr[i,:] = initStacks
        
        # Flip players
        initStacks = initStacks[[1,0]]
        holeCards = holeCards[[1,0]]
        board, players, controlVariables, availableActions = initGame(boardCards, np.abs(smallBlindPlayerIdx-1), 
                                                                      smallBlindAmount, initStacks.copy(), 
                                                                      holeCards)
        boardsArr[(nGames+i),:] = board
        playersArr[(nGames*2 + i*2):(nGames*2 + (i*2+2)),:] = players
        controlVariablesArr[(nGames+i),:] = controlVariables
        availableActionsArr[(nGames+i),:] = availableActions
        initStacksArr[(nGames+i),:] = initStacks
        
    return boardsArr, playersArr, controlVariablesArr, availableActionsArr, initStacksArr
    

class AiModel(nn.Module):
    def __init__(self, winLen):
        super(AiModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(7*(winLen+17), 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
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

    def mutate(self, weights, sigma, ratio):
        w = weights.data.numpy().reshape(-1)
        rndIdx = np.random.choice(len(w), size=max(1,int(ratio*len(w))), replace=0)
        w[rndIdx] += np.random.normal(scale=sigma, size=len(rndIdx))

    def mutateWeightsAllLayers(self, sigma, ratio):
        for i in range(len(self.weights)):
            self.mutate(self.weights[i], sigma, ratio)

    def mutateBiasesAllLayers(self, sigma, ratio):
        for i in range(len(self.biases)):
            self.mutate(self.biases[i], sigma, ratio)

    def mutateWeightsOneLayer(self, sigma, layerIdx, ratio):
        self.mutate(self.weights[layerIdx], sigma, ratio)

    def mutateBiasesOneLayer(self, sigma, layerIdx, ratio):
        self.mutate(self.biases[layerIdx], sigma, ratio)
    
    def mutateOneLayer(self, sigma, ratio=1.0):
        layerIdx = np.random.randint(len(self.weights))
        self.mutateWeightsOneLayer(sigma, layerIdx, ratio)
        self.mutateBiasesOneLayer(sigma, layerIdx, ratio)
        
    def mutateAllLayers(self, sigma, ratio=1.0):
        self.mutateWeightsAllLayers(sigma, ratio=ratio)
        self.mutateBiasesAllLayers(sigma, ratio=ratio)


class Population():
    def __init__(self, size, winLen):
        device = torch.device('cpu')
        
        models = []
        for i in range(size):
            models.append(AiModel(winLen).to(device))
        
        self.models = np.array(models)
        self.bestModel = models[np.random.randint(len(models))]


#if __name__ == "__main__":
    
    # %%
    
    # Initialize agent
    
    #SEED = 123
    N_CORES = 6
    
    N_POPULATIONS = 3
    POPULATION_SIZE = 100
    RATIO_BEST_INDIVIDUALS = 0.10
    MUTATION_SIGMA = 1.0e-4
    MUTATION_RATIO = 1.0
    
    N_HANDS_FOR_EVAL = 20000
#    N_HANDS_FOR_RE_EVAL = 30000
    N_RND_PLAYS_PER_HAND = 1
    
    WIN_LEN = 2
    
    
    device = torch.device('cpu')
    
    populations = np.array([Population(POPULATION_SIZE, WIN_LEN) for _ in range(N_POPULATIONS)])
    


        
# %%
    

    initGameStates, initStacks = initRandomGames(N_HANDS_FOR_EVAL)

    # %%
    
    import time
    
    t1 = time.time()
    
    populationFitness, bestFitness = [], []
    

    
    popIdx = 0
    
    curPopulation = populations[popIdx]
    opponentPopulations = populations[np.delete(np.arange(len(populations)), popIdx)]
    opponentModels = np.array([pop.bestModel for pop in opponentPopulations])
    
    N_OPTIMIZATION_ITERS = 1
    for k in range(N_OPTIMIZATION_ITERS):
        
        
        
#        states, stacks = initRandomGames(int(N_HANDS_FOR_EVAL*0.25))
#        rndIdx = np.random.choice(N_HANDS_FOR_EVAL, size=len(stacks), replace=0)
#        
#        initStacks[rndIdx] = stacks
#        
#        initGameStates.availableActions[rndIdx] = states.availableActions
#        initGameStates.boards[rndIdx] = states.boards
#        initGameStates.controlVariables[rndIdx] = states.controlVariables
#        rndIdx2 = np.repeat(rndIdx*2, 2)
#        rndIdx2[1::2] = rndIdx*2+1
#        initGameStates.players[rndIdx2] = states.players
        
        
        
        # Play games 
        modelWinAmounts = []
        for opponentModel in opponentModels:
            finalGameStates = playGamesParallel(initGameStates, curPopulation.models, opponentModel, 
                                                N_CORES, WIN_LEN)
            assert len(finalGameStates) == POPULATION_SIZE
            modelWinAmounts.append(getWinAmountsForModels(finalGameStates, initStacks, 0))
        modelWinAmounts = np.column_stack((modelWinAmounts))
        
        modelFitness = np.mean(modelWinAmounts,1)
        sorter = np.argsort(modelFitness)
        bestIdx = sorter[-int(len(sorter)*RATIO_BEST_INDIVIDUALS):]
        
        curPopulation.bestModel = curPopulation.models[bestIdx[-1]]
    
        populationFitness.append(np.mean(modelFitness))
        bestFitness.append(np.max(modelFitness))        
        print(k, np.mean(modelFitness), np.max(modelFitness))

        # If last round skip mutation because we want to know which one is the best model in the current
        # population
        if(k == N_OPTIMIZATION_ITERS-1):
            break
        
#        replayGameStates, replayStacks = initRandomGames(N_HANDS_FOR_RE_EVAL)
#        replayFinalGameStates = playGamesParallel(replayGameStates, models[bestIdx], opponent, N_CORES, WIN_LEN)
#        assert len(replayFinalGameStates) == len(bestIdx)
#        replayModelWinAmounts = getWinAmountsForModels(replayFinalGameStates, replayStacks, 0)
#        replayModelFitness = [np.mean(np.concatenate((amounts,amounts2))) \
#            for amounts,amounts2 in zip(replayModelWinAmounts,modelWinAmounts)]
##        print(np.argsort(replayModelFitness))
#        bestIdx = bestIdx[np.argsort(replayModelFitness)]
#        
#        populationFitness.append(np.mean(modelFitness))
#        bestFitness.append(np.max(replayModelFitness))        
#        print(k, np.mean(modelFitness), np.max(replayModelFitness))

    
        # Save data
    
        # Put the best individual without mutation to the next generation
        nextGeneration = []
        nextGeneration = [curPopulation.models[idx] for idx in bestIdx[-3:]]
        
        # Mutate
        for i in range(POPULATION_SIZE-len(nextGeneration)):
            idx = bestIdx[np.random.randint(len(bestIdx))]
            
            model = copy.deepcopy(curPopulation.models[idx])
            model.mutateAllLayers(MUTATION_SIGMA, ratio=MUTATION_RATIO)
#            model.mutateOneLayer(MUTATION_SIGMA, ratio=MUTATION_RATIO)
            
            
            nextGeneration.append(model)
        
        curPopulation.models = np.array(nextGeneration)
            
    
#    n = 0
#    plt.plot(populationFitness[n:])
#    plt.plot(bestFitness[n:])
        
    
    print(time.time() - t1)
    
    
    # %%
    
