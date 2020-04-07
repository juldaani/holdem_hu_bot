#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:08:15 2019

@author: juho
"""



import numpy as np
from numba import jit, njit, prange
import matplotlib.pyplot as plt
import sys, os, copy, shutil
import multiprocessing as mp
from datetime import datetime

import torch
torch.set_num_threads(1)
import torch.nn as nn

#sys.path.append('/home/juho/dev_folder/asdf/')
from holdem_hu_bot.common_stuff import suitsOnehotLut, ranksOnehotLut
from texas_hu_engine.wrappers import executeActions, createActionsToExecute, GameState
from texas_hu_engine.engine_numba import initGame
from holdem_hu_bot.agents import generateRndActions




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


def getWinAmountsForAgents(gameStates, initStacks, agentIdx):
    winAmounts = []
    for states in gameStates:
        smallBlinds = states.boards[:,1]
        tmpWinAmounts = getWinAmounts(states, initStacks)
        assert np.sum(tmpWinAmounts) == 0  # Check that games are zero-sum
        tmpWinAmounts = tmpWinAmounts[:,agentIdx] / smallBlinds
        winAmounts.append(tmpWinAmounts)
        
    return winAmounts


def optimizeWinAmounts(modelWinAmounts):
    for i in range(len(modelWinAmounts)):
        winAmounts = modelWinAmounts[i]
        m = winAmounts > 0
        winAmounts[m] = 3 + np.random.rand(np.sum(m)) * (winAmounts[m]-3)
        
    return modelWinAmounts
 
    
def totalWinRatesForPopulations(popVsPopEvalRes, popIdxEval):
    nPopulations = len(np.unique(popIdxEval[:,0]))
    sorter = np.argsort(popIdxEval[:,0])
    res = popVsPopEvalRes[sorter].reshape((-1,nPopulations))   # Rows contain results # for each population
    
    return np.sum(res,1)

    
def playGames(gameStates, agents, WIN_LEN):
    features = np.zeros((len(gameStates.boards), 7, WIN_LEN+17))
    actionsToExecute = np.zeros((len(gameStates.boards),2), dtype=np.int64) - 999
    
    while(1):
        
        actionsForAgents, masksForAgents = [[] for _ in range(len(agents))], [[] for _ in range(len(agents))]
        for idxAgent, agent in enumerate(agents):
            agentActions, maskAgent, features = agent.getActions(gameStates, idxAgent, features, WIN_LEN)

            actionsForAgents[idxAgent] = agentActions
            masksForAgents[idxAgent] = maskAgent
        
        # Put actions together
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

    
def playGamesWrapper(gameStates, agents, WIN_LEN, idx):
    torch.set_num_threads(1)
    return (playGames(gameStates, agents, WIN_LEN), idx)


def playGamesParallel(pool, gameStates, initStacks, agents1, agents2, winLen):
    result_objects = [pool.apply_async(playGamesWrapper, args=(copy.deepcopy(gameStates), 
                                                               (copy.deepcopy(agents1[i]), 
                                                                copy.deepcopy(agents2[i])), winLen, i)) 
                                                            for i in range(len(agents1))]
    results = [r.get() for r in result_objects]

    # Sort results because it is not quaranteed that apply_async returns them in correct order
    orderNum = np.array([res[1] for res in results])
    sorter = np.argsort(orderNum)
    finalGameStates = np.array([res[0] for res in results])
    finalGameStates = finalGameStates[sorter]
    
    assert len(finalGameStates) == len(agents1)
    
    winAmounts = getWinAmountsForAgents(finalGameStates, initStacks, 0)
    meanWinAmounts = [np.mean(amounts) for amounts in winAmounts]

    return finalGameStates, meanWinAmounts, winAmounts


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
    

def evaluatePopulations(populations, gameStates, initStacks, WIN_LEN, pool):
    popIdxEval = np.column_stack((np.triu_indices(len(populations), k=0)))
    agents1 = [populations[idx].bestAgent for idx in popIdxEval[:,0]]
    agents2 = [populations[idx].bestAgent for idx in popIdxEval[:,1]]
    finalGameStates, meanWinAmounts, winAmounts = playGamesParallel(pool, gameStates, initStacks, 
                                                                    agents1, agents2, WIN_LEN)
    m = ~(popIdxEval[:,0] == popIdxEval[:,1])   # Mask for substracting games against itself
    meanWinAmounts = np.concatenate((meanWinAmounts,np.array(meanWinAmounts)[m]*-1))
    popIdxEval = np.row_stack((popIdxEval, np.roll(popIdxEval[m], shift=1, axis=1)))

    return finalGameStates, meanWinAmounts, winAmounts, popIdxEval


def evaluateAgainstOpponents(populations, gameStates, initStacks, opponents, WIN_LEN, pool):  
    res = {}
    for key, opponent in zip(opponents.keys(), opponents.values()):
        agents = [pop.bestAgent for pop in populations]
        opponentAgents = [opponent for _ in agents]
        
        finalGameStates, meanWinAmounts, winAmounts = playGamesParallel(pool, gameStates, initStacks, 
                                                                        agents, opponentAgents, WIN_LEN)

        res[key] = meanWinAmounts
        
    return res


def ftnessForAgentsInPopulation(population, opponentPopulations, gameStates, initStacks, WIN_LEN, 
                                pool):
    winAmounts = []
    
    for opponentPop in opponentPopulations:
        agentsCurPopulation = population.agents
        agentsOpponent = [opponentPop.bestAgent for _ in range(len(agentsCurPopulation))]
        _, _, tmpWinAmounts = playGamesParallel(pool, gameStates, initStacks, agentsCurPopulation, 
                                                agentsOpponent, WIN_LEN)
        winAmounts.append(tmpWinAmounts)
        
    return np.mean(np.column_stack((winAmounts)),1)


def saveCheckpoint(populations, pastBestAgents, params, path, iteration):
    dictToSave = {'params': params,
                  'populations': [{'models':[agent.model.state_dict() for agent in pop.agents],
                                   'best_model':pop.bestAgent.model.state_dict()} 
                                  for pop in populations],
                  'past_best_agents': {key:pastBestAgents[key].model.state_dict() 
                                       for key in pastBestAgents.keys()}
                  }
    
    torch.save(dictToSave, os.path.join(path, str(iteration)+'_models_for_populations.tar'))


def loadCheckpoint(path):
    checkpoint = torch.load(path)
    params = checkpoint['params']
    populations = np.array([Population(params['POPULATION_SIZE'], params['WIN_LEN']) 
                            for _ in range(params['N_POPULATIONS'])])
    
    # Load models for populations
    for pop, popCheckpoint in zip(populations,checkpoint['populations']):
        for modelCheckpoint, agent in zip(popCheckpoint['models'], pop.agents):
            agent.model.load_state_dict(modelCheckpoint)
        pop.bestAgent.model.load_state_dict(popCheckpoint['best_model'])
    
    populations[0].agents[4].model
    
    # Load models for past best agents
    pastBestAgents = {}
    for key in checkpoint['past_best_agents'].keys():
        tmpModel = AiModel(params['WIN_LEN'])
        tmpModel.load_state_dict(checkpoint['past_best_agents'][key])
        pastBestAgents[key] = AiAgent(tmpModel)
    checkpoint['past_best_agents'].keys()
    
    return populations, pastBestAgents, params


def wrapAgentIntoPopulation(agent, winLen):
    tmpPop = Population(1, winLen)
    tmpPop.agents[0] = agent
    tmpPop.bestAgent = agent
    
    return tmpPop


def getPopulationsToOptimize(popVsPopEvalRes, popIdxEval, pastBestAgentEvalRes, pastBestAgents, 
                             populations, params):
    # Put past agent evaluation results into numpy array
    keys = pastBestAgentEvalRes.keys()
    if(not pastBestAgentEvalRes):
        pastAgentEval, pastAgentIdx = np.zeros(0), np.zeros((0,2), dtype=np.int)
    else:
        pastAgentEval = np.concatenate([pastBestAgentEvalRes[key] for key in keys])
        pastAgentIdx = np.row_stack(([np.column_stack((np.arange(len(pastBestAgentEvalRes[key])), 
                                         np.full(len(pastBestAgentEvalRes[key]), key))) for key in keys]))

    # Generate identifiers to be able to separate which games are population-vs-population and 
    # population-vs-past
    pastIdentifier = ['past'] * len(pastAgentEval)
    popIdentifier = ['pop'] * len(popVsPopEvalRes)
    
    # Stack pop-vs-pop and pop-vs-past
    evalRes = np.concatenate((pastAgentEval,popVsPopEvalRes))
    indexes = np.row_stack((pastAgentIdx,popIdxEval))
    identifiers = np.concatenate((pastIdentifier,popIdentifier))
    
    # Sort
    sorter = np.argsort(evalRes)
    evalRes, indexes, identifiers = evalRes[sorter], indexes[sorter], identifiers[sorter]            
    
    # Create list of populations (to be optimized) and their opponents
    populationsToOptimize, opponentPopulations = [], []
    for i in range(len(evalRes)):
        idxToOptimize = indexes[i,0]
        idxOpponent = indexes[i,1]
        
        if(identifiers[i] == 'pop'):
            populationsToOptimize.append(populations[idxToOptimize])
            opponentPopulations.append(populations[idxOpponent])
        if(identifiers[i] == 'past'):
            populationsToOptimize.append(populations[idxToOptimize])
            tmpPop = wrapAgentIntoPopulation(pastBestAgents[idxOpponent], params['WIN_LEN'])
            opponentPopulations.append(tmpPop)
    
    return populationsToOptimize, opponentPopulations, (evalRes, indexes, identifiers)


def printPopulationWinRates(popVsPopTotalWinRates):
    print('Population win rates: ')
    for i,val in enumerate(popVsPopTotalWinRates):
        print('%3d %5.3f' % (i,val))


class AiModel(nn.Module):
    def __init__(self, winLen):
        super(AiModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(7*(winLen+17), 50), nn.ReLU(),
            # nn.Linear(250, 250), nn.ReLU(),
            # nn.Linear(250, 250), nn.ReLU(),
            # nn.Linear(250, 250), nn.ReLU(),
            # nn.Linear(250, 250), nn.ReLU(),
            # nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(50, 10))
        
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
        
        agents = []
        for i in range(size):
            agents.append(AiAgent(AiModel(winLen).to(device)))
        
        self.agents = np.array(agents)
        self.bestAgent = agents[np.random.randint(len(agents))]


class AiAgent():
    def __init__(self, model):
        self.model = model
    
    def getActions(self, gameStates, idxAgent, features, winLen):
        maskAgent, _, _, _, = getMasks(gameStates, idxAgent)
        
        features = computeFeaturesNb(gameStates.boards, gameStates.players, winLen, maskAgent, features)
        featuresScaled = scaler(features[maskAgent], winLen)
        featuresScaled = featuresScaled.reshape((len(featuresScaled), 
                                                 featuresScaled.shape[1]*featuresScaled.shape[2]))
    
        # Calculate model outputs
        with torch.no_grad():
            featuresScaled = torch.from_numpy(featuresScaled).float()
            modelOutput = self.model(featuresScaled).numpy()
        if(len(modelOutput) == 0):
            modelOutput = np.zeros((0,10), dtype=np.float)
    
        # Convert model outputs into actions
        smallBlinds = gameStates.boards[maskAgent,1]
        pots = features[:, 4, winLen-1][maskAgent]
        pots = (pots * smallBlinds).astype(np.int)
        availableActions = gameStates.availableActions[maskAgent]
        agentActions = modelOutputsToActions(modelOutput, pots, availableActions)
    
        return agentActions, maskAgent, features


class FoldAgent():
    def __init__(self):
        pass
    
    def getActions(self, gameStates, idxAgent, features, winLen):
        maskAgent, _, _, _, = getMasks(gameStates, idxAgent)
        actionsAgent = np.zeros((np.sum(maskAgent),2))-1
        actionsAgent[:,0] = 1

        return actionsAgent, maskAgent, features


class CallAgent():
    def __init__(self):
        pass
    
    def getActions(self, gameStates, idxAgent, features, winLen):
        maskAgent, _, _, _, = getMasks(gameStates, idxAgent)
        actionsAgent = np.zeros((np.sum(maskAgent),2))-1
        actionsAgent[:,1] = gameStates.availableActions[maskAgent,0]   # Only call

        return actionsAgent, maskAgent, features

    
class MinRaiseAgent():
    def __init__(self):
        pass
    
    def getActions(self, gameStates, idxAgent, features, winLen):
        maskAgent, _, _, _, = getMasks(gameStates, idxAgent)
        availableActions = gameStates.availableActions[maskAgent]
        actionsAgent = np.zeros((np.sum(maskAgent),2))-1
        actionsAgent[:,1] = np.max(availableActions[:,:2], 1)   # Pick min raise if available, otherwise call

        return actionsAgent, maskAgent, features


class AllInAgent():
    def __init__(self):
        pass
    
    def getActions(self, gameStates, idxAgent, features, winLen):
        maskAgent, _, _, _, = getMasks(gameStates, idxAgent)
        availableActions = gameStates.availableActions[maskAgent]
        actionsAgent = np.zeros((np.sum(maskAgent),2))-1
        actionsAgent[:,1] = np.max(availableActions[:,[0,-1]], 1)   # Pick all-in if available, otherwise call

        return actionsAgent, maskAgent, features


    

#if __name__ == "__main__":
    
# %%
        
    
    # Parameters        
    # ........................................................................
    PATH_SAVE_RESULTS = '/home/juho/dev_folder/data/poker_ai'
    
    PATH_LOAD_CHECKPOINT = '' # Leave empty if fresh start
    CHECKPOINT_ITER = -1    # Use -1 to load the most recent checkpoint

    N_CORES = 20
    
    params = {
        'WIN_LEN': 2,
        
        'N_POPULATIONS': 1,
        'POPULATION_SIZE': 100,
        'RATIO_BEST_INDIVIDUALS': 0.10,
        # 'MUTATION_SIGMA': 1.0e-2,
        # 'MUTATION_RATIO': 0.25,
        'MUTATION_SIGMA': 2.5e-2,
        'MUTATION_RATIO': 0.01,
        
        'N_HANDS_FOR_EVAL': 50000,
        'N_HANDS_FOR_OPTIMIZATION': 20000,
        
        'N_ITERS_PICK_BEST_AGENT': 10
        
        #'N_OPTIMIZATIONS_BETWEEN_EVALS': 30
        # 'N_ITERS_GENERATE_NEW_HANDS': 10,
        # 'N_ITERS_BETWEEN_EVALS': 10,
        #'OPTIMIZATION_ITERS_PER_POPULATION': 2
    }
    # ........................................................................
    

    # Continue training from checkpoint
    if(len(PATH_LOAD_CHECKPOINT) > 0):
        pathResults = PATH_LOAD_CHECKPOINT
        pathEvalResults = os.path.join(pathResults, 'evaluation')
        pathPopulations = os.path.join(pathResults, 'populations')
        
        # List checkpoint files
        checkpointsFiles = np.array([f for f in os.listdir(pathPopulations)])
        iters = np.array([int(f.split('_')[0]) for f in checkpointsFiles])
        iters = iters[np.argsort(iters)]
        checkpointsFiles = checkpointsFiles[np.argsort(iters)]
        
        # Load latest checkpoint
        iteration = iters[CHECKPOINT_ITER]
        checkpointFile = checkpointsFiles[CHECKPOINT_ITER]
        
        # Load particular checkpoint
        if(CHECKPOINT_ITER != -1):
            idx = np.nonzero(CHECKPOINT_ITER == iters)[0]
            assert len(idx) == 1
            checkpointFile = checkpointsFiles[idx[0]]
            iteration = CHECKPOINT_ITER
            
        checkpointPath = os.path.join(pathPopulations, checkpointFile)
        populations, pastBestAgents, params = loadCheckpoint(checkpointPath)
        
        print('\n........................................................')
        print('Starting training from checkpoint: ' + checkpointPath)
        print('........................................................')
    
    
    # Fresh start for training
    else:
        # Create folder to save the results
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        pathResults = os.path.join(PATH_SAVE_RESULTS,time)
        pathEvalResults = os.path.join(pathResults, 'evaluation')
        pathPopulations = os.path.join(pathResults, 'populations')
        if not os.path.exists(pathResults):
            os.makedirs(pathResults)
        if not os.path.exists(pathEvalResults):
            os.makedirs(pathEvalResults)
        if not os.path.exists(pathPopulations):
            os.makedirs(pathPopulations)
    
        shutil.copy('test_genetic_selfplay.py', os.path.join(pathResults, 'test_genetic_selfplay2.py'))
        
        populations = np.array([Population(params['POPULATION_SIZE'], params['WIN_LEN']) 
                                for _ in range(params['N_POPULATIONS'])])
        
        pastBestAgents = {}
        
        iteration = 0
    
    
    # Dummy opponents serve as absolute reference for evaluation of populations
    DUMMY_OPPONENTS = {'fold_agent': FoldAgent(),
                       'call_agent': CallAgent(),
                       'min_raise_agent': MinRaiseAgent(),
                       'all_in_agent': AllInAgent()}
    
    pool = mp.Pool(N_CORES)
    





# %%
    
    while(1):
        print('\nIteration: ' + str(iteration))    
        print('..................................................')
    
        # Evaluate populations
        # ....................................................................
        
        # Create new fresh games for evaluation
        evalGameStates, evalStacks = initRandomGames(params['N_HANDS_FOR_EVAL'])
        
        # Evaluate populations against each other
        _, popVsPopEvalRes, _, popIdxEval = evaluatePopulations(populations, evalGameStates, evalStacks, 
                                                                params['WIN_LEN'], pool)

        # Compute total win rates for populations
        popVsPopTotalWinRates = totalWinRatesForPopulations(popVsPopEvalRes, popIdxEval)
        assert np.isclose(np.sum(popVsPopTotalWinRates), 0)     # Check that games are zero sum
        printPopulationWinRates(popVsPopTotalWinRates)
            
        # Evaluate populations against dummy opponents
        dummyEvalRes = evaluateAgainstOpponents(populations, evalGameStates, evalStacks, DUMMY_OPPONENTS, 
                                                params['WIN_LEN'], pool)

        # Evaluate populations against past best agents
        pastBestAgentEvalRes = evaluateAgainstOpponents(populations, evalGameStates, evalStacks, 
                                                        pastBestAgents, params['WIN_LEN'], pool)
        
        # Create list of populations (to be optimized) and their opponents
        populationsToOptimize, opponentPopulations, info = getPopulationsToOptimize(popVsPopEvalRes, popIdxEval, 
                                                                                    pastBestAgentEvalRes, 
                                                                                    pastBestAgents, populations, 
                                                                                    params)
        popEvalRes, popIdx, popId = info[0], info[1], info[2]
        
        # Pick the best agent and append to past best agents
        if(iteration % params['N_ITERS_PICK_BEST_AGENT'] == 0):
            bestAgent = populations[np.argmax(popVsPopTotalWinRates)].bestAgent
            pastBestAgents[iteration] = bestAgent
                
        # Save evaluation results
        np.save(os.path.join(pathEvalResults ,str(iteration)+'_eval_dummy_opponents'), dummyEvalRes)
        np.save(os.path.join(pathEvalResults ,str(iteration)+'_eval_past_best_agents'), pastBestAgentEvalRes)
        np.save(os.path.join(pathEvalResults ,str(iteration)+'_eval_population_vs_population'), 
                {'population_index':popIdxEval, 'res':popVsPopEvalRes})
        
        # Save models
        saveCheckpoint(populations, pastBestAgents, params, pathPopulations, iteration)
        
        # ....................................................................
        


        # samplingProbs = (popVsPopEvalRes*-1) + np.abs(np.min(popVsPopEvalRes*-1))
        # samplingProbs /= np.sum(samplingProbs)
        # idx = np.random.choice(np.arange(len(popIdxEval)), size=1, p=samplingProbs)[0]
        # curPopIdx, opponentPopIdx = popIdxEval[idx,0], popIdxEval[idx,1]
        # populationToOptimize, opponentPopulation = populations[curPopIdx], populations[opponentPopIdx]
        
        # for k in range(params['N_OPTIMIZATIONS_BETWEEN_EVALS']):
        # for k in range(len(populationsToOptimize)):
        for k in np.nonzero(popEvalRes < 0.05)[0][::-1]:
            populationToOptimize = populationsToOptimize[k]
            opponentPopulation = copy.deepcopy(opponentPopulations[k])
            
            print('\n==> ' + str(k) )
            print(popId[k], '| idx:  ' + str(popIdx[k]) + ' | init_res: % 5.3f' %((popEvalRes[k])*100))
            
            # Create new games
            # if(k % params['N_ITERS_GENERATE_NEW_HANDS'] == 0):
            initGameStates, initStacks = initRandomGames(params['N_HANDS_FOR_OPTIMIZATION'])
            
            optIter = 0
            while(1):
                # Get fitness for agents in the population
                agentFitness = ftnessForAgentsInPopulation(populationToOptimize, [opponentPopulation], 
                                                           initGameStates, initStacks, params['WIN_LEN'], pool)
    
                # Update the best agent in current population
                populationToOptimize.bestAgent = populationToOptimize.agents[np.argmax(agentFitness)]
    
                populationMeanFitness, populationMaxFitness = np.mean(agentFitness), np.max(agentFitness)
                
                print('% 3d % 5.3f % 5.3f' %(optIter, (populationMeanFitness*100), (populationMaxFitness*100)))
        
                # If max fitness of the population is above zero stop optimizing (mutation is skipped because we 
                # want to know which one is the best agent in the population)
                if((optIter >= 1 and populationMaxFitness > 0.05) or optIter > 40):
                    break
            
                # Put n best agents without mutation to the next generation
                nextGeneration = []
                sorter = np.argsort(agentFitness)
                bestIdx = sorter[-int(len(sorter)*params['RATIO_BEST_INDIVIDUALS']):]
                nextGeneration = [populationToOptimize.agents[idx] for idx in bestIdx]
                
                # Mutate
                for i in range(params['POPULATION_SIZE']-len(nextGeneration)):
                    idx = bestIdx[np.random.randint(len(bestIdx))]
                    
                    agent = copy.deepcopy(populationToOptimize.agents[idx])
                    # model.mutateAllLayers(MUTATION_SIGMA, ratio=MUTATION_RATIO)
                    agent.model.mutateOneLayer(params['MUTATION_SIGMA'], ratio=params['MUTATION_RATIO'])
                    
                    nextGeneration.append(agent)
                
                populationToOptimize.agents = np.array(nextGeneration)
                
                optIter += 1
                    
            
            
        print('')
        iteration += 1
        
        # n = 0
        # plt.plot(populationFitness[n:])
        # plt.plot(bestFitness[n:])
            
    
    
    # %%
    
    pool.close()
    
    
    # %%
    
