#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:08:15 2019

@author: juho
"""



import numpy as np
from numba import jit
import copy
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from holdem_hu_bot.agents import RndAgent, CallAgent, Agent
from holdem_hu_bot.game_data_container import GameDataContainer
from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute
from equity_calculator.equity_calculator import computeEquities




def playGames(agents, gameStates, gameDataContainer):

    while(1):
        actionsAgent0, maskAgent0 = agents[0].getActions(gameStates)
        actionsAgent1, maskAgent1 = agents[1].getActions(gameStates)
        
        actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
        actionsToExecute[maskAgent0] = actionsAgent0
        actionsToExecute[maskAgent1] = actionsAgent1
        
        gameDataContainer.addData(gameStates, actionsToExecute)
    
        gameStates = executeActions(gameStates, actionsToExecute)
        
        nValidGames = np.sum(gameStates.controlVariables[:,1]==0)
        print(nValidGames)
        
        if(nValidGames == 0):
            break
    
    # Save also the last game state
    actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
    gameDataContainer.addData(gameStates, actionsToExecute)
    
    return gameDataContainer


def getWinAmounts(gameContainer, initStacks):
    data, _ = gameContainer.getData()
    lastIndexes, _ = gameContainer.getLastIndexes()
    finalStacks = data['playersData'][lastIndexes][:,[2,10]]
    
    return finalStacks - initStacks


def getEquities(gameStates):
    boardCards = initGameStates.boards[:,8:]
    player0HoleCards = initGameStates.players[::2,:2]
    player1HoleCards = initGameStates.players[1::2,:2]
    
    # Dimensions, 1: game number, 2: preflop/flop/turn/river, 3: player index
    equities = np.zeros((len(initGameStates.boards),4,2))
    equities[:,0,0], equities[:,1,0], equities[:,2,0], equities[:,3,0] = computeEquities(player0HoleCards,
            boardCards)
    equities[:,0,1], equities[:,1,1], equities[:,2,1], equities[:,3,1] = computeEquities(player1HoleCards,
            boardCards)

    return equities


def computeFeatures(boardsData, playersData, availableActions, controlVariables, equities,
                    gameNumbers):
#    boardsData = initGameStates.boards
#    playersData = initGameStates.players
#    availableActions = initGameStates.availableActions
#    controlVariables = initGameStates.controlVariables
#    equities = equities
#    gameNumbers = np.arange(len(boardsData))
    
    smallBlinds = boardsData[:,1] # Small blinds amounts are used for normalization
    actingPlayerIdx = playersData[1::2,6]
    nonActingPlayerIdx = (~actingPlayerIdx.astype(np.bool)).astype(np.int)
    
    isPlayerSmallBlind = np.column_stack((playersData[::2,4],playersData[1::2,4]))
    isPlayerSmallBlind = isPlayerSmallBlind[np.arange(len(actingPlayerIdx)), actingPlayerIdx]
    
    # Pots, stacks etc. money stuffs
    pots = boardsData[:,0]
    bets = playersData[::2,3] + playersData[1::2,3]
    pots = pots + bets
    stacks = np.column_stack((playersData[::2,2], playersData[1::2,2]))
    
    # Normalized pots and stacks
    potsNormalized = pots / smallBlinds
    stacksNormalized = stacks / smallBlinds.reshape((-1,1))
    ownStacksNormalized = stacksNormalized[np.arange(len(stacks)), actingPlayerIdx]
    opponentStacksNormalized = stacksNormalized[np.arange(len(stacks)), nonActingPlayerIdx]
    
    # Betting round
    visibleCardsMask = boardsData[:,3:8].astype(np.bool)
    bettingRound = visibleCardsMask[:,2:].astype(np.int)
    bettingRoundIdx = np.sum(bettingRound,1)
    
    # Equities
    actingPlayerEquities = equities[gameNumbers,bettingRoundIdx,actingPlayerIdx]
    
#    availableActionsNormalized = availableActions / smallBlinds
#    availableActionsNormalized[availableActions < 0] = -1
    
#    gameFinishedMask = controlVariables[:,1] == 1   # Tells if the game has finished succesfully
#    gameFailedMask = controlVariables[:,1] == -999  # Tells if an error has occured
#
#    miscDict = {'availableActionsNormalized':availableActionsNormalized, 
#                'availableActions':availableActions, 'gameNumbers':gameNumbers,
#                'gameFinishedMask':gameFinishedMask, 'gameFailedMask':gameFailedMask, 
#                'actingPlayerIdx':actingPlayerIdx}
    
    features = np.column_stack((isPlayerSmallBlind, bettingRoundIdx, actingPlayerEquities, 
                                potsNormalized, ownStacksNormalized, opponentStacksNormalized))

    return features#, miscDict
    

# %%
    


class AiAgent(Agent):
    
    def __init__(self, playerNumber, featuresFunc, aiModel, equities):
        self.playerNumber = playerNumber
        self.featuresFunc = featuresFunc
        self.aiModel = aiModel
        self.equities = equities
    
    def getActions(self, gameStates):
        mask, _, _, _ = super().getMasks(gameStates)
        aiModel = self.aiModel
        featuresFunc = self.featuresFunc
        equities = self.equities
        
        # Return if all hands are already finished for the current player
        if(np.sum(mask) == 0):
            return np.zeros((0,2), dtype=np.int64), mask

        # %%

        def getMasks(gameStates, playerNum):
#            playerNum = self.playerNumber
            
            actingPlayerNum = np.argmax(gameStates.players[:,6].reshape((-1,2)),1)
            actingPlayerMask = actingPlayerNum == playerNum
            gameEndMask = gameStates.controlVariables[:,1] == 1
            gameFailedMask = gameStates.controlVariables[:,1] == -999
            allMask = actingPlayerMask & ~gameEndMask & ~gameFailedMask
            
            return allMask, actingPlayerMask, gameEndMask, gameFailedMask
        
        gameStates = initGameStates
        playerNum = 0
        mask, _, _, _ = getMasks(gameStates, playerNum)
        playersMask = np.tile(mask, (2,1)).T.flatten()
        featuresFunc = computeFeatures
        equities = equities
        aiModel = adafasfdagasdf
        
        
        playersData = gameStates.players[playersMask]
        boardsData = gameStates.boards[mask]
        availableActs = gameStates.availableActions[mask]
        gameNums = np.nonzero(mask)[0]
        
        features = featuresFunc(boardsData, playersData, availableActs, gameStates.controlVariables[mask],
                                equities, gameNums)
        
        
        
#        return actionsADAFASDfASDF, allMask



# %%

nGames = 7
callPlayerIdx = 0
rndPlayerIdx = 1

initGameStates, initStacks = initRandomGames(nGames, seed=767)
equities = getEquities(initGameStates)

gameCont = GameDataContainer(nGames)

#agents = [RndAgent(0), RndAgent(1, seed=np.random.randint(1,10000))]
#agents = [RndAgent(0), RndAgent(1)]
#agents = [CallAgent(0), RndAgent(1, seed=np.random.randint(1,10000))]
agents = [CallAgent(callPlayerIdx), RndAgent(rndPlayerIdx)]

gameContainers = [playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(gameCont)) for i in range(5)]

# %%

#winAmounts = getWinAmounts(gameContainers[0], initStacks)
#print(winAmounts)
winAmounts = [getWinAmounts(c, initStacks)[:,rndPlayerIdx] for c in gameContainers]
winAmounts = np.column_stack((winAmounts))

highestReturnIdx = np.argmax(winAmounts,1)


print(np.sum(np.max(winAmounts, 1)))
np.max(winAmounts, 1)

#initGameStates.boards[:,1]





# %%

data, _ = gameContainers[0].getData()
data1, _ = gameContainers[1].getData()

i, _ = gameContainers[0].getFirstIndexes()
i1, _ = gameContainers[1].getFirstIndexes()
#i, _ = gameContainers[0].getLastIndexes()
#i1, _ = gameContainers[1].getLastIndexes()

assert np.all(data['playersData'][i] == data1['playersData'][i1])
assert np.all(data['boardsData'][i] == data1['boardsData'][i1])
assert np.all(data['availableActionsData'][i] == data1['availableActionsData'][i1])
assert np.all(data['controlVariablesData'][i] == data1['controlVariablesData'][i1])


# %%







