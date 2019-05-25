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

from holdem_hu_bot.agents import RndAgent, Agent
from holdem_hu_bot.game_data_container import GameDataContainer
from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute




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
    
    # Save also last game state
    actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
    gameDataContainer.addData(gameStates, actionsToExecute)
    
    return gameDataContainer


def getWinAmounts(gameContainer, initStacks):
    data, _ = gameContainer.getData()
    lastIndexes, _ = gameContainer.getLastIndexes()
    finalStacks = data['playersData'][lastIndexes][:,[2,10]]
    
    return finalStacks - initStacks


# %%

nGames = 10

gameStates, initStacks = initRandomGames(nGames, seed=767)


gameCont = GameDataContainer(nGames)

#agents = [RndAgent(0), RndAgent(1, seed=11)]
agents = [RndAgent(0), RndAgent(1)]

gameContainers = [playGames(agents, copy.deepcopy(gameStates), copy.deepcopy(gameCont)) for i in range(2)]

winAmounts = getWinAmounts(gameContainers[1], initStacks)

print(winAmounts)



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







