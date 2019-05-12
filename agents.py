#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:14:19 2019

@author: juho
"""

import numpy as np
from numba import jit
from abc import ABC, abstractmethod

from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute
from holdem_hu_bot.features_rf import RfFeatures


class Agent(ABC):
    
    def __init__(self, playerNumber):
        self.playerNumber = playerNumber
    
    @abstractmethod
    def getActions(self, gameStates):
        pass
    
        
@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def generateRndActions(availableActions):
    foldProb = 0.1
    foldProbArr = np.zeros(100, dtype=np.bool_)
#    foldProbArr[:int(foldProb*100)] = True     # TODO: uncomment
    
    allInRaiseProb = 0.1
    allInRaiseProbArr = np.zeros(100, dtype=np.bool_)
    allInRaiseProbArr[:int(allInRaiseProb*100)] = True
    
    amounts = np.zeros(len(availableActions), dtype=np.int64)
    for i in range(len(availableActions)):
        curActions = availableActions[i]
        
        # Fold ?
        isFold = foldProbArr[np.random.randint(len(foldProbArr))]
        if(isFold):
            amounts[i] = -1
            continue
        
        # Call or raise ?
        callOrRaise = np.random.randint(np.sum(curActions[:2] > -1))  # 0 = call, 1 = raise
        if(callOrRaise == 0):   # Call
            amounts[i] = curActions[callOrRaise]
            continue
        if(callOrRaise == 1):   # Raise
            isAllInRaise = allInRaiseProbArr[np.random.randint(len(allInRaiseProbArr))]
            if(isAllInRaise):
                amounts[i] = curActions[-1]
                continue
            
            lowRaise, highRaise = curActions[1], curActions[2]
            amounts[i] = np.random.randint(lowRaise, highRaise+1)
            
    return createActionsToExecute(amounts)


class RndAgent(Agent):
    
    def getActions(self, gameStates):
        playerNum = self.playerNumber
        
        actingPlayerNum = np.argmax(gameStates.players[:,6].reshape((-1,2)),1)
        playerMask = actingPlayerNum == playerNum
        gameEndMask = gameStates.controlVariables[:,1] == 1
        gameFailedMask = gameStates.controlVariables[:,1] == -999
        mask = playerMask & ~gameEndMask & ~gameFailedMask
        
        availableActions = gameStates.availableActions[mask]
        
        return generateRndActions(availableActions), mask



#
## %%
#
#agents = [RndAgent(0), RndAgent(1)]
#
#N = 10
#rfFeatures = RfFeatures(N)
#
#gameStates = initRandomGames(N)
#rfFeatures.addData(gameStates)
#
#
## %%
#
#
#actionsAgent0, maskAgent0 = agents[0].getActions(gameStates)
#actionsAgent1, maskAgent1 = agents[1].getActions(gameStates)
#
#actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
#actionsToExecute[maskAgent0] = actionsAgent0
#actionsToExecute[maskAgent1] = actionsAgent1
#
#gameStates = executeActions(gameStates, actionsToExecute)
#
#rfFeatures.addData(gameStates)
#
#rfFeatures.getFeatures()
#
#print(np.sum(gameStates.controlVariables[:,1]==0))
#
##gameStates.controlVariables
##np.argmax(gameStates.players[:,6].reshape((-1,2)),1)
#
#
## %%



  
    
    
    

  
    
    
    
    
