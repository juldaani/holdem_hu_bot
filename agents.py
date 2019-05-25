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
    
    def getMasks(self, gameStates):
        playerNum = self.playerNumber
        
        actingPlayerNum = np.argmax(gameStates.players[:,6].reshape((-1,2)),1)
        actingPlayerMask = actingPlayerNum == playerNum
        gameEndMask = gameStates.controlVariables[:,1] == 1
        gameFailedMask = gameStates.controlVariables[:,1] == -999
        allMask = actingPlayerMask & ~gameEndMask & ~gameFailedMask
        
        return allMask, actingPlayerMask, gameEndMask, gameFailedMask
    
    @abstractmethod
    def getActions(self, gameStates):
        pass
    
    
        
@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def generateRndActions(availableActions, seed=-1):
    if(seed != -1):
        np.random.seed(seed)
    
    foldProb = 0.1
    foldProbArr = np.zeros(100, dtype=np.bool_)
    foldProbArr[:int(foldProb*100)] = True
    
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
    
    def __init__(self, playerNumber, seed=-1):
        self.playerNumber = playerNumber
        self.seed = seed
    
    def getActions(self, gameStates):
        allMask, actingPlayerMask, gameEndMask, gameFailedMask = super().getMasks(gameStates)
        availableActions = gameStates.availableActions[allMask]
        
        return generateRndActions(availableActions, seed=self.seed), allMask



  
    
    
    

  
    
    
    
    
