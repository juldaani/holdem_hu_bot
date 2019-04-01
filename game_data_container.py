#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:36:11 2019

@author: juho
"""

import numpy as np


class GameDataContainer:
    
    def __init__(self, nGames):
        self.__indexes = [[] for i in range(nGames)]
        self.__boardsData, self.__playersData, self.__controlVariablesData, \
            self.__availableActionsData = None, None, None, None
    
    def flattenPlayersData(self, playersData):
        flattenedData = np.zeros((int(len(playersData)/2), 
                                  playersData.shape[1]*2), dtype=playersData.dtype)
        flattenedData[:,:playersData.shape[1]] = playersData[::2]
        flattenedData[:,playersData.shape[1]:] = playersData[1::2]
        
        return flattenedData
        
    def addData(self, gameStates):
        validIndexes = np.nonzero(gameStates.validMask)[0]
        nDataPts = 0
        
        if(self.__boardsData is None):
            self.__boardsData = gameStates.boards[validIndexes]
            self.__playersData = self.flattenPlayersData(gameStates.players)[validIndexes]
            self.__controlVariablesData = gameStates.controlVariables[validIndexes]
            self.__availableActionsData = gameStates.availableActions[validIndexes]
        else:
            nDataPts = len(self.__boardsData)
            self.__boardsData = np.row_stack((self.__boardsData, gameStates.boards[validIndexes]))
            self.__playersData = np.row_stack((self.__playersData, 
                                               self.flattenPlayersData(
                                                     gameStates.players)[validIndexes]))
            self.__controlVariablesData = np.row_stack((self.__controlVariablesData, 
                                                      gameStates.controlVariables[validIndexes]))
            self.__availableActionsData = np.row_stack((self.__availableActionsData, 
                                                      gameStates.availableActions[validIndexes]))
        
        dataIndexes = np.arange(len(validIndexes)) + nDataPts
        for gameIdx,dataIdx in zip(validIndexes,dataIndexes):
            self.__indexes[gameIdx].append(dataIdx)
        
    def getIndexes(self): return self.__indexes
    
    def getData(self): return {'boardsData': self.__boardsData,
                               'playersData': self.__playersData,
                               'availableActionsData': self.__availableActionsData,
                               'controlVariablesData': self.__controlVariablesData}
    
    def setIndexes(self, indexes):
        self.__indexes = indexes
    
    def setData(self, data):
        self.__boardsData = data['boardsData']    
        self.__playersData = data['playersData']    
        self.__availableActionsData = data['availableActionsData']   
        self.__controlVariablesData = data['controlVariablesData']   


# %%
#
#from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute
#from texas_hu_engine.wrappers import GameState
#
#
#N = 300
#
#cont = GameDataContainer(N)
#states = initRandomGames(N)
#
#for i in range(100):
#    
#    tmp = np.arange(N).reshape((N,-1))
#    states.availableActions[:] = tmp
#    states.boards[:] = tmp
#    states.controlVariables[:] = tmp
#    states.players[:] = np.tile(tmp.flatten(), (2,1)).T.reshape((N*2,-1))
##    states.validMask = np.array([1,0,1])
##    states.validMask = np.array([1,0,0])
#    states.validMask = np.random.randint(0,2,N)
#    
#    cont.addData(states)
#
#
## %%
#    
#tmpData = cont.getData()
#tmpIdx = cont.getIndexes()
#
#tmpData['boardsData'][tmpIdx[81]]
#
#
#cont = GameDataContainer(1)
#cont.getIndexes()
#cont.getData()
#
#cont.setData(tmpData)
#cont.setIndexes(tmpIdx)
#
