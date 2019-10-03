#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:36:11 2019

@author: juho
"""

import numpy as np
import time
import itertools


class GameDataContainer:
    
    def __init__(self, nGames):
        self.__indexes = [[] for i in range(nGames)]
        self.__boardsData, self.__playersData, self.__controlVariablesData, \
            self.__availableActionsData, self.__actions = None, None, None, None, None
    
    @staticmethod
    def flattenPlayersData(playersData):
        flattenedData = np.zeros((int(len(playersData)/2), 
                                  playersData.shape[1]*2), dtype=playersData.dtype)
        flattenedData[:,:playersData.shape[1]] = playersData[::2]
        flattenedData[:,playersData.shape[1]:] = playersData[1::2]
        
        return flattenedData
    
    @staticmethod
    def unflattenPlayersData(playersData):
        playersData2 = np.zeros((int(playersData.shape[0]*2),int(playersData.shape[1]/2)), 
                                dtype=playersData.dtype)
        playersData2[::2] = playersData[:,:8]
        playersData2[1::2] = playersData[:,8:]
        
        return playersData2
    
    
    def getLastIndexes(self):
        _, indexes = self.getData()
        # Get indexes for the last (the most recent) game states
        lastIndexes = [curIndexes[-1] for curIndexes in indexes]
        gameNumbers = [gameNum for gameNum in range(len(indexes))]
        
        return lastIndexes, gameNumbers
    
    
    def getFirstIndexes(self):
        _, indexes = self.getData()
        # Get indexes for the first game states
        firstIndexes = [curIndexes[0] for curIndexes in indexes]
        gameNumbers = [gameNum for gameNum in range(len(indexes))]
        
        return firstIndexes, gameNumbers
    
    
    def getAllIndexes(self):
        gameData, gameIndexes = self.getData()
        
        gameNumbers = np.arange(len(gameIndexes))
        idxIdx = np.cumsum([len(gameIndexes[i]) for i in range(len(gameIndexes))])
        idxIdx = np.concatenate(([0],idxIdx))
                
#        t = time.time()
#        gameDataIndexes = np.zeros(idxIdx[-1], np.int)
#        c = 0
#        for curGameIndexes in gameIndexes:
#            for idx in curGameIndexes:
#                gameDataIndexes[c] = idx
#                c += 1
        gameDataIndexes = np.array(list(itertools.chain.from_iterable(gameIndexes)))
#        print('loop ' + str(time.time()-t))
        
        return gameDataIndexes, gameNumbers, idxIdx, gameData
    
    
    def getIndexesForGameNums(self, gameNums):
        _, indexes = self.getData()
        
        gameNumbers = np.concatenate([np.full(len(indexes[gameNum]), gameNum) for gameNum in gameNums])
        indexes = np.concatenate(np.array(indexes)[gameNums])
        
        return indexes, gameNumbers
        
    
#    def getWinAmounts(self):
#        firstIndexes, gameNums = self.getFirstIndexes()
#        lastIndexes, _ = self.getLastIndexes()
#        
#        data, _ = self.getData()
#        playersData = data['playersData']
#        stacks = playersData[:,[2,10]]
#        winPlayerIdx = data['controlVariablesData'][lastIndexes,-1]
#        
#        initBets = np.column_stack((playersData[firstIndexes,3],playersData[firstIndexes,11]))
#        initStacks = stacks[firstIndexes] + initBets
#        finalStacks = stacks[lastIndexes]
#        
#        winnerInitStacks = initStacks[np.arange(len(winPlayerIdx)),winPlayerIdx]
#        winnerFinalStacks = finalStacks[np.arange(len(winPlayerIdx)),winPlayerIdx]
#        
#        winAmounts = winnerFinalStacks - winnerInitStacks
#    
#        return winAmounts, winPlayerIdx, gameNums
    
    
    def addData(self, gameStates, actions):
        validIndexes = np.nonzero(gameStates.validMask)[0]
        nDataPts = 0
        
        if(self.__boardsData is None):
            self.__boardsData = gameStates.boards[validIndexes]
            self.__playersData = self.flattenPlayersData(gameStates.players)[validIndexes]
            self.__controlVariablesData = gameStates.controlVariables[validIndexes]
            self.__availableActionsData = gameStates.availableActions[validIndexes]
            self.__actions = actions
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
            self.__actions = np.row_stack((self.__actions, actions[validIndexes]))
        
        dataIndexes = np.arange(len(validIndexes)) + nDataPts
        for gameIdx,dataIdx in zip(validIndexes,dataIndexes):
            self.__indexes[gameIdx].append(dataIdx)
        
        
    def getData(self): return {'boardsData': self.__boardsData,
                               'playersData': self.__playersData,
                               'availableActionsData': self.__availableActionsData,
                               'controlVariablesData': self.__controlVariablesData,
                               'actions': self.__actions}, self.__indexes
    
               
    def setData(self, data, indexes):
        self.__boardsData = data['boardsData']    
        self.__playersData = data['playersData']    
        self.__availableActionsData = data['availableActionsData']   
        self.__controlVariablesData = data['controlVariablesData']
        self.__actions = data['actions']
        self.__indexes = indexes
        
