#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:18:43 2019

@author: juho
"""

import numpy as np
import pandas as pd
from holdem_hu_bot.game_data_container import GameDataContainer

from equity_calculator.equity_calculator import computeEquities
from holdem_hu_bot.common_stuff import suitsOnehotLut, ranksOnehotLut, encodeCardsOnehot



class RfFeatures(GameDataContainer):
    
    def __init__(self, nGames): 
        GameDataContainer.__init__(self, nGames)
        self.equities = None
        
        
    def addData(self, gameStates, actions):
        if(self.equities == None):
            equities = {0:{}, 1:{}}
       
            boardCards = gameStates.boards[:,-5:]
            player0HoleCards = gameStates.players[::2,:2]
            player1HoleCards = gameStates.players[1::2,:2]
            
            equities[0]['preflop'], equities[0]['flop'], equities[0]['turn'], \
                equities[0]['river'] = computeEquities(player0HoleCards, boardCards)
            equities[1]['preflop'], equities[1]['flop'], equities[1]['turn'], \
                equities[1]['river'] = computeEquities(player1HoleCards, boardCards)
            
            self.equities = equities
            
        super().addData(gameStates, actions)
    
    
    def getWinAmountsNormalized(self):
        data, _ = super().getData()
        winAmounts, winPlayerIdx, gameNums = super().getWinAmounts()
        indexes, _ = super().getFirstIndexes()
        
        boardsData = data['boardsData'][indexes]
        smallBlinds = boardsData[:,1]
        
        return winAmounts / smallBlinds, winPlayerIdx, gameNums
        
    
    def getFeaturesForAllGameStates(self):
        indexes, gameNumbers = super().getAllIndexes()
        
        return self.computeFeatures(indexes, gameNumbers)
    
    
    def computeFeatures(self, indexes, gameNumbers):
        data, _ = super().getData()
        
#        indexes, gameNumbers = rfFeatures.getAllIndexes()   # TODO: remove
#        data,_ = rfFeatures.getData()   # TODO: remove
        
        boardsData = data['boardsData'][indexes]
        playersData = data['playersData'][indexes]
        availableActions = data['availableActionsData'][indexes]
        controlVariables = data['controlVariablesData'][indexes]
        executedActions = data['actions'][indexes]
        
        smallBlinds = np.row_stack(boardsData[:,1]) # Small blinds amounts are used for normalization
        actingPlayerIdx = np.argmax(playersData[:,[6,14]], 1)
        nonActingPlayerIdx = (~actingPlayerIdx.astype(np.bool)).astype(np.int)
        isSmallBlindPlayer = playersData[:,[4,12]][np.arange(len(actingPlayerIdx)),actingPlayerIdx].astype(
                np.uint8)
        
        # Normalize executed actions
        smallBlinds = np.full(len(executedActions), 56).reshape((-1,1))
        mask1 = executedActions == -1
        mask2 = executedActions == -999
        mask3 = executedActions[:,0] == 1
        executedActions = executedActions / smallBlinds
        executedActions[mask1] = -1
        executedActions[mask2] = -999
        executedActions[mask3,0] = 1
        
        # Pots, stacks etc. money stuffs
        pots = boardsData[:,0]
        bets = playersData[:,3] + playersData[:,11]
        stacks = playersData[:,[2,10]]
        potsNormalized = (pots + bets) / smallBlinds.flatten()
        availableActionsNormalized = availableActions / smallBlinds
        availableActionsNormalized[availableActions < 0] = -1
        ownStacksNormalized = stacks[np.arange(len(stacks)),actingPlayerIdx] / smallBlinds.flatten()
        opponentStacksNormalized = stacks[np.arange(len(stacks)),nonActingPlayerIdx] / smallBlinds.flatten()
        
        # Encode cards one hot
        boardCards = boardsData[:,8:]
        boardCards[boardCards == -999] = 0  # Assign zero if failure code because otherwise 
            # 'encodeCardsOnehot' will fail
        visibleCardsMask = boardsData[:,3:8].astype(np.bool)
        boardcardSuitsOnehot, boardcardRanksOnehot  = encodeCardsOnehot(boardCards, visibleCardsMask, 
                                                                        ranksOnehotLut, suitsOnehotLut)
        holecards = playersData[:,[0,1,8,9]]
        holecards = holecards.reshape((len(holecards),2,2))
        holecards = holecards[np.arange(len(holecards)),actingPlayerIdx]
        holecards[holecards == -999] = 0  # Assign zero if failure code because otherwise 
            # 'encodeCardsOnehot' will fail
        holecardSuitsOnehot, holecardRanksOnehot = encodeCardsOnehot(holecards,
                                                                     np.ones(holecards.shape, dtype=np.bool), 
                                                                     ranksOnehotLut, suitsOnehotLut)
        
        gameFinishedMask = controlVariables[:,1] == 1   # Tells if the game has finished succesfully
        gameFailedMask = controlVariables[:,1] == -999  # Tells if an error has occured
        bettingRound = visibleCardsMask[:,2:].astype(np.int)

        # Get equities
        bettingRoundNames = ['preflop','flop','turn','river']
        equities = np.zeros(len(smallBlinds), dtype=np.float32)
        for i, gameNum in enumerate(gameNumbers):
            curPlayerIdx = actingPlayerIdx[i]
            curBettingRound = np.sum(bettingRound[i])
            equity = self.equities[curPlayerIdx][bettingRoundNames[curBettingRound]][gameNum]  
            equities[i] = equity
                    
        miscDict = {'availableActionsNormalized':availableActionsNormalized, 
                    'gameFinishedMask':gameFinishedMask, 'gameFailedMask':gameFailedMask}
        
        features = np.column_stack((isSmallBlindPlayer, bettingRound, equities, availableActionsNormalized,
                                    potsNormalized, ownStacksNormalized, opponentStacksNormalized,
                                    boardcardSuitsOnehot, boardcardRanksOnehot, holecardSuitsOnehot, 
                                    holecardRanksOnehot))

        return features, executedActions, miscDict
        
        
        
    