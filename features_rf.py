#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:18:43 2019

@author: juho
"""

import numpy as np
from holdem_hu_bot.game_data_container import GameDataContainer

from equity_calculator.equity_calculator import computeEquities
from holdem_hu_bot.common_stuff import suitsOnehotLut, ranksOnehotLut, encodeCardsOnehot



class RfFeatures(GameDataContainer):
    
    def __init__(self, nGames): 
        GameDataContainer.__init__(self, nGames)
        self.equities = None
        
        
    def addData(self, gameStates):
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
            
        super().addData(gameStates)
        
        
    def getLastIndexes(self):
        _, indexes = super().getData()
        # Get indexes for the last (the most recent) game states
        lastIndexes = [curIndexes[-1] for curIndexes in indexes]
        gameNumbers = [gameNum for gameNum in range(len(indexes))]
        
        return lastIndexes, gameNumbers
    

    def getAllIndexes(self):
        _, indexes = super().getData()
        
        gameNumbers = [np.full(len(indexes[gameNum]), gameNum) for gameNum in range(len(indexes))]
        gameNumbers = np.concatenate(gameNumbers)
        indexes2 = np.concatenate(indexes)
        
        return indexes2, gameNumbers
    
    
    def getFeaturesForAllGameStates(self):
        indexes, gameNumbers = self.getAllIndexes()
        
        return self.computeFeatures(indexes, gameNumbers)
    
    
    def computeFeatures(self, indexes, gameNumbers):
        data, _ = super().getData()
        
#        indexes, gameNumbers = rfFeatures.getAllIndexes()   # TODO: remove
#        data,_ = rfFeatures.getData()   # TODO: remove
        
        boardsData = data['boardsData'][indexes]
        playersData = data['playersData'][indexes]
        availableActions = data['availableActionsData'][indexes]
        controlVariables = data['controlVariablesData'][indexes]
        
        smallBlinds = np.row_stack(boardsData[:,1]) # Small blinds amounts are used for normalization
        actingPlayerIdx = np.argmax(playersData[:,[6,14]], 1)
        nonActingPlayerIdx = (~actingPlayerIdx.astype(np.bool)).astype(np.int)
        isSmallBlindPlayer = playersData[:,[4,12]][np.arange(len(actingPlayerIdx)),actingPlayerIdx].astype(
                np.uint8)
        
        # Pots, stacks etc. money stuffs
        pots = boardsData[:,0]
        bets = playersData[:,3] + playersData[:,11]
        stacks = playersData[:,[2,10]]
        potsNormalized = (pots + bets) / smallBlinds.flatten()
        actionsNormalized = availableActions / smallBlinds
        actionsNormalized[availableActions < 0] = -1
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

        return {'isSmallBlindPlayer':isSmallBlindPlayer, 'bettingRound':bettingRound, 'equities':equities,
                
                'potsNormalized':potsNormalized, 'actionsNormalized':actionsNormalized, 
                'ownStacksNormalized':ownStacksNormalized,
                'opponentStacksNormalized':opponentStacksNormalized, 
                
                'boardcardSuits':boardcardSuitsOnehot, 'boardcardRanks':boardcardRanksOnehot,
                'holecardSuits':holecardSuitsOnehot, 'holecardRanks':holecardRanksOnehot,
                
                'gameFinishedMask':gameFinishedMask, 'gameFailedMask':gameFailedMask}
        
        
        
        
    