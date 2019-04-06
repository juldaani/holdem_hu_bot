#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:18:43 2019

@author: juho
"""

import numpy as np
from holdem_hu_bot.game_data_container import GameDataContainer

"""
- effective stack
- normalize stacks and pot by dividing with small blind
- pot odds
- equities
- encode hole and board cards (one-hot-encoding etc...)
- is small blind
- 
"""


# %%

from texas_hu_engine.wrappers import initRandomGames, GameState
from texas_hu_engine.engine_numba import getBoardCards, getGameEndState
from equity_calculator.equity_calculator import computeEquities
from hand_eval.params import cardToInt, intToCard


cardToSuit = np.zeros((52,4), dtype=np.int8)
cardToRank = np.zeros((52,13), dtype=np.int8)
ranks = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
for cardNum in intToCard:
    card = intToCard[cardNum]
    print(cardNum,card)
    
    cardToSuit[cardNum,0], cardToSuit[cardNum,1] = 'c' in card, 'd' in card
    cardToSuit[cardNum,2], cardToSuit[cardNum,3] = 'h' in card, 's' in card
    cardToRank[cardNum] = [rank in card for rank in ranks]
    
    
    
    

class Features(GameDataContainer):
    
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
        
    
    def getFeatures(self):
        data, indexes = super().getData()
        
        # Get indexes for the last game states
        lastIndexes = [curIndexes[-1] for curIndexes in indexes]
        
        boardsData = data['boardsData'][lastIndexes]
        playersData = data['playersData'][lastIndexes]
        availableActions = data['availableActionsData'][lastIndexes]
        controlVariables= data['controlVariablesData'][lastIndexes]
        
        smallBlinds = np.row_stack(boardsData[:,1]) # Small blinds amounts are used for normalization
        actingPlayerIdx = np.argmax(playersData[:,[6,14]], 1)
        isSmallBlindPlayer = playersData[:,[4,12]][np.arange(len(actingPlayerIdx)),actingPlayerIdx]
        
        potsNormalized = (boardsData[:,0] + playersData[:,3] + playersData[:,11])/smallBlinds.flatten()
        

        actionsNormalized = availableActions / smallBlinds
        

        controlVars = data['controlVariablesData'][lastIndexes]
        gameValidMask = ~controlVars[:,1].astype(np.bool)    # Tells if the game is still ongoing
        
        data['controlVariablesData']
        
        
        return data, indexes
        


N = 100000

gameStates = initRandomGames(N)
asd = Features(N)
asd.addData(gameStates)

asd.equities[1]['flop']

data, indexes = asd.getFeatures()
















# %%
  
    
    
    
    
    
    
    
    
    
    
    
    
    