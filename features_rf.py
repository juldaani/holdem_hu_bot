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

        controlVars = data['controlVariablesData'][lastIndexes]
        gameValidMask = ~controlVars[:,1].astype(np.bool)    # Tells if the game is still going
        
        
        
        
        return data, indexes
        


N = 5

gameStates = initRandomGames(N)
asd = Features(N)
asd.addData(gameStates)

asd.equities[1]['flop']

data, indexes = asd.getFeatures()
















# %%
  
    
    
    
    
    
    
    
    
    
    
    
    
    