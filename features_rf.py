#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:18:43 2019

@author: juho
"""

import numpy as np
from game_data_container import GameDataContainer

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

from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute
from texas_hu_engine.wrappers import GameState


class Features(GameDataContainer):
    
    def __init__(self, nGames): 
        GameDataContainer.__init__(self, nGames)
        
    def addData(self, gameStates):
        super().addData(gameStates)
        


N = 3

states = initRandomGames(N)
asd = Features(N)
asd.addData(states)

asd.getData()
















# %%
  
    
    
    
    
    
    
    
    
    
    
    
    
    