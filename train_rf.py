#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:23:25 2019

@author: juho
"""

import numpy as np

from holdem_hu_bot.agents import RndAgent
from holdem_hu_bot.features_rf import RfFeatures
from texas_hu_engine.wrappers import initRandomGames, executeActions



# %%
# Run random games to initialize random forest agent

nGames = 1000

agents = [RndAgent(0), RndAgent(1)]
rfFeatures = RfFeatures(nGames)
gameStates = initRandomGames(nGames)
#rfFeatures.addData(gameStates)

while(1):
    actionsAgent0, maskAgent0 = agents[0].getActions(gameStates)
    actionsAgent1, maskAgent1 = agents[1].getActions(gameStates)
    
    actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
    actionsToExecute[maskAgent0] = actionsAgent0
    actionsToExecute[maskAgent1] = actionsAgent1
    
    gameStates = executeActions(gameStates, actionsToExecute)
    
#    rfFeatures.addData(gameStates)
#    rfFeatures.getFeatures()
    
    nValidGames = np.sum(gameStates.controlVariables[:,1]==0)
    print(nValidGames)
    if(nValidGames == 0):
        break
    


# %%

