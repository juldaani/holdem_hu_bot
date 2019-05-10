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

#c = 0   # TODO: remove
while(1):
    actionsAgent0, maskAgent0 = agents[0].getActions(gameStates)
    actionsAgent1, maskAgent1 = agents[1].getActions(gameStates)
    
    actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
    actionsToExecute[maskAgent0] = actionsAgent0
    actionsToExecute[maskAgent1] = actionsAgent1
    
    # TODO: remove
#    if(c==0):
#        actionsToExecute[0] = -433453
#    c+=1
    
    rfFeatures.addData(gameStates, actionsToExecute)

    gameStates = executeActions(gameStates, actionsToExecute)
    
    nValidGames = np.sum(gameStates.controlVariables[:,1]==0)
    print(nValidGames)
    if(nValidGames == 0):
        break

# Save also last game state
actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
rfFeatures.addData(gameStates, actionsToExecute)

# %%
# Train extra trees regressor
        
from sklearn.ensemble import ExtraTreesRegressor

features, executedActions, misc = rfFeatures.getFeaturesForAllGameStates()
mask = ~(misc['gameFailedMask'] | misc['gameFinishedMask'])

# Do checks
assert np.sum(misc['gameFailedMask']) == 0  # No failures
assert np.sum(misc['gameFinishedMask']) == nGames   # All games have ended succesfully

etr = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=5, min_samples_split=4, 
                          verbose=2, n_jobs=-1, random_state=0)
etr.fit(features[mask], executedActions[mask])


# %%

ii = 595
etr.verbose = 0
print(etr.predict(features[mask][ii:ii+1]))
print(executedActions[mask][ii])
misc['availableActionsNormalized'][mask][ii]








