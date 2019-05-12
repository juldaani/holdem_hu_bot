#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:23:25 2019

@author: juho
"""

        

import numpy as np

from holdem_hu_bot.agents import RndAgent, Agent
from holdem_hu_bot.features_rf import RfFeatures
from texas_hu_engine.wrappers import initRandomGames, executeActions


def getWinAmountsForFeatures(winAmounts, winPlayerIdx, actingPlayerIdx, gameNumsWinAmounts, gameNumsFeatures, 
                             executedActions, clippingThres=50):
    # Sort
    sorter = np.argsort(gameNumsWinAmounts)
    winAmounts = winAmounts[sorter]
    winPlayerIdx = winPlayerIdx[sorter]
    gameNumsWinAmounts = np.array(gameNumsWinAmounts)[sorter]
    
    # Get win amounts for each game state (feature). This is what we try to predict.
    winAmountsFeatures = winAmounts[gameNumsFeatures]
    winPlayerIdxFeatures = winPlayerIdx[gameNumsFeatures]
    # Set win amount to negative if currently acting player is not the winning player (the player will lose 
    # that amount).
    notWinningPlayerMask = ~(winPlayerIdxFeatures == actingPlayerIdx)
    winAmountsFeatures[notWinningPlayerMask] *= -1
    winAmountsFeatures = np.clip(winAmountsFeatures, a_min=-clippingThres, a_max=clippingThres)
    
#    actionAmounts = executedActions[:,1].copy()
#    actionAmounts[np.isclose(actionAmounts,0)] = 0.001
#    betToWinRatios = winAmountsFeatures / actionAmounts
#    betToWinRatios = np.clip(betToWinRatios, a_min=-clippingThres, a_max=clippingThres)

    return winAmountsFeatures



# %%
# Run random games to initialize random forest agent

nGames = 20000

agents = [RndAgent(0), RndAgent(1)]
gameStates = initRandomGames(nGames)
rfFeatures = RfFeatures(gameStates)

while(1):
    actionsAgent0, maskAgent0 = agents[0].getActions(gameStates)
    actionsAgent1, maskAgent1 = agents[1].getActions(gameStates)
    
    actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
    actionsToExecute[maskAgent0] = actionsAgent0
    actionsToExecute[maskAgent1] = actionsAgent1
    
    rfFeatures.addData(gameStates, actionsToExecute)

    gameStates = executeActions(gameStates, actionsToExecute)
    
    nValidGames = np.sum(gameStates.controlVariables[:,1]==0)
    print(nValidGames)
    if(nValidGames == 0):
        break

# Save also last game state
actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
rfFeatures.addData(gameStates, actionsToExecute)



asd = rfFeatures.getFeaturesForAllGameStates()

features, executedActions, misc = rfFeatures.getFeaturesForAllGameStates()
gameNumsFeatures = misc['gameNumbers']
actingPlayerIdx = misc['actingPlayerIdx']
winAmounts, winPlayerIdx, gameNumsWinAmounts = rfFeatures.getWinAmountsNormalized()

# Do checks
assert np.sum(misc['gameFailedMask']) == 0  # No failures
assert np.sum(misc['gameFinishedMask']) == nGames   # All games have ended succesfully


foldMask = np.isclose(executedActions[:,0], 1)  # Folds are discarded because the result of the game 
    # is then obvious, no need to predict anything.
mask = ~(misc['gameFinishedMask'] | foldMask)   # If the game has ended, we have no need predict the 
    # outcome for that state.
    
winAmountsFeatures = getWinAmountsForFeatures(winAmounts, winPlayerIdx, actingPlayerIdx, gameNumsWinAmounts, 
                                              gameNumsFeatures, executedActions, clippingThres=100)

x = np.column_stack((features[mask], executedActions[mask,1]))
y = winAmountsFeatures[mask]





# %%

# 0.529
# 0.532

from sklearn.ensemble import ExtraTreesRegressor

regressor = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=10, min_samples_split=10, 
                                verbose=2, n_jobs=-1, random_state=0)

regressor.fit(x, y)

pred = regressor.predict(x)
np.sum(pred*y > 0) / len(pred)


# %%

ii = 0

xIn = x[ii:ii+1]

availAct = misc['availableActionsNormalized'][mask][ii]
xIn[:,-1] = availAct[0]+0

regressor.verbose = 0
regressor.predict(xIn)



# %%


class RfAgent():
    
    def __init__(self, playerNumber, rfFeatures):
        self.playerNumber = playerNumber







