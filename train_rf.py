#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:23:25 2019

@author: juho
"""

        

import numpy as np
from numba import jit

from holdem_hu_bot.agents import RndAgent, Agent
from holdem_hu_bot.features_rf import RfFeatures
from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def genIntervals(minRaise, maxRaise, N):
    intervals = np.zeros((len(minRaise),N), dtype=minRaise.dtype)
    for i in range(len(minRaise)):
        intervals[i,:] = np.linspace(minRaise[i], maxRaise[i], N)
    
    return intervals


class RfAgent(Agent):
    
    def __init__(self, playerNumber, rfFeatures, model):
        self.playerNumber = playerNumber
        self.rfFeatures = rfFeatures
        self.model = model
        
        
    def getActions(self, gameStates):
        mask, actingPlayerMask, gameEndMask, gameFailedMask = super().getMasks(gameStates)
        model = self.model
        rfFeatures = self.rfFeatures
        playerNum = self.playerNumber
        
        # Return if all hands are already finished for the current player
        if(np.sum(mask) == 0):
            return np.zeros((0,2), dtype=np.int64), mask
        
#        mask, _, _, _ = agents[1].getMasks(gameStates)
#        model = regressor
#        playerNum = agents[1].playerNumber
        
        playersData = rfFeatures.flattenPlayersData(gameStates.players)[mask]
        boardsData = gameStates.boards[mask]
        availableActs = gameStates.availableActions[mask]
        gameNums = np.nonzero(mask)[0]
        
        features, miscDict = rfFeatures.computeFeatures(boardsData, playersData, availableActs, 
                                                        gameStates.controlVariables[mask], gameNums)

        availableActsNormalized = miscDict['availableActionsNormalized']
        
        # Call and raise amounts
        N = 5
        amounts = np.zeros((len(features),6), dtype=np.int)
        amountsNormalized = np.zeros((len(features),6))
        
        callAmountNorm = availableActsNormalized[:,0]
        minRaiseNorm = availableActsNormalized[:,1]
        maxRaiseNorm = availableActsNormalized[:,2]
        amountsNormalized[:,0] = callAmountNorm
        amountsNormalized[:,1:] = genIntervals(minRaiseNorm, maxRaiseNorm, N)
        
        callAmount = availableActs[:,0]
        minRaise = availableActs[:,1]
        maxRaise = availableActs[:,2]
        amounts[:,0] = callAmount
        amounts[:,1:] = genIntervals(minRaise, maxRaise, N)

        # Predict return (win/lose amount)
        model.verbose = 0
        featureIndexes = np.tile(np.arange(len(amountsNormalized)), (amountsNormalized.shape[1],1)).T
        predictedReturnsNorm = model.predict(np.column_stack((features[featureIndexes.flatten()], 
                                                                       amountsNormalized.flatten() )))
        predictedReturnsNorm = predictedReturnsNorm.reshape(featureIndexes.shape)
        
        # Fold amounts
        smallBlinds = boardsData[:,1]
        pots = boardsData[:,0]
        bets = np.column_stack((playersData[:,3], playersData[:,11]))
        bets = bets + pots.reshape(-1,1)
        foldAmounts = bets[:,playerNum] * -1
        foldAmountsNorm = foldAmounts / smallBlinds
        
        # Get best action (maximum return)        
        predictedReturnsNorm[amounts < 0] = -9999
        highestReturnIdx = np.argmax(np.column_stack((foldAmountsNorm, predictedReturnsNorm)),1)
        amounts = np.column_stack((foldAmounts,amounts))
        amounts2 = amounts[np.arange(len(highestReturnIdx)),highestReturnIdx]
        amounts2[highestReturnIdx == 0] = -1    # Fold
        
        return createActionsToExecute(amounts2), mask
        

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

# 

nGames = 100000

gameStates = initRandomGames(nGames)
rfFeatures = RfFeatures(gameStates)

#agents = [RndAgent(0), RfAgent(1, rfFeatures, regressor)]

#agents = [RfAgent(0, rfFeatures, regressor), RfAgent(1, rfFeatures, regressor)]
#agents = [RfAgent(0, rfFeatures, regressorOld), RfAgent(1, rfFeatures, regressor)]
#agents = [RfAgent(0, rfFeatures, regressor0), RfAgent(1, rfFeatures, regressor)]

#agents = [RndAgent(0), RndAgent(1)]


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

winAmounts, winPlayerIdx, gameNums = rfFeatures.getWinAmounts()

amountPlayer0 = np.sum(winAmounts[winPlayerIdx == 0])
amountPlayer1 = np.sum(winAmounts[winPlayerIdx == 1])

winAmnt = np.max([amountPlayer0, amountPlayer1])
winPlayer = np.argmax([amountPlayer0, amountPlayer1])
loseAmnt = np.min([amountPlayer0, amountPlayer1])

print('\n.........................')
print('win player: ' + str(winPlayer) +', ratio: ' + str(winAmnt / loseAmnt))


# %%


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
                                              gameNumsFeatures, executedActions, clippingThres=1000)

x = np.column_stack((features[mask], executedActions[mask,1]))
y = winAmountsFeatures[mask]


# %%

import copy
from sklearn.ensemble import ExtraTreesRegressor

regressorOld = copy.deepcopy(regressor)
#regressor0 = copy.deepcopy(regressor)

regressor = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=10, min_samples_split=10, 
                                verbose=2, n_jobs=-1, random_state=0)

regressor.fit(x, y)

pred = regressor.predict(x)
np.sum(pred*y > 0) / len(pred)




# %%


 






