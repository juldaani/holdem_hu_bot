#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:23:25 2019

@author: juho
"""

        

import numpy as np
from numba import jit
import copy
from sklearn.ensemble import ExtraTreesRegressor

from holdem_hu_bot.agents import RndAgent, Agent
from holdem_hu_bot.features_rf import RfFeatures
from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute




class RfAgent(Agent):
    
    def __init__(self, playerNumber, rfFeatures, aiModel, randomizationRatio=0.0):
        self.playerNumber = playerNumber
        self.rfFeatures = rfFeatures
        self.aiModel = aiModel
        self.randomizationRatio = randomizationRatio
    
    def getActions(self, gameStates):
        mask, _, _, _ = super().getMasks(gameStates)
        aiModel = self.aiModel
        rfFeatures = self.rfFeatures
        randomizationRatio = self.randomizationRatio
        
        # Return if all hands are already finished for the current player
        if(np.sum(mask) == 0):
            return np.zeros((0,2), dtype=np.int64), mask
        
#        agent = RfAgent(0, rfFeatures, regressor)
#        mask, _, _, _ = agent.getMasks(gameStates)
#        aiModel = regressor
#        randomizationRatio = 1.0

        playersData = rfFeatures.flattenPlayersData(gameStates.players)[mask]
        boardsData = gameStates.boards[mask]
        availableActs = gameStates.availableActions[mask]
        gameNums = np.nonzero(mask)[0]
        
        features, miscDict = rfFeatures.computeFeatures(boardsData, playersData, availableActs, 
                                                        gameStates.controlVariables[mask], gameNums)
        
        aiModel.verbose = 0
        predictedActions = aiModel.predict(features)
        
        # Randomize actions
        rndRowIdx = np.random.choice(len(predictedActions), 
                                     int(len(predictedActions)*randomizationRatio), replace=0)
        rndColIdx = np.random.randint(0, high=predictedActions.shape[1], size=len(rndRowIdx))
        predictedActions[rndRowIdx] = 0
        predictedActions[rndRowIdx, rndColIdx] = 1
        
        # Generate amounts for the actions
        # 0: fold, 1: call, 2: 0.5 pot, 3: 1 pot, 4: 1.5 pot, 5: all-in
        pots = boardsData[:,0]
        bets = playersData[:,3] + playersData[:,11]
        pots += bets
        actionAmounts = np.zeros(predictedActions.shape, dtype=np.int)
        actionAmounts[:,0] = -1     # Fold
        actionAmounts[:,1] = availableActs[:,0]     # Call
        actionAmounts[:,2:5] = pots.reshape(-1,1) * np.tile(np.array([0.5, 1, 1.5]), (len(pots),1))
        actionAmounts[:,-1] = availableActs[:,-1]   # All-in
        
        # If invalid amount, then pick the closest valid amount
        amounts = actionAmounts[np.arange(len(actionAmounts)),np.argmax(predictedActions,1)]
        invalidAmountMask = ~((amounts == availableActs[:,0]) | (amounts == -1) | \
            (amounts >= availableActs[:,1]) & (amounts <= availableActs[:,2]))
        closestIdx = np.argmin(np.abs(availableActs[invalidAmountMask] - 
                                      amounts[invalidAmountMask].reshape((-1,1))),1)
        amounts[invalidAmountMask] = availableActs[invalidAmountMask, closestIdx]
        
        return createActionsToExecute(amounts), mask

        

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


def playGames(agents, gameStates, gameDataContainer):

    while(1):
        actionsAgent0, maskAgent0 = agents[0].getActions(gameStates)
        actionsAgent1, maskAgent1 = agents[1].getActions(gameStates)
        
        actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
        actionsToExecute[maskAgent0] = actionsAgent0
        actionsToExecute[maskAgent1] = actionsAgent1
        
        gameDataContainer.addData(gameStates, actionsToExecute)
    
        gameStates = executeActions(gameStates, actionsToExecute)
        
        nValidGames = np.sum(gameStates.controlVariables[:,1]==0)
        print(nValidGames)
        
        if(nValidGames == 0):
            break
    
    # Save also last game state
    actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
    gameDataContainer.addData(gameStates, actionsToExecute)
    
    return gameDataContainer



# %%
# Run random games to initialize random forest agent

nGames = 5

gameStates = initRandomGames(nGames)
rfFeatures = RfFeatures(gameStates)

#agents = [RndAgent(0), RfAgent(1, rfFeatures, regressor)]
#agents = [RfAgent(0, rfFeatures, regressor), RfAgent(1, rfFeatures, regressor)]
agents = [RndAgent(0), RndAgent(1)]

rfFeatures = playGames(agents, gameStates, rfFeatures)

features, executedActions, misc = rfFeatures.getFeaturesForAllGameStates()

# Do checks
assert np.sum(misc['gameFailedMask']) == 0  # No failures
assert np.sum(misc['gameFinishedMask']) == nGames   # All games have ended succesfully

mask = ~(misc['gameFinishedMask'])   # Discard finished game states

features = features[mask]

# 0: fold, 1: call, 2: 0.5 pot, 3: 1 pot, 4: 1.5 pot, 5: all-in
targetActions = np.zeros((len(features),6))
targetActions[:,1] = 1  # Call always

#regressorOld = copy.deepcopy(regressor)
regressor = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=10, min_samples_split=10, 
                                verbose=2, n_jobs=-1, random_state=0)

regressor.fit(features, targetActions)


# %%
# Use self play to improve agent


nGames = 10
seed = 10
randomizationRatio = 0.5
randomizedPlayerIdx = 0
nonRandomizedPlayerIdx = np.abs(randomizedPlayerIdx-1)

initGameStates = initRandomGames(nGames, seed=seed)
initGameContainer = RfFeatures(copy.deepcopy(initGameStates))
agents = [RfAgent(0, initGameContainer, regressor), RfAgent(1, initGameContainer, regressor)]
gamesNoRandomization = playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(initGameContainer))

gamesRandomized = []
for i in range(5):
#    gameContainer = RfFeatures(copy.deepcopy(initGameStates))
    agents = [RfAgent(randomizedPlayerIdx, initGameContainer, regressor, 
                      randomizationRatio=randomizationRatio), 
              RfAgent(nonRandomizedPlayerIdx, initGameContainer, regressor)]
    gamesRandomized.append(playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(initGameContainer)))
    
# %%

winAmounts, winPlayerIdx, gameNums = gamesNoRandomization.getWinAmounts()
loseMask = winPlayerIdx == nonRandomizedPlayerIdx
winAmounts[loseMask] *= -1

winAmounts1, winPlayerIdx1, gameNums1 = gamesRandomized[2].getWinAmounts()
loseMask = winPlayerIdx1 == nonRandomizedPlayerIdx
winAmounts1[loseMask] *= -1

# %%

data, idx = rfFeatures.getData()

data['actions'][idx[4]]
    
# %%

winAmounts, winPlayerIdx, gameNums = rfFeatures.getWinAmounts()

amountPlayer0 = np.sum(winAmounts[winPlayerIdx == 0])
amountPlayer1 = np.sum(winAmounts[winPlayerIdx == 1])

winAmnt = np.max([amountPlayer0, amountPlayer1])
winPlayer = np.argmax([amountPlayer0, amountPlayer1])
loseAmnt = np.min([amountPlayer0, amountPlayer1])

print('\n.........................')
print('win player: ' + str(winPlayer) +', ratio: ' + str(winAmnt / loseAmnt))



# %%


 






