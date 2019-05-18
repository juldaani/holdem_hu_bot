#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:23:25 2019

@author: juho
"""

        

import numpy as np

from holdem_hu_bot.agents import RndAgent, Agent
from holdem_hu_bot.features_rf import RfFeatures
from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute


class RfAgent(Agent):
    
    def __init__(self, playerNumber, rfFeatures, model):
        self.playerNumber = playerNumber
        self.rfFeatures = rfFeatures
        self.model = model
        
    def getActions(self, gameStates):
        allMask, actingPlayerMask, gameEndMask, gameFailedMask = super().getMasks(gameStates)
        model = self.model
        rfFeatures = self.rfFeatures
        playerNum = self.playerNumber
        
        # Return if all hands are finished for the current player
        if(np.sum(allMask) == 0):
            return np.zeros((0,2), dtype=np.int64), allMask
        
#        allMask, _, _, _ = agents[1].getMasks(gameStates)
#        model = regressor
#        playerNum = agents[1].playerNumber
        
        playersData = rfFeatures.flattenPlayersData(gameStates.players)
        availableActs = gameStates.availableActions
        features, miscDict = rfFeatures.computeFeatures(gameStates.boards, playersData, availableActs, 
                                                        gameStates.controlVariables, 
                                                        np.arange(len(playersData)))

        availableActsNormalized = miscDict['availableActionsNormalized']
        
        # Call and raise amounts
        amounts, amountsNormalized, gameNums = [], [], []
        for i in range(len(availableActsNormalized)):
            
            if(allMask[i] == False):
                continue
            
            callAmountNorm = availableActsNormalized[i,0]
            minRaiseNorm = availableActsNormalized[i,1]
            maxRaiseNorm = availableActsNormalized[i,2]

            callAmount = availableActs[i,0]
            minRaise = availableActs[i,1]
            maxRaise = availableActs[i,2]
            
#            print('....')
#            print(callAmountNorm, minRaiseNorm, maxRaiseNorm)
        
            N = 5
            amountsNormalized.append([callAmountNorm])
            amountsNormalized.append(np.linspace(minRaiseNorm, maxRaiseNorm, num=N))
            gameNums.append(np.full(N+1, i, dtype=np.int))
            
            amounts.append([callAmount])
            amounts.append(np.linspace(minRaise, maxRaise, num=N).astype(np.int))
        
        amountsNormalized = np.concatenate(amountsNormalized)
        amounts = np.concatenate(amounts)
        gameNums = np.concatenate(gameNums)
        mask = amounts < 0
        
        # Fold amounts
        smallBlinds = gameStates.boards[:,1]
        pots = gameStates.boards[:,0]
        bets = np.column_stack((playersData[:,3], playersData[:,11]))
        bets = bets + pots.reshape(-1,1)
        foldAmounts = bets[:,playerNum] * -1
        foldAmountsNorm = foldAmounts / smallBlinds
        
        # Predict win/lose amount        
        model.verbose = 0
        predictedAmounts = model.predict(np.column_stack((features[gameNums], amountsNormalized)))
        
        # Get best action (maximum return)        
        predictedAmounts[mask] = -9999
        predictedAmounts = predictedAmounts.reshape((-1,N+1))
        gameNums = gameNums.reshape((-1,N+1))[:,0]
        
        tmpAmountsNorm, tmpAmounts = np.zeros((len(allMask),7)), np.zeros((len(allMask),7), dtype=np.int)
        tmpAmountsNorm[:,0] = foldAmountsNorm
        tmpAmountsNorm[gameNums,1:] = predictedAmounts
        tmpAmounts[:,0] = foldAmounts
        tmpAmounts[gameNums,1:] = amounts.reshape((-1,N+1))
        
        actionAmountIdx = np.argmax(tmpAmountsNorm, 1)
        actionAmounts = tmpAmounts[np.arange(len(actionAmountIdx)),actionAmountIdx]
        actionAmounts[actionAmountIdx == 0] = -1    # if fold
        
        return createActionsToExecute(actionAmounts[allMask]), allMask
        

#rfAgent1 = RfAgent(0, rfFeatures, regressor)
#rfAgent2 = RfAgent(1, rfFeatures, regressor)
#
#acts, mask = rfAgent1.getActions(gameStates)
#acts1, mask1 = rfAgent2.getActions(gameStates)



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

nGames = 10000

gameStates = initRandomGames(nGames)
rfFeatures = RfFeatures(gameStates)

#agents = [RndAgent(0), RfAgent(1, rfFeatures, regressor)]
agents = [RfAgent(0, rfFeatures, regressor), RfAgent(1, rfFeatures, regressor)]
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
                                              gameNumsFeatures, executedActions, clippingThres=100)

x = np.column_stack((features[mask], executedActions[mask,1]))
y = winAmountsFeatures[mask]


# %%

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


 






