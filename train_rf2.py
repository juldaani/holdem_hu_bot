#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:23:25 2019

@author: juho
"""

        

import numpy as np
from numba import jit
import copy
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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
        
        tmpActions = np.zeros((len(predictedActions),6))
        tmpActions[np.arange(len(predictedActions)),predictedActions] = 1
        predictedActions = tmpActions
        
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

nGames = 10000

gameStates = initRandomGames(nGames)
rfFeatures = RfFeatures(gameStates)

#agents = [RndAgent(0), RfAgent(1, rfFeatures, regressor)]
#agents = [RfAgent(0, rfFeatures, regressor), RfAgent(1, rfFeatures, regressor)]
agents = [RndAgent(0), RndAgent(1)]

rfFeatures = playGames(agents, gameStates, rfFeatures)

features, executedActions, executedActionsNorm, misc = rfFeatures.getFeaturesForAllGameStates()

# Do checks
assert np.sum(misc['gameFailedMask']) == 0  # No failures
assert np.sum(misc['gameFinishedMask']) == nGames   # All games have ended succesfully

mask = ~(misc['gameFinishedMask'])   # Discard finished game states

features = features[mask]

# 0: fold, 1: call, 2: 0.5 pot, 3: 1 pot, 4: 1.5 pot, 5: all-in
#targetActions = np.zeros((len(features),6))
#targetActions[:,1] = 1  # Call always
#targetActions = np.argmax(targetActions,1)
targetActions = np.random.randint(0, high=6, size=len(features))

#regressor = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=10, min_samples_split=10, 
#                                verbose=2, n_jobs=-1, random_state=0)
#regressor = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=10, min_samples_split=10, 
#                                verbose=2, n_jobs=-1)
#regressor.fit(features, targetActions)

regressor = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000, verbose=1)
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
for i in range(100):
    regressor.partial_fit(features, targetActions, classes=np.arange(6))
    print('Iteration: ' + str(i) + '   loss: ' + str(regressor.loss_))

regressor0 = copy.deepcopy(regressor)

# %%
# Use self play to improve agent


def getReturnsForGames(gameContainer, playerIdx):
    winAmounts, winPlayerIdx, gameNums = gameContainer.getWinAmounts()
    loseMask = ~(winPlayerIdx == playerIdx)
    winAmounts[loseMask] *= -1
    
    return winAmounts
#
#def getImprovedReturnGamenums(returnsNoRandomization, returnsRandomized):
#    improvedReturnMask = returnsNoRandomization.reshape((-1,1)) < returnsRandomized
#
#    randomizationIndexes = np.zeros(len(improvedReturnMask), dtype=np.int)-1
#    for i in range(len(improvedReturnMask)):
#        curGame = improvedReturnMask[i]
#        idx = np.where(curGame == 1)[0]
#        
#        if(len(idx) == 0):
#            continue
#        
#        randomizationNum = idx[np.random.randint(0, len(idx))]
#        randomizationIndexes[i] = randomizationNum
#        
#    gameNums = np.arange(len(improvedReturnMask))
#    gameNumsNoRandomization = gameNums[randomizationIndexes == -1]
#    gameNumsRandomized = [gameNums[randomizationIndexes == i] for i in range(improvedReturnMask.shape[1])]
#    
#    return gameNumsNoRandomization, gameNumsRandomized
    
def getImprovedReturnGamenums(returnsNoRandomization, returnsRandomized):
    improvedReturnMask = returnsNoRandomization.reshape((-1,1)) < returnsRandomized
    gameNumsNoRandomization = np.nonzero(np.all(~improvedReturnMask,1))[0]
    gameNumsRandomized = [np.nonzero(improvedReturnMask[:,colIdx])[0] \
        for colIdx in range(improvedReturnMask.shape[1])]
    
    return gameNumsNoRandomization, gameNumsRandomized

def executedActionsToTargetVector(pots, availableActions, executedActions):
    amounts = np.zeros((len(pots),5), dtype=np.int)
    raiseAmounts = (pots * np.array([0.5, 1.0, 1.5]).reshape((-1,1))).T
    amounts[:,0] = availableActions[:,0]    # Call amount
    amounts[:,1:4] = raiseAmounts
    amounts[:,-1] = availableActions[:,-1]  # All-in amount
    # 0: fold, 1: call, 2: 0.5 pot, 3: 1 pot, 4: 1.5 pot, 5: all-in
    targetVector = np.zeros((len(pots),6))
    actionIdx = np.argmin(np.abs(amounts - executedActions[:,1].reshape((-1,1))), 1) + 1
    targetVector[np.arange(len(pots)),actionIdx] = 1
    foldMask = executedActions[:,0] == 1
    targetVector[foldMask] = [1,0,0,0,0,0]  # Fold
    
    return targetVector



nGames = 50000
#seed = 5
randomizationRatio = 0.1
randomizedPlayerIdx = 0
nonRandomizedPlayerIdx = np.abs(randomizedPlayerIdx-1)

initGameStates = initRandomGames(nGames)
#initGameStates = initRandomGames(nGames, seed=seed)
initGameContainer = RfFeatures(copy.deepcopy(initGameStates))
agents = [RfAgent(0, initGameContainer, regressor), RfAgent(1, initGameContainer, regressor)]
gamesNoRandomization = playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(initGameContainer))

gamesRandomized = []
for i in range(10):
    print('')
    print(i)
    print('..................')
    agents = [RfAgent(randomizedPlayerIdx, copy.deepcopy(initGameContainer), regressor, 
                      randomizationRatio=randomizationRatio), 
              RfAgent(nonRandomizedPlayerIdx, copy.deepcopy(initGameContainer), regressor)]
    gamesRandomized.append(playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(initGameContainer)))
    

returnsNoRandomization = getReturnsForGames(gamesNoRandomization, randomizedPlayerIdx)
returnsRandomized = np.column_stack(([getReturnsForGames(games, randomizedPlayerIdx) \
    for games in gamesRandomized]))
gameNumsNoRandomization, gameNumsRandomized = getImprovedReturnGamenums(returnsNoRandomization,
                                                                        returnsRandomized)

features, executedActions, availableActions, actingPlayerIdx, pots, gameFailedMask, \
    gameFinishedMask = [], [], [], [], [], [], []

if(len(gameNumsNoRandomization) > 0):
    tmpFeatures, tmpExecutedActions, _, tmpPots, miscDict = \
        gamesNoRandomization.getFeaturesForGameNums(gameNumsNoRandomization)
    features.append(tmpFeatures)
    executedActions.append(tmpExecutedActions)
    pots.append(tmpPots)
    gameFailedMask.append(miscDict['gameFailedMask'])
    gameFinishedMask.append(miscDict['gameFinishedMask'])
    availableActions.append(miscDict['availableActions'])
    actingPlayerIdx.append(miscDict['actingPlayerIdx'])

for gameNums, gameContainer in zip(gameNumsRandomized, gamesRandomized):
    if(len(gameNums) > 0):
        tmpFeatures, tmpExecutedActions, _, tmpPots, miscDict = \
            gameContainer.getFeaturesForGameNums(gameNums)
        features.append(tmpFeatures)
        pots.append(tmpPots)
        executedActions.append(tmpExecutedActions) 
        gameFailedMask.append(miscDict['gameFailedMask'])
        gameFinishedMask.append(miscDict['gameFinishedMask'])
        availableActions.append(miscDict['availableActions'])
        actingPlayerIdx.append(miscDict['actingPlayerIdx'])

features = np.row_stack(features)
executedActions = np.row_stack(executedActions)
availableActions = np.row_stack(availableActions)
pots = np.concatenate(pots)
gameFinishedMask = np.concatenate(gameFinishedMask)
gameFailedMask = np.concatenate(gameFailedMask)
actingPlayerIdx = np.concatenate(actingPlayerIdx)

# Checks
assert np.sum(gameFailedMask) == 0  # No failures
assert np.sum(gameFinishedMask) == len(gameNumsNoRandomization) + \
    np.sum([len(g) for g in gameNumsRandomized])   # All games have ended succesfully

targetActions = executedActionsToTargetVector(pots, availableActions, executedActions)
targetActions = np.argmax(targetActions,1)

m = (~gameFinishedMask) & (actingPlayerIdx == randomizedPlayerIdx)

regressorOld = copy.deepcopy(regressor)
#regressor = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=1, min_samples_split=2, 
#                                verbose=2, n_jobs=-1, random_state=0)
#regressor = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=3, min_samples_split=2, 
#                                verbose=2, n_jobs=-1, class_weight='balanced')
#regressor.fit(features[m], targetActions[m])

#regressor = MLPClassifier(hidden_layer_sizes=(100,100,),max_iter=100,verbose=1)
#regressor.fit(features2[m], targetActions[m])


rndIdx = np.random.choice(len(features), size=len(features), replace=0)
features2 = scaler.transform(features)
for i in range(100):
    regressor.partial_fit(features2[rndIdx][m], targetActions[rndIdx][m])
    print('Iteration: ' + str(i) + '   loss: ' + str(regressor.loss_))



# %%

np.histogram(targetActions[m], 6)

from sklearn.metrics import confusion_matrix

pred = regressor.predict(features[m])

cm = confusion_matrix(targetActions[m], pred)
cm22 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(np.around(cm22,2))

# %%
# Evaluate

nGames = 50000
seed = 200


#initGameStates = initRandomGames(nGames)
initGameStates = initRandomGames(nGames, seed=seed)
gameContainer = RfFeatures(copy.deepcopy(initGameStates))

agents = [RfAgent(0, gameContainer, regressor0), RfAgent(1, gameContainer, regressor)]
#agents = [RndAgent(0), RfAgent(1, gameContainer, regressor)]
gameContainer = playGames(agents, initGameStates, gameContainer)


winAmounts, winPlayerIdx, gameNums = gameContainer.getWinAmounts()

amountPlayer0 = np.sum(winAmounts[winPlayerIdx == 0])
amountPlayer1 = np.sum(winAmounts[winPlayerIdx == 1])

winAmnt = np.max([amountPlayer0, amountPlayer1])
winPlayer = np.argmax([amountPlayer0, amountPlayer1])
loseAmnt = np.min([amountPlayer0, amountPlayer1])

print('\n.........................')
print('win player: ' + str(winPlayer) +', ratio: ' + str(winAmnt / loseAmnt))



# %%


 






