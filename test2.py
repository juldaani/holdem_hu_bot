#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:08:15 2019

@author: juho
"""



import numpy as np
from numba import jit
import copy
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from holdem_hu_bot.agents import RndAgent, CallAgent, Agent, generateRndActions
from holdem_hu_bot.game_data_container import GameDataContainer
from texas_hu_engine.wrappers import initRandomGames, executeActions, createActionsToExecute
from equity_calculator.equity_calculator import computeEquities




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
    
    # Save also the last game state
    actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
    gameDataContainer.addData(gameStates, actionsToExecute)
    
    return gameDataContainer


def getWinAmounts(gameContainer, initStacks):
    data, _ = gameContainer.getData()
    lastIndexes, _ = gameContainer.getLastIndexes()
    finalStacks = data['playersData'][lastIndexes][:,[2,10]]
    
    return finalStacks - initStacks


def getEquities(gameStates, seed=-1):
    boardCards = initGameStates.boards[:,8:]
    player0HoleCards = initGameStates.players[::2,:2]
    player1HoleCards = initGameStates.players[1::2,:2]
    
    # Dimensions, 1: game number, 2: preflop/flop/turn/river, 3: player index
    equities = np.zeros((len(initGameStates.boards),4,2))
    equities[:,0,0], equities[:,1,0], equities[:,2,0], equities[:,3,0] = \ 
        computeEquities(player0HoleCards, boardCards, seed=seed)
    equities[:,0,1], equities[:,1,1], equities[:,2,1], equities[:,3,1] = \
        computeEquities(player1HoleCards, boardCards, seed=seed)

    return equities


def computeFeatures(boardsData, playersData, availableActions, controlVariables, equities,
                    gameNumbers):
#    boardsData = initGameStates.boards
#    playersData = initGameStates.players
#    availableActions = initGameStates.availableActions
#    controlVariables = initGameStates.controlVariables
#    equities = equities
#    gameNumbers = np.arange(len(boardsData))
    
    smallBlinds = boardsData[:,1] # Small blinds amounts are used for normalization
    actingPlayerIdx = playersData[1::2,6]
    nonActingPlayerIdx = (~actingPlayerIdx.astype(np.bool)).astype(np.int)
    
    isPlayerSmallBlind = np.column_stack((playersData[::2,4],playersData[1::2,4]))
    isPlayerSmallBlind = isPlayerSmallBlind[np.arange(len(actingPlayerIdx)), actingPlayerIdx]
    
    # Pots, stacks etc. money stuffs
    pots = boardsData[:,0]
    bets = playersData[::2,3] + playersData[1::2,3]
    pots = pots + bets
    stacks = np.column_stack((playersData[::2,2], playersData[1::2,2]))
    
    # Normalized pots and stacks
    potsNormalized = pots / smallBlinds
    stacksNormalized = stacks / smallBlinds.reshape((-1,1))
    ownStacksNormalized = stacksNormalized[np.arange(len(stacks)), actingPlayerIdx]
    opponentStacksNormalized = stacksNormalized[np.arange(len(stacks)), nonActingPlayerIdx]
    
    # Betting round
    visibleCardsMask = boardsData[:,3:8].astype(np.bool)
    bettingRound = visibleCardsMask[:,2:].astype(np.int)
    bettingRoundIdx = np.sum(bettingRound,1)
    
    # Equities
    actingPlayerEquities = equities[gameNumbers,bettingRoundIdx,actingPlayerIdx]
    
#    availableActionsNormalized = availableActions / smallBlinds
#    availableActionsNormalized[availableActions < 0] = -1
    
#    gameFinishedMask = controlVariables[:,1] == 1   # Tells if the game has finished succesfully
#    gameFailedMask = controlVariables[:,1] == -999  # Tells if an error has occured
#
#    miscDict = {'availableActionsNormalized':availableActionsNormalized, 
#                'availableActions':availableActions, 'gameNumbers':gameNumbers,
#                'gameFinishedMask':gameFinishedMask, 'gameFailedMask':gameFailedMask, 
#                'actingPlayerIdx':actingPlayerIdx}
    
    features = np.column_stack((isPlayerSmallBlind, bettingRoundIdx, actingPlayerEquities, 
                                potsNormalized, ownStacksNormalized, opponentStacksNormalized))

    return features#, miscDict
    


class AiAgent(Agent):
    
    def __init__(self, playerNumber, featuresFunc, aiModel, equities, foldThres, 
                 randomizationRatio=0.0):
        self.playerNumber = playerNumber
        self.featuresFunc = featuresFunc
        self.aiModel = aiModel
        self.equities = equities
        self.foldThres = foldThres
        self.randomizationRatio = randomizationRatio
    
    def getActions(self, gameStates):
        mask, _, _, _ = super().getMasks(gameStates)
        aiModel = self.aiModel
        aiModel.verbose = 0
        featuresFunc = self.featuresFunc
        equities = self.equities
        foldThres = self.foldThres
        randomizationRatio = self.randomizationRatio
        
        # Return if all hands are already finished for the current player
        if(np.sum(mask) == 0):
            return np.zeros((0,2), dtype=np.int64), mask
#
#        def getMasks(gameStates, playerNum):
##            playerNum = self.playerNumber
#            
#            actingPlayerNum = np.argmax(gameStates.players[:,6].reshape((-1,2)),1)
#            actingPlayerMask = actingPlayerNum == playerNum
#            gameEndMask = gameStates.controlVariables[:,1] == 1
#            gameFailedMask = gameStates.controlVariables[:,1] == -999
#            allMask = actingPlayerMask & ~gameEndMask & ~gameFailedMask
#            
#            return allMask, actingPlayerMask, gameEndMask, gameFailedMask
#        
#        gameStates = initGameStates
#        playerNum = 1
#        mask, _, _, _ = getMasks(gameStates, playerNum)
#        featuresFunc = computeFeatures
#        equities = equities
#        aiModel = regressor
#        foldThres = 0.67
#        randomizationRatio = 0.5
        
        playersMask = np.tile(mask, (2,1)).T.flatten()
        playersData = gameStates.players[playersMask]
        boardsData = gameStates.boards[mask]
        availableActs = gameStates.availableActions[mask]
        gameNums = np.nonzero(mask)[0]
        
        features = featuresFunc(boardsData, playersData, availableActs, 
                                gameStates.controlVariables[mask], equities, gameNums)
        actions = aiModel.predict(features)
        
        foldMask = actions[:,0] > foldThres
        
        smallBlinds = boardsData[:,1]
        actionAmounts = (actions[:,1] * smallBlinds).astype(np.int)
        validAmountsMask = (actionAmounts == availableActs[:,0]) | \
            (actionAmounts >= availableActs[:,1]) & (actionAmounts <= availableActs[:,2])
        closestIdx = np.argmin(np.abs(availableActs - actionAmounts.reshape((-1,1))),1)
        invalidIdx = np.nonzero(~validAmountsMask)[0]
        actionAmounts[~validAmountsMask] = availableActs[invalidIdx, closestIdx[invalidIdx]]
        actionAmounts[foldMask] = -1
        acts = createActionsToExecute(actionAmounts)
        
        # Randomization
        rndIdx = np.random.choice(len(availableActs), size=int(len(availableActs)*randomizationRatio),
                                  replace=0)        
        acts[rndIdx] = generateRndActions(availableActs[rndIdx])
        
        return acts, mask



# %%
# Initialize agent

nGames = 1000

initGameStates, initStacks = initRandomGames(nGames)
smallBlinds = initGameStates.boards[:,1]
equities = getEquities(initGameStates)

gameDataCont = GameDataContainer(nGames)
agents = [RndAgent(0), RndAgent(1)]
gameDataContainer = playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(gameDataCont))

data, _ = gameDataContainer.getData()
features = computeFeatures(data['boardsData'], 
                           GameDataContainer.unflattenPlayersData(data['playersData']),
                           data['availableActionsData'], data['controlVariablesData'], equities,
                           np.random.randint(0, high=nGames, size=len(data['boardsData'])))

regressor = ExtraTreesRegressor(n_estimators=10, min_samples_leaf=10, min_samples_split=4, 
                                verbose=2, n_jobs=-1)
regressor.fit(features, data['actions'])



# %%
# Self-play

nGames = 20000
playerIdxToOptimize = 1
nOptimizationRounds = 5
foldThres = 0.8
seed = -1

initGameStates, initStacks = initRandomGames(nGames, seed=seed)
smallBlinds = initGameStates.boards[:,1]
equities = getEquities(initGameStates, seed=seed)

gameDataCont = GameDataContainer(nGames)

agents = [AiAgent(0, computeFeatures, regressor, equities, foldThres), 
          AiAgent(playerIdxToOptimize, computeFeatures, regressor, equities, foldThres)]
gameDataOriginal = playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(gameDataCont))

agents = [AiAgent(0, computeFeatures, regressor, equities, foldThres), 
          AiAgent(playerIdxToOptimize, computeFeatures, regressor, equities, foldThres, 
                  randomizationRatio=0.7)]
gameDataRandomized = [playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(gameDataCont)) \
    for i in range(nOptimizationRounds)]

# Add also game data without randomization
gameDataRandomized.append(gameDataOriginal)


## %%
# Create training data

winAmounts = [getWinAmounts(c, initStacks)[:,playerIdxToOptimize] for c in gameDataRandomized]
winAmounts = np.column_stack((winAmounts))
winAmounts = winAmounts / smallBlinds.reshape((-1,1))

highestReturnGameContainerIdx = np.argmax(winAmounts,1)
gameNums = np.arange(nGames)
gameNumsForGameContainers, winAmounts2 = [[] for i in range(len(gameDataRandomized))], \
    [[] for i in range(len(gameDataRandomized))] 
for idx, gameNum in zip(highestReturnGameContainerIdx, gameNums):
    gameNumsForGameContainers[idx].append(gameNum)
    winAmounts2[idx].append(winAmounts[gameNum,idx])

boardsData, playersData, availableActions, controlVariables, actions, gameNumbers, winAmounts3 = \
    [], [], [], [], [], [], []
for gameNum, winAmount, containerNum in zip(gameNumsForGameContainers, winAmounts2, 
                                            np.arange(len(winAmounts2))):
    if(len(gameNum) > 0):
        tmpIndexes, tmpGameNumbers = gameDataRandomized[containerNum].getIndexesForGameNums(gameNum)
        data, _ = gameDataRandomized[containerNum].getData()
        boardsData.append(data['boardsData'][tmpIndexes])
        playersData.append(data['playersData'][tmpIndexes])
        availableActions.append(data['availableActionsData'][tmpIndexes])
        controlVariables.append(data['controlVariablesData'][tmpIndexes])
        actions.append(data['actions'][tmpIndexes])
        gameNumbers.append(tmpGameNumbers)
        winAmountDict = {gameN:winAmnt for winAmnt, gameN in zip(winAmount, gameNum)}
        winAmounts3.append([winAmountDict[gameN] for gameN in tmpGameNumbers])

boardsData = np.row_stack(boardsData)
playersData = np.row_stack(playersData)
availableActions = np.row_stack(availableActions)
controlVariables = np.row_stack(controlVariables)
actions = np.row_stack(actions)
gameNumbers = np.concatenate(gameNumbers)
winAmounts3 = np.concatenate(winAmounts3)

winAmountsOriginal = getWinAmounts(gameDataOriginal, initStacks)[:,playerIdxToOptimize]
winAmountOptimized = winAmounts[np.arange(len(winAmounts)), highestReturnGameContainerIdx]
print('\nwin amounts original: ' + str(np.sum(winAmountsOriginal/smallBlinds)/nGames))
print('win amounts optimized: ' + str(np.sum(winAmountOptimized)/nGames))

## %%

gameNotEndMask = ~(controlVariables[:,1] != 0)

boardsData = boardsData[gameNotEndMask]
playersData = GameDataContainer.unflattenPlayersData(playersData[gameNotEndMask])
availableActions = availableActions[gameNotEndMask]
controlVariables = controlVariables[gameNotEndMask]
actions = actions[gameNotEndMask]
gameNumbers = gameNumbers[gameNotEndMask]
winAmounts3 = winAmounts3[gameNotEndMask]

features = computeFeatures(boardsData, playersData, availableActions, controlVariables, equities, 
                           gameNumbers)


## %%
# Train regressor

actingPlayerIdx = playersData[1::2,6]
optimizePlayerMask = actingPlayerIdx == playerIdxToOptimize

smallBlinds = boardsData[:,1]
targetActions = actions / np.row_stack(smallBlinds)
targetActions[actions == -1] = 0
targetActions[actions[:,0] == 1] = [1,0]

# Upsample folds
upsampleRatio = 5
foldMask = actions[:,0] == 1
foldFeatures = np.tile(features[foldMask], (upsampleRatio,1))
foldTargetActions = np.tile(targetActions[foldMask], (upsampleRatio,1))

x = np.row_stack((features[optimizePlayerMask],foldFeatures))
y = np.row_stack((targetActions[optimizePlayerMask],foldTargetActions))
shuffler = np.arange(len(x))
np.random.shuffle(shuffler)

regressorOld = copy.deepcopy(regressor)
regressor = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=10, min_samples_split=4, 
                                verbose=2, n_jobs=-1)
regressor.fit(x[shuffler], y[shuffler])


# Plot
preds = regressor.predict(features[optimizePlayerMask])
m = targetActions[optimizePlayerMask,0] == 1
np.mean(preds[m,0])
np.percentile(preds[m,0], 15)
np.mean(preds[~m,0])
np.percentile(preds[~m,0], 97)
plt.hist(preds[m,0], 20, alpha=0.5, color='red')
plt.hist(preds[~m,0], 40, alpha=0.5, color='blue')
plt.show()

#plt.hist(preds[:,0], 30, alpha=0.5, color='red')



# %%
# Evaluate

nGames = 10000
seed = 12445
seed = np.random.randint(456488)

initGameStates, initStacks = initRandomGames(nGames, seed=seed)
smallBlinds = initGameStates.boards[:,1]
equities = getEquities(initGameStates, seed=seed)

gameCont = GameDataContainer(nGames)
agents = [AiAgent(0, computeFeatures, regressorOld, equities, 0.8),
          AiAgent(playerIdxToOptimize, computeFeatures, regressor, equities, 0.8)]
#agents = [RndAgent(0), AiAgent(playerIdxToOptimize, computeFeatures, regressor, equities, 0.8)]
#agents = [CallAgent(0), AiAgent(playerIdxToOptimize, computeFeatures, regressor, equities, 0.8)]
          
gameCont = playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(gameCont))


winAmounts = getWinAmounts(gameCont, initStacks)[:,playerIdxToOptimize]
winAmountsNormalized = winAmounts / smallBlinds

winRateSmallBlindsPerGame = np.sum(winAmountsNormalized) / nGames

print('win rate: ' + str(winRateSmallBlindsPerGame) + ' small blinds / hand')



#
## Old regressor
#gameCont = GameDataContainer(nGames)
#agents = [AiAgent(0, computeFeatures, regressorOld, equities, 0.9),
#          AiAgent(playerIdxToOptimize, computeFeatures, regressorOld, equities, 0.9)]
#gameCont = playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(gameCont))
#
#winAmounts = getWinAmounts(gameCont, initStacks)[:,playerIdxToOptimize]
#winAmountsNormalized = winAmounts / smallBlinds
#
#winRateSmallBlindsPerGame = np.sum(winAmountsNormalized) / nGames
#
#print('win rate: ' + str(winRateSmallBlindsPerGame) + ' small blinds / hand')
#




# %%




preds = regressor.predict(features[rndPlayerMask])

#asd = np.column_stack((preds, targetActions[rndPlayerMask], features[rndPlayerMask,2]))

m = targetActions[rndPlayerMask,0] == 1

np.mean(preds[m,0])
np.percentile(preds[m,0], 15)

np.mean(preds[~m,0])
np.percentile(preds[~m,0], 97)


plt.hist(preds[m,0], 20, alpha=0.5, color='red')
plt.hist(preds[~m,0], 40, alpha=0.5, color='blue')
plt.show()

















