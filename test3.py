#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:08:15 2019

@author: juho
"""



import numpy as np
from numba import jit, njit, prange
import copy
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from holdem_hu_bot.agents import RndAgent, CallAgent, Agent, generateRndActions
from holdem_hu_bot.game_data_container import GameDataContainer
from holdem_hu_bot.common_stuff import suitsOnehotLut, ranksOnehotLut, encodeCardsOnehotNb
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


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def computeFeaturesNb(boardsData, playersData, winLen, ranksOnehotLut, suitsOnehotLut):
    eventFeats = np.zeros((6,winLen))
    
    smallBlind = boardsData[-1,1] # Small blinds amounts are used for normalization
    actingPlayerIdx = playersData[-1,14]
    nonActingPlayerIdx = np.abs(actingPlayerIdx-1)
    
    # Pots, stacks etc. money stuffs
    pots = boardsData[:,0]
    bets = playersData[:,3] + playersData[:,11]
    pots = pots + bets
    stacks = np.asfortranarray(playersData[:,np.array([2,10])])
    
    # Normalized pots and stacks
    potsNormalized = pots / smallBlind
    stacksNormalized = stacks / smallBlind
    
    ownStacksNormalized = stacksNormalized[:, actingPlayerIdx]
    opponentStacksNormalized = stacksNormalized[:, nonActingPlayerIdx]
    
    # Betting round
    visibleCardsMask = boardsData[:,3:8].astype(np.bool_)
    bettingRound = np.sum(visibleCardsMask[:,2:].astype(np.int64),1)
    
    # Put features into array
    idx = min(len(potsNormalized), winLen)
    eventFeats[0, -idx:] = ownStacksNormalized
    eventFeats[1, -idx:-1] = np.abs(np.diff(ownStacksNormalized))
    eventFeats[2, -idx:] = opponentStacksNormalized
    eventFeats[3, -idx:-1] = np.abs(np.diff(opponentStacksNormalized))
    eventFeats[4, -idx:] = potsNormalized
    eventFeats[5, -idx:] = bettingRound+1    # Add 1 so we make difference to default value 0

    # Encode cards one hot
    boardCards = boardsData[-1,8:].reshape((1,-1))
#    boardCards[boardCards == -999] = 0  # Assign zero if failure code because otherwise 
        # 'encodeCardsOnehot' will fail
    visibleCardsMask = boardsData[-1,3:8].astype(np.bool_).reshape((1,-1))
    boardcardSuitsOnehot, boardcardRanksOnehot = encodeCardsOnehotNb(boardCards, visibleCardsMask, 
                                                                     ranksOnehotLut, suitsOnehotLut)
    holecards = playersData[-1:,np.array([0,1,8,9])]
    holecards = holecards.reshape((2,2))[actingPlayerIdx]
#    holecards[holecards == -999] = 0  # Assign zero if failure code because otherwise 
        # 'encodeCardsOnehot' will fail
    holecardSuitsOnehot, holecardRanksOnehot = encodeCardsOnehotNb(holecards.reshape(1,-1), 
                                                                   np.ones(holecards.shape, dtype=np.bool_), 
                                                                   ranksOnehotLut, suitsOnehotLut)

    return eventFeats, boardcardSuitsOnehot[0], boardcardRanksOnehot[0], holecardSuitsOnehot[0], \
        holecardRanksOnehot[0]


#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
#def computeFeaturesWrapperNb(boardsData, playersData, gameDataIndexes, idxIdx, winLen):
#    features = np.zeros((len(idxIdx)-1, 7, winLen+17), dtype=np.float32)
#    
#    for i in prange(1,len(idxIdx)):
#        prevIdx, curIdx = idxIdx[i-1], idxIdx[i]
#        curGameDataIdx = gameDataIndexes[prevIdx:curIdx]
#    
#        eventFeats, boardSuits, boardRanks, holeSuits, holeRanks \
#            = computeFeaturesNb(boardsData[curGameDataIdx], playersData[curGameDataIdx], 
#                                winLen, ranksOnehotLut, suitsOnehotLut)
#        
#        features[i-1,:eventFeats.shape[0],:eventFeats.shape[1]] = eventFeats
#        
#        features[i-1,:2,eventFeats.shape[1]:eventFeats.shape[1]+4] = holeSuits
#        features[i-1,2:,eventFeats.shape[1]:eventFeats.shape[1]+4] = boardSuits
#        features[i-1,:2,eventFeats.shape[1]+4:] = holeRanks
#        features[i-1,2:,eventFeats.shape[1]+4:] = boardRanks
#
#    return features


# %%
# Initialize agent

N_GAMES = 10
RND_AGENT_NUM = 0
AI_AGENT_NUM = np.abs(RND_AGENT_NUM-1)
SEED = 123
WIN_LEN = 20

initGameStates, initStacks = initRandomGames(N_GAMES)
smallBlinds = initGameStates.boards[:,1]
#equities = getEquities(initGameStates)

gameDataContainer = GameDataContainer(N_GAMES)
#agents = [RndAgent(0), RndAgent(1)]
#gameDataContainer = playGames(agents, copy.deepcopy(initGameStates), copy.deepcopy(gameDataCont))

#rndAgent = RndAgent(0)
curGameStates = initGameStates

mockActions = np.zeros((len(curGameStates.availableActions),2), dtype=np.int64) - 999
actionsToExecute = np.zeros((len(curGameStates.availableActions),2), dtype=np.int64) - 999


# %%


@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
#@jit(nopython=True, fastmath=True, nogil=True)
#@njit(parallel=True)
def computeFeaturesWrapperNb(boardsData, playersData, gameDataIndexes, idxIdx, winLen, mask):
#    features = np.zeros((len(idxIdx)-1, 7, winLen+17), dtype=np.float32)
    
    maskIndexes = np.nonzero(mask)[0] + 1
    features = np.zeros((len(maskIndexes), 7, winLen+17), dtype=np.float32)

    stIdx, endIdx = np.zeros(np.sum(mask), dtype=np.uint64), np.zeros(np.sum(mask), np.uint64)
    for k in range(len(stIdx)):
        iMask = maskIndexes[k]
        stIdx[k], endIdx[k] = idxIdx[iMask-1], idxIdx[iMask]
        
        
    
##    for i in prange(1,len(idxIdx)):
##    for i in range(1,len(idxIdx)):
#    for i in range(len(maskIndexes)):
#        iMask = maskIndexes[i]
##        print(iMask)
##        prevIdx, curIdx = idxIdx[i-1], idxIdx[i]
#        prevIdx, curIdx = idxIdx[iMask-1], idxIdx[iMask]
##        print(prevIdx,curIdx)
#        curGameDataIdx = gameDataIndexes[prevIdx:curIdx]

#        eventFeats, boardSuits, boardRanks, holeSuits, holeRanks \
#            = computeFeaturesNb(boardsData[curGameDataIdx], playersData[curGameDataIdx], 
#                                winLen, ranksOnehotLut, suitsOnehotLut)
#        
#        features[i-1,:eventFeats.shape[0],:eventFeats.shape[1]] = eventFeats
#        
#        features[i-1,:2,eventFeats.shape[1]:eventFeats.shape[1]+4] = holeSuits
#        features[i-1,2:,eventFeats.shape[1]:eventFeats.shape[1]+4] = boardSuits
#        features[i-1,:2,eventFeats.shape[1]+4:] = holeRanks
#        features[i-1,2:,eventFeats.shape[1]+4:] = boardRanks


    for i in range(len(stIdx)):
        print(i)
        prevIdx, curIdx = stIdx[i], endIdx[i] 
        curGameDataIdx = gameDataIndexes[prevIdx:curIdx]

        eventFeats, boardSuits, boardRanks, holeSuits, holeRanks \
            = computeFeaturesNb(boardsData[curGameDataIdx], playersData[curGameDataIdx], 
                                winLen, ranksOnehotLut, suitsOnehotLut)
        
        features[i,:eventFeats.shape[0],:eventFeats.shape[1]] = eventFeats
        
        features[i,:2,eventFeats.shape[1]:eventFeats.shape[1]+4] = holeSuits
        features[i,2:,eventFeats.shape[1]:eventFeats.shape[1]+4] = boardSuits
        features[i,:2,eventFeats.shape[1]+4:] = holeRanks
        features[i,2:,eventFeats.shape[1]+4:] = boardRanks

    return features


mask = maskAiAgent
features = computeFeaturesWrapperNb(gameData['boardsData'], gameData['playersData'], gameDataIndexes,
                                    idxIdx, WIN_LEN, mask)

#print(features)

#np.sum(mask)

# %%

    def getMasks(gameStates, playerNumber):
        playerNum = playerNumber
        
        actingPlayerNum = np.argmax(gameStates.players[:,6].reshape((-1,2)),1)
        actingPlayerMask = actingPlayerNum == playerNum
        gameEndMask = gameStates.controlVariables[:,1] == 1
        gameFailedMask = gameStates.controlVariables[:,1] == -999
        allMask = actingPlayerMask & ~gameEndMask & ~gameFailedMask
        
        return allMask, actingPlayerMask, gameEndMask, gameFailedMask


#while(1):
        
    gameDataContainer.addData(curGameStates, mockActions)

    # Rnd agent actions
    maskRndAgent, _, _, _, = getMasks(curGameStates, RND_AGENT_NUM)
    actionsRndAgent = generateRndActions(curGameStates.availableActions[maskRndAgent], seed=SEED)
    
    # Ai agent actions
    maskAiAgent, _, _, _, = getMasks(curGameStates, AI_AGENT_NUM)
    gameDataIndexes, gameNums, idxIdx = gameDataContainer.getAllIndexes()
    gameData, _ = gameDataContainer.getData()
    features = computeFeaturesWrapperNb(gameData['boardsData'], gameData['playersData'], gameDataIndexes,
                                        idxIdx, WIN_LEN)
    
    
    actionsToExecute[:] = -999
    actionsToExecute[maskRndAgent] = actionsRndAgent
    actionsToExecute[maskAgent1] = actionsAgent1
    

    curGameStates = executeActions(curGameStates, actionsToExecute)
    
    nValidGames = np.sum(curGameStates.controlVariables[:,1]==0)
    print(nValidGames)
    
    if(nValidGames == 0):
        break
    

# Save also the last game state
actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
gameDataContainer.addData(gameStates, actionsToExecute)


# %%


WIN_LEN = 20


gameDataIndexes, gameNums, idxIdx = gameDataContainer.getAllIndexes()
gameData, _ = gameDataContainer.getData()
boardsData, playersData = gameData['boardsData'], gameData['playersData']




aa = computeFeaturesWrapperNb(boardsData, playersData, gameDataIndexes, idxIdx)


    
    

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

















