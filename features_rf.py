#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:18:43 2019

@author: juho
"""

import numpy as np
import pandas as pd
from holdem_hu_bot.game_data_container import GameDataContainer
from numba import jit

"""
- effective stack
- normalize stacks and pot by dividing with small blind
- pot odds
- equities
- encode hole and board cards (one-hot-encoding etc...)
- is small blind
- 
"""



from texas_hu_engine.wrappers import initRandomGames, GameState, executeActions, createActionsToExecute
from texas_hu_engine.engine_numba import getBoardCards, getGameEndState
from equity_calculator.equity_calculator import computeEquities
from holdem_hu_bot.common_stuff import suitsOnehotLut, ranksOnehotLut, encodeCardsOnehot


    



class Features(GameDataContainer):
    
    def __init__(self, nGames): 
        GameDataContainer.__init__(self, nGames)
        self.equities = None
        
        
    def addData(self, gameStates):
        if(self.equities == None):
            equities = {0:{}, 1:{}}
       
            boardCards = gameStates.boards[:,-5:]
            player0HoleCards = gameStates.players[::2,:2]
            player1HoleCards = gameStates.players[1::2,:2]
            
            equities[0]['preflop'], equities[0]['flop'], equities[0]['turn'], \
                equities[0]['river'] = computeEquities(player0HoleCards, boardCards)
            equities[1]['preflop'], equities[1]['flop'], equities[1]['turn'], \
                equities[1]['river'] = computeEquities(player1HoleCards, boardCards)
            
            self.equities = equities
            
        super().addData(gameStates)
        
    
    def getFeatures(self):
        data, indexes = super().getData()
        
        lastIndexes = [curIndexes[-1] for curIndexes in indexes]    # Get indexes for the last game states
        boardsData = data['boardsData'][lastIndexes]
        playersData = data['playersData'][lastIndexes]
        availableActions = data['availableActionsData'][lastIndexes]
        controlVariables = data['controlVariablesData'][lastIndexes]
        
        smallBlinds = np.row_stack(boardsData[:,1]) # Small blinds amounts are used for normalization
        actingPlayerIdx = np.argmax(playersData[:,[6,14]], 1)
        nonActingPlayerIdx = (~actingPlayerIdx.astype(np.bool)).astype(np.int)
        isSmallBlindPlayer = playersData[:,[4,12]][np.arange(len(actingPlayerIdx)),actingPlayerIdx].astype(
                np.uint8)
        
        # Pots, stacks etc. money stuffs
        pots = boardsData[:,0]
        bets = playersData[:,3] + playersData[:,11]
        stacks = playersData[:,[2,10]]
        potsNormalized = (pots + bets) / smallBlinds.flatten()
        actionsNormalized = availableActions / smallBlinds
        ownStacksNormalized = stacks[np.arange(len(stacks)),actingPlayerIdx] / smallBlinds.flatten()
        opponentStacksNormalized = stacks[np.arange(len(stacks)),nonActingPlayerIdx] / smallBlinds.flatten()
        
        # Encode cards one hot
        boardCards = boardsData[:,8:]
        visibleCardsMask = boardsData[:,3:8].astype(np.bool)
        boardcardSuitsOnehot, boardcardRanksOnehot  = encodeCardsOnehot(boardCards, visibleCardsMask, 
                                                                        ranksOnehotLut, suitsOnehotLut)
        holecards = playersData[:,[0,1,8,9]]
        holecards = holecards.reshape((len(holecards),2,2))
        holecards = holecards[np.arange(len(holecards)),actingPlayerIdx]
        holecardSuitsOnehot, holecardRanksOnehot = encodeCardsOnehot(holecards,
                                                                     np.ones(holecards.shape, dtype=np.bool), 
                                                                     ranksOnehotLut, suitsOnehotLut)

        gameValidMask = ~controlVariables[:,1].astype(np.bool)    # Tells if the current game is still on
        bettingRound = visibleCardsMask[:,2:].astype(np.int)

        # Get equities
        bettingRoundNames = ['preflop','flop','turn','river']
        equities = np.zeros(len(smallBlinds), dtype=np.float32)
        for i in range(len(actingPlayerIdx)):
            curPlayerIdx = actingPlayerIdx[i]
            curBettingRound = np.argmax(bettingRound[i])
            equity = self.equities[curPlayerIdx][bettingRoundNames[curBettingRound]][i]
            equities[i] = equity
        
        return {'isSmallBlindPlayer':isSmallBlindPlayer, 'bettingRound':bettingRound, 'equities':equities,
                
                'potsNoarmalized':potsNormalized, 'actionsNormalized':actionsNormalized, 
                'ownStacksNormalized':ownStacksNormalized,
                'opponentStacksNormalized':opponentStacksNormalized, 
                
                'boardcardSuits':boardcardSuitsOnehot, 'boardcardRanks':boardcardRanksOnehot,
                'holecardSuits':holecardSuitsOnehot, 'holecardRanks':holecardRanksOnehot,
                
                'gameValidMask':gameValidMask}
        
        
        
        


from abc import ABC, abstractmethod


class Agent(ABC):
    
    def __init__(self, playerNumber):
        self.playerNumber = playerNumber
    
    @abstractmethod
    def getActions(self, gameStates):
        pass
    
        
@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def generateRndActions(availableActions):
    foldProb = 0.1
    foldProbArr = np.zeros(100, dtype=np.bool_)
    foldProbArr[:int(foldProb*100)] = True
    
    allInRaiseProb = 0.1
    allInRaiseProbArr = np.zeros(100, dtype=np.bool_)
    allInRaiseProbArr[:int(allInRaiseProb*100)] = True
    
    amounts = np.zeros(len(availableActions), dtype=np.int64)
    for i in range(len(availableActions)):
        curActions = availableActions[i]
        
        # Fold ?
        isFold = foldProbArr[np.random.randint(len(foldProbArr))]
        if(isFold):
            amounts[i] = -1
            continue
        
        # Call or raise ?
        callOrRaise = np.random.randint(np.sum(curActions[:2] > -1))  # 0 = call, 1 = raise
        if(callOrRaise == 0):   # Call
            amounts[i] = curActions[callOrRaise]
            continue
        if(callOrRaise == 1):   # Raise
            isAllInRaise = allInRaiseProbArr[np.random.randint(len(allInRaiseProbArr))]
            if(isAllInRaise):
                amounts[i] = curActions[-1]
                continue
            
            lowRaise, highRaise = curActions[1], curActions[2]
            amounts[i] = np.random.randint(lowRaise, highRaise+1)
            
    return createActionsToExecute(amounts)


class RndAgent(Agent):
    
    def getActions(self, gameStates):
        playerNum = self.playerNumber
        
        actingPlayerNum = np.argmax(gameStates.players[:,6].reshape((-1,2)),1)
        playerMask = actingPlayerNum == playerNum
        gameEndMask = gameStates.controlVariables[:,1] == 1
        gameFailedMask = gameStates.controlVariables[:,1] == -999
        mask = playerMask & ~gameEndMask & ~gameFailedMask
        
        availableActions = gameStates.availableActions[mask]
        
        return generateRndActions(availableActions), mask



class PlayGames:

    def __init__(self, agents):
        self.agents = agents
    
    def initRandomGames(self, numGames):    
        self.gameStates = initRandomGames(numGames)

#availableActions = gameStates.availableActions[[1,4,-1,-2]]
#aaa = generateRndActions(np.tile(availableActions,(1000000,1)))


# %%

agents = [RndAgent(0), RndAgent(1)]
#agent.getActions(gameStates)



N = 10

gameStates = initRandomGames(N)

# %%

actionsAgent0, maskAgent0 = agents[0].getActions(gameStates)
actionsAgent1, maskAgent1 = agents[1].getActions(gameStates)

actionsToExecute = np.zeros((len(gameStates.availableActions),2), dtype=np.int64)-999
actionsToExecute[maskAgent0] = actionsAgent0
actionsToExecute[maskAgent1] = actionsAgent1

gameStates = executeActions(gameStates, actionsToExecute)

print(np.sum(gameStates.controlVariables[:,1]==0))

#gameStates.controlVariables
#np.argmax(gameStates.players[:,6].reshape((-1,2)),1)


# %%



  
    
    
    
    
    
    
    
    
    
    
    
    
    