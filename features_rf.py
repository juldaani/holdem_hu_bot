#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:18:43 2019

@author: juho
"""

import numpy as np
from holdem_hu_bot.game_data_container import GameDataContainer

"""
- effective stack
- normalize stacks and pot by dividing with small blind
- pot odds
- equities
- encode hole and board cards (one-hot-encoding etc...)
- is small blind
- 
"""


# %%

from texas_hu_engine.wrappers import initRandomGames, GameState
from texas_hu_engine.engine_numba import getBoardCards, getGameEndState
from equity_calculator.equity_calculator import computeEquities
from holdem_hu_bot.common_stuff import suitsOnehotLut, ranksOnehotLut, encodeCardsOnehot


#%%    
#
#def encodeCardsOnehot(cards, visibleCardsMask, ranksOnehotLut, suitsOnehotLut):
#    visibleCardsMaskRanks = np.repeat(visibleCardsMask, ranksOnehotLut.shape[1], axis=1)
#    visibleCardsMaskSuits = np.repeat(visibleCardsMask, suitsOnehotLut.shape[1], axis=1)
#    cardSuitsOnehot = suitsOnehotLut[cards].reshape((len(cards), cards.shape[1]*suitsOnehotLut.shape[1]))
#    cardRanksOnehot = ranksOnehotLut[cards].reshape((len(cards), cards.shape[1]*ranksOnehotLut.shape[1]))
#    cardSuitsOnehot[~visibleCardsMaskSuits.astype(np.bool)] = 0    # Set nonvisible cards to zero
#    cardRanksOnehot[~visibleCardsMaskRanks.astype(np.bool)] = 0
#    
#    return cardSuitsOnehot, cardRanksOnehot



    
    

# %%


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
        isSmallBlindPlayer = playersData[:,[4,12]][np.arange(len(actingPlayerIdx)),actingPlayerIdx]
        
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

        gameValidMask = ~controlVariables[:,1].astype(np.bool)    # Tells if the game is still on
        bettingRound = visibleCardsMask[:,2:].astype(np.int)
        
        return data, indexes
        


N = 3

gameStates = initRandomGames(N)
asd = Features(N)
asd.addData(gameStates)

asd.equities[1]['flop']

data, indexes = asd.getFeatures()

# %%


cardToSuit = np.zeros((52,4), dtype=np.int8)
cardToRank = np.zeros((52,13), dtype=np.int8)
ranks = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
for cardNum in intToCard:
    card = intToCard[cardNum]
    print(cardNum,card)
    
    cardToSuit[cardNum,0], cardToSuit[cardNum,1] = 'c' in card, 'd' in card
    cardToSuit[cardNum,2], cardToSuit[cardNum,3] = 'h' in card, 's' in card
    cardToRank[cardNum] = [rank in card for rank in ranks]














# %%
  
    
    
    
    
    
    
    
    
    
    
    
    
    