#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:30:35 2019

@author: juho
"""

import numpy as np
from hand_eval.params import cardToInt, intToCard


def getCardOneHotLuts():
    suitsOnehotLut = np.zeros((52,4), dtype=np.int8)
    ranksOnehotLut = np.zeros((52,13), dtype=np.int8)
    ranks = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
    
    for cardNum in intToCard:
        card = intToCard[cardNum]
        
        suitsOnehotLut[cardNum,0], suitsOnehotLut[cardNum,1] = 'c' in card, 'd' in card
        suitsOnehotLut[cardNum,2], suitsOnehotLut[cardNum,3] = 'h' in card, 's' in card
        ranksOnehotLut[cardNum] = [rank in card for rank in ranks]

    return suitsOnehotLut, ranksOnehotLut


def encodeCardsOnehot(cards, visibleCardsMask, ranksOnehotLut, suitsOnehotLut):
    visibleCardsMaskRanks = np.repeat(visibleCardsMask, ranksOnehotLut.shape[1], axis=1)
    visibleCardsMaskSuits = np.repeat(visibleCardsMask, suitsOnehotLut.shape[1], axis=1)
    cardSuitsOnehot = suitsOnehotLut[cards].reshape((len(cards), cards.shape[1]*suitsOnehotLut.shape[1]))
    cardRanksOnehot = ranksOnehotLut[cards].reshape((len(cards), cards.shape[1]*ranksOnehotLut.shape[1]))
    cardSuitsOnehot[~visibleCardsMaskSuits.astype(np.bool)] = 0    # Set nonvisible cards to zero
    cardRanksOnehot[~visibleCardsMaskRanks.astype(np.bool)] = 0
    
    return cardSuitsOnehot, cardRanksOnehot


suitsOnehotLut, ranksOnehotLut = getCardOneHotLuts()



# %%