#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:03:13 2019

@author: juho
"""

# %%

import os
import numpy as np
import matplotlib.pyplot as plt


fNames = np.array([f for f in os.listdir('data') if 'win_amounts.npy' in f])

# Sort filenames according to generation number
generationNums = np.array([int(f.split('_')[0]) for f in fNames])
sorter = np.argsort(generationNums)
fNames = fNames[sorter]


generationMeanFitness, generationBestFitness = [], []
for fName in fNames:
    winAmounts = np.load('data/' + fName)
    generationMeanFitness.append(np.mean(winAmounts))
    generationBestFitness.append(np.max(np.mean(winAmounts,1)))
    
    print(fName)
    

plt.plot(generationMeanFitness)
plt.plot(generationBestFitness)

# %%