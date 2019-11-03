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


PATH = '/home/juho/dev_folder/asdf/data/2019-11-02_18-49-52/'

pathData = os.path.join(PATH,'evaluation')
fNames = np.array([f for f in os.listdir(pathData) if 'win_amounts.npy' in f])

# Sort filenames according to generation number
generationNums = np.array([int(f.split('_')[0]) for f in fNames])
sorter = np.argsort(generationNums)
fNames = fNames[sorter]


winAmounts = []
for fName in fNames:
    winAmounts.append(np.load(os.path.join(pathData, fName)))
winAmounts = np.row_stack(winAmounts)



plt.plot(winAmounts)
#plt.plot(winAmounts[:,4])


# %%