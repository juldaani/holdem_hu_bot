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


PATH = '/home/juho/dev_folder/data/poker_ai/2020-03-09_10-19-07/'

pathData = os.path.join(PATH,'evaluation')
fNames = np.array([f for f in os.listdir(pathData) if 'eval_dummy_opponents.npy' in f])

# Sort filenames according to generation number
iteration = np.array([int(f.split('_')[0]) for f in fNames])
sorter = np.argsort(iteration)

iteration = iteration[sorter]
fNames = fNames[sorter]


dummyEvalRes = []
for fName in fNames:
    dummyEvalRes.append(np.load(os.path.join(pathData, fName), allow_pickle=1).item())

tmp = {key:[] for key in dummyEvalRes[0].keys()}

for data in dummyEvalRes:
    for key,val in zip(data.keys(), data.values()):
        tmp[key].append(val)

for key,val in zip(tmp.keys(), tmp.values()):
   tmp[key] = np.row_stack((val))
dummyEvalRes = tmp




plt.plot(iteration, dummyEvalRes['all_in_agent'])
#plt.plot(winAmounts[:,4])


# %%