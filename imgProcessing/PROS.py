# -*- coding: utf-8 -*-
"""
Perpendicular Robot Optimization System - PROS

Created on Thu Jul 25 22:38:21 2024

@author: Mateo-drr
"""

import byble as byb
from pathlib import Path
import numpy as np

#from scipy.optimize import minimize
from confidenceMap import confidenceMap
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from skimage.transform import resize

#PARAMS
date='05Jul'
confname='05julyconfig.json'

#Get files in the directory
current_dir = Path(__file__).resolve().parent.parent 
datapath = current_dir / 'data' / 'acquired' / date / 'processed'
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

#load the configuration of the experiment
conf = byb.loadConf(datapath,confname)

#organize the data as [coord,q rot, id]
positions = []
for i,coord in enumerate(conf['tcoord']):
    positions.append(coord + conf['quater'][i] + [i])
    
xmove = np.array(positions[:41])
ymove = np.array(positions[41:])

#create fake x and y path positions
center = xmove[20] # same as ymove[20]

#Duplicate the positions to make a grid of movements
xfake = np.reshape(np.tile(xmove,(41,1)), (41,41,8))
yfake = np.reshape(np.tile(ymove,(41,1)), (41,41,8))

#Basically I copy the y position to all the elements in the first set
#Effectively shifting all the acquisition cordinates in y (same for x)
#fake x positions
for i,pos in enumerate(ymove):
    xfake[i, :, 1] = pos[1]
#fake x positions
for i,pos in enumerate(xmove):
    yfake[i, :, 0] = pos[0]


def costFunc(img):
    
    #Calculate confidence map
    cmap = confidenceMap(img,rsize=True)

    ###########################################################################
    #Image based metrics
    ###########################################################################        

    #Apply hilbert transform    
    #img = byb.envelope(img)
    
    #Collapse the axis of the image
    yhist,xhist = byb.getHist(img,tensor=False)
    
    #Get the variance of the summed data
    yvar = np.var(yhist)
    
    ###########################################################################
    #Confidence map based metrics
    ###########################################################################
    
    #Collapse the axis of the cmap
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
    cyhist,cxhist = byb.getHist(cmap,tensor=False)

    ###########################################################################
    
    cost = np.mean(np.diff(cyhist[100:-100]))

    return cost
   
###############################################################################
# Bayesian Optimization
###############################################################################

#define the limits of the space of search
space = [
    Real(0, 0, name='x0'),  # x
    Real(0, 0, name='x1'),  # y
    Real(0, 0, name='x2'),  # z
    Real(0, 0, name='x3'),  # q1 (quaternion component 1)
    Real(0, 0, name='x4'),  # q2 (quaternion component 2)
    Real(0, 0, name='x5'),  # q3 (quaternion component 3)
    Real(0, 0, name='x6')   # q4 (quaternion component 4)
]

# preds = []
# for k in range(0,10):

#     result = gp_minimize(function, space,
#                           n_calls=50, random_state=None, verbose=True)
    
#     from skopt.plots import plot_convergence, plot_objective, plot_evaluations
#     best_params = result.x
#     best_value = result.fun
    
#     print("Best parameters:", best_params)
#     print("Best function value:", best_value)
    
#     theta = findClosestPosition(best_params)
#     print(theta)
#     res = loadImg(int(theta[-1]), datapath)
#     byb.plotUS(res,norm=True)
#     plt.show()
    
#     # Optionally, you can plot the convergence and other diagnostics
#     plot_convergence(result)
#     plt.show()
#     plot_objective(result)
#     plt.show()
#     plot_evaluations(result)
#     plt.show()
    
#     pos=int(theta[-1])
#     print(k, min(abs(20-pos),abs(61-pos)))
#     preds.append((pos,best_value))

