# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:54:11 2024

@author: Mateo-drr
"""


import numpy as np
import networkx as nx
import scipy
from skimage.transform import resize

#Uncomment for manual testing/usage
'''
from pathlib import Path
import matplotlib.pyplot as plt
from filtering import plotUS

#PARAMS
d='/acquired/processed/'
alpha=1
beta=90
gamma=0.05

#Get files in the directory
current_dir = Path(__file__).resolve().parent.parent / 'data'
path = current_dir.as_posix()
datapath = Path(path+d)
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

#Load file
img = np.load(path + d + fileNames[60])[:,:]
h,w = img.shape
#img = 20*np.log10(np.abs(img)+1)
'''

def confidenceMap(img,alpha=3,beta=90,gamma=0.05,rsize=False):
    #alpha controls depth attenuation
    
    if rsize:
        img = resize(img, (img.shape[0]//3, img.shape[1]), anti_aliasing=True)
    
    h,w = img.shape
    #create a 4x4 grid
    G = nx.grid_2d_graph(h, w)
    
    # Manually add the connections to make it 8-connected
    for x in range(w):
        for y in range(h):
            if x + 1 < w and y + 1 < h:
                G.add_edge((y, x), (y + 1, x + 1), weight=1)
            if x + 1 < w and y - 1 >= 0:
                G.add_edge((y, x), (y - 1, x + 1), weight=1)
         
    #Calculate normalized intensities
    g = (img - img.min()) / (img.max() - img.min())
            
    # Set all weights to the values from the paper
    for edge in G.edges:
        
        y1,x1 = edge[0]
        y2,x2 = edge[1]
        
        #Attenuated signal of each node 
        #The depth of the node (y+1) times coef alpha, all times the node intensity
        
        #Normalize the distance
        i1 = (y1+1)/h
        i2 = (y2+1)/h
        
        c1 = g[y1,x1] * np.exp(-alpha * i1)
        c2 = g[y2,x2] * np.exp(-alpha * i2)
        
        xstep,ystep = abs(x1-x2),abs(y1-y2)
        
        #Check diagonals first since for every node there are 4 diag and 2 side
        if xstep == 1 and ystep == 1:  
            # Diagonal edges
            weight = np.exp(-beta * (abs(c1 - c2) + np.sqrt(2) * gamma))
        
        elif xstep == 1 and ystep == 0:  
            # Horizontal edges
            weight = np.exp(-beta * (abs(c1 - c2) + gamma))
            
        elif xstep == 0 and ystep == 1:  
            # Vertical edges
            weight = np.exp(-beta * (abs(c1 - c2)))
        
        else:
            print('ERROR')
            weight = 0  # This should not happen in an 8-connected grid
        
        #assing the weight
        G.edges[edge]['weight'] = weight
    
    #RANDOM WALK ALGO

    # Set boundary conditions    
    probabilities = np.zeros(h * w)
    probabilities[:w] = 1  # First row (virtual transducer elements) set to unity
    probabilities[-w:] = 0  # Last row (no signal region) set to zero
    
    # Laplacian matrix (verified its the same as in paper)
    L = nx.laplacian_matrix(G).astype(float)
    #Get the indices of the top and bottom of the graph ie the ones initialized to 0 and 1
    boundary_indices = np.concatenate([np.arange(w), np.arange((h-1) * w, h * w)])
    #Get the indices of all the rest of the nodes
    interior_indices = np.setdiff1d(np.arange(h * w), boundary_indices)
    
    # Solve for unknown probabilities
    L_unmarked = L[interior_indices, :][:, interior_indices]
    B = L[interior_indices, :][:, boundary_indices]
    xU = np.zeros(h * w)
    
    xU[interior_indices] = scipy.sparse.linalg.spsolve(L_unmarked,
                                                       -B @ probabilities[boundary_indices])
        
    # Fill known probabilities
    # same as xU[boundary_indices] = probabilities[boundary_indices]
    replace = np.where(probabilities == 0)
    probabilities[replace] = xU[replace]
    
    # Reshape to 2D grid for visualization
    confidence_map = probabilities.reshape((h, w))
    
    return confidence_map

'''
# Visualize the result
plt.imshow(confidence_map, cmap='hot', interpolation='nearest', aspect='auto')
plt.colorbar()
plt.title("Confidence Map")
plt.show()

plotUS(20*np.log10(np.abs(img)+1))
plt.show()
'''