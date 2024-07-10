# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:54:11 2024

@author: Mateo-drr
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import grid
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from filtering import plotUS

#PARAMS
d='/acquired/processed/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alpha=1
beta=90
gamma=0.05
# torch.set_num_threads(8)
# torch.set_num_interop_threads(8)
# torch.backends.cudnn.benchmark=True

#Get files in the directory
current_dir = Path(__file__).resolve().parent.parent / 'data'
path = current_dir.as_posix()
datapath = Path(path+d)
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

#Load file
img = np.load(path + d + fileNames[60])[:,:]
h,w = img.shape
#img = 20*np.log10(np.abs(img)+1)

import networkx as nx
import scipy

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
    #The depth of the node (y+1) times coef, all times the node intensity
    
    #Normalize the distance
    i1 = (y1+1)/h
    i2 = (y2+1)/h
    
    c1 = g[y1,x1] * np.exp(-alpha * i1)
    c2 = g[y2,x2] * np.exp(-alpha * i2)
    
    if abs(x1 - x2) == 1 and abs(y1 - y2) == 0:  
        # Horizontal edges
        weight = np.exp(-beta * (abs(c1 - c2) + gamma))
        
    elif abs(x1 - x2) == 0 and abs(y1 - y2) == 1:  
        # Vertical edges
        weight = np.exp(-beta * (abs(c1 - c2)))
        
    elif abs(x1 - x2) == 1 and abs(y1 - y2) == 1:  
        # Diagonal edges
        weight = np.exp(-beta * (abs(c1 - c2) + np.sqrt(2) * gamma))
        
    else:
        print('ERROR')
        weight = 0  # This should not happen in an 8-connected grid
    
    #print(edge, round(weight,2))
    
    #assing the weight
    G.edges[edge]['weight'] = weight#int(weight)
    
labels = nx.get_edge_attributes(G,'weight')
ww = [G[u][v]['weight'] for u, v in G.edges()]

# for edge in G.edges:       
#     G.edges[edge]['weight'] = 1
    
# pos=nx.spring_layout(G) # pos = nx.nx_agraph.graphviz_layout(G)
# nx.draw_networkx(G,pos)
# # labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)


#RANDOM WALK ALGO

# Set boundary conditions
probabilities = np.zeros(h * w)
for x in range(w):
    probabilities[x] = 1  # First row (virtual transducer elements) set to unity
for x in range((h-1) * w, h * w):
    probabilities[x] = 0  # Last row (no signal region) set to zero

# Construct Laplacian matrix (verified its the same as in paper)
L = nx.laplacian_matrix(G).astype(float)
#Get the indices of the top and bottom of the graph ie the ones initialized to 0 and 1
boundary_indices = np.concatenate([np.arange(w), np.arange((h-1) * w, h * w)])
#Get the indices of all the rest of the nodes
interior_indices = np.setdiff1d(np.arange(h * w), boundary_indices)

# Solve for unknown probabilities
L_unmarked = L[interior_indices, :][:, interior_indices]
B = L[interior_indices, :][:, boundary_indices]
xU = np.zeros(h * w)
xM = np.arange(w)

xU[interior_indices] = scipy.sparse.linalg.spsolve(L_unmarked,
                                                   -B @ probabilities[boundary_indices])
    
# Fill known probabilities
#xU[boundary_indices] = probabilities[boundary_indices]
replace = np.where(probabilities == 0)
probabilities[replace] = xU[replace]

# Reshape to 2D grid for visualization
confidence_map = probabilities.reshape((h, w))#xU.reshape((h, w))

# Visualize the result
plt.imshow(confidence_map, cmap='hot', interpolation='nearest', aspect='auto')
plt.colorbar()
plt.title("Confidence Map")
plt.show()

plotUS(20*np.log10(np.abs(img)+1))
plt.show()