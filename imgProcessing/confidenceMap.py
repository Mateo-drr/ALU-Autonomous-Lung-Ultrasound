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

#PARAMS
d='/acquired/processed/'

#Get files in the directory
current_dir = Path(__file__).resolve().parent.parent / 'data'
path = current_dir.as_posix()
datapath = Path(path+d)
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

#Load file
img = np.load(path + d + fileNames[0])
h,w = img.shape

#Create grid -> 2xnumEdges & numNodesx2
edge_index, pos = grid(h, w)

#Add weigthts to the edges
alpha = 0.99  # Depth attenuation
beta = 0.1  # Penalty parameter
gamma = 1.0  # Diagonal penalty parameter


edge_weights = []
for i in range(edge_index.shape[1]):
    x1, y1 = pos[edge_index[0, i]]
    x2, y2 = pos[edge_index[1, i]]
    
    depth = (y1 + y2) / 2  # Average depth
    attenuation = np.exp(-alpha * depth) 
    intensity = (img[int(y1), int(x1)] + img[int(y2), int(x2)]) / 2
    
    # Calculate beam width influence and penalties
    if abs(x1 - x2) == 1 and abs(y1 - y2) == 0:  # Horizontal edge
        beam_weight = np.exp(beta * (abs(x1 - x2) + gamma))
    elif abs(x1 - x2) == 0 and abs(y1 - y2) == 1:  # Vertical edge
        beam_weight = np.exp(beta * abs(y1 - y2))
    elif abs(x1 - x2) == 1 and abs(y1 - y2) == 1:  # Diagonal edge
        beam_weight = np.exp(beta * (abs(x1 - x2) + np.sqrt(2) * gamma))
    else:
        beam_weight = 0.0
    
    # Combine attenuation and beam width influence
    edge_weight = attenuation * beam_weight * intensity
    edge_weights.append(edge_weight)
    
edge_weights = torch.tensor(edge_weights, dtype=torch.float)

# Create a PyTorch Geometric Data object
data = Data(edge_index=edge_index, edge_attr=edge_weights, pos=pos)

class RandomWalks(MessagePassing):
    def __init__(self):
        super(RandomWalks, self).__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

# Create a feature vector for the seeds
x = torch.zeros((h * w, 1))
for i in range(w):
    x[i, 0] = 1  # Start seeds
    x[-i-1, 0] = 0  # End seeds
    
# Initialize the Random Walks class
rw = RandomWalks()

# Perform the random walk
output = rw(x, data.edge_index, edge_weights)

# Normalize the output
output = F.softmax(output, dim=0)

confidence_map = output.view(h, w).detach().numpy()

plt.imshow(confidence_map, cmap='hot', interpolation='nearest')
plt.title('Ultrasound Confidence Map')
plt.colorbar()
plt.show()