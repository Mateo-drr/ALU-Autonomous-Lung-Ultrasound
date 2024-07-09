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
alpha=2
beta=90
gamma=0.01
# torch.set_num_threads(8)
# torch.set_num_interop_threads(8)
# torch.backends.cudnn.benchmark=True

#Get files in the directory
current_dir = Path(__file__).resolve().parent.parent / 'data'
path = current_dir.as_posix()
datapath = Path(path+d)
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

#Load file
img = np.load(path + d + fileNames[20])[:,:]
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
    c1 = g[y1,x1] * np.exp(-alpha * (y1+1))
    c2 = g[y2,x2] * np.exp(-alpha * (y2+1))
    
    if abs(x1 - x2) == 1 and abs(y1 - y2) == 0:  # Horizontal edges
        weight = np.exp(beta * (abs(c1 - c2) + gamma))
        
    elif abs(x1 - x2) == 0 and abs(y1 - y2) == 1:  # Vertical edges
        weight = np.exp(beta * (abs(c1 - c2)))
        
    elif abs(x1 - x2) == 1 and abs(y1 - y2) == 1:  # Diagonal edges
        weight = np.exp(beta * (abs(c1 - c2) + np.sqrt(2) * gamma))
        
    else:
        print('ERROR')
        weight = 0  # This should not happen in an 8-connected grid
    
    #print(edge, round(weight,2))
    
    #assing the weight
    G.edges[edge]['weight'] = weight#int(weight)
    
labels = nx.get_edge_attributes(G,'weight')

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

# Construct Laplacian matrix
L = nx.laplacian_matrix(G).astype(float)
boundary_indices = np.concatenate([np.arange(w), np.arange((h-1) * w, h * w)])
interior_indices = np.setdiff1d(np.arange(h * w), boundary_indices)

# Solve for unknown probabilities
L_interior = L[interior_indices, :][:, interior_indices]
B = L[interior_indices, :][:, boundary_indices]
xU = np.zeros(h * w)
xU[interior_indices] = scipy.sparse.linalg.spsolve(L_interior, -B.dot(probabilities[boundary_indices]))
    
# Fill known probabilities
xU[boundary_indices] = probabilities[boundary_indices]

# Reshape to 2D grid for visualization
confidence_map = xU.reshape((h, w))

# Visualize the result
plt.imshow(confidence_map, cmap='hot', interpolation='nearest', aspect='auto')
plt.colorbar()
plt.title("Confidence Map")
plt.show()




'''
#Create grid -> 2xnumEdges & numNodesx2
edge_index, pos = grid(h, w)

#Add weigthts to the edges
alpha = 2  # Depth attenuation
beta = 90  # Penalty parameter
gamma = 0.05  # Diagonal penalty parameter

# Normalized closest distance from node vi to the virtual transducer elements (top row)
l = np.zeros((h, w))
for i in range(h):
    l[i, :] = i / (h - 1)
    
# Image intensities and scaled intensities
gi = (img - img.min()) / (img.max() - img.min())
ci = gi * np.exp(-alpha * l)

edge_weights = []
for i in range(edge_index.shape[1]):
    x1, y1 = pos[edge_index[0, i]]
    x2, y2 = pos[edge_index[1, i]]
    
    #depth = (y1 + y2) / 2  # Average depth
    #attenuation = np.exp(-alpha * depth) 
    # Difference in intensity between pixel 1 and 2 
    intensity = abs(img[int(y1), int(x1)] - img[int(y2), int(x2)])
    
    if abs(x1 - x2) == 1 and abs(y1 - y2) == 0:  # Horizontal edges
        wij = np.exp(beta * (abs(ci[int(y1), int(x1)] - ci[int(y2), int(x2)]) + gamma))
    elif abs(x1 - x2) == 0 and abs(y1 - y2) == 1:  # Vertical edges
        wij = np.exp(beta * (abs(ci[int(y1), int(x1)] - ci[int(y2), int(x2)])))
    elif abs(x1 - x2) == 1 and abs(y1 - y2) == 1:  # Diagonal edges
        wij = np.exp(beta * (abs(ci[int(y1), int(x1)] - ci[int(y2), int(x2)]) + np.sqrt(2) * gamma))
    else:
        wij = 0  # This should not happen in an 8-connected grid
    
    # Combine attenuation and beam width influence
    edge_weight = wij
    edge_weights.append(edge_weight)
    
edge_weights = torch.tensor(edge_weights, dtype=torch.float).to(device)
# Normalize edge weights to prevent explosion of values
edge_weights = edge_weights / edge_weights.max()
#edge_weights = torch.log10(edge_weights)

# Create a PyTorch Geometric Data object
data = Data(edge_index=edge_index, edge_attr=edge_weights, pos=pos).to(device)

x = torch.zeros((h * w, 1),device=device)
# Identify the indices of the top and bottom row nodes
top_row_indices = np.arange(0, w)
bottom_row_indices = np.arange((h - 1) * w, h * w)
# Set the seeds
x[top_row_indices] = 1  # Start seeds at the transducer elements
x[bottom_row_indices] = 0  # End seeds at the bottom boundary

# Perform the random walk
# output = rw(x, data.edge_index, edge_weights)

# Function to perform one iteration of the random walk
# def random_walk_step(x, edge_index, edge_weight):
#     row, col = edge_index
#     out = torch.zeros_like(x)
#     for i in range(x.size(0)): #number of nodes
#         #find the indices of edges that start w node i
#         neighbors = (row == i).nonzero().view(-1)
#         for n in neighbors:
#             j = col[n] #get the neighbour
#             out[i] += edge_weight[n] * x[j]
#     out /= out.sum(dim=0, keepdim=True)
#     return out

def random_walk_step(x, edge_index, edge_weight):
    row, col = edge_index
    out = torch.zeros_like(x)
    num_nodes = x.size(0)
    
    for i in range(num_nodes):
        # Find neighbors of node i
        neighbors = (row == i).nonzero().view(-1)
        
        if neighbors.numel() > 0:
            # Compute probabilities based on edge weights
            probabilities = edge_weight[neighbors] / edge_weight[neighbors].sum()
            
            # Select a random neighbor based on computed probabilities
            #print(probabilities)
            selected_neighbor = torch.multinomial(probabilities, 1).item()
            
            # Update the output for node i based on the selected neighbor
            out[i] += x[col[neighbors[selected_neighbor]]]  # x[j] where j is the selected neighbor
    
    return out / out.sum(dim=0, keepdim=True)

# Perform the random walk iteratively until convergence
max_iterations = 1000
tolerance = 1e-6
prev_output = x
best = 1
for _ in range(max_iterations):
    output = random_walk_step(prev_output, data.edge_index, edge_weights)
    score = torch.max(torch.abs(output - prev_output)).item()
    print(score)
    if score <= best:
        final = output.clone()    
        best = score
    if score < tolerance:
        break
    if np.isnan(score):
        #reset to last best
        prev_output = final.clone()
    else:
        prev_output = output.clone()


# Normalize the output
output = F.softmax(final, dim=0)

confidence_map = output.view(h, w).detach().cpu().numpy()

plt.imshow(confidence_map, cmap='hot', interpolation='nearest', aspect='auto')
plt.title('Ultrasound Confidence Map')
plt.colorbar()
plt.show()

plt.imshow(20*np.log10(confidence_map+1), cmap='hot', interpolation='nearest', aspect='auto')
plt.title('Ultrasound Confidence Map')
plt.colorbar()
plt.show()
'''