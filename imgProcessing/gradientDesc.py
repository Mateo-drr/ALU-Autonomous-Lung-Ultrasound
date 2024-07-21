# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:01:38 2024

@author: Mateo-drr
"""

import torch
import torch.optim as optim
import filtering as filt
from scipy.signal import hilbert
from pathlib import Path
import numpy as np
import json
from scipy.optimize import minimize
from confidenceMap import confidenceMap
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from skimage.transform import resize

CONFMAP=None
IMAGE=None
POS=None

d='/acquired/processed/'
#Get files in the directory
current_dir = Path(__file__).resolve().parent.parent / 'data'
path = current_dir.as_posix()
datapath = Path(path+d)
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

def loadImg(idx,datapath):
    print('Loading image',fileNames[idx])
    img = np.load(datapath.as_posix() + '/' + fileNames[idx])[:,:]
    #img = torch.tensor(img,requires_grad=True)
    return img

def loadConf(confpath,name):
    # Load the JSON data
    with open(confpath + name, 'r') as file:
        data = json.load(file)
    return data

conf = loadConf(datapath.parent.as_posix(), '/05julyconfig.json')
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
#Effectively shifting all the acquisition cordinates in y
#fake x positions
for i,pos in enumerate(ymove):
    xfake[i, :, 1] = pos[1]
#fake x positions
for i,pos in enumerate(xmove):
    yfake[i, :, 0] = pos[0]

def findClosestPosition(target_coord):
    target_coord = np.array(target_coord)
    
    distances_xfake = np.linalg.norm(xfake[:, :, :7] - target_coord, axis=-1)  # Only consider first 7 values
    closest_index_xfake = np.unravel_index(np.argmin(distances_xfake), distances_xfake.shape)
    closest_xfake = xfake[closest_index_xfake]
    min_distance_xfake = distances_xfake[closest_index_xfake]
    
    distances_yfake = np.linalg.norm(yfake[:, :, :7] - target_coord, axis=-1)  # Only consider first 7 values
    closest_index_yfake = np.unravel_index(np.argmin(distances_yfake), distances_yfake.shape)
    closest_yfake = yfake[closest_index_yfake]
    min_distance_yfake = distances_yfake[closest_index_yfake]
    
    # Output the results
    #print("Closest point in xfake:", closest_xfake)
    #print("Minimum distance in xfake:", min_distance_xfake)
    #print("Closest point in yfake:", closest_yfake)
    #print("Minimum distance in yfake:", min_distance_yfake)
    
    return closest_xfake if min_distance_xfake < min_distance_yfake else closest_yfake


def function(theta):
    global IMAGE, CONFMAP, POS
    #find closest recorded position to current
    pos = int(findClosestPosition(theta)[-1])
    
    if POS is None or pos != POS:   
    
        #Load the image closest to the current theta
        img = loadImg(int(pos), datapath)
        
        #Calculate confidence map
        cmap = confidenceMap(img,rsize=True)
        
        POS = pos
        IMAGE = img
        CONFMAP = cmap
        
    else:
        
        img = IMAGE
        cmap = CONFMAP
    
    #Collapse the axis of the image
    yhist,xhist = filt.getHist(img,tensor=False)
    
    #Collapse the axis of the cmap
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
    cyhist,cxhist = filt.getHist(cmap,tensor=False)
    
    #Get the variance of the data so we can minimize it
    #Inturn making the pleura perpendicular to the probe
    #yvar = np.var(yhist)#torch.var(yhist)
    yvar = np.average((yhist-np.average(yhist,weights=cyhist))**2,weights=cyhist)
    
    ########
    ########
    # filt.plotUS(img,True)
    # plt.show()
    
    # plt.plot(yhist)
    # plt.show()
    
    # fimg = 20*np.log10(abs(img)+1)
    # ydis,_ = filt.getHist(fimg)
    
    # plt.plot(ydis)
    # plt.show()
    
    ########
    #######
    
    #Get the mean value of the confidence map to maximize it
    #Since it get high confidence (1s) when it's perpendicular
    #avgConf = np.mean(cmap)#torch.mean(cmap)
    threshold = 0.8
    confC = np.sum(cmap < threshold)
    
    # Normalize the high confidence count by the total number of pixels
    confRT = confC / cmap.size
    
    # Create a meshgrid of coordinates (X and Y)
    x_coords, y_coords = np.meshgrid(np.arange(cmap.shape[1]),
                                     np.arange(cmap.shape[0]))
    # Calculate the total weight (sum of all values in the heatmap)
    total_weight = np.sum(cmap)

    # Calculate the weighted average for x and y coordinates
    x_center_of_mass = np.sum(x_coords * cmap) / total_weight
    y_center_of_mass = np.sum(y_coords * cmap) / total_weight
    
    # Distance of the center of mass from the center of the image
    # ignore y data since we want x data aligned
    dist = cmap.shape[1]/2 - x_center_of_mass
    
    #The collapsed x axis gives us a clean segmented line from the image
    #And we can get an angle from it
    l1,ang1,_,_ = filt.regFit(cxhist,tensor=False)
    
    #We can also calculate the angle from the image itself
    # Normalize to 255
    nimg = filt.normalize(img,tensor=False)
    # Find the peak of each line
    peaks = filt.findPeaks(nimg,tensor=False)
    #And we can get an angle from it
    l2,ang2,_,_ = filt.regFit(peaks,tensor=False)
    
    #Putting everything together
    #Minimize variance
    #Maximize average confidence
    #Reach angle 0
    ang = min((abs(ang1)+abs(ang2))/2, 20)/20
    ang = abs((ang1+ang2)/2)#(abs(ang1)+abs(ang2))/2 
    ang = np.sqrt(abs(ang1*ang2))
    
    loss = (confRT)# confRT #yvar * (1-avgConf) * ang #+ min((abs(ang1)+abs(ang2))/2, 20)/20
    print(f"Center d: {min(abs(20-pos),abs(61-pos))}, Variance: {yvar:.4f}, Conf: {confRT:.4f}, Loss: {loss:.4f}\n")

    ###########################################################################
    #visual servoing
    ###########################################################################
    #Get ROI based on cmap high confidence zone
    # high_confidence_mask = cmap > 0.7
    
    # #Calculate gradients
    # gradients = np.gradient(cmap.astype(float))
    # grad_x, grad_y = gradients[1], gradients[0]
    # y, x = np.indices(cmap.shape)
    
    # high_confidence_y = y[high_confidence_mask]
    # high_confidence_x = x[high_confidence_mask]
    # high_confidence_grad_x = grad_x[high_confidence_mask]
    # high_confidence_grad_y = grad_y[high_confidence_mask]
    
    # #Interaction matrix
    # Lstrack = filt.compute_interaction_matrix_for_high_confidence(cmap,
    #                                                     high_confidence_mask,
    #                                                     grad_x,
    #                                                     grad_y)
    
    # # Define initial_features and current_features appropriately
    # initial_features = ...  # Initial set of visual features
    # current_features = ...  # Current set of visual features
    # lambda_track = 0.5  # Example gain
    # vR = calculate_movement(Lstrack, initial_features, current_features, lambda_track)

    return np.var(yhist) #-100*np.var(np.diff(cyhist[100:-100]))
 
def calculate_movement(Lstrack, initial_features, current_features, lambda_track):
    error = current_features - initial_features
    pseudo_inverse = np.linalg.pinv(Lstrack)
    vR = -lambda_track * np.dot(pseudo_inverse, error)
    return vR
   
# Get an initial random theta, ie position and rotation within the movement margins
i = np.random.randint(0,len(xmove))
j = np.random.randint(0,len(xmove))
xORy = np.random.randint(0,2)
i = 40
j = 40

target = xfake[i,j,:7] if xORy else yfake[i,j,:7]

theta = np.array(target)#torch.tensor(target, requires_grad=True)
#print(theta)

space = [
    Real(min(np.min(xfake[:, :, 0]), np.min(yfake[:, :, 0])),
         max(np.max(xfake[:, :, 0]), np.max(yfake[:, :, 0])), name='x0'),
    Real(min(np.min(xfake[:, :, 1]), np.min(yfake[:, :, 1])),
         max(np.max(xfake[:, :, 1]), np.max(yfake[:, :, 1])), name='x1'),
    Real(min(np.min(xfake[:, :, 2]), np.min(yfake[:, :, 2])),
         max(np.max(xfake[:, :, 2]), np.max(yfake[:, :, 2])), name='x2'),
    Real(min(np.min(xfake[:, :, 3]), np.min(yfake[:, :, 3])),
         max(np.max(xfake[:, :, 3]), np.max(yfake[:, :, 3])), name='x3'),
    Real(min(np.min(xfake[:, :, 4]), np.min(yfake[:, :, 4])),
         max(np.max(xfake[:, :, 4]), np.max(yfake[:, :, 4])), name='x4'),
    Real(min(np.min(xfake[:, :, 5]), np.min(yfake[:, :, 5])),
         max(np.max(xfake[:, :, 5]), np.max(yfake[:, :, 5])), name='x5'),
    Real(min(np.min(xfake[:, :, 6]), np.min(yfake[:, :, 6])),
         max(np.max(xfake[:, :, 6]), np.max(yfake[:, :, 6])), name='x6')
]

#loss = function(theta)


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
#     filt.plotUS(res,norm=True)
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


# if True:
    
#     # Compute objective function value
    
    
#     #Minimize it 
#     result = minimize(function, theta, method='Nelder-Mead',
#                       options={'xatol': 1e-2, 'fatol': 1e-2, 'disp': True, 'maxiter': 1000, 'adaptive': True})
    
#     print(theta, result)
 
###############################################################################
#Terrain Visualization
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def evaluate_function_on_positions(positions):
    num_positions = len(positions)
    result = np.zeros(num_positions)
    for i in range(num_positions):
        theta = positions[i, :7]
        result[i] = function(theta)
    return result

# Evaluate the objective function on the xmove and ymove positions
xmove_results = evaluate_function_on_positions(xmove)
ymove_results = evaluate_function_on_positions(ymove)

# Replicate the results along the fake axis to match the shape of xfake and yfake
xfake_results = np.tile(xmove_results[:, None], (1, 41))
yfake_results = np.tile(ymove_results[:, None], (1, 41))

# Plotting the results as heatmaps
def plot_heatmap(data, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap=cm.viridis, interpolation='nearest')
    plt.colorbar(label='Objective Function Value')
    plt.title(title)
    plt.xlabel('Position Index')
    plt.ylabel('Position Index')
    plt.show()

plot_heatmap(xfake_results, 'Objective Function Values for xfake')
plot_heatmap(yfake_results, 'Objective Function Values for yfake')

# Rotate yfake_results to align with xfake_results
yfake_rotated = np.rot90(yfake_results)

# Multiply the maps
combined_map_multiply = xfake_results * yfake_rotated

# Plot the combined map
plot_heatmap(combined_map_multiply, 'Combined Objective Function Values (Multiplication)')

plt.plot(xmove_results)
plt.show()
plt.plot(ymove_results)
plt.show()