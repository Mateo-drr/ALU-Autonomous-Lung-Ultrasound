# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:01:38 2024

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

#GLOBAL VARS
CONFMAP=None
IMAGE=None
POS=None

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


def costFunc(theta,xfake,yfake):
    
    global IMAGE, CONFMAP, POS
    
    #find closest recorded position to current
    pos = int(byb.findClosestPosition(theta,xfake,yfake)[-1])
    
    if POS is None or pos != POS:   
    
        #Load the image closest to the current theta
        img = byb.loadImg(fileNames,int(pos), datapath)
        
        #Calculate confidence map
        cmap = confidenceMap(img,rsize=True)
        
        POS = pos
        IMAGE = img
        CONFMAP = cmap
        
    else: #dont reload current image 
        
        img = IMAGE
        cmap = CONFMAP
        
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
    
    #Get the mean value
    #avgConf = np.mean(cmap)
    
    #Confidence ratio
    # threshold = 0.8
    # confC = np.sum(cmap < threshold)
    # # Normalize the high confidence count by the total number of pixels
    # confRT = confC / cmap.size
    
    #CENTER OF MASS OF CONFIDENCE
    
    # # Create a meshgrid of coordinates (X and Y)
    # x_coords, y_coords = np.meshgrid(np.arange(cmap.shape[1]),
    #                                  np.arange(cmap.shape[0]))
    # # Calculate the total weight (sum of all values in the heatmap)
    # total_weight = np.sum(cmap)

    # # Calculate the weighted average for x and y coordinates
    # x_center_of_mass = np.sum(x_coords * cmap) / total_weight
    # y_center_of_mass = np.sum(y_coords * cmap) / total_weight
    
    # # Distance of the center of mass from the center of the image
    # # ignore y data since we want x data aligned
    # dist = cmap.shape[1]/2 - x_center_of_mass
    
    ###########################################################################
    #Angle predictions
    ###########################################################################
    
    # #The collapsed x axis gives us a clean segmented line from the image
    # #And we can get an angle from it
    # l1,ang1,_,_ = byb.regFit(cxhist,tensor=False)
    
    # #We can also calculate the angle from the image itself
    # # Normalize to 255
    # nimg = byb.normalize(img,tensor=False)
    # # Find the peak of each line
    # peaks = byb.findPeaks(nimg,tensor=False)
    # #And we can get an angle from it
    # l2,ang2,_,_ = byb.regFit(peaks,tensor=False)
    
    #Joining the angles
    # ang = min((abs(ang1)+abs(ang2))/2, 20)/20
    # ang = abs((ang1+ang2)/2)#(abs(ang1)+abs(ang2))/2 
    # ang = np.sqrt(abs(ang1*ang2))

    ###########################################################################
    
    cost = np.mean(np.diff(cyhist[100:-100]))

    return cost
   
###############################################################################
# Bayesian Optimization
###############################################################################

#define the limits of the space of search
lim=1e-16
space = [
    Real(min(np.min(xfake[:, :, 0]), np.min(yfake[:, :, 0]-lim)),
         max(np.max(xfake[:, :, 0]), np.max(yfake[:, :, 0])), name='x0'),
    Real(min(np.min(xfake[:, :, 1]-lim), np.min(yfake[:, :, 1])),
         max(np.max(xfake[:, :, 1]), np.max(yfake[:, :, 1])), name='x1'),
    Real(min(np.min(xfake[:, :, 2]-lim), np.min(yfake[:, :, 2])),
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

###############################################################################
# Others
###############################################################################

# Get an initial random theta, ie position and rotation within the movement margins
# i = np.random.randint(0,len(xmove))
# j = np.random.randint(0,len(xmove))
# xORy = np.random.randint(0,2)
# i = 40
# j = 40

# target = xfake[i,j,:7] if xORy else yfake[i,j,:7]

# theta = np.array(target)

# if True:
    
#     # Compute objective function value
    
    
#     #Minimize it 
#     result = minimize(function, theta, method='Nelder-Mead',
#                       options={'xatol': 1e-2, 'fatol': 1e-2, 'disp': True, 'maxiter': 1000, 'adaptive': True})
    
#     print(theta, result)
 

###############################################################################
#Terrain Visualization
###############################################################################

def evaluate_function_on_positions(positions,xfake,yfake):
    num_positions = len(positions)
    result = np.zeros(num_positions)
    for i in range(num_positions):
        theta = positions[i, :7]
        result[i] = costFunc(theta,xfake,yfake)
    return result

# Evaluate the objective function on the xmove and ymove positions
xmove_results = evaluate_function_on_positions(xmove,xfake,yfake)
ymove_results = evaluate_function_on_positions(ymove,xfake,yfake)

# Replicate the results along the fake axis to match the shape of xfake and yfake
xfake_results = np.tile(xmove_results[:, None], (1, 41))
yfake_results = np.tile(ymove_results[:, None], (1, 41))

# Plotting the results as heatmaps
def plot_heatmap(data, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis', interpolation='nearest')
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


###############################################################################
#ALL data plot
###############################################################################
# Determine the number of images and grid dimensions
# num_images = len(ymove)
# cols = 14  # Number of columns in the grid
# rows = (num_images + cols - 1) // cols  # Calculate rows needed

# # Create a figure with subplots arranged in a grid
# fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
# axes = axes.flatten()  # Flatten the 2D array of axes to easily index it

# # Plot each yhist in its respective subplot
# for i, pos in enumerate(ymove):
#     img = byb.loadImg(int(pos[-1]), datapath)  # Load the image
#     cmap = confidenceMap(img,rsize=True)
#     cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
#     cyhist,cxhist = byb.getHist(cmap,tensor=False)  
#     yhist,xhist = byb.getHist(img,tensor=False)
#     #_,g = fit_polynomial(hilb(yhist),10)#fit_gaussian(hilb(yhist))
#     #axes[i].plot(yhist, np.arange(len(yhist)))  # Plot yhist
#     #axes[i].plot(g, np.arange(len(g)), linewidth=6)
#     axes[i].plot(cyhist[100:-100],np.arange(len(cyhist[100:-100])))
#     axes[i].invert_yaxis()
#     axes[i].axis('off')
# # Hide any unused subplots
# for j in range(len(ymove), len(axes)):
#     axes[j].axis('off')

# plt.tight_layout()
# plt.show()

# ###############################################################################
# #
# ###############################################################################
xdata = []
for pos,x in enumerate(xmove):
    img = byb.loadImg(fileNames, int(x[-1]), datapath)
    cmap = confidenceMap(img,rsize=True)
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
    yhist,xhist = byb.getHist(img)
    cyhist,cxhist = byb.getHist(cmap)
    xdata.append([img,cmap,yhist,xhist,cyhist,cxhist])

ydata = []
for pos,x in enumerate(ymove):
    img = byb.loadImg(fileNames, int(x[-1]), datapath)
    cmap = confidenceMap(img,rsize=True)
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
    yhist,xhist = byb.getHist(img)
    cyhist,cxhist = byb.getHist(cmap)
    ydata.append([img,cmap,yhist,xhist,cyhist,cxhist])

# windows=[]
# qwer=[]
# for w in windows:
#     t = w[1600:2200]
#     qwer.append([t.max(),t.var()])

# data =qwer
# means = [item[0] for item in data]
# variances = [item[1] for item in data]

# # Convert variances to standard deviations
# std_devs = np.sqrt(variances)

# # Create x values that go from -20 to 20
# x_values = list(range(-20, 21))

# # Adjust means and std_devs to match the x_values length
# extended_means = means[:len(x_values)]
# extended_std_devs = std_devs[:len(x_values)]

# # Plot
# plt.figure(figsize=(10, 6))
# plt.errorbar(x_values, extended_means, yerr=extended_std_devs, fmt='-o', ecolor='r', capsize=5, capthick=2)
# plt.xlabel('Index')
# plt.ylabel('Max Value')
# plt.grid(True)
# plt.show()

