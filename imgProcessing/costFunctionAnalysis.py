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
from scipy.ndimage import laplace

#GLOBAL VARS
CONFMAP=None
IMAGE=None
POS=None

#PARAMS
date='01Aug6'
# confname='05julyconfig.json'
ptype='cl'
ptype2conf = {
    'cl': 'curvedlft_config.json',
    'cf': 'curvedfwd_config.json',
    'rf': 'rotationfwd_config.json',
    'rl': 'rotationlft_config.json'
}
confname=ptype2conf[ptype]


#Get files in the directory
current_dir = Path(__file__).resolve().parent.parent.parent
# datapath = current_dir / 'data' / 'acquired' / date / 'processed'
datapath = current_dir / 'data' / 'acquired' / date / 'processed' / ptype
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
    # img = byb.envelope(img)
    
    #Collapse the axis of the image
    yhist,xhist = byb.getHist(img)
    
    #Get the variance of the summed data
    yvar = np.var(yhist[2000:2800])
    
    #Variance of laplacian
    lvar = variance_of_laplacian(img[2000:2800])
    
    ###########################################################################
    #Confidence map based metrics
    ###########################################################################
    
    #Collapse the axis of the cmap
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
    cyhist,cxhist = byb.getHist(cmap[2000:2800,:],tensor=False)
    
    #Get the mean value
    #avgConf = np.mean(cmap)
    #Derivative
    gcyhist = np.diff(cyhist)
    
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
    l1,ang1,_,_ = byb.regFit(cxhist,tensor=False)
    
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
    
    # cost = np.var(gcyhist)
    # cost = np.mean(abs(gcyhist))
    # cost = yvar
    cost = lvar

    return cost
   
def variance_of_laplacian(gray):
    # use grayscale
    laplacian = laplace(gray)
    variance = laplacian.var()
    return variance
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
'''
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

#'''
###############################################################################
#ALL data plot
###############################################################################
# '''
# Determine the number of images and grid dimensions
side=ymove
num_images = len(side)
cols = 14  # Number of columns in the grid
rows = (num_images + cols - 1) // cols  # Calculate rows needed

# Create a figure with subplots arranged in a grid
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), dpi=300)
axes = axes.flatten()  # Flatten the 2D array of axes to easily index it

#LABELS
idx=3
lblpth = current_dir / 'ALU---Autonomous-Lung-Ultrasound' / 'imgProcessing' / 'ml' / 'lines'
top = np.load(lblpth / f'top_lines_{idx}.npy')
btm = np.load(lblpth / f'btm_lines_{idx}.npy')

# Plot each yhist in its respective subplot
a,b,c=2,90,0.05
print(a,b,c)
for i, pos in enumerate(side):
    img = byb.loadImg(fileNames,int(pos[-1]), datapath)#[2000:2800]  # Load the image
    # img = byb.envelope(img)
    # cmap = confidenceMap(img,alpha=a,beta=b,gamma=c,rsize=True)
    # cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
    # cyhist,cxhist = byb.getHist(cmap,tensor=False)  
    # gcyhist = np.diff(cyhist)
    # yhist,xhist = byb.getHist(img,tensor=False)
    # yhist = byb.hilb(yhist)
    # lap = laplace(img)
    
    # gy,gx = np.gradient(img)
    # yhist,_ = byb.getHist(gy)
    # yhist = byb.hilb(yhist)
    #_,g = fit_polynomial(hilb(yhist),10)#fit_gaussian(hilb(yhist))
    # axes[i].plot(gcyhist, np.arange(len(gcyhist)))  # Plot yhist
    #axes[i].plot(g, np.arange(len(g)), linewidth=6)
    # axes[i].plot(yhist,np.arange(len(yhist)))
    axes[i].imshow(20*np.log10(abs(img)+1),aspect='auto',cmap='viridis')
    # axes[i].imshow(cmap,aspect='auto',cmap='viridis')
    # axes[i].imshow(20*np.log10(abs(lap)+1),aspect='auto',cmap='viridis')
    # axes[i].invert_yaxis()
    
    #labels plot
    axes[i].axhline(top[int(pos[-1])], color='r')
    axes[i].axhline(btm[int(pos[-1])], color='b')
    
    axes[i].axis('off')
# Hide any unused subplots
for j in range(len(side), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
#'''
# ###############################################################################
# #Load all data in memory
# ###############################################################################
'''

# strt,end=1600,2200
strt,end=2000,2800
subsec = False

xdata = []
for pos,x in enumerate(xmove):
    img = byb.loadImg(fileNames, int(x[-1]), datapath)#[100:]
    cmap = confidenceMap(img,rsize=True)
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]
    
    if subsec:
        img,cmap=img[strt:end],cmap[strt:end]
    
    yhist,xhist = byb.getHist(img)
    cyhist,cxhist = byb.getHist(cmap)
    xdata.append([img,cmap,yhist,xhist,cyhist,cxhist])

ydata = []
for pos,x in enumerate(ymove):
    img = byb.loadImg(fileNames, int(x[-1]), datapath)#[100:]
    cmap = confidenceMap(img,rsize=True)
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]
    
    if subsec:
        img,cmap=img[strt:end],cmap[strt:end]
    
    yhist,xhist = byb.getHist(img)
    cyhist,cxhist = byb.getHist(cmap)
    ydata.append([img,cmap,yhist,xhist,cyhist,cxhist])
#'''
# ###############################################################################
# calculate all features
# ###############################################################################
'''
import numpy as np

from sklearn.preprocessing import MinMaxScaler


xcost=[]
for pos in xdata:
    metrics=[]
    
    img,cmap,yhist,xhist,cyhist,cxhist = pos
    
    #yhist variance without hilbert
    metrics.append(np.var(yhist))
    #yhist variance with hilbert of sum
    metrics.append(np.var(byb.hilb(yhist)))
    #yhist variance with hilbert of each line
    hb = byb.envelope(img)
    hbyh,_ = byb.getHist(hb)
    metrics.append(np.var(hbyh))
    #mean abs of confidence deriv
    metrics.append(np.mean(np.abs(np.diff(cyhist))))
    #variance of confidence deriv
    metrics.append(np.var(np.diff(cyhist)))
    #mean abs luminocity
    metrics.append(np.mean(np.abs(img)))
    #variance of laplacian 
    metrics.append(variance_of_laplacian(img))
    #variance of laplacian of cmap
    metrics.append(variance_of_laplacian(cmap))
    #cxhist angle prediction
    l1,ang1,_,_ = byb.regFit(cxhist)
    metrics.append(abs(ang1))
    #peaks angle prediction
    nimg = byb.normalize(img)
    peaks = byb.findPeaks(nimg)
    l2,ang2,_,_ = byb.regFit(peaks)
    metrics.append(abs(ang2))
    #xhist variance
    metrics.append(np.var(xhist))
    #cxhist variance
    metrics.append(np.var(cxhist))
    
    xcost.append(metrics)

ycost=[]
for pos in ydata:
    metrics=[]
    
    img,cmap,yhist,xhist,cyhist,cxhist = pos
    
    #yhist variance without hilbert
    metrics.append(np.var(yhist))
    #yhist variance with hilbert of sum
    metrics.append(np.var(byb.hilb(yhist)))
    #yhist variance with hilbert of each line
    hb = byb.envelope(img)
    hbyh,_ = byb.getHist(hb)
    metrics.append(np.var(hbyh))
    #mean abs of confidence deriv
    metrics.append(np.mean(np.abs(np.diff(cyhist))))
    #variance of confidence deriv
    metrics.append(np.var(np.diff(cyhist)))
    #mean abs luminocity
    metrics.append(np.mean(np.abs(img)))
    #variance of laplacian 
    metrics.append(variance_of_laplacian(img))
    #variance of laplacian of cmap
    metrics.append(variance_of_laplacian(cmap))
    #cxhist angle prediction
    l1,ang1,_,_ = byb.regFit(cxhist)
    metrics.append(abs(ang1))
    #peaks angle prediction
    nimg = byb.normalize(img)
    peaks = byb.findPeaks(nimg)
    l2,ang2,_,_ = byb.regFit(peaks)
    metrics.append(abs(ang2))
    #xhist variance
    metrics.append(np.var(xhist))
    #cxhist variance
    metrics.append(np.var(cxhist))
    
    ycost.append(metrics)
    
#Create goal metric values
#sharp
triangle_array = np.concatenate((np.arange(21), np.arange(19, -1, -1)))
#gaussian
x = np.linspace(-1, 1, 41)
sigma = 0.3  # Adjust sigma for the desired smoothness
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)

goal=gaussian_array

# Plot the Gaussian target array for visualization
plt.plot(gaussian_array)
plt.title("Gaussian Target Array")
plt.show()

inx,iny = np.array(xcost), np.array(ycost)

# Normalize the metrics using Min-Max scaling
scalerx = MinMaxScaler()
inxn = scalerx.fit_transform(inx)
scalery = MinMaxScaler()
inyn = scalery.fit_transform(iny)
    
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train a linear regression model
modelx = LinearRegression()
modelx.fit(inxn, goal)

# Train a linear regression model
modely = LinearRegression()
modely.fit(inyn, goal)

# Get the weights
weights_x = modelx.coef_
print("Learned weights:", weights_x)
weights_y = modely.coef_
print("Learned weights:", weights_y)

weights_x_percentage = np.round(100 * weights_x / np.sum(np.abs(weights_x)), 2)
weights_y_percentage = np.round(100 * weights_y / np.sum(np.abs(weights_y)), 2)
# Convert the weights to string format to avoid scientific notation
weights_x_str = [f"{w:.2f}" for w in weights_x_percentage]
weights_y_str = [f"{w:.2f}" for w in weights_y_percentage]
print("Learned weights (x) as percentages:\n", weights_x_str)
print("Learned weights (y) as percentages:\n", weights_y_str)
    
predx=[]
for i in xcost:
    met = scalerx.transform(np.array(i).reshape(1,-1))
    predx.append(modelx.predict(met))
    
predy=[]
for i in ycost:
    met = scalery.transform(np.array(i).reshape(1,-1))
    predy.append(modely.predict(met))
        
plt.plot(predx)
plt.plot(goal)
plt.show()
plt.plot(predy)    
plt.plot(goal)
plt.show()

print(round(mean_squared_error(goal, predx),6))
print(round(mean_squared_error(goal, predy),6))
    
labels = [
    'ysum var w/o hilbert',
    'yhist var w/ hilbert sum',
    'ysum var w/ hilbert lines', 
    'mean abs conf deriv',
    'var conf deriv',
    'mean abs intensity', 
    'var laplacian',
    'var laplacian of conf',
    'confxsum angle pred', 
    'peaks angle pred',
    'xsum var',
    'confxsum var'
]

# Convert absolute weights for pie chart
abs_weights_x_percentage = [abs(w) for w in weights_x_percentage]
abs_weights_y_percentage = [abs(w) for w in weights_y_percentage]

# Sort by absolute values in decreasing order
sorted_indices = np.argsort(abs_weights_x_percentage)[::-1]
sorted_weights = [abs_weights_x_percentage[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]

# Create a horizontal bar chart
plt.figure(figsize=(10, 7))
y_pos = np.arange(len(sorted_labels))
plt.barh(y_pos, sorted_weights, color='skyblue')
plt.yticks(y_pos, sorted_labels)
plt.xlabel('Importance (Absolute Value)')
plt.title('Importance of Metrics Based on Learned Weights')
plt.grid(axis='x', linestyle='--')

# Add data labels to each bar
for index, value in enumerate(sorted_weights):
    plt.text(value, index, f'{value:.2f}', va='center')

# Reverse the order of the y-ticks to have larger values on top
plt.gca().invert_yaxis()

plt.show()

# Sort by absolute values in decreasing order
sorted_indices = np.argsort(abs_weights_y_percentage)[::-1]
sorted_weights = [abs_weights_y_percentage[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]

# Create a horizontal bar chart
plt.figure(figsize=(10, 7))
y_pos = np.arange(len(sorted_labels))
plt.barh(y_pos, sorted_weights, color='skyblue')
plt.yticks(y_pos, sorted_labels)
plt.xlabel('Importance (Absolute Value)')
plt.title('Importance of Metrics Based on Learned Weights')
plt.grid(axis='x', linestyle='--')

# Add data labels to each bar
for index, value in enumerate(sorted_weights):
    plt.text(value, index, f'{value:.2f}', va='center')

# Reverse the order of the y-ticks to have larger values on top
plt.gca().invert_yaxis()

plt.show()
#'''
###############################################################################
# Loop all posible feature combinations
###############################################################################
'''
import itertools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def extract_metrics(data):
    metrics_list = []
    for pos in data:
        metrics = []
        
        img, cmap, yhist, xhist, cyhist, cxhist = pos
        
        metrics.append(np.var(yhist))  # yhist variance without hilbert
        metrics.append(np.var(byb.hilb(yhist)))  # yhist variance with hilbert of sum
        hb = byb.envelope(img)
        hbyh, _ = byb.getHist(hb)
        metrics.append(np.var(hbyh))  # yhist variance with hilbert of each line
        metrics.append(np.mean(np.abs(np.diff(cyhist))))  # mean abs of confidence deriv
        metrics.append(np.var(np.diff(cyhist)))  # variance of confidence deriv
        metrics.append(np.mean(np.abs(img)))  # mean abs luminocity
        metrics.append(variance_of_laplacian(img))  # variance of laplacian
        metrics.append(variance_of_laplacian(cmap))  # variance of laplacian of cmap
        l1, ang1, _, _ = byb.regFit(cxhist)
        metrics.append(abs(ang1))  # cxhist angle prediction
        nimg = byb.normalize(img)
        peaks = byb.findPeaks(nimg)
        l2, ang2, _, _ = byb.regFit(peaks)
        metrics.append(abs(ang2))  # peaks angle prediction
        metrics.append(np.var(xhist))  # xhist variance
        metrics.append(np.var(cxhist))  # cxhist variance
        
        metrics_list.append(metrics)
    return np.array(metrics_list)

xcost = extract_metrics(xdata)
ycost = extract_metrics(ydata)

# Create goal metric values
x = np.linspace(-1, 1, 41)
sigma = 0.3
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
goal = gaussian_array

# Normalize the metrics using Min-Max scaling
scalerx = MinMaxScaler()
inxn = scalerx.fit_transform(xcost)
scalery = MinMaxScaler()
inyn = scalery.fit_transform(ycost)

# Get the number of features
num_features = inxn.shape[1]

# Store MSE scores for each combination
mse_scores_x = []
mse_scores_y = []

# Train and evaluate for all combinations of features
for r in range(1, num_features + 1):
    for combination in itertools.combinations(range(num_features), r):
        # Select the columns for this combination
        inxn_subset = inxn[:, combination]
        inyn_subset = inyn[:, combination]
        
        # Train the linear regression model
        modelx = LinearRegression()
        modelx.fit(inxn_subset, goal)
        
        modely = LinearRegression()
        modely.fit(inyn_subset, goal)
        
        # Predict the goal shape
        predx = modelx.predict(inxn_subset)
        predy = modely.predict(inyn_subset)
        
        # Calculate the MSE for this combination
        mse_x = mean_squared_error(goal, predx)
        mse_y = mean_squared_error(goal, predy)
        
        # Store the MSE and the combination of features
        mse_scores_x.append((mse_x, combination))
        mse_scores_y.append((mse_y, combination))

# Sort the results by MSE
mse_scores_x.sort(key=lambda x: x[0])
mse_scores_y.sort(key=lambda x: x[0])

# Print the results
print("Top 5 combinations for xdata with lowest MSE:")
for i in range(5):
    print(f"Combination {mse_scores_x[i][1]}: MSE = {mse_scores_x[i][0]}")

print("\nTop 5 combinations for ydata with lowest MSE:")
for i in range(5):
    print(f"Combination {mse_scores_y[i][1]}: MSE = {mse_scores_y[i][0]}")


# Identify the combination with the least features and lowest MSE
least_features_x = min(mse_scores_x, key=lambda x: (len(x[1]), x[0]))
least_features_y = min(mse_scores_y, key=lambda x: (len(x[1]), x[0]))

print("Combination for xdata with the least amount of features and lowest MSE:")
print(f"Combination {least_features_x[1]}: MSE = {least_features_x[0]}")

print("\nCombination for ydata with the least amount of features and lowest MSE:")
print(f"Combination {least_features_y[1]}: MSE = {least_features_y[0]}")
#'''
###############################################################################
#plotting of best combinations
###############################################################################
'''
import matplotlib.pyplot as plt
import numpy as np

# Function to count feature usage in combinations
def count_feature_usage(combinations, num_features):
    feature_counts = np.zeros(num_features)
    for combo in combinations:
        for feature in combo:
            feature_counts[feature] += 1
    return feature_counts

# Select the top 100 combinations for both xdata and ydata
top_n = 100
top_combinations_x = mse_scores_x[:top_n]
top_combinations_y = mse_scores_y[:top_n]

# Extract the feature combinations
combinations_x = set(combo[1] for combo in top_combinations_x)
combinations_y = set(combo[1] for combo in top_combinations_y)

# Find the intersection of the top combinations
common_combinations = combinations_x.intersection(combinations_y)

# Get the MSE values for the common combinations
common_mse_scores_x = [combo for combo in mse_scores_x if combo[1] in common_combinations]
common_mse_scores_y = [combo for combo in mse_scores_y if combo[1] in common_combinations]

# Sort the common combinations by the sum of MSE values from both xdata and ydata
common_mse_scores = sorted(
    [(combo_x[0] + combo_y[0], combo_x[1]) for combo_x in common_mse_scores_x for combo_y in common_mse_scores_y if combo_x[1] == combo_y[1]],
    key=lambda x: x[0]
)

# Select the top 100 common combinations
top_common_combinations = common_mse_scores[:top_n]

# Extract the feature combinations for counting
combinations = [combo[1] for combo in top_common_combinations]

# Count the occurrences of each feature in the top 100 combinations
num_features = len(xcost[0])  # Number of features
feature_counts = count_feature_usage(combinations, num_features)

# Labels for the features
labels = [
    'ysum var w/o hilbert',
    'yhist var w/ hilbert sum',
    'ysum var w/ hilbert lines', 
    'mean abs conf deriv',
    'var conf deriv',
    'mean abs intensity', 
    'var laplacian',
    'var laplacian of conf',
    'confxsum angle pred', 
    'peaks angle pred',
    'xsum var',
    'confxsum var'
]

# Plotting the top 100 common combinations with their MSE values
combinations_for_plot = [' + '.join(map(str, combo[1])) for combo in top_common_combinations]
mse_values_for_plot = [combo[0] for combo in top_common_combinations]

plt.figure(figsize=(14, 7))
plt.plot(combinations_for_plot, mse_values_for_plot, 'o-')
plt.xticks(rotation=90, fontsize=8)
plt.xlabel('Feature Combinations')
plt.ylabel('MSE')
plt.title('Top 100 Common MSE for Feature Combinations (xdata & ydata)')
plt.tight_layout()
plt.show()

# Plotting the frequency of feature usage in the top 100 common combinations
plt.figure(figsize=(14, 7))
plt.bar(range(num_features), feature_counts, color='purple', alpha=0.7)
plt.xticks(range(num_features), labels, rotation=90, fontsize=10)
plt.xlabel('Features')
plt.ylabel('Frequency')
plt.title('Frequency of Feature Usage in Top 100 Common Combinations (xdata & ydata)')
plt.tight_layout()
plt.show()

# Print the best combination
best_combination = top_common_combinations[0]
print(f"Best combination: {best_combination[1]} with combined MSE = {best_combination[0]}")


# Find the combination with the least number of features
min_features_combination = min(top_common_combinations, key=lambda x: len(x[1]))

# Print the combination with the least number of features and its combined MSE
print(f"Combination with the least features: {min_features_combination[1]} with combined MSE = {min_features_combination[0]}")
print(f"Number of features used: {len(min_features_combination[1])}")

#'''
################################################################################
# Compare sigmas as target function
################################################################################

'''
# Initialize lists to store learned weights
learned_weights_x = []
learned_weights_y = []

# Define sigma values to experiment with
sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Adjust sigma values as needed

# Loop through each sigma value
for sigma in sigma_values:
    # Create Gaussian target array
    x = np.linspace(-1, 1, 41)
    gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)

    inx, iny = np.array(xcost), np.array(ycost)

    # Normalize the metrics using Min-Max scaling
    scalerx = MinMaxScaler()
    inxn = scalerx.fit_transform(inx)
    scalery = MinMaxScaler()
    inyn = scalery.fit_transform(iny)
    
    # Train a linear regression model for x
    modelx = LinearRegression()
    modelx.fit(inxn, gaussian_array)
    
    # Train a linear regression model for y
    modely = LinearRegression()
    modely.fit(inyn, gaussian_array)
    
    # Store learned weights
    learned_weights_x.append(modelx.coef_)
    learned_weights_y.append(modely.coef_)

# Convert lists to numpy arrays for averaging
learned_weights_x = np.array(learned_weights_x)
learned_weights_y = np.array(learned_weights_y)

# Calculate the average weights across all sigma values
average_weights_x = np.mean(learned_weights_x, axis=0)
average_weights_y = np.mean(learned_weights_y, axis=0)

# Print the average weights
print("Average Learned Weights (x):", average_weights_x)
print("Average Learned Weights (y):", average_weights_y)

# Optionally, plot the learned weights for different sigmas
plt.figure(figsize=(12, 6))

# Plot learned weights for x
plt.subplot(1, 2, 1)
for i, sigma in enumerate(sigma_values):
    plt.plot(learned_weights_x[i], label=f'Sigma = {sigma}')
plt.title('Learned Weights for Different Sigma Values (x)')
plt.xlabel('Metric Index')
plt.ylabel('Learned Weights')
plt.legend()

# Plot learned weights for y
plt.subplot(1, 2, 2)
for i, sigma in enumerate(sigma_values):
    plt.plot(learned_weights_y[i], label=f'Sigma = {sigma}')
plt.title('Learned Weights for Different Sigma Values (y)')
plt.xlabel('Metric Index')
plt.ylabel('Learned Weights')
plt.legend()

plt.tight_layout()
plt.show()

#'''
###############################################################################
# Angle mean varince graphs
###############################################################################
'''
windows=np.array(alldat)[:41]
qwer=[]
for w in windows:
    img = w[0]
    img = byb.envelope(img)
    yhist,_=byb.getHist(img)#
    # yhist,_=getHist(20*np.log10(abs(img)+1))
    t=yhist[2000:2800]
    # t=byb.hilb(yhist)[2000:2800]
    qwer.append([t.mean(),t.var()])

data =qwer
means = [item[0] for item in data]
variances = [item[1] for item in data]

# Convert variances to standard deviations
std_devs = np.sqrt(variances)

# Create x values that go from -20 to 20
x_values = list(range(-20, 21))

# Adjust means and std_devs to match the x_values length
extended_means = means[:len(x_values)]
extended_std_devs = std_devs[:len(x_values)]

# Plot
plt.figure(figsize=(10, 6),dpi=300)
plt.errorbar(x_values, extended_means, yerr=extended_std_devs, fmt='-o', ecolor='r', capsize=5, capthick=2)
plt.xlabel('Degrees')
plt.ylabel('Mean Value')
plt.grid(True)
plt.show()



###
#average among all paths
###

# Group the data into 8 paths with 41 images each
paths = np.array(alldat)
paths = np.array(np.split(paths,8,axis=0))

#Labels
lines = np.array(lines)  # Ensure 'lines' is a NumPy array if it's not already
new_lines = []
# Loop through each array of shape (2, 82)
for array in lines:
    # Split the array into two halves of 41 each
    first_half = array[:, :41]
    second_half = array[:, 41:]
    # Append both halves
    new_lines.append(first_half)
    new_lines.append(second_half)
# Convert the list to a NumPy array with shape (8, 2, 41)
new_lines = np.array(new_lines)

# Initialize lists to store means and variances for all paths
all_means = []
all_variances = []

# Process each path
for i,path in enumerate(paths):
    qwer = []
    for j,w in enumerate(path):
        img = w[0]
        img = byb.envelope(img)
        yhist, _ = byb.getHist(img)
        t = yhist[new_lines[i,0,j]:new_lines[i,1,j]]
        qwer.append([t.mean(), t.var()])
    
    data = qwer
    means = [item[0] for item in data]
    variances = [item[1] for item in data]
    
    all_means.append(means)
    all_variances.append(variances)

# Convert to NumPy arrays for easier manipulation
all_means = np.array(all_means)
all_variances = np.array(all_variances)

# Compute the average means and variances across the 8 paths
avg_means = all_means.mean(axis=0)
avg_variances = all_variances.mean(axis=0)

# Convert variances to standard deviations
avg_std_devs = np.sqrt(avg_variances)

# Create x values that go from -20 to 20
x_values = list(range(-20, 21))

# Adjust means and std_devs to match the x_values length
extended_means = avg_means[:len(x_values)]
extended_std_devs = avg_std_devs[:len(x_values)]

# Plot
plt.figure(figsize=(10, 6), dpi=300)
plt.errorbar(x_values, extended_means, yerr=extended_std_devs, fmt='-o', ecolor='r', capsize=5, capthick=2)
plt.xlabel('Degrees')
plt.ylabel('Mean Value')
plt.grid(True)
plt.show()

# Set up the subplot grid (3 rows x 3 cols, 8 for paths)
fig, axes = plt.subplots(4, 2, figsize=(15, 15), dpi=300)
axes = axes.ravel()  # Flatten the axes for easy iteration

# Plot individual paths
for i, (means, variances) in enumerate(zip(all_means, all_variances)):
    std_devs = np.sqrt(variances[:len(x_values)])  # Convert variance to std dev
    axes[i].errorbar(x_values, means[:len(x_values)], yerr=std_devs, fmt='-o', ecolor='r', capsize=5, capthick=2)
    axes[i].set_title(f'Path {i+1}')
    axes[i].set_xlabel('Degrees')
    axes[i].set_ylabel('Mean Value')
    axes[i].grid(True)

# Adjust layout
plt.tight_layout()

# Show the subplots for individual paths
plt.show()

#'''
###############################################################################
# Load all data from all acquisitions
###############################################################################
#'''

# PARAMS
date = '01Aug6'
ptype2conf = {
    'cl': 'curvedlft_config.json',
    'cf': 'curvedfwd_config.json',
    'rf': 'rotationfwd_config.json',
    'rl': 'rotationlft_config.json'
}

# Get the base directory
current_dir = Path(__file__).resolve().parent.parent.parent

# Initialize lists to store all loaded data
all_filenames = []
all_conf = []
all_positions = []

# Loop over each ptype to load the corresponding data
for ptype, confname in ptype2conf.items():
    # Set the path for the current ptype
    datapath = current_dir / 'data' / 'acquired' / date / 'processed' / ptype
    # Get file names in the current directory
    fileNames = [f.name for f in datapath.iterdir() if f.is_file()]
    all_filenames.append([datapath,fileNames])
    
    # Load the configuration of the experiment
    conf = byb.loadConf(datapath, confname)
    all_conf.append(conf)
    
    # Organize the data as [coord, q rot, id]
    positions = []
    for i, coord in enumerate(conf['tcoord']):
        positions.append(coord + conf['quater'][i] + [i])
    
    all_positions.append(np.array(positions))

# If you need to concatenate positions or other data across ptpyes, you can do so here
allmove = np.concatenate(all_positions, axis=0)

###############################################################################
datapath = Path(__file__).resolve().parent / 'ml' / 'lines'
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

lines = []
for i in range(0, 4):
    btm = np.load(datapath / fileNames[i])
    top = np.load(datapath / fileNames[i+4])
    lines.append([top, btm])

lbls = np.concatenate(np.transpose(lines, (0, 2, 1)))
###############################################################################

# strt,end=1600,2200
strt,end=2000,2800
subsec = False

alldat = []

datapath = all_filenames[0][0]
fileNames = all_filenames[0][1]
for pos,x in enumerate(allmove):
    
    if pos%82 == 0:
        datapath = all_filenames[pos//82][0]
        fileNames = all_filenames[pos//82][1]
    
    img = byb.loadImg(fileNames, int(x[-1]), datapath)#[100:]
    cmap = confidenceMap(img,rsize=True)
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]
    
    if subsec:
        img,cmap=img[strt:end],cmap[strt:end]
    
    yhist,xhist = byb.getHist(img)
    cyhist,cxhist = byb.getHist(cmap)
    alldat.append([img,cmap,yhist,xhist,cyhist,cxhist])
    
#'''
###############################################################################
#Calculate features
###############################################################################
# '''
def calculate_metrics(data):
    cost = []
    
    for idx,pos in enumerate(data):
        metrics = []
        
        img, cmap, yhist, xhist, cyhist, cxhist = pos
        
        # print(img.shape,
        #       cmap.shape,
        #       yhist.shape,
        #       xhist.shape,
        #       cyhist.shape,
        #       cxhist.shape)
        
        strt,end = lbls[idx]
        
        #crop the data
        img = img[strt:end, :]
        cmap = cmap[strt:end, :]
        yhist,xhist = byb.getHist(img)
        cyhist,cxhist = byb.getHist(cmap)

        # print(cxhist.shape)
        
        # yhist variance without Hilbert
        metrics.append(np.var(yhist))
        
        # yhist variance with Hilbert of sum
        metrics.append(np.var(byb.hilb(yhist)))
        
        # yhist variance with Hilbert of each line
        hb = byb.envelope(img)
        hbyh, _ = byb.getHist(hb)
        metrics.append(np.var(hbyh))
        # metrics.append(np.var(yhist)+np.var(byb.hilb(yhist))*np.var(hbyh))
        
        # mean abs of confidence deriv
        metrics.append(np.mean(np.abs(np.diff(cyhist))))
        
        # variance of confidence deriv
        metrics.append(np.var(np.diff(cyhist)))
        
        # mean abs luminance
        metrics.append(np.mean(np.abs(img)))
        
        # variance of Laplacian 
        metrics.append(variance_of_laplacian(img))
        
        # variance of Laplacian of cmap
        metrics.append(variance_of_laplacian(cmap))
        
        # cxhist angle prediction
        l1, ang1, _, _ = byb.regFit(cxhist)
        metrics.append(abs(ang1))
        
        # peaks angle prediction
        nimg = byb.normalize(img)
        peaks = byb.findPeaks(nimg)
        l2, ang2, _, _ = byb.regFit(peaks)
        metrics.append(abs(ang2))
        
        # xhist variance
        metrics.append(np.var(xhist))
        
        # cxhist variance
        metrics.append(np.var(cxhist))
        
        cost.append(metrics)
    
    return cost

cost = calculate_metrics(alldat)
#'''
###############################################################################
# Linear model joint data
###############################################################################
# '''
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

indata = np.array(cost)

#gaussian
x = np.linspace(-1, 1, 41)
sigma = 0.3  # Adjust sigma for the desired smoothness
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
goal=np.tile(gaussian_array,8)

# Normalize the metrics using Min-Max scaling
scalerx = MinMaxScaler()
inxn = scalerx.fit_transform(indata)

# Train a linear regression model
model = LinearRegression()
# model = DecisionTreeRegressor()
model.fit(inxn, goal)

predx=[]
for i in cost:
    met = scalerx.transform(np.array(i).reshape(1,-1))
    predx.append(model.predict(met))

plt.plot(predx)
plt.plot(goal)
plt.show()


# Get the weights
weights = model.coef_
print("Avg Learned weights:", weights)

# Convert to percentages
percentages = np.round(100 * weights / np.sum(np.abs(weights)), 2)
weights_x_percentage = percentages

    
labels = [
    'ysum var w/o hilbert',
    'yhist var w/ hilbert sum',
    'ysum var w/ hilbert lines', 
    'mean abs conf deriv',
    'var conf deriv',
    'mean abs intensity', 
    'var laplacian',
    'var laplacian of conf',
    'confxsum angle pred', 
    'peaks angle pred',
    'xsum var',
    'confxsum var'
]

# Convert absolute weights for chart
abs_weights_x_percentage = [abs(w) for w in weights_x_percentage]

# Sort by absolute values in decreasing order
sorted_indices = np.argsort(abs_weights_x_percentage)[::-1]
sorted_weights = [abs_weights_x_percentage[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]

# Create a horizontal bar chart
plt.figure(figsize=(10, 7))
y_pos = np.arange(len(sorted_labels))
plt.barh(y_pos, sorted_weights, color='skyblue')
plt.yticks(y_pos, sorted_labels)
plt.xlabel('Importance (Absolute Value)')
plt.title('Importance of Metrics Based on Learned Weights')
plt.grid(axis='x', linestyle='--')

# Add data labels to each bar
for index, value in enumerate(sorted_weights):
    plt.text(value, index, f'{value:.2f}', va='center')

# Reverse the order of the y-ticks to have larger values on top
plt.gca().invert_yaxis()

plt.show()

#'''

#####################
# feature combination 
#####################
# '''
import itertools

# Get the number of features
num_features = inxn.shape[1]

# Store MSE scores for each combination
mse_scores_x = []
preds=[]

# Train and evaluate for all combinations of features
for r in range(1, num_features + 1):
    for combination in itertools.combinations(range(num_features), r):
        # Select the columns for this combination
        inxn_subset = inxn[:, combination]
        
        # Train the linear regression model
        modelx = LinearRegression()
        modelx.fit(inxn_subset, goal)
        
        # Predict the goal shape
        predx = modelx.predict(inxn_subset)
        
        # Calculate the MSE for this combination
        mse_x = mean_squared_error(goal, predx)
        
        # Store the MSE and the combination of features
        mse_scores_x.append((mse_x, combination))
        
        # Store the prediction
        preds.append((predx, mse_x, combination))
        
        
#Plot of the worst prediction
# Find the prediction with the highest MSE in the preds list
worst_pred, worst_mse, worst_combination = max(preds, key=lambda x: x[1])
plt.plot(worst_pred)
plt.plot(goal)
plt.show()

# Sort the results by MSE
mse_scores_x.sort(key=lambda x: x[0])

# Print the results
print("Top 5 combinations for xdata with lowest MSE:")
for i in range(5):
    print(f"Combination {mse_scores_x[i][1]}: MSE = {mse_scores_x[i][0]}")

# Identify the combination with the least features and lowest MSE
least_features_x = min(mse_scores_x, key=lambda x: (len(x[1]), x[0]))

print("Combination for xdata with the least amount of features and lowest MSE:")
print(f"Combination {least_features_x[1]}: MSE = {least_features_x[0]}")

#############
#plotting
#############

# Function to count feature usage in combinations
def count_feature_usage(combinations, num_features):
    feature_counts = np.zeros(num_features)
    for combo in combinations:
        for feature in combo:
            feature_counts[feature] += 1
    return feature_counts

# Select the top 100 combinations for both xdata and ydata
top_n = None
top_combinations_x = mse_scores_x[:top_n]

# Extract the feature combinations
combinations = [combo[1] for combo in top_combinations_x]

# Count the occurrences of each feature in the top 100 combinations
feature_counts = count_feature_usage(combinations, num_features)

# Labels for the features
labels = [
    'ysum var w/o hilbert',
    'yhist var w/ hilbert sum',
    'ysum var w/ hilbert lines', 
    'mean abs conf deriv',
    'var conf deriv',
    'mean abs intensity', 
    'var laplacian',
    'var laplacian of conf',
    'confxsum angle pred', 
    'peaks angle pred',
    'xsum var',
    'confxsum var'
]

# Plotting the top 100 combinations with their MSE values
combinations_for_plot = [' + '.join(map(str, combo)) for combo in combinations]
mse_values_for_plot = [combo[0] for combo in top_combinations_x]

plt.figure(figsize=(14, 7))
plt.plot(combinations_for_plot, mse_values_for_plot, 'o-')
plt.xticks([], fontsize=8)
plt.xlabel('Feature Combinations')
plt.ylabel('MSE')
plt.title('Top MSE for Feature Combinations')
plt.tight_layout()
plt.show()

# Plotting the frequency of feature usage in the top 100 combinations
plt.figure(figsize=(14, 7))
plt.bar(range(num_features), feature_counts, color='purple', alpha=0.7)
plt.xticks(range(num_features), labels, rotation=90, fontsize=10)
plt.xlabel('Features')
plt.ylabel('Frequency')
plt.title('Frequency of Feature Usage in Top Combinations ')
plt.tight_layout()
plt.show()

# Print the best combination
best_combination = top_combinations_x[0]
print(f"Best combination: {best_combination[1]} with MSE = {best_combination[0]}")

# Find the combination with the least number of features
min_features_combination = min(top_combinations_x, key=lambda x: len(x[1]))

# Print the combination with the least number of features and its MSE
print(f"Combination with the least features: {min_features_combination[1]} with MSE = {min_features_combination[0]}")
print(f"Number of features used: {len(min_features_combination[1])}")
# '''
################################3##############################################
# Separate models separate data
#############################################################33#########################3#
'''
indata = np.reshape(np.array(cost), (8,41,-1))
linmods=[]
preds=[]
normdata=[]
for acq in indata:
    
    #gaussian
    x = np.linspace(-1, 1, len(acq))
    sigma = 0.3  # Adjust sigma for the desired smoothness
    gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
    
    goal=gaussian_array
    
    # Normalize the metrics using Min-Max scaling
    scalerx = MinMaxScaler()
    inxn = scalerx.fit_transform(acq)
    
    # Train a linear regression model
    model = LinearRegression()
    # model = DecisionTreeRegressor()
    model.fit(inxn, goal)
    
    predx=[]
    for i in acq:
        met = scalerx.transform(np.array(i).reshape(1,-1))
        predx.append(model.predict(met))
    
    #store the trained model
    linmods.append(model)
    #store the predictions
    preds.append(predx)
    #store the normalized data
    normdata.append(inxn)
    

# Get the weights
weights = [model.coef_ for model in linmods]
# get the average weight among all models
weights_x = np.mean(weights, axis=0)
print("Avg Learned weights:", weights_x)

# Convert to percentages
percentages = [np.round(100 * w / np.sum(np.abs(w)), 2) for w in weights]
weights_x_percentage = np.mean(percentages, axis=0)

# Convert the weights to string format to avoid scientific notation
# weights_x_str = [f"{w:.2f}" for w in weights_x_percentage]

# print("Learned weights (x) as percentages:\n", weights_x_str)

#PLOTTING
for p in preds:        
    plt.plot(p)
plt.plot(goal)
plt.show()

#print(round(mean_squared_error(goal, predx),6))

    
labels = [
    'ysum var w/o hilbert',
    'yhist var w/ hilbert sum',
    'ysum var w/ hilbert lines', 
    'mean abs conf deriv',
    'var conf deriv',
    'mean abs intensity', 
    'var laplacian',
    'var laplacian of conf',
    'confxsum angle pred', 
    'peaks angle pred',
    'xsum var',
    'confxsum var'
]

# Convert absolute weights for chart
abs_weights_x_percentage = [abs(w) for w in weights_x_percentage]

# Sort by absolute values in decreasing order
sorted_indices = np.argsort(abs_weights_x_percentage)[::-1]
sorted_weights = [abs_weights_x_percentage[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]

# Create a horizontal bar chart
plt.figure(figsize=(10, 7))
y_pos = np.arange(len(sorted_labels))
plt.barh(y_pos, sorted_weights, color='skyblue')
plt.yticks(y_pos, sorted_labels)
plt.xlabel('Importance (Absolute Value)')
plt.title('Importance of Metrics Based on Learned Weights')
plt.grid(axis='x', linestyle='--')

# Add data labels to each bar
for index, value in enumerate(sorted_weights):
    plt.text(value, index, f'{value:.2f}', va='center')

# Reverse the order of the y-ticks to have larger values on top
plt.gca().invert_yaxis()

plt.show()

#######
#Feature combination
########
topx = None

# Get the number of features
num_features = indata.shape[2]  # Number of features in each acquisition

# Initialize the list to store MSE scores for each model
mse_scores = [[] for _ in range(len(linmods))]

# Loop over each model (each acquisition)
for model_idx, (model, acq) in enumerate(zip(linmods, normdata)):
    
    # Loop over each combination of features (r is the number of features in the combination)
    for r in range(1, num_features + 1):
        for combination in itertools.combinations(range(num_features), r):
            
            # Select the columns for this combination
            inxn_subset = acq[:, combination]
            
            # Predict the goal shape
            model = LinearRegression()
            model.fit(inxn_subset, goal)
            pred = model.predict(inxn_subset)
            
            # Calculate the MSE for this combination
            mse = mean_squared_error(goal, pred)
            
            # Store the MSE and the combination of features
            mse_scores[model_idx].append((mse, combination))

# Sort the MSE scores for each model
for i in range(len(mse_scores)):
    mse_scores[i].sort(key=lambda x: x[0])

# Extract the feature combinations for each model's top combinations
top_combinations_sets = [set(combo[1] for combo in mse_scores[i][:topx]) for i in range(len(mse_scores))]

# Find the intersection of the top combinations across all models
common_combinations = top_combinations_sets[0]
for i in range(1, len(top_combinations_sets)):
    common_combinations = common_combinations.intersection(top_combinations_sets[i])

print(f"Found {len(common_combinations)} common feature combinations among top {topx} of each model.")

# If no common combinations are found, this could be a sign that topx might be too small.
if not common_combinations:
    print("No common combinations found. Consider increasing topx or revising criteria.")

# Get the average MSE values for the common combinations
common_mse_scores = []
for common_combo in common_combinations:
    mse_sum = 0
    count = 0
    for model_scores in mse_scores:
        for score, combo in model_scores:
            if combo == common_combo:
                mse_sum += score
                count += 1
    if count > 0:
        average_mse = mse_sum / count  # Average MSE across models
        common_mse_scores.append((average_mse, common_combo))

# Sort the common combinations by their average MSE values
common_mse_scores.sort(key=lambda x: x[0])

# Select the top combinations based on their average MSE
top_common_combinations = common_mse_scores[:topx]

# Extract the feature combinations for further analysis or plotting
combinations_for_plot = [combo[1] for combo in top_common_combinations]
mse_values_for_plot = [combo[0] for combo in top_common_combinations]

# Count the occurrences of each feature in the top common combinations
def count_feature_usage(combinations, num_features):
    feature_counts = np.zeros(num_features)
    for combo in combinations:
        for feature in combo:
            feature_counts[feature] += 1
    return feature_counts

feature_counts = count_feature_usage(combinations_for_plot, num_features)

# Labels for the features (assuming the same labels as before)
labels = [
    'ysum var w/o hilbert',
    'yhist var w/ hilbert sum',
    'ysum var w/ hilbert lines', 
    'mean abs conf deriv',
    'var conf deriv',
    'mean abs intensity', 
    'var laplacian',
    'var laplacian of conf',
    'confxsum angle pred', 
    'peaks angle pred',
    'xsum var',
    'confxsum var'
]

# Plotting the top common combinations with their MSE values
combination_labels_for_plot = [' + '.join(map(str, combo)) for combo in combinations_for_plot]

plt.figure(figsize=(14, 7))
plt.plot(combination_labels_for_plot, mse_values_for_plot, 'o-')
plt.xticks([], fontsize=8)
plt.xlabel('Feature Combinations')
plt.ylabel('MSE')
plt.title('Top Common MSE for Feature Combinations')
plt.tight_layout()
plt.show()

# Plotting the frequency of feature usage in the top common combinations
plt.figure(figsize=(14, 7))
plt.bar(range(num_features), feature_counts, color='purple', alpha=0.7)
plt.xticks(range(num_features), labels, rotation=90, fontsize=10)
plt.xlabel('Features')
plt.ylabel('Frequency')
plt.title('Frequency of Feature Usage in Top Common Combinations')
plt.tight_layout()
plt.show()

# Print the best combination
if top_common_combinations:
    best_combination = top_common_combinations[0]
    print(f"Best combination: {best_combination[1]} with combined MSE = {best_combination[0]}")

    # Find the combination with the least number of features
    min_features_combination = min(top_common_combinations, key=lambda x: len(x[1]))
    
    # Print the combination with the least number of features and its combined MSE
    print(f"Combination with the least features: {min_features_combination[1]} with combined MSE = {min_features_combination[0]}")
    print(f"Number of features used: {len(min_features_combination[1])}")
else:
    print("No common combinations were found.")
#'''

###############################################################################
# MSE lin reg error video
###############################################################################
'''
import matplotlib.pyplot as plt
import imageio
import os

# Sort the preds list by MSE in ascending order
sorted_preds = sorted(preds, key=lambda x: x[1])

# Directory to save frames
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

# Set a higher DPI for better resolution
dpi = 200

# Generate and save frames for each prediction
for i, (pred, mse, combination) in enumerate(sorted_preds):
    plt.figure(figsize=(6, 4), dpi=dpi)  # Increased DPI for higher resolution
    plt.plot(pred, label='Prediction')
    plt.plot(goal, label='Goal', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Prediction vs. Goal\nCombination: {combination}\nMSE: {mse:.4f}')
    plt.legend()
    plt.tight_layout()

    # Save the frame as an image
    frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
    plt.savefig(frame_path)
    plt.close()

# Create a video from the saved frames
video_path = "predictions_mse.mp4"
with imageio.get_writer(video_path, fps=2) as video_writer:  # fps controls the frame rate of the video
    for i in range(len(sorted_preds)):
        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        video_writer.append_data(imageio.imread(frame_path))

# Optionally, remove the frames directory after creating the video
# shutil.rmtree(frames_dir)

print(f"Video saved as {video_path}")

'''
###############################################################################
# Correlation analysis
###############################################################################
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'indata' is your feature matrix
# Convert it to a DataFrame for easier manipulation
feature_df = pd.DataFrame(indata, columns=labels)

# Calculate the correlation matrix
correlation_matrix = feature_df.corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = feature_df.columns
vif_data["VIF"] = [variance_inflation_factor(feature_df.values, i) for i in range(len(feature_df.columns))]

print(vif_data)
#'''




