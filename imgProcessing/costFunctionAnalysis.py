# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:01:38 2024

@author: Mateo-drr

Initial analysis of US data. All code is deprecated and replaced by the code in testDataAnalysis.
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
from scipy.signal import savgol_filter

#GLOBAL VARS
CONFMAP=None
IMAGE=None
POS=None

#PARAMS
date='01Aug0'
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

#%%
'''
Initial bayessian optimization testing, using default gp_minimize (deprecated)
'''

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

#%%
'''
ALL data plot
'''

# '''
# Determine the number of images and grid dimensions
side=ymove
num_images = len(side)
step=20

iok = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
# cols = 42  # Number of columns in the grid
cols = len(iok)  # Number of columns in the grid
rows = 1#(num_images + cols - 1) // cols  # Calculate rows needed

# Create a figure with subplots arranged in a grid
fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), dpi=200)
axes = axes.flatten()  # Flatten the 2D array of axes to easily index it

#LABELS
idx=3
lblpth = current_dir / 'ALU---Autonomous-Lung-Ultrasound' / 'imgProcessing' / 'ml' / 'lines'
top = np.load(lblpth / f'top_lines_{idx}.npy')
btm = np.load(lblpth / f'btm_lines_{idx}.npy')

ang = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ang2 = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16,  20]

maxv = np.max(np.mean(byb.logS(byb.envelope(byb.loadImg(fileNames,int(side[len(side)//2][-1]),
                                               datapath))),axis=1))

# Determine the global minimum and maximum values across all images
global_min = float('inf')
global_max = float('-inf')

for count, pos in enumerate(side):
    img = byb.loadImg(fileNames, int(pos[-1]), datapath)
    img = byb.envelope(img)
    img = 20 * np.log10(abs(img) + 1)
    global_min = min(global_min, img.min())
    global_max = max(global_max, img.max())

#For cmap min and max is 0 and 1
global_min=0
global_max=1

# Plot each yhist in its respective subplot
a,b,c=3,90,0.05
print(a,b,c)
i=0
for count, pos in enumerate(side):
    if count not in iok:
        continue
    img = byb.loadImg(fileNames,int(pos[-1]), datapath)#[2000:2800]  # Load the image
    # img = byb.envelope(img)
    # img = 20* np.log10(abs(img)+1)
    cmap = confidenceMap(img,alpha=a,beta=b,gamma=c,rsize=True)
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
    
    crop = cmap
    lineMean = np.mean(crop,axis=1)
    deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//125,
                               polyorder=2, deriv=1))
    # cyhist,cxhist = byb.getHist(cmap[2000:2800,:],tensor=False)  
    # gcyhist = np.diff(cyhist)
    # yhist,xhist = byb.getHist(img,tensor=False)
    
    # yhist = byb.hilb(yhist)
    # lap = laplace(img)
    
    #envelope of whole image
    # hb = byb.envelope(img)
    #normalization of each line
    # himgCopy = hb.copy()
    # for k in range(hb.shape[1]):
    #     line = hb[:,k]
        
    #     min_val = np.min(line)
    #     max_val = np.max(line)
    #     line = (line - min_val) / (max_val - min_val)
        
    #     himgCopy[:,k] = line
    #crop the image
    # imgCrop = hb#[2000:2800,:]
    
    # yhist = np.mean(img,axis=1)
    
    # gy,gx = np.gradient(img)
    # yhist,_ = byb.getHist(gy)
    # yhist = byb.hilb(yhist)
    # _,g = fit_polynomial(hilb(yhist),10)#fit_gaussian(hilb(yhist))
    # axes[i].plot(yhist, np.arange(len(yhist)))  # Plot yhist
    axes[i].plot(deriv, np.arange(len(deriv)))  # Plot yhist
    #axes[i].plot(g, np.arange(len(g)), linewidth=6)
    # axes[i].plot(yhist,np.arange(len(yhist)))
    # axes[i].imshow(img,aspect='auto',cmap='viridis')
    # lol= axes[i].imshow(img,aspect='auto',cmap='viridis',vmin=global_min, vmax=global_max)
    # lol = axes[i].imshow(cmap,aspect='auto',cmap='viridis',vmin=global_min, vmax=global_max)
    # axes[i].imshow(20*np.log10(abs(lap)+1),aspect='auto',cmap='viridis')
    # axes[i].invert_yaxis()
    
    #labels plot
    # axes[i].axhline(top[int(pos[-1])], color='r')
    # axes[i].axhline(btm[int(pos[-1])], color='b')
    
    
    # Set y-axis visibility for the first subplot only
    if i == 0:
        axes[i].yaxis.set_visible(True)
        axes[i].set_ylabel('Depth [px]', fontsize=18)
        # axes[i].set_ylabel('Depth [px]', fontsize=18)
    else:
        axes[i].yaxis.set_visible(False)
    # axes[i].set_xticks([])    
    #for sporadic angles
    axes[i].set_xticks(np.arange(0,0.025,0.01))
    print(deriv.max())
    axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0)) 
    axes[i].xaxis.get_offset_text().set_fontsize(14)

    # Set x-axis label for each subplot
    # axes[i].set_xlabel(f"{ang[i]}", fontsize=16)  # You can change the label text
    # axes[i].set_xlabel(f"Deriv.",fontsize=16, labelpad=30)  # You can change the label text
    axes[i].set_title(f"{ang2[i]}", fontsize=16)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)
    axes[i].spines['left'].set_visible(False)
    axes[i].tick_params(axis='both', labelsize=16)
    # axes[i].grid(axis='x')
    
    i+=1

# # Hide any unused subplots
# for j in range(len(side), len(axes)):
#     axes[j].axis('off')
    
#for all plots w colorbar
# axes[i].yaxis.set_visible(False)
# axes[i].set_xticks([])
# axes[i].spines['top'].set_visible(False)
# axes[i].spines['right'].set_visible(False)
# axes[i].spines['bottom'].set_visible(False)
# axes[i].spines['left'].set_visible(False)
# # Add a global colorbar linked to all plots
# cbar = fig.colorbar(lol, ax=axes[i], location='right', fraction=1)
# cbar.set_label('Confidence', fontsize=18, labelpad=10)
# cbar.ax.tick_params(labelsize=16)


# for sporadic angles
plt.figtext(0.5, 1.01, 'Degrees', ha='center', va='center', fontsize=18)
plt.figtext(0.5, -0.05, 'Derivative Amplitude', ha='center', va='center', fontsize=18)

#for all angles
# plt.figtext(0.5, -0.02, 'Degrees', ha='center', va='center', fontsize=18)

# plt.tight_layout(pad=0.001)

plt.show()

raise Exception("Stopping execution here")

#%%
'''
Initial analysis of the data and some initial features. Deprecated
'''

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
To use this part of the code when loading all data comment out all the features and cmap
just append the images to the alldat list
'''
'''

from sklearn.preprocessing import MinMaxScaler,QuantileTransformer
from scipy.signal import savgol_filter

#OR
#run this
justimgs = alldat
# justimgs = np.array([[img[0]] for img in alldat])

#for cmap
justimgs = np.array([[img[1]] for img in alldat0])



windows=np.array(justimgs)[:41]
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
paths = np.array(justimgs)
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
arrays=[]
# Process each path
for i,path in enumerate(paths):
    qwer = []
    hists=[]
    for j,w in enumerate(path):
        img = w[0]#[2000:2800]
        raw = w[0]#[2000:2800]
        # img = byb.envelope(img)[2000:2800]
        
        # img = 20*np.log10(abs(img)+1)
        
        # img = img - np.max(img)
        
        # minv = np.min(img)
        # maxv = np.max(img)
        # img = (img - minv)/(maxv-minv)
        
        # imgn = byb.envelope(imgn)#[2000:2800,:])
        
        
        imgc = img.copy()

        # for k in range(img.shape[1]):
        #     line = img[:,k]
            
        #     min_val = np.min(line)
        #     max_val = np.max(line)
        #     line = (line - min_val) / (max_val - min_val)
            
        #     imgc[:,k] = line 
            # imgc[:,k] = line - np.max(line)
        
        
        # imgc = imgc[2000:2800,:]
        yhist = np.mean(imgc,axis=1)#, _ = byb.getHist(img)
        #t = yhist[new_lines[i,0,j]:new_lines[i,1,j]]
        t = yhist
        # t = byb.hilb(yhist)#[2000:2800]
        # t = 20*np.log10(abs(t)+1)
        # avg = t.mean()
        # var = t.var()
        # avg = var
        
        # imgc = imgc/np.max(imgc)
        
        # avg = np.mean((imgc))
        # nimg = byb.normalize(20*np.log1p(abs(img)))
        # peaks = byb.findPeaks(img[2000:2800,:])
        # l2, ang2, _, _ = byb.regFit(peaks)
        # avg = abs(ang2)
        
        # cmap = confidenceMap(raw,rsize=False)
        # cmap = resize(cmap, (raw.shape[0], raw.shape[1]), anti_aliasing=True)#[strt:end]
        # cyhist,cxhist = byb.getHist(img[2000:2800])
        cyhist = np.mean(img, axis=1)[2000:2800]
        #cyhist = (cyhist - cyhist.min())/(cyhist.max()-cyhist.min())
        # cyhists = np.convolve(cyhist, np.ones(5)/5, mode='same')
        # deriv= abs(np.gradient(cyhist))
        
        deriv = abs(savgol_filter(cyhist, window_length=len(cyhist)//16, polyorder=2, deriv=1))#[2000:2800]
        # mad = np.median(np.abs(deriv - np.median(deriv)))
        # l2_norm = np.linalg.norm(deriv)
        # deriv = (deriv - deriv.mean())/deriv.std()
        # deriv = np.exp(deriv)/np.sum(np.exp(deriv))
        # deriv = deriv/np.percentile(deriv, 80)
        # deriv = cyhist

        # vardif = np.mean(deriv)
        
        # vardif = np.mean(abs(np.diff(cyhist)/cyhist[:-1]))
        
        # vardif = np.var(abs(np.diff(cyhist)))
        # avg = vardif
        # var = avg
        t = deriv#abs(np.diff(cyhist))
        
        ##############
        #distribution
        ##############
        probs = t/np.sum(t)
        x = np.arange(len(probs))

        avg = np.sum(x*probs)
        var = np.sum((x - avg)**2 * probs)
        avg = var
        #############
        
        # imgc=(imgc-imgc.min())/(imgc.max()-imgc.min())
        # p25, p75 = np.percentile(imgc, [50, 95])
        # imgc= (imgc-p25)/(p75-p25)
        
        # imgc=imgc[2000:2800]
        # lum = np.mean(imgc)
        logim = 20*np.log10(imgc+1)
        lum = np.mean((logim-np.max(logim)))#[2000:2800])
        # lum = np.mean(imgc/imgc.max())
        # lum = np.mean(imgc)/imgc.max()
        # lum = np.mean(imgc[2000:2800]/imgc.max())
        # p25, p75 = np.percentile(imgc, [25, 75])
        # lum = np.mean((imgc-p25)/(p75-p25))
        # lum = np.mean(imgc)
        # lum = (lum - np.mean(imgc)) / np.std(imgc)
        # lum = np.mean((imgc - np.mean(imgc)) / np.std(imgc))
        # lum = np.sqrt(np.mean(np.square(imgc)[2000:2800]))
        # lum = np.mean(imgc)
        
        # hist, bin_edges = np.histogram(imgc[2000:2800], bins=256, range=(0, 255))
        # lum = np.sum(hist * bin_edges[:-1]) / np.sum(hist)

        # p75 = np.percentile(imgc, 90)
        # lum = np.sum(imgc[2000:2800] > p75)/imgc[2000:2800].size
        
        avg=lum
        var=lum

        # avg = laplace(20*np.log10(abs(img[2000:2800,:])+1)).var()
        # avg = laplace(imgc).var()#[2000:2800,:]).var()
        # var=avg
        
        # qwer.append([t.mean(), t.var()])
        qwer.append([avg, var])
        hists.append(t)
    
    data = qwer
    means = [item[0] for item in data]
    variances = [item[1] for item in data]
    # arrays.append(hists)
    
    all_means.append(means)
    all_variances.append(variances)

plt.figure(dpi=200)
for k in hists:
    plt.plot(k)
plt.show()

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
plt.figure(figsize=(10, 6), dpi=200)
# plt.errorbar(x_values, extended_means, yerr=extended_std_devs, fmt='-o', ecolor='r',capsize=5, capthick=2, label='Mean of joint lines w. std')
plt.plot(x_values, extended_means, '-o' )
# plt.title('prob = counts/sum(counts) | counts = mean(lines in img), img = crop(20log(hilbert(RFfiltered)))')
plt.xlabel('Degrees')
plt.ylabel('Average Intensity')
plt.legend()
plt.grid(True)
plt.show()

# Set up the subplot grid (3 rows x 3 cols, 8 for paths)
fig, axes = plt.subplots(4, 2, figsize=(15, 15), dpi=200) 
axes = axes.ravel()  # Flatten the axes for easy iteration

# Plot individual paths
for i, (means, variances) in enumerate(zip(all_means, all_variances)):
    std_devs = np.sqrt(variances[:len(x_values)])  # Convert variance to std dev
    # axes[i].errorbar(x_values, means[:len(x_values)], yerr=std_devs, fmt='-o', ecolor='r', capsize=5, capthick=2)
    axes[i].plot(x_values, means[:len(x_values)], '-o' )
    axes[i].set_title(f'Path {i+1}')
    axes[i].set_xlabel('Degrees')
    axes[i].set_ylabel('Mean Value')
    axes[i].grid(True)

# Adjust layout
plt.tight_layout()

# Show the subplots for individual paths
plt.show()
# '''
##############################
#BOXPLOT OF HISTS
#############################
'''

import numpy as np
import matplotlib.pyplot as plt

# Example arrays (replace with your actual data)
arrays = np.array(arrays)

avg_medians = []
avg_q1s = []
avg_q3s = []
avg_whisker_mins = []
avg_whisker_maxs = []
avg_outliers = []

# Set up subplots (2x4 grid)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()  # Flatten the 2x4 axes grid for easier access

x_values = np.arange(len(arrays[0]))  # Replace with actual x-values if different

for hists_idx, hists in enumerate(arrays):
    # Initialize lists to store statistics for each array in the group
    medians = []
    q1s = []
    q3s = []
    whisker_mins = []
    whisker_maxs = []
    outliers = []
    
    # Loop through each array and compute proper box plot statistics
    for arr in hists:
        # Step 1: Compute quartiles and IQR
        q1 = np.percentile(arr, 25)
        median = np.percentile(arr, 50)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        
        # Step 2: Calculate whiskers (1.5 * IQR rule)
        whisker_min = np.min(arr[arr >= (q1 - 1.5 * iqr)])  # Lowest value within whisker range
        whisker_max = np.max(arr[arr <= (q3 + 1.5 * iqr)])  # Highest value within whisker range
        
        # Step 3: Identify outliers (values beyond the whiskers)
        outliers_arr = arr[(arr < whisker_min) | (arr > whisker_max)]
        
        # Append statistics to the lists
        q1s.append(q1)
        medians.append(median)
        q3s.append(q3)
        whisker_mins.append(whisker_min)
        whisker_maxs.append(whisker_max)
        outliers.append(outliers_arr)
    
    # Plot each statistic as a separate line (or scatter) on the current subplot
    ax = axes[hists_idx]
    
    # Plot Q1, Median (Q2), Q3, Min, and Max
    ax.plot(x_values, q1s, marker='o', color='blue', label='Q1 (25th percentile)')
    ax.plot(x_values, medians, marker='o', color='green', label='Median (Q2)')
    ax.plot(x_values, q3s, marker='o', color='red', label='Q3 (75th percentile)')
    ax.plot(x_values, whisker_mins, marker='o', color='black', label='Whisker Min')
    ax.plot(x_values, whisker_maxs, marker='o', color='orange', label='Whisker Max')
    
    # Scatter plot for outliers
    for i, outliers_arr in enumerate(outliers):
        if len(outliers_arr) > 0:
            ax.scatter([x_values[i]] * len(outliers_arr), outliers_arr, color='purple', marker='x', label='Outliers' if i == 0 else "")
    
    # Add labels and grid
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Value')
    ax.set_title(f'Path {hists_idx+1}')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Only show legends on the first plot
    # if hists_idx == 0:
    #     ax.legend()
    
    # Append values for averaging later
    avg_medians.append(medians)
    avg_q1s.append(q1s)
    avg_q3s.append(q3s)
    avg_whisker_mins.append(whisker_mins)
    avg_whisker_maxs.append(whisker_maxs)
    avg_outliers.append(outliers)

# Adjust layout
plt.tight_layout()

# Show the subplot figure
plt.show()

# Calculate average statistics across all paths
avg_medians = np.mean(avg_medians, axis=0)
avg_q1s = np.mean(avg_q1s, axis=0)
avg_q3s = np.mean(avg_q3s, axis=0)
avg_whisker_mins = np.mean(avg_whisker_mins, axis=0)
avg_whisker_maxs = np.mean(avg_whisker_maxs, axis=0)

# Plot the averages
plt.figure(figsize=(10, 6))

# Plot Q1, Median (Q2), Q3, Min, and Max (Averaged)
plt.plot(x_values, avg_q1s, marker='o', color='blue', label='Avg Q1 (25th percentile)')
plt.plot(x_values, avg_medians, marker='o', color='green', label='Avg Median (Q2)')
plt.plot(x_values, avg_q3s, marker='o', color='red', label='Avg Q3 (75th percentile)')
plt.plot(x_values, avg_whisker_mins, marker='o', color='black', label='Avg Whisker Min')
plt.plot(x_values, avg_whisker_maxs, marker='o', color='orange', label='Avg Whisker Max')

k=0
for path in avg_outliers:
    for i, outliers_arr in enumerate(path):
        
        if len(outliers_arr) > 0:
            plt.scatter([x_values[i]] * len(outliers_arr), outliers_arr, color='purple', marker='x', label='Outliers' if k == 0 else "")
        k=1

# Add labels and grid
plt.xlabel('Degrees')
plt.ylabel('Value')
plt.title('Average Box Plot Statistics Across All Paths')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Show the average plot
plt.show()


#'''

#%%
'''
 More analysis of all the acquisition data, again deprecated.
'''
# '''

# PARAMS
date = '01Aug0'
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
    
    # if subsec:
        # img,cmap=img[strt:end],cmap[strt:end]
    
    yhist,xhist = byb.getHist(img)
    cyhist,cxhist = byb.getHist(cmap)
    alldat.append((img,cmap,yhist,xhist,cyhist,cxhist))
    # alldat.append([img])
    
#'''
raise Exception("Stopping execution here")
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
        
        strt,end = 2000,2800#lbls[idx]
        
        '''
        # yhist variance with Hilbert of each line
        #############################################
        himg = byb.envelope(img)
        himgCopy = himg.copy()
        for k in range(himg.shape[1]):
            line = himg[:,k]
            
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            
            himgCopy[:,k] = line
        himgCopyCrop = himgCopy[strt:end, :]
        hbyh = np.mean(himgCopyCrop,axis=1)
        metrics.append(np.var(hbyh))
        #############################################
        # variance/mean of confidence deriv
        #############################################
        cmap = cmap[strt:end, :]
        cyhist,cxhist = byb.getHist(cmap)
        # metrics.append(np.var(abs(np.diff(cyhist))))
        metrics.append(np.var(np.diff(cyhist)))
        # metrics.append(np.mean(abs(np.diff(cyhist))))
        # metrics.append(np.mean((np.diff(cyhist))))
        #############################################
        # lum mean abs img
        #############################################
        metrics.append(np.mean(np.abs(himg[strt:end,:])))
        
        #############################################
        # varlap 20log abs hilb
        #############################################
        metrics.append(laplace(20*np.log10(abs(himg[strt:end,:])+1)).var())
        #############################################
        
        '''
        # Var of mean of loghilb img
        #############################################
        himg = byb.envelope(img)
        himgCopyCrop = 20*np.log10(himg[strt:end, :]+1)
        hbyh = np.mean(himgCopyCrop,axis=1)
        metrics.append(np.var(hbyh))
        #############################################
        # mean of confidence deriv
        #############################################
        cmap = cmap[strt:end]
        # cyhist,cxhist = byb.getHist(cmap)
        cyhist = np.mean(cmap,axis=1)
        # cyhist = np.sum(cmap,axis=1)
        # metrics.append(np.var(np.diff(cyhist)[strt:end]))
        # metrics.append(np.mean(abs(np.diff(cyhist))))
        
        deriv = abs(savgol_filter(cyhist, window_length=len(cyhist)//16, polyorder=2, deriv=1))#[strt:end]
        
        t=deriv
        probs = t/np.sum(t)
        x = np.arange(len(probs))

        avg = np.sum(x*probs)
        var = np.sum((x - avg)**2 * probs)
        
        metrics.append(var)
        #############################################
        # lum mean hilb img
        #############################################
        # metrics.append(np.mean(himg[strt:end,:]))
        
        # p25, p75 = np.percentile(himg[strt:end,:], [25, 75])
        # lum = np.mean((himg[strt:end,:]-p25)/(p75-p25))
        # metrics.append(lum)
        
        logim = himgCopyCrop#20*np.log10(imgc+1)
        lum = np.mean((logim-np.max(logim)))#[2000:2800])
        metrics.append(lum)
        
        #############################################
        # varlap 20log abs hilb
        #############################################
        metrics.append(laplace(20*np.log10(abs(himg[strt:end,:])+1)).var())
        #############################################
        # '''
        
        #crop the data
        img = img[strt:end, :]
        
        yhist,xhist = byb.getHist(img)
        

        # print(cxhist.shape)
        
        # yhist variance without Hilbert
        # metrics.append(np.var(yhist))
        
        # yhist variance with Hilbert of sum
        # metrics.append(np.var(byb.hilb(yhist)))
        
        # min_val = np.min(hbyh)
        # max_val = np.max(hbyh)
        # hbyh = (hbyh - min_val) / (max_val - min_val)
        
        # hbyh, _ = byb.getHist(hb)
        
        # metrics.append(np.var(yhist)+np.var(byb.hilb(yhist))*np.var(hbyh))
        
        # mean abs of confidence deriv
        # metrics.append(np.mean(np.abs(np.diff(cyhist))))
        
        
        
        
        
        # variance of Laplacian 
        
        
        # variance of Laplacian of cmap
        # metrics.append(variance_of_laplacian(cmap))
        
        # cxhist angle prediction
        l1, ang1, _, _ = byb.regFit(cxhist)
        # metrics.append(abs(ang1))
        
        # peaks angle prediction
        nimg = byb.normalize(img)
        peaks = byb.findPeaks(nimg)
        l2, ang2, _, _ = byb.regFit(peaks)
        # metrics.append(abs(ang2))
        
        # xhist variance
        # metrics.append(np.var(xhist))
        
        # cxhist variance
        # metrics.append(np.var(cxhist))
        
        cost.append(metrics)
        
    
    return cost

cost = calculate_metrics(alldat)

labels = [
    # 'ysum var w/o hilbert',
    # 'yhist var w/ hilbert sum',
    'Variance of joint lines',#'ysum var w/ hilbert lines', 
    'mean abs conf deriv',
    # 'var conf deriv',
    'mean hilb intensity', 
    'var laplacian hilb+log',
    # 'var laplacian of conf',
    # 'confxsum angle pred', 
    # 'peaks angle pred',
    # 'xsum var',
    # 'confxsum var'
]

# cost = np.log1p(cost)
#'''
###############################################################################
# Linear model joint data
###############################################################################
# '''
from sklearn.preprocessing import MinMaxScaler,QuantileTransformer, RobustScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler,PowerTransformer,Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import itertools


indata = np.array(cost)


#gaussian
x = np.linspace(-1, 1, 41)
sigma = 0.3  # Adjust sigma for the desired smoothness
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
goal=np.tile(gaussian_array,8)

# Normalize the metrics using Min-Max scaling
# scalerx = MinMaxScaler()
scalerx = QuantileTransformer()
# scalerx = RobustScaler()
# scalerx = StandardScaler()

# scalerx = MaxAbsScaler()
# scalerx = PowerTransformer()
# scalerx = Normalizer()


inxn = scalerx.fit_transform(indata)
plt.plot(inxn)
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  # 2x2 grid of subplots
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

# Loop through each column and plot
for i in range(4):
    axes[i].plot(inxn[:, i])  # Plot each column
    axes[i].plot(goal)
    axes[i].set_title(labels[i])  # Set title for each subplot
    axes[i].set_xlabel('Index')  # Set x-axis label
    axes[i].set_ylabel('Value')  # Set y-axis label

# Adjust layout
plt.tight_layout()
plt.show()

# '''
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

# Convert absolute weights for chart
#abs_weights_x_percentage = [abs(w) for w in weights_x_percentage]
abs_weights_x_percentage = [w for w in weights_x_percentage]

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
        preds.append((predx, mse_x, combination, model.coef_, model.intercept_))
        
        
#Plot of the worst prediction
# Find the prediction with the highest MSE in the preds list
worst_pred, worst_mse, worst_combination, worst_c, worst_i = max(preds, key=lambda x: x[1])
plt.plot(worst_pred)
plt.plot(goal)
plt.show()

worst_pred, worst_mse, worst_combination, worst_c, worst_i = min(preds, key=lambda x: x[1])
print('Best:', min(preds, key=lambda x: x[1])[1:])
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

fig, axs = plt.subplots(2, 2, dpi=200, figsize=(10,8))
axs = axs.flatten()
axs[0].plot(indata[:,0])
axs[1].plot(indata[:,1])
axs[2].plot(indata[:,2])
axs[3].plot(indata[:,3])

print(np.min(indata[:,0]),np.max(indata[:,0]))
print(np.min(indata[:,1]),np.max(indata[:,1]))
print(np.min(indata[:,2]),np.max(indata[:,2]))
print(np.min(indata[:,3]),np.max(indata[:,3]))

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
    #scalerx = MinMaxScaler()
    #inxn = scalerx.fit_transform(acq)
    #use scaler fitted on all data
    inxn = scalerx.transform(acq)
    
    #################
    #PCA
    
    # pca = PCA(n_components=4)
    # inxn = pca.fit_transform(inxn)
    
    # #################
    
    # Train a linear regression model
    model = LinearRegression()
    # model = DecisionTreeRegressor()
    model.fit(inxn, goal)
    
    predx=[]
    for i in acq:
        met = scalerx.transform(np.array(i).reshape(1,-1))
        
        #PCA
        # met = pca.transform(met)
        
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

#For unnormalized weights
######3
# Get the weights
weights = [model.coef_ for model in linmods]

# Calculate the average weights among all models
weights_x = np.mean(weights, axis=0)

# Calculate each model's weight contribution as a percentage of the total (using the sum of absolute weights)
percentages = [
    np.round(100 * w / np.sum(np.abs(weights_x)), 2) if np.sum(np.abs(weights_x)) != 0 else np.zeros_like(w) 
    for w in weights
]

# Get the average percentage contribution among all models
weights_x_percentage = np.mean(percentages, axis=0)

print("Avg Learned weights as percentages:", weights_x_percentage)
########

# Convert the weights to string format to avoid scientific notation
# weights_x_str = [f"{w:.2f}" for w in weights_x_percentage]

# print("Learned weights (x) as percentages:\n", weights_x_str)

#PLOTTING
for p in preds:        
    plt.plot(p)
plt.plot(goal)
plt.show()

#print(round(mean_squared_error(goal, predx),6))

    


# Convert absolute weights for chart
abs_weights_x_percentage = [w for w in weights_x_percentage]

# Sort by absolute values in decreasing order
sorted_indices = np.argsort(abs_weights_x_percentage)[::-1]
sorted_weights = [abs_weights_x_percentage[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]

# Create a horizontal bar chart
plt.figure(figsize=(10, 7))
y_pos = np.arange(len(sorted_labels))
plt.barh(y_pos, sorted_weights, color='skyblue')
plt.yticks(y_pos, sorted_labels)
plt.xlabel('Importance %')
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

modelslist = []

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
            
            #store models
            modelslist.append([model.coef_,model.intercept_])

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
# labels = [
#     'ysum var w/o hilbert',
#     'yhist var w/ hilbert sum',
#     'ysum var w/ hilbert lines', 
#     'mean abs conf deriv',
#     'var conf deriv',
#     'mean abs intensity', 
#     'var laplacian',
#     'var laplacian of conf',
#     'confxsum angle pred', 
#     'peaks angle pred',
#     'xsum var',
#     'confxsum var'
# ]

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

#SAVING
# '''
import pickle
with open('qt_3.2.pkl', 'wb') as f:
    pickle.dump(scalerx, f)
print(weights_x)
avgbias = [model.intercept_ for model in linmods]
print(np.mean(avgbias))
# '''
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

###############################################################################
# scalers test
###############################################################################
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler, QuantileTransformer, RobustScaler,
    StandardScaler, MaxAbsScaler, PowerTransformer, Normalizer
)

# Original training data within range [0, 10]
train_data = np.array([[0], [2], [4], [6], [8], [10]])
# New data with an out-of-range value [15]
test_data = np.array([[-5], [5], [15]])

# Scalers to test
scalers = {
    "MinMaxScaler": MinMaxScaler(),
    "QuantileTransformer": QuantileTransformer(),
    "RobustScaler": RobustScaler(),
    "StandardScaler": StandardScaler(),
    "MaxAbsScaler": MaxAbsScaler(),
    "PowerTransformer": PowerTransformer(),
    "Normalizer": Normalizer()
}

# Apply each scaler and show how it transforms out-of-range data
for name, scaler in scalers.items():
    scaler.fit(train_data)  # Fit to the initial data range
    transformed = scaler.transform(test_data)
    print(f"{name} transformed data:\n{transformed}\n")
    print(f'{scaler.transform(train_data)}')



###############################################################################
# Load test images and compare feature scores
###############################################################################
import json

print(current_dir) # documents
datap = current_dir / 'data' / 'dataChest' 
# Get all folder names in the directory
folder_names = [f.name for f in datap.iterdir() if f.is_dir()]

filtRF = []
for folder in folder_names:
    runpath = datap / folder / 'runData'
    #load RF data
    imgs=[]
    matching_files = list(runpath.rglob('UsRaw*.npy'))
    for run in matching_files:
        imgs.append(np.load(runpath / run))
        
    with open(runpath / 'variables.json', 'r') as file:
        data = json.load(file)
        
    # scores = []    
    # for i in range(len(data['scores'])-1):
    #     scores.append(data['scores'][i][1])
        
    # filtRF.append([imgs,scores])
    filtRF.append(imgs)
    
'''
Calculate scores
'''
import pickle

#og
# w = [ 1.50142378, -0.50806179,  0.18003294, -0.0502186 ]
# b = -0.15638810337191103

#mean sum
# w = [ 0.86027377, -0.14450117,  0.16682434,  0.08422899]
# b = -0.11681850124481369

#absmean sum
# w = [0.86027377, 0.14450117, 0.16682434, 0.08422899]
# b = -0.2613196696236939

#var sum --> same as abs var sum
# w = [ 1.17015782, -0.30825557,  0.28480443,  0.0843479 ]
# b= -0.2489328223188194

#0
# w = [0.65500939, 0.32719537, 0.43761955, 0.15642175]
# b = -0.42152856075265666

#1
# w = np.array([0.55678448, 0.2334604,  0.6368764,  0.02270388])
# b = -0.3510284141084585

#2
# w = np.array([ 0.33068497, -0.59401001,  0.39815741,  0.06128418])
# b = 0.26853618721981365
# w = np.array([ 0.32262789, -0.60305398,  0.39742833,  0.0580219 ])
# b = 0.2790823942730414

#3
# w = [ 0.19353861, -0.579008  ,  0.33328544, -0.08233752]
# b = 0.4338551949004921
# w = [ 0.18352962, -0.58834553,  0.33555851, -0.08508549]
# b = 0.4437659116780795

#3.2
# w = [ 0.2068774 , -0.73112137, -0.18226259, -0.07498166]
# b = 0.7573385756716293
w = [ 0.1938455 , -0.73462668, -0.19355259, -0.07888275]
b = 0.7732027267451909


with open(current_dir / 'data' / 'scalers' / 'qt_3.2.pkl', 'rb') as f:
    scaler = pickle.load(f)

newScores=[]

rawfeat=[]
scafeat=[]
weifeat=[]

crops=[[100,-250],
       [100,-250],
       [100,-150],
       [125,-200],
       [75,-275],
       [75,-250],
       [75,-250],
       [75,-250],
       [175,-150],
       [175,-150]]

for i,group in enumerate(filtRF):
    imgs,scores = group 
    temp=[]
    
    for img, score in zip(imgs, scores):
        metrics=[]
        
        strt,end=crops[i]#175,-175
        
        ##
        sfactor = 6292/512
        strt = round(strt*sfactor)
        end = round((512-end)*sfactor)
        imgog = img.copy()
        img = resize(img, (6292,129), anti_aliasing=True)
        ##
        
        '''
        # yhist variance with Hilbert of each line
        #############################################
        himg = byb.envelope(img)
        himgCopy = himg.copy()
        for k in range(himg.shape[1]):
            line = himg[:,k]
            
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            
            himgCopy[:,k] = line
        himgCopyCrop = himgCopy[strt:end, :]
        hbyh = np.mean(himgCopyCrop,axis=1)
        metrics.append(np.var(hbyh))
        #############################################
        # variance/mean of confidence deriv
        #############################################
        cmap = confidenceMap(img[10:,:], rsize=False)[strt-10:end, :]
        cyhist,cxhist = byb.getHist(cmap)
        # metrics.append(np.var(abs(np.diff(cyhist))))
        metrics.append(np.var(np.diff(cyhist)))
        # metrics.append(np.mean(abs(np.diff(cyhist))))
        # metrics.append(np.mean((np.diff(cyhist))))
        #############################################
        # lum mean abs img
        #############################################
        metrics.append(np.mean(np.abs(himg[strt:end,:])))
        
        #############################################
        # varlap 20log abs hilb
        #############################################
        metrics.append(laplace(20*np.log10(abs(himg[strt:end,:])+1)).var())
        #############################################
        
        '''
        # Var of mean of loghilb img
        #############################################
        himg = byb.envelope(img)
        himgCopyCrop = 20*np.log10(himg[strt:end, :]+1)
        hbyh = np.mean(himgCopyCrop,axis=1)
        metrics.append(np.var(hbyh))
        #############################################
        # mean of confidence deriv
        #############################################
        cmap = confidenceMap(imgog[10:,:],rsize=False)
        cmap = resize(cmap, (img.shape[0]-10, img.shape[1]), anti_aliasing=True)[strt:end,:]
        # cyhist,cxhist = byb.getHist(cmap)
        # cyhist = np.sum(cmap,axis=1)
        cyhist = np.mean(cmap,axis=1)
        # metrics.append(np.mean(abs(np.diff(cyhist))))
        cyhist = resize(cyhist, [800], anti_aliasing=True)
        deriv = abs(savgol_filter(cyhist, window_length=len(cyhist)//16, polyorder=2, deriv=1))#[strt-10:end]
        # metrics.append(np.var(np.diff(cyhist)))
        
        t=deriv
        probs = t/np.sum(t)
        x = np.arange(len(probs))

        avg = np.sum(x*probs)
        var = np.sum((x - avg)**2 * probs)
        
        # metrics.append(np.mean(deriv))
        metrics.append(var)
        #############################################
        # lum mean hilb img
        #############################################
        # p25, p75 = np.percentile(himg[strt:end,:], [25, 75])
        # lum = np.mean((himg[strt:end,:]-p25)/(p75-p25))
        # metrics.append(np.mean(himg[strt:end,:]))
        logim = himgCopyCrop#20*np.log10(imgc+1)
        lum = np.mean((logim-np.max(logim)))#[2000:2800])
        metrics.append(lum)
        #############################################
        # varlap 20log abs hilb
        #############################################
        metrics.append(laplace(20*np.log10(abs(himg[strt:end,:])+1)).var())
        #############################################
        # '''

        rawfeat.append(metrics)

        metrics = scaler.transform([metrics])[0]

        scafeat.append(metrics)

        metrics = np.array(metrics)
        linmod = metrics*w
        weifeat.append(linmod)
        
        temp.append(-1*(np.sum(linmod) + b))
    
    newScores.append(temp)
    
for i,run in enumerate(newScores):
    print(f'Run{i}')
    for s in run:
        print(f'{s:<10.4f}')
    print('best: ',np.min(run), np.argmin(run))
    print('-'*10)
    
for i in range(10):
    # Extract and negate the arrays for newScores
    plt.plot(np.array(newScores[i]), label=f'newScores {i+1}', color='orange', linestyle='--')

    # Extract and plot the arrays from filtRF
    plt.plot(np.array(filtRF[i][1]), label=f'filtRF {i+1}', color='blue')

    # Add labels, title, and legend
    plt.title(f'Comparison between filtRF and newScores for Array {i+1}')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()

    # Display the plot
    plt.show()
    
rawfeat,scafeat,weifeat = np.array(rawfeat),np.array(scafeat),np.array(weifeat)

# Plot rawfeat
plt.figure(figsize=(12, 6))
plt.plot(rawfeat[:, 0], label='rawfeat: linevar')
plt.plot(rawfeat[:, 1], label='rawfeat: cmap')
plt.plot(rawfeat[:, 2], label='rawfeat: int')
plt.plot(rawfeat[:, 3], label='rawfeat: lap')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Rawfeat')
plt.legend()
plt.show()

# Plot scafeat
plt.figure(figsize=(12, 6))
plt.plot(scafeat[:, 0], label='scafeat: linevar')
plt.plot(scafeat[:, 1], label='scafeat: cmap')
plt.plot(scafeat[:, 2], label='scafeat: int')
plt.plot(scafeat[:, 3], label='scafeat: lap')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Scafeat')
plt.legend()
plt.show()

# Plot weifeat
plt.figure(figsize=(12, 6))
plt.plot(weifeat[:, 0], label='weifeat: linevar')
plt.plot(weifeat[:, 1], label='weifeat: cmap')
plt.plot(weifeat[:, 2], label='weifeat: int')
plt.plot(weifeat[:, 3], label='weifeat: lap')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Weifeat')
plt.legend()
plt.show()

fig, axs = plt.subplots(2, 2, dpi=200, figsize=(10,8))
axs = axs.flatten()
axs[0].plot(rawfeat[:,0])
axs[1].plot(rawfeat[:,1])
axs[2].plot(rawfeat[:,2])
axs[3].plot(rawfeat[:,3])

fig, axs = plt.subplots(2, 2, dpi=200, figsize=(10,8))
axs = axs.flatten()
axs[0].plot(scafeat[:,0])
axs[1].plot(scafeat[:,1])
axs[2].plot(scafeat[:,2])
axs[3].plot(scafeat[:,3])

print(np.min(rawfeat[:,0]),np.max(rawfeat[:,0]))
print(np.min(rawfeat[:,1]),np.max(rawfeat[:,1]))
print(np.min(rawfeat[:,2]),np.max(rawfeat[:,2]))
print(np.min(rawfeat[:,3]),np.max(rawfeat[:,3]))
    


#%%
'''
 Code used to find the correct crop zone of the chest phantom data
'''

import numpy as np
import matplotlib.pyplot as plt

# Define unique horizontal line positions for each figure
line_positions_list = [
    [20, 180],
    [70, 180],
    [50, 150],
    [30, 180],
    [40, 150],
    [30, 130],
    [25, 120],
    [50, 200],
    [40, 160],
    [60, 120],
]

# Loop through each item in filtRF and create separate figures
for fig_idx, images in enumerate(filtRF):
    # Get the unique line positions for the current figure
    line_positions = line_positions_list[fig_idx]

    # Create a new figure for each set of images
    fig, axs = plt.subplots(1, len(images), dpi=200)

    # If there's only one image, axs won't be an array
    if len(images) == 1:
        axs = [axs]

    # Loop through each image in the set and create a subplot
    for img_idx, (ax, img) in enumerate(zip(axs, images)):
        
        # sfactor = 6292/512
        # strt = round(strt*sfactor)
        # end = round((512-end)*sfactor)
        # imgog = img.copy()
        # img = resize(img, (6292,129), anti_aliasing=True, order=5)
        
        #calculate cmap
        cmap = confidenceMap(img[round(10):], rsize=False)
        cmap = resize(cmap, img.shape, anti_aliasing=True)
        confC = np.sum(cmap < 0.85)
        confRT = confC / cmap.size
        
        # img = resize(img, imgog.shape, anti_aliasing=True, order=5)
        if confRT <= 0.9:
            # Display the image
            ax.imshow(20 * np.log10(byb.envelope(img) + 1))
            
        else:
            ax.imshow(20 * np.log10(byb.envelope(img) + 1), cmap='grey')
        
        ax.set_xlabel(f'{confRT:.2f}', fontsize=5)

        # Add horizontal lines
        for y in line_positions:
            ax.axhline(y=y, color='red', linestyle='--', linewidth=1)

        # Remove axis labels for clarity
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Adjust layout
    # plt.subplots_adjust(wspace=0.05, hspace=0.02)

    # Show the current figure
    plt.show()




