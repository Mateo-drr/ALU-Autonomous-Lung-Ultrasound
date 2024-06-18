#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:23:15 2024

@author: mateo-drr
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
from imgProcessing.filtering import bandFilt,envelope,plotfft,getHist
import imgProcessing.filtering as filt
import numpy as np
import random
from pathlib import Path

#first_freq_Filt_Norm.mat
fc=3e6 #hz
fs=50e6
postProc = False

file0 = 'first_freq_Filt_Norm.mat'
# Get the current file's directory
current_dir = Path(__file__).resolve().parent / 'data'
path = current_dir.as_posix()


datapath = Path(path+'/raw/')
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

idx=7
file = fileNames[idx]

mat_data = loadmat(path + '/raw/' + file)

try:
    img = mat_data[file[:-4]][0][0]
except:
    print('data with wrong key')
    img = mat_data[file0[:-4]][0][0]

def plotUS(img):
    # Plot the data
    plt.figure(dpi=300)
    plt.imshow(img, aspect='auto', cmap='viridis')  # Adjust the colormap as needed
    plt.xlabel('Time')
    plt.ylabel('Index')
    plt.title('US')
    plt.colorbar(label='Intensity')

plotUS(img)
plt.show()

if postProc:
    #filter the data
    a = bandFilt(img, highcut=fc+0.5e6, lowcut=fc-0.5e6, fs=fs, N=len(img[:,0]), order=6)
    
    #plot fouriers
    plotfft(img[:,0], fs)
    plotfft(a[:,0], fs)
    
    #hilbert 
    ah = envelope(np.array(a))
    ah = np.array(a)
    #normalize
    ahn = 20*np.log10(np.abs(ah)+1) # added small value to avoid log 0
    
    # Plot the data
    plt.figure(dpi=300)
    plt.imshow(ahn, aspect='auto', cmap='viridis')  # Adjust the colormap as needed
    plt.xlabel('Time')
    plt.ylabel('Index')
    plt.title('US')
    plt.colorbar(label='Intensity')
    plt.show()


###############################################################################
#Make the original image flat
###############################################################################
# Resize the image into a square
rimg = filt.rsize(img)
#copy the image
imgc = rimg[0,0].numpy()
# Normalize to 255
nimg = filt.normalize(imgc)
# Find the peak of each line
peaks = filt.findPeaks(nimg)
# Regression on the peaks 
line, angle, x, y = filt.regFit(peaks)


# Optionally, plot the data and the fitted line
plt.scatter(x, y, label='Data points')
plt.plot(x, line, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
#plot the line on the us image
plotUS(imgc)
plt.plot(line)
plt.show()

#Rotate the image
imgcr = filt.rotate(imgc, angle)
# Crop the rotated image
rotimg,x0,y0 = filt.rotatClip(imgcr, imgc, angle, cropidx=True)
# plot area to crop
plotUS(imgcr)
plt.axhline(y0)
plt.axvline(x0)
plt.axhline(imgcr.shape[0] - y0)
plt.axvline(imgcr.shape[1] - x0)
plt.show()

plotUS(rotimg)
plt.show()

# copy it
flatimg = rotimg.numpy()
# save it
#np.save(path+'numpy/'+f'img{idx}.npy', filt.normalize(flatimg))

###############################################################################
# Sum axes
###############################################################################

#
ydim,xdim = getHist(flatimg)
x_values = np.arange(len(ydim))

# Plot the data
plotUS(flatimg)
plt.plot(ydim,x_values)
plt.plot(xdim)
plt.show()

###############################################################################
#Rotate image for training
##############################################################################
# Rotate it
ang = random.randint(-10, 10)
rtimg = filt.rotate(flatimg, ang)
# Crop it
finalimg,x0,y0 = filt.rotatClip(rtimg, flatimg, ang, cropidx=True)
# plot area to crop
plotUS(rtimg)
plt.axhline(y0)
plt.axvline(x0)
plt.axhline(rtimg.shape[0] - y0)
plt.axvline(rtimg.shape[1] - x0)
plt.show()
#
plotUS(finalimg)
plt.show()

#resize
finalimg = filt.rsize(finalimg.numpy(), x=flatimg.shape[1], y = flatimg.shape[0])[0][0]
plotUS(finalimg)
plt.show()

###############################################################################
#Translate image
###############################################################################
# Maximum area crop in %
area = 0.75
maxx = int((finalimg.shape[1] - finalimg.shape[1]*area))
maxy = int((finalimg.shape[0] - finalimg.shape[0]*area))
# Get a random translation for x and y
newx = random.randint(0, maxx)
newy = random.randint(0, maxy)
endx = maxx - newx
endy = maxy - newy
# Plot area to crop
plotUS(finalimg)
plt.axhline(newy)
plt.axvline(newx)
plt.axhline(finalimg.shape[0] - endy)
plt.axvline(finalimg.shape[1] - endx)
plt.show()
# Crop image
crop = finalimg[newy:-endy,newx:-endx]
plotUS(crop)
plt.show()

###############################################################################
# Sum axes
###############################################################################
# copy it
npcrop = crop.numpy()
# sum axes
ydim,xdim = getHist(npcrop)
x_values = np.arange(len(ydim))
# Plot the data
plotUS(npcrop)
plt.plot(ydim,x_values)
plt.plot(xdim)
plt.show()


###############################################################################
#Gym
###############################################################################

from gym.cstmGym import LungUS
import matplotlib.pyplot as plt
import matplotlib

env = LungUS(path + '/numpy/', res=20)
env.reset()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

done=False

action = 0
while not done:
    new_state, reward, done, info = env.step(action)
    env.render(mode='human')
    print(str(np.round(reward,4)), str(env.action_map), info)
    action=int(input(': '))
    print('moving...')

env.close()








