#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:23:15 2024

@author: mateo-drr
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import byble as byb
import numpy as np
import random
from pathlib import Path

#first_freq_Filt_Norm.mat
fc=3e6 #hz
fs=50e6
postProc = False

file0 = 'first_freq_Filt_Norm.mat'
# Get the current file's directory
current_dir = Path(__file__).resolve().parent.parent / 'data'
path = current_dir.as_posix()

datapath = Path(path+'/pre-made/raw/')
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

idx=7
file = fileNames[idx]

mat_data = loadmat(path + '/pre-made/raw/' + file)

try:
    img = mat_data[file[:-4]][0][0]
except:
    print('data with wrong key')
    img = mat_data[file0[:-4]][0][0]

byb.plotUS(img)
plt.show()

if postProc:
    #filter the data
    a = byb.bandFilt(img, highcut=fc+0.5e6, lowcut=fc-0.5e6, fs=fs, N=len(img[:,0]), order=6)
    
    #plot fouriers
    byb.plotfft(img[:,0], fs)
    byb.plotfft(a[:,0], fs)
    
    #hilbert 
    ah = byb.envelope(np.array(a))
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
rimg = byb.rsize(img)
#copy the image
imgc = rimg[0,0].numpy()
# Normalize to 255
nimg = byb.normalize(imgc)
# Find the peak of each line
peaks = byb.findPeaks(nimg)
# Regression on the peaks 
line, angle, x, y = byb.regFit(peaks)


# Optionally, plot the data and the fitted line
plt.scatter(x, y, label='Data points')
plt.plot(x, line, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
#plot the line on the us image
byb.plotUS(imgc)
plt.plot(line)
plt.show()

#Rotate the image
imgcr = byb.rotate(imgc, angle)
# Crop the rotated image
rotimg,x0,y0 = byb.rotatClip(imgcr, imgc, angle, cropidx=True)
# plot area to crop
byb.plotUS(imgcr)
plt.axhline(y0)
plt.axvline(x0)
plt.axhline(imgcr.shape[0] - y0)
plt.axvline(imgcr.shape[1] - x0)
plt.show()

byb.plotUS(rotimg)
plt.show()

# copy it
flatimg = rotimg.numpy()
# save it
#np.save(path+'numpy/'+f'img{idx}.npy', byb.normalize(flatimg))

###############################################################################
# Sum axes
###############################################################################

#
ydim,xdim = byb.getHist(flatimg)
x_values = np.arange(len(ydim))

# Plot the data
byb.plotUS(flatimg)
plt.plot(ydim,x_values)
plt.plot(xdim)
plt.show()

###############################################################################
#Rotate image for training
##############################################################################
# Rotate it
ang = random.randint(-10, 10)
rtimg = byb.rotate(flatimg, ang)
# Crop it
finalimg,x0,y0 = byb.rotatClip(rtimg, flatimg, ang, cropidx=True)
# plot area to crop
byb.plotUS(rtimg)
plt.axhline(y0)
plt.axvline(x0)
plt.axhline(rtimg.shape[0] - y0)
plt.axvline(rtimg.shape[1] - x0)
plt.show()
#
byb.plotUS(finalimg)
plt.show()

#resize
finalimg = byb.rsize(finalimg.numpy(), x=flatimg.shape[1], y = flatimg.shape[0])[0][0]
byb.plotUS(finalimg)
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
byb.plotUS(finalimg)
plt.axhline(newy)
plt.axvline(newx)
plt.axhline(finalimg.shape[0] - endy)
plt.axvline(finalimg.shape[1] - endx)
plt.show()
# Crop image
crop = finalimg[newy:-endy,newx:-endx]
byb.plotUS(crop)
plt.show()

###############################################################################
# Sum axes
###############################################################################
# copy it
npcrop = crop.numpy()
# sum axes
ydim,xdim = byb.getHist(npcrop)
x_values = np.arange(len(ydim))
# Plot the data
byb.plotUS(npcrop)
plt.plot(ydim,x_values)
plt.plot(xdim)
plt.show()

