#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:23:15 2024

@author: mateo-drr
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
from imgProcessing.filtering import bandFilt,envelope,plotfft
import numpy as np
import torch
import random
from torchvision import transforms
from scipy.ndimage import rotate
import math

#first_freq_Filt_Norm.mat
fc=3e6 #hz
fs=50e6
postProc = False

file = 'first_freq_Filt_Norm.mat'
#file = 'first_freq_NFilt.mat'
path = '/home/mateo-drr/Documents/Trento/ALU---Autonomous-Lung-Ultrasound/data/'

mat_data = loadmat(path + file)
img = mat_data[file[:-4]][0][0]

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
    
def getHist(sec):
    #normalize
    min_val = np.min(sec)
    max_val = np.max(sec)
    sec = (sec - min_val) / (max_val - min_val)
    #collapse to 1d
    histY = np.sum(sec, axis=1) #same results with mean
    histX = np.sum(sec, axis=0) #same results with mean
    return histY, histX

ydim,xdim = getHist(img)
x_values = np.arange(len(ydim))

# Plot the data
plotUS(img)
plt.plot(ydim,x_values)
plt.plot(xdim)
plt.show()


def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr


# Resize the image into a square
timg = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
resize = transforms.Resize((img.shape[0],img.shape[0]))
rimg = resize(timg)
# Rotate it
ang = random.randint(-10, 10)
rtimg = transforms.functional.rotate(rimg, ang, expand=True)[0][0]
# Calculate the rotated crop
xr,yr = rotatedRectWithMaxArea(img.shape[0], img.shape[0], np.deg2rad(ang))
x0,y0 = int(np.ceil((rtimg.shape[1]-xr)/2)), int(np.ceil((rtimg.shape[0]-yr)/2))
#plot
plotUS(rtimg)
plt.axhline(y0)
plt.axvline(x0)
plt.axhline(rtimg.shape[0] - y0)
plt.axvline(rtimg.shape[1] - x0)
plt.show()
#crop image
rotimg = rtimg[y0:-y0,x0:-x0]
plotUS(rotimg)


# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image_path = '/home/mateo-drr/Desktop/lena.png'  # Replace with your image path
# img = Image.open(image_path)

# # Convert the image to black and white
# bw_img = img.convert('L')

# # Convert the black and white image to a NumPy array
# bw_array = np.array(bw_img)

# # Display the black and white image to verify
# plt.imshow(bw_array, cmap='gray')
# plt.axis('off')
# plt.show()

# # Print the shape of the array
# print(bw_array.shape)









