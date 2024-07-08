# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:01:38 2024

@author: Mateo-drr
"""

import filtering as filt
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import numpy as np

#PARAMS
fc=6e6
fs=50e6
idx=0
d ='/acquired/pydata/'
depthCut=6292
highcut=fc+1e6
lowcut=fc-1e6
frameLines=129
save=True

#Get list of files in directory
current_dir = Path(__file__).resolve().parent.parent / 'data'
path = current_dir.as_posix()
datapath = Path(path+d)
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

for idx in range(0,len(fileNames)):
    print('Working on img', idx)
    ###############################################################################
    #Load matlab data
    ###############################################################################
    #load a selected file
    file = fileNames[idx]
    mat_data = loadmat(path + d + file)
    img = mat_data['rf'][:6292,:]
    
    ###############################################################################
    #Processing
    ###############################################################################
    #filter the data
    imgfilt = filt.bandFilt(img, highcut=highcut, lowcut=lowcut, fs=fs, N=len(img[:,0]), order=6)
    #plot fouriers
    filt.plotfft(img[:,0], fs)
    filt.plotfft(imgfilt[:,0], fs)
    
    #hilbert 
    imghilb = filt.envelope(np.array(imgfilt))
    imgfilt = np.array(imgfilt)
    #normalize
    imgfiltnorm = 20*np.log10(np.abs(imgfilt)+1) # added small value to avoid log 0
    
    # Plot the data
    filt.plotUS(20*np.log10(np.abs(imgfilt)+1))
    plt.show()
    filt.plotUS(imgfiltnorm)
    plt.show()
    
    ###############################################################################
    #Frame crop
    ###############################################################################
    #Find index of the frames
    fidx = filt.findFrame(imgfiltnorm,frameLines,getframes=False)
    #Crop the frames without normalization
    frames = filt.cropFrames(imgfilt, fidx)
    #Merge frames to remove noise
    image = np.mean(frames,axis=0)
    #Plot
    filt.plotUS(20*np.log10(np.abs(image)+1))
    
    ###############################################################################
    #Save file
    ###############################################################################
    if save:
        np.save(datapath.parent.as_posix()+f'/processed/cf_{idx:03d}_{fidx[0]+1}',image)


