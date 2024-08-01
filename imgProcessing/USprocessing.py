# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:01:38 2024

@author: Mateo-drr
"""

import byble as byb
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import numpy as np

#PARAMS
fc=6e6
fs=50e6
idx=0
date ='30Jul'
ptype = 'rl0' #scan path to load
imgtype = 'Rf' #image type to be loaded
fkey = 'rf' #dictionary key of the mat data
depthCut=6292
highcut=fc+0.5e6
lowcut=fc-2e6
frameLines=129
save=True
frameCrop=False #manually crop frames

#Get list of files in directory
current_dir = Path(__file__).resolve().parent.parent 
datapath = current_dir / 'data' / 'acquired' / date / 'pydata' / ptype
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

# _=input()
for idx in range(0,len(fileNames)):
    print('Working on img', idx)
    ###############################################################################
    #Load matlab data
    ###############################################################################
    #load a selected file
    file = fileNames[idx]
    mat_data = loadmat(datapath / file)
    if frameCrop:
        img = mat_data[fkey][:depthCut,:]
    else:
        #img = mat_data[fkey][:depthCut,:,0] #take only first frame to simulate RL trans.
        img = np.mean(mat_data[fkey][:depthCut,:,:], axis=2)
    
    ###############################################################################
    #Processing
    ###############################################################################
    #filter the data
    imgfilt = byb.bandFilt(img, highcut=highcut, lowcut=lowcut, fs=fs, N=len(img[:,0]), order=6)
    #plot fouriers
    byb.plotfft(img[:,0], fs)
    byb.plotfft(imgfilt[:,0], fs)
    
    #normalize
    imgfiltnorm = 20*np.log10(np.abs(imgfilt)+1) # added small value to avoid log 0
    
    # Plot the data
    byb.plotUS(20*np.log10(np.abs(imgfilt)+1))
    plt.show()
    byb.plotUS(byb.envelope(imgfiltnorm))
    plt.show()
    
    ###############################################################################
    #Frame crop
    ###############################################################################
    if frameCrop:
        #Find index of the frames
        fidx = byb.findFrame(imgfiltnorm,frameLines,getframes=False)
        #Crop the frames without normalization
        frames = byb.cropFrames(imgfilt, fidx)
        #Merge frames to remove noise
        image = np.mean(frames,axis=0)
        #Plot
        byb.plotUS(20*np.log10(np.abs(image)+1))
        plt.show()
    
    ###############################################################################
    #Save file
    ###############################################################################
    if save:
        if frameCrop:
            np.save(datapath.parent.parent / 'processed' / ptype / f'{imgtype.lower()}_{ptype}_{idx:03d}_{fidx[0]+1}',image)
        else:
            np.save(datapath.parent.parent / 'processed' / ptype / f'{imgtype.lower()}_{idx:03d}',imgfilt)
        


