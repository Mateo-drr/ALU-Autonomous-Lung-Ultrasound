#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 01:31:24 2024

@author: mateo-drr
"""

from pathlib import Path
import sys
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(current_dir.as_posix())

import byble as byb
from confidenceMap import confidenceMap
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from copy import deepcopy

# PARAMS
date = '01Aug6'
ptype2conf = {
    'cl': 'curvedlft_config.json',
    'cf': 'curvedfwd_config.json',
    'rf': 'rotationfwd_config.json',
    'rl': 'rotationlft_config.json'
}

# Get the base directory
current_dir = Path(__file__).resolve().parent.parent.parent.parent

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


alldat = []

assert len(alldat) % 82 == 0, "alldat does not have a length that is a multiple of 82"

datapath = all_filenames[0][0]
fileNames = all_filenames[0][1]
for pos,x in enumerate(allmove):
    
    if pos%82==0:
        datapath = all_filenames[pos//82][0]
        fileNames = all_filenames[pos//82][1]
    
    img = byb.loadImg(fileNames, int(x[-1]), datapath)#[100:]
    cmap = confidenceMap(img,rsize=True)
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]

    alldat.append([img,cmap])
    
acqdat = []
temp = []

for pos, acq in enumerate(alldat):
    if pos % 82 == 0 and pos != 0:
        
        acqdat.append(deepcopy(temp))
        temp = []
    temp.append(acq)

# Append the last batch of acquisitions
acqdat.append(temp)

    
    
top, btm = [], []

tline, bline = 2100, 2600


#Change the acqdat to each path
pth = 0
for i,acq in enumerate(acqdat[pth]):
    temp = 20*np.log10(abs(acq[0])+1)
    while True:
        # Create a subplot for the image and the confidence map
        fig, axs = plt.subplots(1, 2, figsize=(12, 6),dpi=300)

        # Plot the image with the top and bottom lines
        axs[0].imshow(temp,aspect='auto',cmap='viridis')
        axs[0].axhline(tline, color='r')
        axs[0].axhline(bline, color='b')
        axs[0].set_title(f'Image {i}')

        # Plot the confidence map with the same lines
        axs[1].imshow(acq[1],aspect='auto')
        axs[1].axhline(tline, color='r')
        axs[1].axhline(bline, color='b')
        axs[1].set_title('Confidence Map')

        # Plot the result of byb.hilb(byb.getHist(img)) on the confidence map
        hilb_result = byb.hilb(byb.getHist(acq[0])[0])
        axs[1].plot(hilb_result,np.arange(len(hilb_result)), color='green')

        plt.show()

        cmd = input('Press Enter to exit, or enter +num or -num to adjust the top line: ')
        if cmd == "":
            top.append(tline)
            break
        else:
            try:
                num = int(cmd)
                tline += num
            except ValueError:
                print("Invalid input, please enter a number.")

    while True:
        # Create a subplot for the image and the confidence map
        fig, axs = plt.subplots(1, 2, figsize=(12, 6),dpi=300)

        # Plot the image with the top and bottom lines
        axs[0].imshow(temp,aspect='auto')
        axs[0].axhline(tline, color='r')
        axs[0].axhline(bline, color='b')
        axs[0].set_title(f'Image {i}')

        # Plot the confidence map with the same lines
        axs[1].imshow(acq[1],aspect='auto')
        axs[1].axhline(tline, color='r')
        axs[1].axhline(bline, color='b')
        axs[1].set_title('Confidence Map')

        # Plot the result of byb.hilb(byb.getHist(img)) on the confidence map
        hilb_result = byb.hilb(byb.getHist(acq[0])[0])
        axs[1].plot(hilb_result,np.arange(len(hilb_result)), color='green')

        plt.show()

        cmd = input('Press Enter to exit, or enter +num or -num to adjust the bottom line: ')
        if cmd == "":
            btm.append(bline)
            break
        else:
            try:
                num = int(cmd)
                bline += num
            except ValueError:
                print("Invalid input, please enter a number.")
    print('top-btm',tline,bline)

# Convert lists to numpy arrays
top = np.array(top)
btm = np.array(btm)

# Save the arrays as files
np.save(current_dir / 'data' / 'acquired' / date / f'top_lines_{pth}.npy', top)
np.save(current_dir / 'data' / 'acquired' / date / f'btm_lines_{pth}.npy', btm)    
    
    
    
    
    
    
    
    
    