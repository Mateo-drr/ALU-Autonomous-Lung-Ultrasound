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

###############################################################################
#Load images with the meat
###############################################################################

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

###############################################################################
# Load images withou meat
###############################################################################    
# PARAMS
date = '01Aug0'
ptype2conf = {
    'cl': 'curvedlft_config.json',
    'cf': 'curvedfwd_config.json',
    'rf': 'rotationfwd_config.json',
    'rl': 'rotationlft_config.json'
}

# Get the base directory
#current_dir = Path(__file__).resolve().parent.parent.parent.parent

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
allmove0 = np.concatenate(all_positions, axis=0)


alldat0 = []

assert len(alldat) % 82 == 0, "alldat does not have a length that is a multiple of 82"

datapath = all_filenames[0][0]
fileNames = all_filenames[0][1]
for pos,x in enumerate(allmove0):
    
    if pos%82==0:
        datapath = all_filenames[pos//82][0]
        fileNames = all_filenames[pos//82][1]
    
    img = byb.loadImg(fileNames, int(x[-1]), datapath)#[100:]
    #cmap = confidenceMap(img,rsize=True)
    #cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]

    alldat0.append(img)
    
acqdat0 = []
temp = []

for pos, acq in enumerate(alldat0):
    if pos % 82 == 0 and pos != 0:
        
        acqdat0.append(deepcopy(temp))
        temp = []
    temp.append(acq)

# Append the last batch of acquisitions
acqdat0.append(temp)

###############################################################################
#Loop to identify mask
###############################################################################
def findlineplot(temp,acq,ref,tline,bline):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 11), dpi=300)

    # [0,0] Plot the image with the top and bottom lines
    axs[0, 0].imshow(temp[:4000,:], aspect='auto', cmap='viridis')
    axs[0, 0].axhline(tline, color='r')
    axs[0, 0].axhline(bline, color='b')
    axs[0, 0].set_title(f'Image {i}')

    # [0,1] Plot the confidence map with the same lines
    axs[0, 1].imshow(acq[1][:4000,:], aspect='auto')
    axs[0, 1].axhline(tline, color='r')
    axs[0, 1].axhline(bline, color='b')
    axs[0, 1].set_title('Confidence Map')

    # Plot the result of byb.hilb(byb.getHist(acq[0])) on the confidence map
    hilb_result = byb.normalize(byb.hilb(byb.getHist(acq[0])[0]), scale=60)[:4000]
    axs[0, 1].plot(hilb_result, np.arange(len(hilb_result)), color='m')

    pad=200
    # [1,0] Plot the chopped temp image
    chopped_temp = temp[tline-pad:bline+pad, :]
    axs[1, 0].imshow(chopped_temp, cmap='viridis', aspect=0.05)
    axs[1, 0].axhline(pad, color='r')
    axs[1, 0].axhline(len(chopped_temp)-1-pad, color='b')
    # axs[1, 0].set_title('Chopped Image')

    # [1,1] Plot the chopped ref image
    chopped_ref = ref[tline-pad:bline+pad, :]
    axs[1, 1].imshow(chopped_ref, cmap='viridis', aspect=0.05)
    axs[1, 1].axhline(pad, color='r')
    axs[1, 1].axhline(len(chopped_ref)-1-pad, color='b')
    # axs[1, 1].set_title('Chopped Ref')

    plt.tight_layout(pad=1.0, h_pad=-6, w_pad=0.5)
    plt.show()


top, btm = [], []
tline, bline = 2100, 2600

# Change the acqdat to each path
pth = 3
for i, acq in enumerate(acqdat[pth]):
    temp = 20 * np.log10(abs(acq[0]) + 1)
    ref = 20 * np.log10(abs(byb.envelope(acqdat0[pth][i])) + 1)
    while True:
        findlineplot(temp, acq, ref, tline, bline)

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
        findlineplot(temp, acq, ref, tline, bline)

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
    print('top-btm', tline, bline)
    
    
# Convert lists to numpy arrays
top = np.array(top)
btm = np.array(btm)

date = '01Aug6'
# Save the arrays as files
np.save(current_dir / 'data' / 'acquired' / date / f'top_lines_{pth}.npy', top)
np.save(current_dir / 'data' / 'acquired' / date / f'btm_lines_{pth}.npy', btm)    
    
    
    
    
    
    
    
    
    