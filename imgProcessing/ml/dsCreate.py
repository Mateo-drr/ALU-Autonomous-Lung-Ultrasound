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


alldat = []

datapath = all_filenames[0][0]
fileNames = all_filenames[0][1]
for pos,x in enumerate(allmove):
    
    if pos%82:
        datapath = all_filenames[pos//82][0]
        fileNames = all_filenames[pos//82][1]
    
    img = byb.loadImg(fileNames, int(x[-1]), datapath)#[100:]
    cmap = confidenceMap(img,rsize=True)
    cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]

    alldat.append([img,cmap])
    
top, btm = [], []

tline, bline = 1000, 2000
maybe = None

for acq in alldat:
    while True:
        byb.plotUS(acq[0], True)
        plt.axhline(tline, color='r')
        plt.axhline(bline, color='b')
        plt.show()

        if maybe is None:
            maybe = input('See confidence map? (y or any other key): ')
        if maybe == 'y':
            byb.plotUS(acq[1], False)
            plt.axhline(tline, color='r')
            plt.axhline(bline, color='b')
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
        byb.plotUS(acq[0], True)
        plt.axhline(tline, color='r')
        plt.axhline(bline, color='b')
        plt.show()

        if maybe == 'y':
            byb.plotUS(acq[1], False)
            plt.axhline(tline, color='r')
            plt.axhline(bline, color='b')
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

top = np.array(top)
btm = np.array(btm)

np.save(current_dir / 'data' / 'acquired' / date , top)
np.save(current_dir / 'data' / 'acquired' / date , bt)
    
    
    
    
    
    
    
    
    
    