#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:55:45 2024

@author: mateo-drr
"""

from torch.utils.data import Dataset

import sys
from pathlib import Path
from torchvision import transforms
import torch

class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data
        #TODO check a good resize 
        self.rsize = transforms.Resize((6292,128),antialias=True)

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)
    
    def augment(self, inimg, lbl):
        
        #alter pixel values
        inimg,lbl = jitter(inimg, lbl)
        inimg,lbl = noise(inimg, lbl)
        
        #shift pixel positions
        inimg,lbl = rolling(inimg, lbl)
        inimg,lbl = flipping(inimg, lbl)
        inimg,lbl = rdmLocalRotation(inimg, lbl)
        
        #mask
        inimg,lbl = masking(inimg, lbl)
        
        return inimg,lbl

    def __getitem__(self, idx):    
    #TAKE ONE ITEM FROM THE DATASET
        img = torch.tensor(self.data[idx]).unsqueeze(0)
        
        #TODO wtf is this?
        img = self.rsize(img)[:,4:,:]
        
        #Normalize the data to 0 and 1
        min_val = torch.min(img)
        max_val = torch.max(img)
        img = (img - min_val) / (max_val - min_val)

        return img.to(torch.float32)
    
###############################################################################

"""
Created on Sun Dec 31 13:25:41 2023

@author: Mateo-drr

input tensors have to be shape [c,x,y]
"""

import torch
import random
from PIL import ImageDraw
import PIL.Image as Image
import numpy as np
import torchvision.transforms.functional as tf

random.seed(8)
np.random.seed(7)
torch.manual_seed(6)

def rolling(inimg, lbl):
    if random.random() < 0.33:
        width = inimg.size(1)
        shift_amount = random.randint(1, width - 1)
        inimg = torch.roll(inimg, shifts=shift_amount, dims=1)
        lbl = torch.roll(lbl, shifts=shift_amount, dims=1)
        
    if random.random() < 0.33:
        width = inimg.size(2)
        shift_amount = random.randint(1, width - 1)
        inimg = torch.roll(inimg, shifts=shift_amount, dims=2)
        lbl = torch.roll(lbl, shifts=shift_amount, dims=2)
    
    return inimg, lbl

def flipping(inimg, lbl, flipChan=False):
    if random.random() < 0.33:
        inimg = torch.flip(inimg, dims=[1])
        lbl = torch.flip(lbl, dims=[1])
        
    if random.random() < 0.33:
        inimg = torch.flip(inimg, dims=[2])
        lbl = torch.flip(lbl, dims=[2])
        
    if flipChan:
        if random.random() < 0.5:
            inimg = torch.flip(inimg, dims=[0])
            #add lbl if necessary
    
    return inimg, lbl

def jitter(inimg, lbl):
    if random.random() < 0.5:
        r = 0.8 + random.random() * 0.4
        inimg = adjust_contrast(adjust_intensity(inimg, r), r)
        # lbl = adjust_contrast(adjust_intensity(lbl, r), r)
    
    return inimg, lbl
        
def adjust_intensity(image, factor):
    return torch.clamp(image * factor, 0, 1)

def adjust_contrast(image, factor):
    return torch.clamp((image - 0.5) * factor + 0.5, 0, 1)

def masking(inimg, lbl, mask_prob=0.5, mask_size=64):
    if random.random() < mask_prob:
        mask_size = (mask_size, mask_size)
        height, width = inimg.shape[1:]
        h_start = random.randint(0, height - mask_size[0])
        w_start = random.randint(0, width - mask_size[1])

        mask = torch.ones_like(inimg)
        mask[:, h_start:h_start + mask_size[0], w_start:w_start + mask_size[1]] = 0

        minimg = inimg * mask
        #mlbl = lbl * mask[0]
    else:
        minimg = inimg
    
    mlbl = lbl

    return minimg, mlbl

def rdmLocalRotation(inimg, lbl, radius=64):
    if random.random() < 0.5:
        diam = radius * 2
        x, y = inimg.shape[1:]

        xcenter = random.randint(0, x - diam)
        ycenter = random.randint(0, y - diam)
        ang = random.randint(20, 340)

        lum_img = Image.new('L', [diam, diam], 0)
        draw = ImageDraw.Draw(lum_img)
        draw.pieslice([(0, 0), (diam-1, diam-1)], 0, 360, fill=255)
        circmaks = torch.tensor(np.array(lum_img) / 255)
        invcircmask = (tf.rotate(circmaks.unsqueeze(0), ang) - 1) * -1

        circCrop = inimg[:, xcenter:xcenter + diam, ycenter:ycenter + diam] * circmaks
        circCrop = tf.rotate(circCrop, ang)
        invCircCrop = inimg[:, xcenter:xcenter + diam, ycenter:ycenter + diam] * invcircmask
        inimg[:, xcenter:xcenter + diam, ycenter:ycenter + diam] = circCrop + invCircCrop

        circCrop_lbl = lbl[:,xcenter:xcenter + diam, ycenter:ycenter + diam] * circmaks
        circCrop_lbl = tf.rotate(circCrop_lbl, ang)
        invCircCrop_lbl = lbl[:,xcenter:xcenter + diam, ycenter:ycenter + diam] * invcircmask
        lbl[:,xcenter:xcenter + diam, ycenter:ycenter + diam] = circCrop_lbl + invCircCrop_lbl.squeeze()
        
        #Find the first and last zero in the vertical axis to expand the mask if a rotation took place
        zero_mask = (lbl == 0).any(dim=2).squeeze()  # Check for any 0s in each row

        if zero_mask.any():
            first_zero_row = zero_mask.nonzero(as_tuple=True)[0][0]  # First row with 0
            last_zero_row = zero_mask.nonzero(as_tuple=True)[0][-1]  # Last row with 0

            # Fill the space between first and last zero rows with 0s
            lbl[:, first_zero_row:last_zero_row + 1, :] = 0

    return inimg, lbl

def noise(inimg, lbl, noise_level=0.1):
    noise = torch.randn_like(inimg) * noise_level
    ninimg = inimg + noise
    return ninimg.clip(inimg.min(), inimg.max()), lbl
