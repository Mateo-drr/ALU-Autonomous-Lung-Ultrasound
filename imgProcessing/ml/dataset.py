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
import matplotlib.pyplot as plt

prob = 0.7 # used in random.random() < prob

class CustomDataset(Dataset):

    def __init__(self, data, lbl, cmap, valid=False):
        self.data = data
        self.lbl = lbl
        self.cmap=cmap
        self.valid=valid
        
        #TODO check a good resize 
        #self.rsize = transforms.Resize((512,128),antialias=True)
        self.rsize = transforms.Resize((512,128),antialias=True)

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)
    
    def augment(self, inimg, lbl, cmap):
        
        #alter pixel values
        inimg,lbl = jitter(inimg, lbl, intensity=0.01)
        inimg,lbl,cmap = noise(inimg, lbl, cmap, noise_level=0.005)
        
        #shift pixel positions
        inimg,lbl,cmap = rolling(inimg, lbl, cmap)
        inimg,lbl,cmap = flipping(inimg, lbl, cmap)
        inimg,lbl,cmap = rdmLocalRotation(inimg, lbl, cmap)
        
        #mask
        inimg,lbl,cmap = masking(inimg, lbl, cmap, mask_prob=prob)
        
        return inimg,lbl,cmap

    def __getitem__(self, idx):    
    #TAKE ONE ITEM FROM THE DATASET
        img = torch.tensor(self.data[idx]).unsqueeze(0) #[C,H,W]
        cmap = torch.tensor(self.cmap[idx]).unsqueeze(0)
        
        #get the 2 points of the pleura mask
        p1,p2 = self.lbl[idx]
        #Create the binary mask image
        mask = torch.zeros_like(img, dtype=torch.uint8)
        # Set the region between p1 and p2 to 1
        mask[:,p1:p2, :] = 1
        
        #randomly cut the image in he height axis
        if not self.valid:
            img, mask, cmap = hcut(img, mask, cmap)
        
        #resizze the image
        img = self.rsize(img)
        mask = self.rsize(mask)
        cmap = self.rsize(cmap)
        
        #Normalize the data to 0 and 1
        min_val = torch.min(img)
        max_val = torch.max(img)
        imgn = (img - min_val) / (max_val - min_val)
        #Normalize the data to 0 and 1
        minc = torch.min(cmap)
        maxc = torch.max(cmap)
        cmapn = (cmap - minc) / (maxc - minc)
        
        #augment the data
        if not self.valid:
            imgn,mask,cmapn = self.augment(imgn, mask, cmapn)
        
        #find the new points of the mask
        p1mod,p2mod = masklim(mask) 
        
        #debug: plotting
        # img_denorm = imgn * (max_val - min_val) + min_val
        # plt.imshow(20*np.log10(abs(img_denorm)+1)[0],aspect='auto')
        # plt.axhline(p1mod)
        # plt.axhline(p2mod)
        
        # plt.show()
        # plt.imshow(20*np.log10(abs(img)+1)[0],aspect='auto')
        # plt.show()
        imgn = torch.stack([imgn,cmapn],dim=0)

        #bin multiclass label
        blbl = torch.zeros(512)
        blbl[p1mod] = 1
        blbl[p2mod] = 1

        #TODO imgn not being used??
        return imgn.to(torch.float32),torch.tensor([p1mod,p2mod],dtype=torch.float32),mask,blbl, [min_val,max_val]
    
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

def masklim(lbl):
    # Find the first and last row with a 1 in the mask
    one_mask = (lbl == 1).any(dim=2).squeeze()  # Check for any 1s in each row
    first_one_row = one_mask.nonzero(as_tuple=True)[0][0]  # First row with 1
    last_one_row = one_mask.nonzero(as_tuple=True)[0][-1]  # Last row with 1
    return first_one_row, last_one_row

def hcut(inimg, lbl, cmap):
    inimgc, lblc, cmapc = inimg, lbl, cmap
    if random.random() < prob:
        first_one_row, last_one_row = masklim(lbl)
        cuttop = random.randint(0, first_one_row)
        cutbtm = random.randint(0, inimg.size(1) - (last_one_row + 1))
        
        if random.random() < prob:
            inimgc = inimg[:, cuttop:, :]
            lblc = lbl[:, cuttop:, :]
            cmapc = cmap[:, cuttop:, :]
        if random.random() < prob:
            inimgc = inimg[:, :-cutbtm, :]
            lblc = lbl[:, :-cutbtm, :]
            cmapc = cmap[:, :-cutbtm, :]
        
        if inimgc.shape[1] == 0 or inimgc.shape[2] == 0:
            #something went wrong
            pass
        else:
            inimg,lbl,cmap = inimgc,lblc,cmapc
            
    return inimg,lbl,cmap

def rolling(inimg, lbl, cmap):
    first_one_row, last_one_row = masklim(lbl)

    if random.random() < prob:
        height = inimg.size(1)
        shiftdown = height - (last_one_row + 1) 
        shiftup = -first_one_row

        #safety checsk for scenarios where the mask strip is in the edge of the picture
        if shiftdown >0:
            sdown = random.randint(1,shiftdown)
        else:
            sdown=0
        if first_one_row > 0:
            sup = random.randint(shiftup, -1)
        else:
            sup=0
            
        shift_amount = random.choice([sdown, sup])

        inimg = torch.roll(inimg, shifts=shift_amount, dims=1)
        lbl = torch.roll(lbl, shifts=shift_amount, dims=1)
    
    if random.random() < prob:
        width = inimg.size(2)
        shift_amount = random.randint(1, width - 1)
        inimg = torch.roll(inimg, shifts=shift_amount, dims=2)
        lbl = torch.roll(lbl, shifts=shift_amount, dims=2)
        cmap = torch.roll(cmap, shifts=shift_amount, dims=2)

    return inimg, lbl, cmap

def flipping(inimg, lbl, cmap, flipChan=False):
    if random.random() < prob:
        inimg = torch.flip(inimg, dims=[1])
        lbl = torch.flip(lbl, dims=[1])
        cmap = torch.flip(cmap, dims=[1])
        
    if random.random() < prob:
        inimg = torch.flip(inimg, dims=[2])
        lbl = torch.flip(lbl, dims=[2])
        cmap = torch.flip(cmap, dims=[2])
        
    #UNUSED        
    # if flipChan and random.random() < 0.5:
    #     inimg = torch.flip(inimg, dims=[0])
    #     #add lbl and cmap if necessary
    
    return inimg, lbl, cmap

def jitter(inimg, lbl, intensity=0.4):
    if random.random() < prob:
        r = 0.8 + random.random() * intensity
        inimg = adjust_contrast(adjust_intensity(inimg, r), r)
    
    return inimg, lbl

def adjust_intensity(image, factor):
    return torch.clamp(image * factor, 0, 1)

def adjust_contrast(image, factor):
    return torch.clamp((image - 0.5) * factor + 0.5, 0, 1)

def masking(inimg, lbl, cmap, mask_prob=0.5, mask_size=64):
    if random.random() < mask_prob:
        mask_size = (mask_size, mask_size)
        height, width = inimg.shape[1:]
        h_start = random.randint(0, height - mask_size[0])
        w_start = random.randint(0, width - mask_size[1])

        mask = torch.ones_like(inimg)
        mask[:, h_start:h_start + mask_size[0], w_start:w_start + mask_size[1]] = 0
        minimg = inimg * mask
        cmap = cmap * mask
    else:
        minimg = inimg
        
    
    return minimg, lbl, cmap

def rdmLocalRotation(inimg, lbl, cmap, radius=64):
    
    if random.random() < prob:
        
        bkpinimg, bkplbl, bkpcmap = inimg.clone(), lbl.clone(), cmap.clone()
        
        first_one_row, last_one_row = masklim(lbl)
        
        diam = radius * 2
        x, y = inimg.shape[1:]
        xcenter = random.randint(0, x - diam)
        ycenter = random.randint(0, y - diam)
        ang = random.randint(20, 340)

        lum_img = Image.new('L', [diam, diam], 0)
        draw = ImageDraw.Draw(lum_img)
        draw.pieslice([(0, 0), (diam - 1, diam - 1)], 0, 360, fill=255)
        circmaks = torch.tensor(np.array(lum_img) / 255)
        invcircmask = (tf.rotate(circmaks.unsqueeze(0), ang) - 1) * -1

        circCrop = inimg[:, xcenter:xcenter + diam, ycenter:ycenter + diam] * circmaks
        circCrop = tf.rotate(circCrop, ang)
        invCircCrop = inimg[:, xcenter:xcenter + diam, ycenter:ycenter + diam] * invcircmask
        inimg[:, xcenter:xcenter + diam, ycenter:ycenter + diam] = circCrop + invCircCrop

        circCrop_cmap = cmap[:, xcenter:xcenter + diam, ycenter:ycenter + diam] * circmaks
        circCrop_cmap = tf.rotate(circCrop_cmap, ang)
        invCircCrop_cmap = cmap[:, xcenter:xcenter + diam, ycenter:ycenter + diam] * invcircmask
        cmap[:, xcenter:xcenter + diam, ycenter:ycenter + diam] = circCrop_cmap + invCircCrop_cmap

        circCrop_lbl = lbl[:, xcenter:xcenter + diam, ycenter:ycenter + diam] * circmaks
        circCrop_lbl = tf.rotate(circCrop_lbl, ang)
        invCircCrop_lbl = lbl[:, xcenter:xcenter + diam, ycenter:ycenter + diam] * invcircmask
        lbl[:, xcenter:xcenter + diam, ycenter:ycenter + diam] = circCrop_lbl + invCircCrop_lbl.squeeze()

        #expan area if target zone was rotated
        nfirst_one_row, nlast_one_row = masklim(lbl)
        
        if first_one_row != nfirst_one_row or last_one_row != nlast_one_row:
            inimg,lbl,cmap = bkpinimg,bkplbl,bkpcmap

    return inimg, lbl, cmap

def noise(inimg, lbl, cmap, noise_level=0.1):
    if random.random() < prob:
        noise = torch.randn_like(inimg) * noise_level
        ninimg = inimg + noise
        ncmap = cmap + noise
        return ninimg.clip(inimg.min(), inimg.max()), lbl, ncmap.clip(cmap.min(), cmap.max())
    return inimg, lbl, cmap
