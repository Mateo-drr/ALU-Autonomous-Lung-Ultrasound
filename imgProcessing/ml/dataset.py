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
        self.rsize = transforms.Resize((6292,128),antialias=True)

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):    
    #TAKE ONE ITEM FROM THE DATASET
        img = torch.tensor(self.data[idx]).unsqueeze(0)
        
        img = self.rsize(img)[:,4:,:]
        
        min_val = torch.min(img)
        max_val = torch.max(img)
        img = (img - min_val) / (max_val - min_val)

        return img.to(torch.float32)