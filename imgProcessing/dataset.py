#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:55:45 2024

@author: mateo-drr
"""

from torch.utils.data import DataLoader, Dataset
import random
from filtering import getHist
from torchvision import transforms
import torch

class CustomDataset(Dataset):

    def __init__(self, data, size=1024, angle=20):
        self.data = data
        self.rsz = transforms.Resize([size,size], antialias=True)
        self.angle = angle

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):    
    #TAKE ONE ITEM FROM THE DATASET
        img = self.data[idx][0]

        img = self.rsz(img)
        
        yhist,yhistH=getHist(img)
        
        ang = random.randint(-self.angle, self.angle)
        img = transforms.functional.rotate(img, ang)
        
        xhist,xhistH=getHist(img)
        
        return {'img':img,
                'ang':torch.tensor([ang],dtype=torch.float32),
                'yhist':yhist,
                'yhistH':yhistH,
                'xhist':xhist,
                'xhistH':xhistH,
                }