#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:43:02 2024

@author: mateo-drr
"""

import torch
import torch.nn as nn
from dataset import CustomDataset
import sys
from pathlib import Path

#imgprocessing folder
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(current_dir.as_posix())

import byble as byb

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import copy

import torchvision.ops as tvo
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F


from torchvision.models import vit_b_16, ViT_B_16_Weights

def conv(ni, nf, ks=3, stride=1, padding=1, **kwargs):
    _conv = nn.Conv2d(ni, nf, kernel_size=ks,stride=stride,padding=padding, **kwargs)
    nn.init.kaiming_normal_(_conv.weight, mode='fan_out')
    return _conv

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class plExtractor(nn.Module):
    def __init__(self):
        super(plExtractor, self).__init__()
        
        #encoder
        self.enc1 = nn.Sequential(conv(1, 16, 3, 1, 1,padding_mode='reflect'),
                                  nn.Mish(inplace=True),
                                  nn.PixelUnshuffle(2),
                                  )
        self.enc2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1,padding_mode='reflect'),
                                  nn.PixelUnshuffle(2),
                                  )
        # self.RinR = RRDB(nf=128, gc=256)
        
        self.enc3 = nn.Sequential(nn.Conv2d(128, 4, 3, 1, 1,padding_mode='reflect'),
                                  nn.PixelUnshuffle(2),
                                  ) 
        
        self.join = nn.Conv2d(16, 1, 3,1,1,padding_mode='reflect')
        
        #decoder
        self.px1 = nn.Sequential(nn.Conv2d(16, 512, 3, stride=1, padding=1, padding_mode='reflect'), 
                                  nn.PixelShuffle(2)
                                  )
        
        # self.RinRdec = RRDB(nf=128, gc=256)

        self.px2 = nn.Sequential(conv(128, 256, 3, stride=1, padding=1,padding_mode='reflect'), 
                                  nn.PixelShuffle(2),
                                  nn.LeakyReLU(inplace=True)
                                  )
        
        self.px3 = nn.Sequential(conv(64, 4, 3, stride=1, padding=1,padding_mode='reflect'),
                                  nn.PixelShuffle(2),
                                  )
        
        self.muxweights = nn.Sequential(nn.Linear(1024,512),
                                        nn.Mish(inplace=True))
        self.out = nn.Sequential(nn.Linear(512,2),
                                 nn.Mish(inplace=True))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2, batch_first=True)
        self.autobot = nn.TransformerEncoder(encoder_layer, 1)
    
    def encoder(self,x):
        #[b,c,h,w]
        j1 = self.enc1(x) #/2    
        j2 = self.enc2(j1) #/2
        # oute = self.RinR(oute)
        oute = self.enc3(j2)
        return oute, j1,j2
    
    def decoder(self,unqlat,j1,j2):
        outd = self.px1(unqlat) + 0.2*j2
        # outd = self.RinRdec(outd)
        outd = self.px2(outd) + 0.2*j1
        out = self.px3(outd) 
        return out.clamp(0,1)
        
    def forward(self,x):
        latent,j1,j2 = self.encoder(x)
        out = self.decoder(latent,j1,j2)
        
        oute = self.join(latent)
        #[b,1,64,16]
        oute = oute.reshape([-1,1,64*16])
        #[b,cxhxw]
        yhist = x.sum(dim=3)
        yhist = self.autobot(yhist)
        #[b,1,512]
        att = self.muxweights(oute)
        #[b,1,512]
        oute = att + 0.2*yhist 
        oute = self.out(oute).squeeze(1)
        #[b,!1,2]
        return out,oute.clamp(min=0)
    
class vitPl(nn.Module):
    def __init__(self):
        super(vitPl, self).__init__()
    
        if False:
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1  # Choose weights as per your need
            self.backbone = vit_b_16(weights=weights)
        else:
            self.backbone = vit_b_16(weights=None,num_classes=224)
            
        self.c1 = conv(1, 1,3,1,1,padding_mode='reflect')
        self.c2 = conv(1, 1,7,1,3,padding_mode='reflect')
        # self.l1 = nn.Linear(224, 2)
            
    def forward(self,x):
        x1=self.c1(x)
        x2=self.c2(x1)
        x = torch.cat((x1,x2,x),dim=1)
        x=self.backbone(x)
        # x = self.l1(x)
        x = F.sigmoid(x)
        return x,x
    
#PARAMS
date='01Aug6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
#LOAD IMAGES 
###############################################################################

ptype2conf = {
    'cl': 'curvedlft_config.json',
    'cf': 'curvedfwd_config.json',
    'rf': 'rotationfwd_config.json',
    'rl': 'rotationlft_config.json'
}

# Initialize lists to store all loaded data
all_filenames = []
all_conf = []
all_positions = []
# Loop over each ptype to load the corresponding data
for ptype, confname in ptype2conf.items():
    # Set the path for the current ptype
    datapath = current_dir.parent.parent / 'data' / 'acquired' / date / 'processed' / ptype
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

allmove = np.concatenate(all_positions, axis=0)
alldat = []
datapath = all_filenames[0][0]
fileNames = all_filenames[0][1]
for pos,x in enumerate(allmove):
    
    if pos%82==0:
        datapath = all_filenames[pos//82][0]
        fileNames = all_filenames[pos//82][1]
    
    img = byb.loadImg(fileNames, int(x[-1]), datapath)#[100:]
    #cmap = confidenceMap(img,rsize=True)
    #cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]

    alldat.append(img)
############################################################################### 
#LOAD LABELS
###############################################################################


datapath = current_dir / 'ml' / 'lines' 
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

lines=[]
for i in range(0,4):
    btm = np.load(datapath / fileNames[i])
    top = np.load(datapath / fileNames[i+4])
    lines.append([top,btm])
    
lbls = np.concatenate(np.transpose(lines,(0,2,1)))

train_ds = CustomDataset(alldat, lbls)

train_ds[0]

train_dl = DataLoader(train_ds, batch_size=32, pin_memory=True, shuffle=True)

lr=1e-4
numEpochs=50
# Instantiate the model
model = vitPl()#plExtractor()
model.to(device)
# Define a loss function and optimizer
criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

for epoch in range(numEpochs):    
    trainLoss=0
    for sample in tqdm(train_dl, desc=f"Epoch {epoch+1}/{numEpochs}"):
        img,lbl,mask,blbl,[minv,maxv] = sample
        img = img.to(device)
        lbl = lbl.to(device)
        mask = mask.to(device).to(torch.float32)
        
        pmask, out = model(img)
        
        optimizer.zero_grad()    
        
        # Create tensors with the same batch size as `out` for x1 and x2
        fake_x1 = torch.full((out.size(0), 1), 128, dtype=torch.float32, requires_grad=True, device=device)  # Shape [b, 1]
        fake_x2 = torch.full((out.size(0), 1), 128, dtype=torch.float32, requires_grad=True, device=device)  # Shape [b, 1]
        # Split out into y1 and y2
        y1, y2 = torch.chunk(out, 2, dim=1)
        outb = torch.cat([fake_x1,y1,fake_x2,y2], dim=1)
        
        #same for label
        y1, y2 = torch.chunk(lbl, 2, dim=1)
        lblb = torch.cat([fake_x1,y1,fake_x2,y2],dim=1)
        
        # if random.random() < 0.5:
        #     loss = criterion(pmask,mask)#criterion(out, lbl) + criterion(pmask,mask) # Compute loss
        # else:
        #     loss = criterion(out, lbl) + criterion(pmask,mask) # Compute loss
        
        loss= criterion(out,blbl.to(device))
            
        # loss = tvo.generalized_box_iou_loss(outb,lblb, reduction='sum') #+ 0.1*criterion(out,lbl)
        loss.backward()             # Backward pass
    
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm()
        #         print(f"Layer: {name} | Gradient Norm: {grad_norm.item()}")
    
        optimizer.step()        

        trainLoss += loss.item()
        
    # print("Model output:", out)
    # print("Labels:", lbl)
    avg_loss = trainLoss / len(train_dl)    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    

print(out,lbl)    

# plt.imshow(pmask.cpu().detach().numpy()[0][0])
# plt.show()
# plt.imshow(mask.cpu().detach().numpy()[0][0])
# plt.show()
# plt.imshow(img.cpu().detach().numpy()[0][0])
# plt.show()