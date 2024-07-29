# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:44:23 2024

@author: Mateo-drr
"""

import torch
import torch.nn as nn
from dataset import CustomDataset
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent
sys.path.append(current_dir.as_posix())

import byble as byb

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

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

class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        
        #encoder
        self.enc1 = nn.Sequential(conv(1, 16, 3, 1, 1,padding_mode='reflect'),
                                  nn.LeakyReLU(inplace=True),
                                  nn.PixelUnshuffle(2),
                                  )
        self.enc2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1,padding_mode='reflect'),
                                  nn.PixelUnshuffle(2),
                                  )
        self.RinR = RRDB(nf=128, gc=256)
        
        self.enc3 = nn.Sequential(nn.Conv2d(128, 4, 3, 1, 1,padding_mode='reflect'),
                                  nn.PixelUnshuffle(2),
                                  ) 
        
        self.join = nn.Conv2d(16, 1, 3,1,1,padding_mode='reflect')
        
        #decoder
        self.px1 = nn.Sequential(nn.Conv2d(4, 512, 3, stride=1, padding=1, padding_mode='reflect'), 
                                 nn.PixelShuffle(2)
                                 )
        
        self.RinRdec = RRDB(nf=128, gc=256)

        self.px2 = nn.Sequential(conv(128, 256, 3, stride=1, padding=1,padding_mode='reflect'), 
                                 nn.PixelShuffle(2),
                                 nn.LeakyReLU(inplace=True)
                                 )
        
        self.px3 = nn.Sequential(conv(64, 4, 3, stride=1, padding=1,padding_mode='reflect'),
                                 nn.PixelShuffle(2),
                                 )
    
    def encoder(self,x):
        oute = self.enc1(x) #/2    
        oute = self.enc2(oute) #/2
        oute = self.RinR(oute)
        oute = self.enc3(oute)
        oute = self.join(oute)
        return oute
    
    def decoder(self,unqlat):
        outd = self.px1(unqlat)
        outd = self.RinRdec(outd)
        outd = self.px2(outd)
        out = self.px3(outd)
        return out.clamp(0,1)
        
    def forward(self,x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent
    
#PARAMS
lr=0.01
numEpochs=100
date='05Jul'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datapath = current_dir.parent / 'data' / 'acquired' / date / 'processed'
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

data=[]
for f in range(len(fileNames)):
    temp = byb.loadImg(fileNames,int(f), datapath)
    data.append(temp)

train_ds = CustomDataset(data)

train_dl = DataLoader(train_ds)

# Instantiate the model
model = SampleNet()
model.to(device)
# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

for epoch in range(numEpochs):    
    trainLoss=0
    for sample in tqdm(train_dl, desc=f"Epoch {epoch+1}/{numEpochs}"):
        inp = sample.to(device)
        
        out,latent = model(inp)
        
        optimizer.zero_grad()    
        loss = criterion(out, inp)  # Compute loss
        loss.backward()             # Backward pass
        optimizer.step()        

        trainLoss += loss.item()
        
    avg_loss = trainLoss / len(train_dl)    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')