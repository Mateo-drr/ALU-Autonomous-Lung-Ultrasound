#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 18:43:02 2024

@author: mateo-drr
"""

import math
import wandb
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import torchvision.ops as tvo
import copy
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
# imgprocessing folder

current_dir = Path(__file__).resolve().parent.parent

sys.path.append(current_dir.as_posix())

import byble as byb
import torch
import torch.nn as nn
from dataset import CustomDataset

from plExtractor import plExtractor

# PARAMS
date = '01Aug6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''
###############################################################################
# LOAD IMAGES
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
    datapath = current_dir.parent.parent / 'data' / \
        'acquired' / date / 'processed' / ptype
    # Get file names in the current directory
    fileNames = [f.name for f in datapath.iterdir() if f.is_file()]
    all_filenames.append([datapath, fileNames])

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
allcmap = []
datapath = all_filenames[0][0]
fileNames = all_filenames[0][1]
for pos, x in enumerate(allmove):

    if pos % 82 == 0:
        datapath = all_filenames[pos//82][0]
        fileNames = all_filenames[pos//82][1]

    img = byb.loadImg(fileNames, int(x[-1]), datapath)  # [100:]
    cmap = np.load(datapath.parent.parent / 'cmap' / f'cmap_{pos}.npy')
    # cmap = confidenceMap(img,rsize=True)
    # cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]

    alldat.append(img)
    allcmap.append(cmap)
###############################################################################
# LOAD LABELS
###############################################################################

datapath = current_dir / 'ml' / 'lines'
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

lines = []
for i in range(0, 4):
    btm = np.load(datapath / fileNames[i])
    top = np.load(datapath / fileNames[i+4])
    lines.append([top, btm])

lbls = np.concatenate(np.transpose(lines, (0, 2, 1)))

train_dts = CustomDataset(alldat, lbls, allcmap)
valid_dts = CustomDataset(alldat, lbls, allcmap, valid=True)

split_ratio = 0.2
dataset_size = len(train_dts)
indices = np.arange(dataset_size)
np.random.shuffle(indices)
split = int(split_ratio * dataset_size)
train_indices, val_indices = indices[:split], indices[split:]
# Create subsets
train_ds = Subset(train_dts, train_indices)
valid_ds = Subset(valid_dts, val_indices)

if True:

    batch=16
    train_dl = DataLoader(train_ds, batch_size=batch,
                          pin_memory=True, shuffle=True)  # , num_workers=2)

    valid_dl = DataLoader(valid_ds, batch_size=batch,
                          pin_memory=True, shuffle=False)  # , num_workers=2)

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    torch.backends.cudnn.benchmark = True

    lr = 1e-4
    numEpochs = 1
    # Instantiate the model
    model = plExtractor(device)
    model.to(device)
    # Define a loss function and optimizer
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    import segmentation_models_pytorch as smp
    dice = smp.losses.DiceLoss(mode='binary',from_logits=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    wb = False

    if wb:
        wandb.init(project="ALU",
                   config={
                       'lr': lr,
                   })

    bestmodel = None
    bestLoss = 1e32
    bestLossV = 1e32
    
    def tvl(pred, lbl):
         # Compute the gradient differences along the x-axis
         dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).pow(2).mean()  # RMS step
         dxt = (lbl[:, :, :, 1:] - lbl[:, :, :, :-1]).pow(2).mean()   # RMS step
                     
         # Compute the RMS difference instead of absolute value
         return torch.sqrt((dx - dxt).pow(2) + 1e-6)  # Adding a small epsilon for stability
    
    def p01(output):
        # Penalize values less than 0
        penalty_below = torch.relu(-output)  # relu(-x) penalizes x < 0
        # Penalize values greater than 1
        penalty_above = torch.relu(output - 1)  # relu(x - 1) penalizes x > 1
        # Combine penalties
        penalty = penalty_below + penalty_above
        return penalty.mean()
    
    for epoch in range(numEpochs):
        trainLoss = 0
        validLoss = 0

        model.train()
        for sample in tqdm(train_dl, desc=f"Epoch {epoch+1}/{numEpochs}"):
            img, lbl, mask, blbl, [minv, maxv] = sample
            img = img.to(device)
            lbl = lbl.to(device)
            mask = mask.to(device).to(torch.float32)

            pmask, out = model(img)

            optimizer.zero_grad()

            mask = torch.mean(mask, dim=3)

            background = (mask == 0).float()
            foreground = (mask == 1).float()
            mask = torch.stack([background, foreground], dim=1)
                
            loss = dice(pmask, mask) #+ 0.1*torch.mean((mask[:,1].sum(dim=2) - pmask[:,1].sum(dim=2)).clamp(0,None)) # * bce(pmask,mask)*l1(pmask,mask)
            #loss += tvl(pmask,mask)
            # loss += p01(pmask)
            loss+=bce(pmask,mask)

            loss.backward()             # Backward pass

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm()
            #         print(f"Layer: {name} | Gradient Norm: {grad_norm.item()}")

            optimizer.step()

            trainLoss += loss.item()
        # print(tvl(pmask,mask))
        # print("Model output:", out)
        # print("Labels:", lbl)
        avg_loss = trainLoss / len(train_dl)
        print(f'Epoch {epoch+1}, Loss: {loss.item()} {avg_loss}')
        # if wb:
        #     wandb.log({"Loss": avg_loss})  # Log epoch loss to W&B

        model.eval()
        with torch.no_grad():
            for sample in valid_dl:
                img, lbl, mask, blbl, [minv, maxv] = sample
                img = img.to(device)
                lbl = lbl.to(device)
                mask = mask.to(device).to(torch.float32)

                pmask, out = model(img)
                mask = torch.mean(mask, dim=3)

                background = (mask == 0).float()
                foreground = (mask == 1).float()
                mask = torch.stack([background, foreground], dim=1)#.squeeze(2)

                pmask = torch.round(pmask.clamp(0,1))
                loss = dice(pmask, mask)  # * bce(pmask,mask)*l1(pmask,mask)

                validLoss += loss.item()

        avg_lossV = validLoss / len(valid_dl)
        print(f'Epoch {epoch+1}, Loss: {loss.item()} {avg_lossV}')
        if wb:
            # Log epoch loss to W&B
            wandb.log({"VLoss": avg_lossV, "Loss": avg_loss})

        if avg_lossV < bestLossV and avg_loss < bestLoss:
            bestLoss = avg_loss
            bestLossV = avg_lossV
            bestmodel = copy.deepcopy(model)
            e = epoch

    # run the best model once
    pmask, out = bestmodel(img)
    pmask = torch.round(pmask.clamp(0,1))
    print('best', bestLoss, e)

    if wb:
        wandb.finish()
        
    path = current_dir.parent.parent / 'data' / 'models'
    bestmodel.load_state_dict(torch.load(path / 'model.pth'))
    bestmodel.eval()
    pmask, out = bestmodel(img)

    # print(torch.topk(out,2,dim=1).values,lbl)
    a,b = mask[:,1].cpu().detach().numpy(), pmask[:,1].cpu().detach().numpy()
    for i in range(0, len(a)):
        plt.plot(a[i, 0],'r')
        plt.plot(b[i, 0],'b')
        plt.show()

    plt.imshow(pmask.cpu().detach().numpy()[0][0])
    plt.show()
    plt.imshow(mask.cpu().detach().numpy()[0][0])
    plt.show()
    plt.imshow(img.cpu().detach().numpy()[0][0])
    plt.show()


if __name__ == '__main__':
    main()
'''


torch.save(bestmodel.state_dict(), path / 'model.pth')



'''