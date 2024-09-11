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




def conv(ni, nf, ks=3, stride=1, padding=1, padding_mode='reflect', **kwargs):
    _conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride,
                      padding=padding, padding_mode='reflect', **kwargs)
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

        # encoder
        self.enc1 = nn.Sequential(conv(1, 16, 3, 1, 1, padding_mode='reflect'),
                                  # nn.Mish(inplace=True),
                                  nn.PixelUnshuffle(2),
                                  )
        self.enc2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, padding_mode='reflect'),
                                  nn.PixelUnshuffle(2),
                                  )
        self.RinR = RRDB(nf=128, gc=64)

        self.enc3 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, padding_mode='reflect'),
                                  nn.Mish(inplace=True),
                                  nn.PixelUnshuffle(2),
                                  )

        # self.join = nn.Conv2d(16, 1, 3,1,1,padding_mode='reflect')
        self.RinRb = RRDB(nf=256, gc=256)

        # decoder
        self.px1 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1, padding_mode='reflect'),
                                 nn.Mish(inplace=True),
                                 nn.PixelShuffle(2)
                                 )

        self.RinRdec = RRDB(nf=128*2, gc=64)

        self.px2 = nn.Sequential(conv(128*2, 256, 3, stride=1, padding=1, padding_mode='reflect'),
                                 nn.Mish(inplace=True),
                                 nn.PixelShuffle(2),
                                 )

        self.px3 = nn.Sequential(conv(64*2, 4, 3, stride=1, padding=1, padding_mode='reflect'),
                                 nn.PixelShuffle(2),
                                 )

        # self.muxweights = nn.Sequential(nn.Linear(1024,512),
        #                                 nn.Mish(inplace=True))
        # self.out = nn.Sequential(nn.Linear(512,2),
        #                          nn.Mish(inplace=True))

        # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2, batch_first=True)
        # self.autobot = nn.TransformerEncoder(encoder_layer, 1)

    def encoder(self, x):
        # [b,c,h,w]
        j1 = self.enc1(x)  # /2
        j2 = self.enc2(F.mish(j1))  # /2
        # j2 = self.RinR(j2)
        oute = self.enc3(F.mish(j2))
        return oute, j1, j2

    def decoder(self, unqlat, j1, j2):
        outd = torch.cat([self.px1(unqlat), j2], dim=1)
        # outd = self.RinRdec(outd)
        outd = torch.cat([self.px2(outd), j1], dim=1)
        out = self.px3(outd)
        return out.clamp(0, 1)

    def forward(self, x):
        latent, j1, j2 = self.encoder(x)
        latent = self.RinRb(latent)
        out = self.decoder(latent, j1, j2)

        # oute = self.join(latent)
        # #[b,1,64,16]
        # oute = oute.reshape([-1,1,64*16])
        # #[b,cxhxw]
        # yhist = x.sum(dim=3)
        # yhist = self.autobot(yhist)
        # #[b,1,512]
        # att = self.muxweights(oute)
        # #[b,1,512]
        # oute = att + 0.2*yhist
        # oute = self.out(oute).squeeze(1)
        # [b,!1,2]
        return out, out  # oute.clamp(min=0)


class vitPl(nn.Module):
    def __init__(self):
        super(vitPl, self).__init__()

        if False:
            # Choose weights as per your need
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
            self.backbone = vit_b_16(weights=weights)
        else:
            self.backbone = vit_b_16(weights=None, num_classes=224)

        self.c1 = conv(1, 1, 3, 1, 1, padding_mode='reflect')
        self.c2 = conv(1, 1, 7, 1, 3, padding_mode='reflect')
        # self.l1 = nn.Linear(224, 2)
        self.m = nn.Mish(inplace=True)

    def forward(self, x):
        x1 = self.m(self.c1(x))
        x2 = self.m(self.c2(x1))
        x = torch.cat((x1, x2, x), dim=1)
        x = self.backbone(x)
        # x = self.l1(x)
        x = F.softmax(x, dim=1)
        return x, x


class MoDEEncoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv_more = MoDESubNet2Conv(
            num_experts, num_tasks, in_chan, out_chan)
        # self.conv_down = torch.nn.Sequential(
        #     torch.nn.Conv2d(out_chan, out_chan, kernel_size=2, stride=2, bias=False),
        #     nn.BatchNorm2d(out_chan, affine=True),
        #     torch.nn.Mish(inplace=True),
        # )

    def forward(self, x, t):
        x_skip = self.conv_more(x, t)
        # x = self.conv_down(x_skip)
        return x, x_skip


class MoDEDecoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.convt = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chan, affine=True),
            torch.nn.Mish(inplace=True),
        )
        self.conv_less = MoDESubNet2Conv(
            num_experts, num_tasks, in_chan, out_chan)

    def forward(self, x, x_skip, t):
        x = self.convt(x)
        x_cat = torch.cat((x_skip, x), 1)  # concatenate
        x_cat = self.conv_less(x_cat, t)
        return x_cat


class MoDESubNet2Conv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, n_in, n_out):
        super().__init__()
        self.conv1 = MoDEConv(num_experts, num_tasks, n_in,
                              n_out, kernel_size=5, padding='same')
        self.conv2 = MoDEConv(num_experts, num_tasks, n_out,
                              n_out, kernel_size=5, padding='same')

    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        return x


class MoDEConv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan, kernel_size=5, stride=1, padding='same', conv_type='normal'):
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.stride = stride
        self.padding = padding

        self.expert_conv5x5_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 5)
        self.expert_conv3x3_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 3)
        self.expert_conv1x1_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg3x3_pool', self.gen_avgpool_kernel(3))
        self.expert_avg3x3_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg5x5_pool', self.gen_avgpool_kernel(5))
        self.expert_avg5x5_conv = self.gen_conv_kernel(
            self.out_chan, self.in_chan, 1)

        assert self.conv_type in ['normal', 'final']
        if self.conv_type == 'normal':
            self.subsequent_layer = torch.nn.Sequential(
                nn.InstanceNorm2d(out_chan, affine=True),
                torch.nn.Mish(inplace=True),
            )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(
            num_tasks, num_experts * self.out_chan, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def gen_conv_kernel(self, Co, Ci, K):
        weight = torch.nn.Parameter(torch.empty(Co, Ci, K, K))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode='fan_out')
        return weight

    def gen_avgpool_kernel(self, K):
        weight = torch.ones(K, K).mul(1.0 / K ** 2)
        return weight

    def trans_kernel(self, kernel, target_size):
        Hp = (target_size - kernel.shape[2]) // 2
        Wp = (target_size - kernel.shape[3]) // 2
        return F.pad(kernel, [Wp, Wp, Hp, Hp])

    def routing(self, g, N):
        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(
            self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(
            self.expert_conv1x1_conv, self.kernel_size)
        expert_avg3x3 = self.trans_kernel(
            torch.einsum('oihw,hw->oihw', self.expert_avg3x3_conv,
                         self.expert_avg3x3_pool),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum(
            'oihw,hw->oihw', self.expert_avg5x5_conv, self.expert_avg5x5_pool)

        weights = []
        for n in range(N):
            weight_nth_sample = (
                torch.einsum('oihw,o->oihw', expert_conv5x5, g[n, 0, :])
                + torch.einsum('oihw,o->oihw', expert_conv3x3, g[n, 1, :])
                + torch.einsum('oihw,o->oihw', expert_conv1x1, g[n, 2, :])
                + torch.einsum('oihw,o->oihw', expert_avg3x3, g[n, 3, :])
                + torch.einsum('oihw,o->oihw', expert_avg5x5, g[n, 4, :])
            )
            weights.append(weight_nth_sample)
        weights = torch.stack(weights)

        return weights

    def forward(self, x, t):
        N = x.shape[0]  # batch size

        g = self.gate(t)
        g = g.view((N, self.num_experts, self.out_chan))
        g = self.softmax(g)

        w = self.routing(g, N)  # mix expert kernels

        if self.training:
            y = [F.conv2d(x[i].unsqueeze(0), w[i], bias=None,
                          stride=1, padding='same') for i in range(N)]
            y = torch.cat(y, dim=0)
        else:
            y = F.conv2d(x, w[0], bias=None, stride=1, padding='same')

        y = self.subsequent_layer(y)

        return y


class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        
        numc = 16
        
        self.re1c = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=1,out_chan=numc,)

        self.re1 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=1,out_chan=numc,)
        self.re2 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=numc*4,out_chan=numc*2,)
        self.re3 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=numc*2*4,out_chan=numc*4,)
        self.re4 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=numc*4*4,out_chan=numc*8,)
        
        self.rb = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=numc*8*4,out_chan=numc*16,)

        self.rd1 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=(numc*16)//4 + numc*8,out_chan=numc*8,)
        self.rd2 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=(numc*8)//4 + numc*4,out_chan=numc*4,)
        self.rd3 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=(numc*4)//4 + numc*2,out_chan=numc*2,)
        self.rd4 = MoDEEncoderBlock(num_experts=5,num_tasks=1,in_chan=(numc*2)//4 + numc,out_chan=numc,)

        self.dw = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2, batch_first=True)
        self.autobot = nn.TransformerEncoder(encoder_layer, 1)

        # self.e1 = nn.Sequential(conv(1,64,3,1,1),
        #                         nn.Mish(inplace=True),
        #                         #nn.BatchNorm2d(64),
        #                         conv(64,64,3,1,1),
        #                         nn.Mish(inplace=True),
        #                         #nn.BatchNorm2d(64,)
        #                         )

        # self.e2 = nn.Sequential(nn.PixelUnshuffle(2),  # uncomment if not using repmode
        #                         conv(64*4, 128, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(128),
        #                         conv(128, 128, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(128),
        #                         )

        # self.e3 = nn.Sequential(nn.PixelUnshuffle(2),
        #                         conv(128*4, 256, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(256),
        #                         conv(256, 256, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(256),
        #                         )

        # self.e4 = nn.Sequential(nn.PixelUnshuffle(2),
        #                         conv(256*4, 512, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(512),
        #                         conv(512, 512, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(512),
        #                         )

        # self.b = nn.Sequential(nn.PixelUnshuffle(2),
        #                        conv(512*4, 1024, 3, 1, 1),
        #                        nn.Mish(inplace=True),
        #                        # nn.BatchNorm2d(1024),
        #                        conv(1024, 1024, 3, 1, 1),
        #                        nn.Mish(inplace=True),
        #                        # nn.BatchNorm2d(1024),
        #                        nn.PixelShuffle(2))

        # self.d1 = nn.Sequential(conv(1024//4 + 512, 512, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(512),
        #                         conv(512, 512, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(512),
        #                         nn.PixelShuffle(2))

        # self.d2 = nn.Sequential(conv(512//4 + 256, 256, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(256),
        #                         conv(256, 256, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(256),
        #                         nn.PixelShuffle(2))

        # self.d3 = nn.Sequential(conv(256//4 + 128, 128, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(128),
        #                         conv(128, 128, 3, 1, 1),
        #                         nn.Mish(inplace=True),
        #                         # nn.BatchNorm2d(128),
        #                         nn.PixelShuffle(2))

        # self.d4 = nn.Sequential(conv(128//4 + 64,64,3,1,1),
        #                         nn.Mish(inplace=True),
        #                         #nn.BatchNorm2d(64),
        #                         conv(64,64,3,1,1),
        #                         nn.Mish(inplace=True),
        #                         #nn.BatchNorm2d(64)
        #                         )#nn.PixelShuffle(2))

        self.out = nn.Sequential(conv(numc, 2, 1, 1, 0),
                                 )  # nn.Sigmoid())

        self.dotmat = nn.Parameter(torch.empty(2,128, 1))
        nn.init.kaiming_normal_(
            self.dotmat, mode='fan_out', nonlinearity='relu')

    def one_hot_task_embedding(self, task_id):
        N = task_id.shape[0]
        task_embedding = torch.zeros((N, 1))
        for i in range(N):
            task_embedding[i, task_id[i]] = 1
        return task_embedding.to(device)

    def forward(self, x):

        
        cmap = 0#x[:,1]
        x = x[:,0]
        
        t = self.one_hot_task_embedding(torch.zeros(x.shape[0], dtype=int))
        #_, cmap = self.re1c(cmap,t)
        _, x1 = self.re1(x, t)  # [b,64*4,256,64]
        
        x2 = self.dw(x1+cmap)
        _, x2 = self.re2(x2, t)  # [b,64*4,256,64]
        x3 = self.dw(x2)
        _, x3 = self.re3(x3, t)  # [b,64*4,256,64]
        x4 = self.dw(x3)
        _, x4 = self.re4(x4, t)  # [b,64*4,256,64]
        lat = self.dw(x4)
        
        _, lat = self.rb(lat,t)
        
        lat = self.up(lat)
        _, x4 = self.rd1(torch.cat([lat, x4], dim=1),t)  # [b,128,64,16]
        x4 = self.up(x4)
        _, x3 = self.rd2(torch.cat([x4, x3], dim=1),t)  # [b,64,128,32]
        x3 = self.up(x3)
        _, x2 = self.rd3(torch.cat([x3, x2], dim=1),t)  # [b,32,256,64]
        x2 = self.up(x2)

        # x1 = self.e1(x)
        # x2 = self.e2(x1)  # [b,128,128,32]
        # x3 = self.e3(x2)  # [b,256,64,16]
        # x4 = self.e4(x3)  # [b,512,32,8]

        # lat = self.b(x4)  # [b,256,32,8]

        # x4 = self.d1(torch.cat([lat, x4], dim=1))  # [b,128,64,16]
        # x3 = self.d2(torch.cat([x4, x3], dim=1))  # [b,64,128,32]
        # x2 = self.d3(torch.cat([x3, x2], dim=1))  # [b,32,256,64]
        # x1 = self.d4(torch.cat([x2,x1],dim=1)) #[b,64,256,64]

        _, x1 = self.rd4(torch.cat([x2, x1+cmap], dim=1), t)

        x = self.out(x1)#.clamp(0, 1)

        #x = torch.mean(x, dim=3)
        temp=[]
        for chan in range(0,2):
            dotmat = self.dotmat[chan].repeat(x.size(0),1,1)
            temp.append( torch.bmm(x[:,chan], dotmat).clamp(0,1).squeeze(2).unsqueeze(1))

        x = torch.stack(temp,dim=1).clamp(0,1)

        return x, x


# PARAMS
date = '01Aug6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

split_ratio = 0.9
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

    lr = 1e-3
    numEpochs = 1500
    # Instantiate the model
    # model = vitPl()
    # model = plExtractor()
    model = unet()
    model.to(device)
    # Define a loss function and optimizer
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    import segmentation_models_pytorch as smp
    dice = smp.losses.DiceLoss(mode='binary')

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

            # # Create tensors with the same batch size as `out` for x1 and x2
            # fake_x1 = torch.full((out.size(0), 1), 128, dtype=torch.float32, requires_grad=True, device=device)  # Shape [b, 1]
            # fake_x2 = torch.full((out.size(0), 1), 128, dtype=torch.float32, requires_grad=True, device=device)  # Shape [b, 1]
            # # Split out into y1 and y2
            # y1, y2 = torch.chunk(out, 2, dim=1)
            # outb = torch.cat([fake_x1,y1,fake_x2,y2], dim=1)

            # #same for label
            # y1, y2 = torch.chunk(lbl, 2, dim=1)
            # lblb = torch.cat([fake_x1,y1,fake_x2,y2],dim=1)

            # if random.random() < 0.5:
            #     loss = criterion(pmask,mask)#criterion(out, lbl) + criterion(pmask,mask) # Compute loss
            # else:
            #     loss = criterion(out, lbl) + criterion(pmask,mask) # Compute loss

            mask = torch.mean(mask, dim=3)

            background = (mask == 0).float()
            foreground = (mask == 1).float()
            mask = torch.stack([background, foreground], dim=1)#.squeeze(2)

            # if mask[:,1].sum(dim=2) > pmask[:,1].sum(dim=2) :
                
            loss = dice(pmask, mask) + 0.1*torch.mean((mask[:,1].sum(dim=2) - pmask[:,1].sum(dim=2)).clamp(0,None)) # * bce(pmask,mask)*l1(pmask,mask)
            #
            loss += tvl(pmask,mask)
            
            # loss= criterion(out,blbl.to(device))#*l1(torch.topk(out,2,dim=1).values,lbl)

            # loss = tvo.generalized_box_iou_loss(outb,lblb, reduction='sum') #+ 0.1*criterion(out,lbl)
            loss.backward()             # Backward pass

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm()
            #         print(f"Layer: {name} | Gradient Norm: {grad_norm.item()}")

            optimizer.step()

            trainLoss += loss.item()
        print(tvl(pmask,mask))
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
    pmask, out = model(img)
    print('best', bestLoss, e)

    if wb:
        wandb.finish()

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
