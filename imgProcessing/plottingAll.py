# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:06:19 2024

@author: Mateo-drr
"""

import byble as byb
from pathlib import Path
import numpy as np
from confidenceMap import confidenceMap
from skimage.transform import resize
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from sklearn.preprocessing import MinMaxScaler,QuantileTransformer, RobustScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler,PowerTransformer,Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
import pandas as pd
import gc

#LOAD THE DATA USING TESTDATAANALYSIS

# Group the data into 8 paths with 41 images each
paths = np.array(np.split(np.concatenate((alldat, alldat0), axis=1), 8, axis=0))

def feature(w,feat,rsize=False):
    strt,end = 2000,2800
    rsize=True
    coords=None
    if feat == 0:
        #Hilbert → Crop → Mean Lines → Prob. → Variance
        himg = byb.envelope(w)
        crop = himg[strt:end, :]
        t = np.mean(crop,axis=1)
        if rsize:
            t = resize(t, [800], anti_aliasing=True)
        probs = t/np.sum(t)
        x = np.arange(len(probs))
        avg = np.sum(x*probs)
        var = np.sum((x - avg)**2 * probs)
        r0 = var
    elif feat ==1: 
        #Hilbert → Log → Crop → Mean Lines → Prob. → Variance    
        himg = byb.envelope(w)
        loghimg = 20*np.log10(himg+1)
        crop = loghimg[strt:end, :]
        t = np.mean(crop,axis=1)
        if rsize:
            t = resize(t, [800], anti_aliasing=True)
        probs = t/np.sum(t)
        x = np.arange(len(probs))
        avg = np.sum(x*probs)
        var = np.sum((x - avg)**2 * probs)
        r0 = var  
    elif feat ==2:
        #Hilbert → Crop → Mean Lines → Variance
        himg = byb.envelope(w)
        crop = himg[strt:end, :]
        lineMean = np.mean(crop,axis=1)
        if rsize:
            lineMean = resize(lineMean, [800], anti_aliasing=True)
        r0 = lineMean.var()
    elif feat==3:
        #Hilbert → Log → Crop → Mean Lines → Variance
        himg = byb.envelope(w)
        loghimg = 20*np.log10(himg+1)
        crop = loghimg[strt:end, :]
        lineMean = np.mean(crop,axis=1)
        if rsize:
            lineMean = resize(lineMean, [800], anti_aliasing=True)
        r0 = lineMean.var()
    elif feat==4:
        #Hilbert → Crop → MinMax Line → Mean Lines → var
        himg = byb.envelope(w)
        crop = himg[strt:end, :]
        img = crop
        imgc = crop.copy()
        for k in range(img.shape[1]):
            line = img[:,k]
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            imgc[:,k] = line 
        lineMean = np.mean(imgc,axis=1)
        if rsize:
            lineMean = resize(lineMean, [800], anti_aliasing=True)
        r0 = np.var(lineMean)
    elif feat==5:
        #Hilbert → Log → Crop → MinMax Line → Mean Lines → var
        himg = byb.envelope(w)
        loghimg = 20*np.log10(himg+1)
        crop = loghimg[strt:end, :]
        img = crop
        imgc = crop.copy()
        for k in range(img.shape[1]):
            line = img[:,k]
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            imgc[:,k] = line 
        lineMean = np.mean(imgc,axis=1)
        if rsize:
            lineMean = resize(lineMean, [800], anti_aliasing=True)
        r0 = np.var(lineMean)
        
    elif feat==6:
        #Confidence Map → Crop → Mean Lines → Deriv. → Abs → Prob. → Variance
        if coords is not None:
            crop = w[strt-10:end]
        else:
            crop = w[strt:end]
        lineMean = np.mean(crop,axis=1)
        deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//16,
                                    polyorder=2, deriv=1))
        t=deriv
        if rsize:
            t = resize(t, [800], anti_aliasing=True)
        probs = t/np.sum(t)
        x = np.arange(len(probs))
        avg = np.sum(x*probs)
        var = np.sum((x - avg)**2 * probs)
        r0 = var
    elif feat==7:
        #Confidence Map → Crop → Mean Lines → Deriv. → Abs → Mean
        if coords is not None:
            crop = w[strt-10:end]
        else:
            crop = w[strt:end]
        lineMean = np.mean(crop,axis=1)
        deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//16,
                                    polyorder=2, deriv=1))
        if rsize:
            deriv = resize(deriv, [800], anti_aliasing=True)
        r0 = deriv.mean()
    elif feat==8:
        #Confidence Map → Crop → Mean Lines → Deriv. → Abs → Variance
        if coords is not None:
            crop = w[strt-10:end]
        else:
            crop = w[strt:end]
        lineMean = np.mean(crop,axis=1)
        deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//16,
                                    polyorder=2, deriv=1))
        if rsize:
            deriv = resize(deriv, [800], anti_aliasing=True)
        r0 = deriv.var()
        
    elif feat==9:
        #Hilbert → Crop → Mean
        himg = byb.envelope(w)
        crop = himg[strt:end, :]
        if rsize:
            crop = resize(crop, [800,129], anti_aliasing=True)
        r0 = crop.mean()
    elif feat==10:
        #Hilbert → Log → Crop → Mean
        himg = byb.envelope(w)
        loghimg = 20*np.log10(himg+1)
        crop = loghimg[strt:end, :]
        if rsize:
            crop = resize(crop, [800,129], anti_aliasing=True)
        r0 = crop.mean()
    elif feat==11:
        #Hilbert → Log → Crop → Log norm. → Mean
        himg = byb.envelope(w)
        loghimg = 20*np.log10(himg+1)
        crop = loghimg[strt:end, :]
        if rsize:
            crop = resize(crop, [800,129], anti_aliasing=True)
        crop = crop - crop.max()
        r0 = crop.mean()
    elif feat==12:
        #Hilbert → Crop → MinMax Line → mean
        himg = byb.envelope(w)
        crop = himg[strt:end, :]
        img = crop
        imgc = crop.copy()
        for k in range(img.shape[1]):
            line = img[:,k]
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            imgc[:,k] = line 
        if rsize:
            imgc = resize(imgc, [800,129], anti_aliasing=True)
        r0 = imgc.mean()
    elif feat==13:
        #Hilbert → Log → Crop → MinMax Line → mean
        himg = byb.envelope(w)
        loghimg = 20*np.log10(himg+1)
        crop = loghimg[strt:end, :]
        img = crop
        imgc = crop.copy()
        for k in range(img.shape[1]):
            line = img[:,k]
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            imgc[:,k] = line 
        if rsize:
            imgc = resize(imgc, [800,129], anti_aliasing=True)
        r0 = imgc.mean()
    
    
    elif feat==14:
        #Crop → Laplace → Variance
        crop = w[strt:end,:]
        if rsize:
            crop = resize(crop, [800,129], anti_aliasing=True)
        # print(crop.shape)
        lap = laplace(crop)
        r0 = lap.var()
    elif feat==15:
        #Hilbert → Crop → Laplace → Variance
        himg = byb.envelope(w)
        crop = himg[strt:end, :]
        if rsize:
            crop = resize(crop, [800,129], anti_aliasing=True)
        # print(crop.shape)
        lap = laplace(crop)
        r0 = lap.var()
    elif feat==16:
        #Hilbert → Log → Crop → Laplace → Variance
        himg = byb.envelope(w)
        loghimg = 20*np.log10(himg+1)
        crop = loghimg[strt:end, :]
        if rsize:
            crop = resize(crop, [800,129], anti_aliasing=True)
        # print(crop.shape)
        lap = laplace(crop)
        r0 = lap.var()
        
    return r0

def plotres(all_means,all_means0, feat):
    all_means = np.array(all_means)
    avg_means = all_means.mean(axis=0) #means among paths
    variance_among_paths = np.var(all_means, axis=0)
    x_values = list(range(-20, 21))
    extended_means = avg_means[:len(x_values)]
    
    all_means0 = np.array(all_means0)
    avg_means0 = all_means0.mean(axis=0) #means among paths
    variance_among_paths0 = np.var(all_means0, axis=0)
    x_values0 = list(range(-20, 21))
    extended_means0 = avg_means0[:len(x_values0)]

    if feat in [7,9,10,11,12,13]:
        name='Mean'
    else:
        name='Variance'
        
    xlbl = 'Degrees'#f'Degrees\nCoefficient of variation: {cv_per_angle.mean():.2f}%'
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=200)
    # Plot for the first subplot
    axes[0].errorbar(
        x_values0, extended_means0, yerr=np.sqrt(variance_among_paths0), fmt='-o',
        ecolor='r', capsize=5, capthick=2, color='#1f77b4'
    )
    axes[0].set_xlabel(xlbl, fontsize=18)
    axes[0].set_ylabel(name, fontsize=18)
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].tick_params(axis='y', labelsize=16)
    # axes[0].legend(["Plot 1"], fontsize=16)
    axes[0].grid(True)
    
    # Plot for the second subplot
    axes[1].errorbar(
        x_values, extended_means, yerr=np.sqrt(variance_among_paths), fmt='-o',
        ecolor='r', capsize=5, capthick=2, color='#ff7f0e'
    )
    axes[1].set_xlabel(xlbl, fontsize=18)
    axes[1].tick_params(axis='x', labelsize=16)
    axes[1].tick_params(axis='y', labelsize=16)
    # axes[1].legend(["Plot 2"], fontsize=16)
    axes[1].grid(True)
    
    # Adjust layout and display the figure
    plt.tight_layout()
    
    plt.savefig(f'C:/Users/Mateo-drr/Documents/data/figures/feats/f{feat}var.png')
    
    plt.show()

def calcmean(feat):
    all_means = []
    all_means0 = []
    for i, path in enumerate(paths):
        qwer,qwer0 = [],[]
        for j, w in enumerate(path):
            
            w,cmap,w0,cmap0 = w
            if feat in [6,7,8]:
                w,w0 = cmap,cmap0
            
            qwer.append(feature(w,feat))
            qwer0.append(feature(w0,feat))
    
        all_means.append(qwer)
        all_means0.append(qwer0)
    return all_means, all_means0


nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
for feat in nums:
    all_means, all_means0 = calcmean(feat)
    plotres(all_means,all_means0,feat)
    gc.collect()
    

