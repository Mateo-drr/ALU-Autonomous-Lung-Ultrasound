# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:01:38 2024

@author: Mateo-drr
"""

import numpy as np
import filtering as filt
from scipy.signal import hilbert

def function(img,cmap):
    #Collapse the axis of the image
    yhist,xhist = filt.getHist(img)
    
    #Get the variance of the data so we can minimize it
    #Inturn making the pleura perpendicular to the probe
    yvar = np.var(yhist)
    
    #Get the mean value of the confidence map to maximize it
    #Since it get high confidence (1s) when it's perpendicular
    avgConf = np.mean(cmap)
    
    #Collapse the axis of the cmap
    cyhist,cxhist = filt.getHist(cmap)
    
    #The collapsed x axis gives us a clean segmented line from the image
    #And we can get an angle from it
    _,ang1,_,_ = filt.regFit(cxhist)
    
    #We can also calculate the angle from the image itself
    # Normalize to 255
    nimg = filt.normalize(img)
    # Find the peak of each line
    peaks = filt.findPeaks(nimg)
    #And we can get an angle from it
    _,ang2,_,_ = filt.regFit(peaks)
    
    
    