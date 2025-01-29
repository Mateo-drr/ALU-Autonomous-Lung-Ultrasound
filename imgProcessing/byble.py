#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:27:30 2024

@author: mateo-drr

Collection of various functions used all along the code
"""

from scipy.signal import butter
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.stats import mode
import torch
from torchvision import transforms
import math
import random
import json
from pathlib import Path
import json
from confidenceMap import confidenceMap
from skimage.transform import resize

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

def plotUS(img, norm=False):
    if norm:
        img = 20*np.log10(np.abs(img)+1)
    plt.figure(dpi=300)
    plt.imshow(img, aspect='auto', cmap='viridis')  # Adjust the colormap as needed
    plt.title('US')
    plt.colorbar(label='Intensity')

def hilb(data):
    hb = hilbert(data - data.mean())
    return np.abs(hb)

def envelope(data):
    """
    Computes the envelope of a 2D array using the Hilbert transform.

    Parameters:
    data (ndarray): 2D array of input data.

    Returns:
    ndarray: 2D array of the envelope of the input data.
    """
    env = []
    for idx in range(0,data.shape[1]):
        line = data[:,idx]
        hb = hilbert(line - line.mean())
        env.append(np.abs(hb))
    return np.array(env).transpose()

def getHist(sec,tensor=False,norm=False):
    """
    Computes the normalized sum of the image in both axes.

    Parameters:
    sec (ndarray): 2D array of input data.

    Returns:
    tuple: Two 1D arrays representing the normalized sum along the y-axis and x-axis.
    """
    
    if tensor:
        # Ensure input is a tensor
        sec = torch.tensor(sec, dtype=torch.float32)
        #normalize
        if norm:
            min_val = torch.min(sec)
            max_val = torch.max(sec)
            sec = (sec - min_val) / (max_val - min_val)
        #collapse to 1d
        histY = torch.sum(sec, axis=1) #same results with mean
        histX = torch.sum(sec, axis=0) #same results with mean
    else:
        # Ensure input is a NumPy array
        sec = np.array(sec, dtype=np.float32)
        #normalize
        if norm:
            min_val = np.min(sec)
            max_val = np.max(sec)
            sec = (sec - min_val) / (max_val - min_val)
        #collapse to 1d
        histY = np.sum(sec, axis=1) #same results with mean
        histX = np.sum(sec, axis=0) #same results with mean
    
    return histY, histX

def normalize(imgc, scale=255, tensor=False):
    """
    Normalizes the input image to a range of 0 to 255.

    Parameters:
    imgc (ndarray or Tensor): 2D array or tensor representing the input image.
    tensor (bool): Flag indicating whether the input is a PyTorch tensor.

    Returns:
    ndarray or Tensor: Normalized image as an 8-bit unsigned integer array or tensor.
    """
    if tensor:
        # Ensure input is a tensor
        imgc = torch.tensor(imgc, dtype=torch.float32)
        # Normalize to 0-255 range
        nimg = imgc + imgc.abs().min()
        nimg = (scale * nimg / nimg.max()).to(torch.uint8)
    else:
        # Ensure input is a NumPy array
        imgc = np.array(imgc, dtype=np.float32)
        # Normalize to 0-255 range
        nimg = imgc + abs(imgc.min())
        nimg = (scale * nimg / nimg.max()).astype(np.uint8)
    
    return nimg

def rotate(imgc, angle):
    """
    Rotates the input image by a specified angle.

    Parameters:
    imgc (ndarray): 2D array representing the input image.
    angle (float): Angle by which to rotate the image.

    Returns:
    Tensor: Rotated image.
    """
    #Rotate the image
    imgct = torch.tensor(imgc).unsqueeze(0).unsqueeze(0)
    imgcr = transforms.functional.rotate(imgct,angle,expand=True)[0][0]
    return imgcr

def rotatClip(imgcr, imgc, angle, cropidx=False):
    if angle == 0:
        return imgcr,0,0
    # Calculate the rotated crop
    xr,yr = rotatedRectWithMaxArea(imgc.shape[0], imgc.shape[0], np.deg2rad(angle))
    x0,y0 = int(np.ceil((imgcr.shape[1]-xr)/2)), int(np.ceil((imgcr.shape[0]-yr)/2))
    # crop and plot
    rotimg = imgcr[y0:-y0,x0:-x0]
    if cropidx:
        return rotimg,x0,y0
    return rotimg

def rotcrop(imgcr, xr,yr , cropidx=False):
    x0,y0 = int(np.ceil((imgcr.shape[1]-xr)/2)), int(np.ceil((imgcr.shape[0]-yr)/2))
    # crop and plot
    rotimg = imgcr[y0:-y0,x0:-x0]
    if cropidx:
        return rotimg,x0,y0
    return rotimg

def moveClip(finalimg, area = 0.75, newx=None,newy=None,cropidx=False):
    # Maximum area crop in %
    maxx = int((finalimg.shape[1] - finalimg.shape[1]*area))
    maxy = int((finalimg.shape[0] - finalimg.shape[0]*area))
    # Get a random translation for x and y
    if newx is None or newy is None:
        newx = random.randint(0, maxx)
        newy = random.randint(0, maxy)
    else:
        newx = min(newx,maxx)
        newy = min(newy,maxy)
    endx = maxx - newx
    endy = maxy - newy
    crop = finalimg[newy:finalimg.shape[0]-endy,newx:finalimg.shape[1]-endx]
    if cropidx:
        return crop,newx,endx,newy,endy,maxx,maxy   
    return crop

def rsize(img,y=None,x=None):
    if y is None or x is None:
        x = img.shape[0]
        y = img.shape[0]
    timg = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    resize = transforms.Resize((x,y), antialias=True)
    rimg = resize(timg)
    return rimg

import numpy as np
import torch

def regFit(peaks, tensor=False):
    """
    Calculates the linear fit for a given set of peaks, the angle of inclination, and fit certainty metrics.

    Parameters:
    peaks (array-like or Tensor): 1D array of peak positions.
    tensor (bool): Flag indicating whether the input is a PyTorch tensor.

    Returns:
    tuple: Contains the fitted line (as an array or tensor), angle in degrees, x-values, y-values, R² score, and RMSE.
    """

    if tensor:
        # Ensure input is a tensor
        y = torch.tensor(peaks, dtype=torch.float32)
        x = torch.arange(0, len(peaks), dtype=torch.float32)
        # Perform linear regression using PyTorch
        A = torch.vstack([x, torch.ones_like(x)]).T
        slope, intercept = torch.linalg.lstsq(A, y.unsqueeze(1)).solution[:2].squeeze()
        line = slope * x + intercept
        # Convert the slope to an angle in degrees
        angle = torch.atan(slope).item() * (180 / np.pi)
        
        # Calculate R² score
        y_mean = torch.mean(y)
        ss_total = torch.sum((y - y_mean) ** 2)
        ss_residual = torch.sum((y - line) ** 2)
        r_squared = 1 - (ss_residual / ss_total).item()
        
        # Calculate RMSE
        rmse = torch.sqrt(torch.mean((y - line) ** 2)).item()
        
    else:
        # Ensure input is a NumPy array
        y = np.array(peaks, dtype=np.float32)
        x = np.arange(0, len(peaks), step=1)
        # Perform linear regression using NumPy
        slope, intercept = np.polyfit(x, y, 1)
        line = slope * x + intercept
        # Convert the slope to an angle in degrees
        angle = np.arctan(slope) * (180 / np.pi)
        
        # Calculate R² score
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - line) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y - line) ** 2))

    return line, angle, x, y, r_squared, rmse


def bandFilt(data,highcut,lowcut,fs,N,order=10,plot=True):
    
    """
    Applies a Butterworth bandpass filter to the input data.

    Parameters:
    data (ndarray): 2D array of input data.
    highcut (float): High cutoff frequency (Hz).
    lowcut (float): Low cutoff frequency (Hz).
    fs (float): Sampling frequency.
    N (int): Length of the data.
    order (int, optional): Filter order. Default is 10.

    Returns:
    ndarray: Filtered data.
    """
    
    fdata = []
    for idx in range(0,data.shape[1]):
        
        # Sample spacing (inverse of the sampling frequency)
        T = 1.0 / fs
        # Compute the FFT frequency range
        frequencies = np.fft.fftfreq(N, T)
        
        lw,hg = lowcut/(0.5*fs),highcut/(0.5*fs)
        
        # Design Butterworth filter
        b, a = butter(order, [lw, hg], btype='bandpass')
        
        #Apply the filter
        filtered_signal = filtfilt(b, a, data[:,idx])
        
        fdata.append(filtered_signal)
        
        if plot:
            plt.plot(frequencies,np.log10(np.abs(np.fft.fft(filtered_signal))))
            plt.xlim(-0.5e7,0.5e7)
        
    return np.transpose(np.array(fdata),[1,0])

def findPeaks(nimg,tensor=False):
    
    #Loop each line and get the maximum value
    peaks=[]
    
    if tensor:
        for i in range(nimg.shape[1]):
            line = nimg[:, i]
            maxidx = torch.where(line == torch.max(line))[0]
            step = torch.diff(maxidx)
            
            # Check if all steps are equal to 1
            stepidx = torch.nonzero(step != 1).squeeze()
            
            # If all of them are equal, pick the middle value as the top
            if stepidx.numel() == 0:
                peaks.append(maxidx[len(maxidx) // 2].item())
            else:
                # Temporary solution: add the past peak as the current one
                try:
                    peaks.append(peaks[-1])
                except IndexError:
                    peaks.append(maxidx[0].item())  # If fail, just use the first one
    else:
        for i in range(nimg.shape[1]):
            line = nimg[:, i]
            maxidx = np.where(line == np.max(line))[0]
            step = np.diff(maxidx)
            
            # Check if all steps are equal to 1
            stepidx = np.where(step != 1)[0]
            
            # If all of them are equal, pick the middle value as the top
            if len(stepidx) == 0:
                peaks.append(maxidx[len(maxidx)//2])
            else:
                # Temporary solution: add the past peak as the current one
                try:
                    peaks.append(int(np.mean(maxidx).item())) 
                    # peaks.append(peaks[-1])
                except IndexError:
                    peaks.append(maxidx[0])  # If fail, just use the first one

    return peaks

def findFrame(data,lineFrame,wind=1000,getframes=True):
    #Collapse height axis
    flat = np.sum(data, axis=0)
    plt.plot(flat[:wind],linewidth=1)
    plt.show()
    
    # Get the indices that would sort the array
    sorted_indices = np.argsort(flat)
    # Get the indices of the three smallest values
    three_smallest_indices = sorted_indices[:10]
    
    deriv = np.diff(flat)
    derivClean = np.power(np.clip(deriv,0,deriv.max()),2) #apply a power to make the maximum values stand out more

    plt.plot(derivClean[:wind])
    plt.show()

    # find indexes where the derivative is maximum
    fidx = np.where(derivClean>=derivClean.max()*0.2)[0] 
    #double check if frames where identified correctly
    fSize = np.diff(fidx)
    fmode = mode(fSize)[0]
    favg = np.mean(fSize)

    strt = fidx[0]

    while True:

        plt.imshow(data[:,0:strt+lineFrame+20], aspect='auto', cmap='viridis')
        plt.axvline(x=strt, color='r', linestyle='--')
        plt.axvline(x=strt+lineFrame, color='r', linestyle='--')
        plt.show()
        
        print(three_smallest_indices, fidx)
        check = input(f'Current index: {strt}, mean: {favg}, mode: {fmode}. Enter new value (]) or 0 to exit ')
        if check == '0' or check == '':
            break
        else:
            strt=int(check)
            #create frame index start array
            fidx = np.arange(strt,strt+len(flat),lineFrame)
            if fidx[-1] >= data.shape[1]:
                print('Selected start is over image size (max 499):',fidx[-1],data.shape[1])

    if getframes:
        #Crop the correct frames -> one frame is lost 
        frames = cropFrames(data, fidx)
        return frames
    else:
        return fidx

def cropFrames(data,fidx):
    frames = []
    for i in range(1,len(fidx)):
        if fidx[i] >= data.shape[1]:
            print('One extra frame lost!')
            continue
        frames.append(data[:,fidx[i-1]+1:fidx[i]+1])
    return frames

def plotfft(data, fs, log=True):
    """
    Plot the Fourier transform of an array.
    
    Parameters:
    - data: array-like, the input signal.
    - fs: float, the sampling frequency.
    - log: bool, whether to plot the magnitude on a logarithmic scale.
    """
    # Compute the FFT
    fft_result = np.fft.fft(data)
    
    # Compute the frequencies corresponding to the FFT result
    N = len(data)
    T = 1.0 / fs
    frequencies = np.fft.fftfreq(N, T)
    
    # Only plot the positive frequencies
    positive_freq_indices = np.where(frequencies >= 0)
    frequencies = frequencies[positive_freq_indices]
    fft_result = fft_result[positive_freq_indices]
    
    # Compute the magnitude of the FFT result
    magnitude = np.abs(fft_result)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    if log:
        plt.semilogy(frequencies[:-500], magnitude[:-500])
        plt.ylabel('Magnitude (dB)')
    else:
        plt.plot(frequencies[:-500], magnitude[:-500])
        plt.ylabel('Magnitude')
    
    plt.xlabel('Frequency (Hz)')
    plt.title('Fourier Transform')
    plt.grid()
    plt.show()

def loadImg(fileNames,idx,datapath):
    print('Loading image',datapath / fileNames[idx])
    img = np.load(datapath / fileNames[idx])[:,:]
    return img

def loadConf(datapath,confname):
    with open(datapath.parent / confname, 'r') as file:
        data = json.load(file)
    return data

def findClosestPosition(target_coord,xfake,yfake):
    target_coord = np.array(target_coord)
    
    distances_xfake = np.linalg.norm(xfake[:, :, :7] - target_coord, axis=-1)  # Only consider first 7 values
    closest_index_xfake = np.unravel_index(np.argmin(distances_xfake), distances_xfake.shape)
    closest_xfake = xfake[closest_index_xfake]
    min_distance_xfake = distances_xfake[closest_index_xfake]
    
    distances_yfake = np.linalg.norm(yfake[:, :, :7] - target_coord, axis=-1)  # Only consider first 7 values
    closest_index_yfake = np.unravel_index(np.argmin(distances_yfake), distances_yfake.shape)
    closest_yfake = yfake[closest_index_yfake]
    min_distance_yfake = distances_yfake[closest_index_yfake]
    
    return closest_xfake if min_distance_xfake < min_distance_yfake else closest_yfake

def convert_to_native_types(data):
    """Recursively convert NumPy arrays and scalars to native Python types."""
    if isinstance(data, dict):
        # Convert each key-value pair in the dictionary
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Recursively convert each item in the list
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, tuple):
        # Convert tuples to lists
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        # Convert NumPy arrays to lists of native Python types
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        # Convert NumPy scalars to float
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        # Convert NumPy integers to int
        return int(data)
    else:
        # Return the data as is if it's already a native Python type
        return data
    
def loadAllData(baseDir, withMeat=True, withoutMeat=True, testMeat=True, testChest=False,
                withMeatFolder='01Aug6', loadCmap=True):
    '''
    Parameters
    ----------
    baseDir : String
        Directory where the repo is located (ex: C:/Users/Name/Documents). Assuming the data folder is in the same dir.
    withMeat : Bool, optional
        Load train data with meat. The default is True.
    withoutMeat : Bool, optional
        Load train data without meat. The default is True.
    testMeat : Bool, optional
        Load test data with meat. The default is True.
    testChest : Bool, optional
        Load test data of chest phantom. The default is False.
    withMeatFolder : String, optional
        Acquisition folder to load. The default is '01Aug6'.
    loadCmap : Bool, optional
        Flag to load or not the confidence map

    Returns
    -------
    alldat : List
        List with all the train data with the meat + conf. map.
    alldat0 : List
        List with all the train data without the meat + conf. map. 
    alltestMeat : List
        List with all the test data with the meat. 
    alltestChest : List
        List with all the test data with the chest phantom. 
    '''
    
    alldat,alldat0,alltestMeat,alltestChest=[],[],[],[]
    
    if withMeat:
        date = withMeatFolder
        ptype2conf = {
            'cl': 'curvedlft_config.json',
            'cf': 'curvedfwd_config.json',
            'rf': 'rotationfwd_config.json',
            'rl': 'rotationlft_config.json'
        }
        
        # Get the base directory
        current_dir = Path(__file__).resolve().parent.parent.parent
        
        # Initialize lists to store all loaded data
        all_filenames = []
        all_conf = []
        all_positions = []
        
        # Loop over each ptype to load the corresponding data
        for ptype, confname in ptype2conf.items():
            # Set the path for the current ptype
            datapath = current_dir / 'data' / 'acquired' / date / 'processed' / ptype
            # Get file names in the current directory
            fileNames = [f.name for f in datapath.iterdir() if f.is_file()]
            all_filenames.append([datapath,fileNames])
            
            # Load the configuration of the experiment
            conf = loadConf(datapath, confname)
            all_conf.append(conf)
            
            # Organize the data as [coord, q rot, id]
            positions = []
            for i, coord in enumerate(conf['tcoord']):
                positions.append(coord + conf['quater'][i] + [i])
            
            all_positions.append(np.array(positions))
        
        # If you need to concatenate positions or other data across ptpyes, you can do so here
        allmove = np.concatenate(all_positions, axis=0)
        
        # alldat = []
        
        datapath = all_filenames[0][0]
        fileNames = all_filenames[0][1]
        for pos,x in enumerate(allmove):
            
            if pos%82 == 0:
                datapath = all_filenames[pos//82][0]
                fileNames = all_filenames[pos//82][1]
            
            img = loadImg(fileNames, int(x[-1]), datapath)#[100:]
            if loadCmap:
                cmap = confidenceMap(img,rsize=True)
                cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]
                alldat.append((img,cmap))
            else:
                alldat.append((img))
            
    if withoutMeat:    
        # PARAMS
        date = '01Aug0'
        ptype2conf = {
            'cl': 'curvedlft_config.json',
            'cf': 'curvedfwd_config.json',
            'rf': 'rotationfwd_config.json',
            'rl': 'rotationlft_config.json'
        }
        
        # Get the base directory
        current_dir = Path(__file__).resolve().parent.parent.parent
        
        # Initialize lists to store all loaded data
        all_filenames = []
        all_conf = []
        all_positions = []
        
        # Loop over each ptype to load the corresponding data
        for ptype, confname in ptype2conf.items():
            # Set the path for the current ptype
            datapath = current_dir / 'data' / 'acquired' / date / 'processed' / ptype
            # Get file names in the current directory
            fileNames = [f.name for f in datapath.iterdir() if f.is_file()]
            all_filenames.append([datapath,fileNames])
            
            # Load the configuration of the experiment
            conf = loadConf(datapath, confname)
            all_conf.append(conf)
            
            # Organize the data as [coord, q rot, id]
            positions = []
            for i, coord in enumerate(conf['tcoord']):
                positions.append(coord + conf['quater'][i] + [i])
            
            all_positions.append(np.array(positions))
        
        # If you need to concatenate positions or other data across ptpyes, you can do so here
        allmove = np.concatenate(all_positions, axis=0)
        
        # alldat0 = []
        
        datapath = all_filenames[0][0]
        fileNames = all_filenames[0][1]
        for pos,x in enumerate(allmove):
            
            if pos%82 == 0:
                datapath = all_filenames[pos//82][0]
                fileNames = all_filenames[pos//82][1]
            
            img = loadImg(fileNames, int(x[-1]), datapath)#[100:]
            if loadCmap:
                cmap = confidenceMap(img,rsize=True)
                cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]
                alldat0.append((img,cmap))
            else:
                alldat0.append((img))
        
    '''
    Test data
    '''
    
    if testMeat:
        datap = current_dir / 'data' / 'dataMeat' 
        print(f'Loading test data: {datap}')
        # Get all folder names in the directory
        folder_names = [f.name for f in datap.iterdir() if f.is_dir()]
        
        # filtRF = []
        for folder in folder_names:
            runpath = datap / folder / 'runData'
            #load RF data
            imgs=[]
            matching_files = list(runpath.rglob('UsRaw*.npy'))
            for run in matching_files:
                imgs.append(np.load(runpath / run))
                
            with open(runpath / 'variables.json', 'r') as file:
                data = json.load(file)
                
            scores = []    
            for i in range(len(data['scores'])-1):
                scores.append(data['scores'][i][1])
                
            angle = []
            for i in range(len(data['results']['all'])):
                temp = data['results']['all'][i]['position'][-3:-1]
                assert len(temp) == 2
                # if (temp[0] >= 0 and temp[1] >=0) or (temp[0] < 0 and temp[1] < 0):    
                #     angle.append(np.mean(temp))
                # else:
                angle.append(np.mean(np.abs(temp)))
                
            alltestMeat.append([imgs,scores,angle])    
            
    if testChest:
        datap = current_dir / 'data' / 'dataChest' 
        print(f'Loading test data: {datap}')
        # Get all folder names in the directory
        folder_names = [f.name for f in datap.iterdir() if f.is_dir()]
        
        # filtRF = []
        for folder in folder_names:
            runpath = datap / folder / 'runData'
            #load RF data
            imgs=[]
            matching_files = list(runpath.rglob('UsRaw*.npy'))
            for run in matching_files:
                imgs.append(np.load(runpath / run))
                
            with open(runpath / 'variables.json', 'r') as file:
                data = json.load(file)
                
            scores = []    
            for i in range(len(data['scores'])-1):
                scores.append(data['scores'][i][1])
                
            angle = []
            for i in range(len(data['results']['all'])):
                temp = data['results']['all'][i]['position'][-3:-1]
                assert len(temp) == 2
                # if (temp[0] >= 0 and temp[1] >=0) or (temp[0] < 0 and temp[1] < 0):    
                #     angle.append(np.mean(temp))
                # else:
                angle.append(np.mean(np.abs(temp)))
                
            alltestChest.append([imgs,scores,angle])    
            
    return alldat,alldat0,alltestMeat,alltestChest
        
    
def logS(data):
    return 20*np.log10(abs(data)+1)