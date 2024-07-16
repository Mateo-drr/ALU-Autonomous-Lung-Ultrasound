#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:27:30 2024

@author: mateo-drr
"""

from scipy.signal import butter, freqz
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.stats import mode
import torch
from torchvision import transforms
import math
import random

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
    # Plot the data
    plt.figure(dpi=300)
    plt.imshow(img, aspect='auto', cmap='viridis')  # Adjust the colormap as needed
    plt.xlabel('Time')
    plt.ylabel('Index')
    plt.title('US')
    plt.colorbar(label='Intensity')

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
    return np.array(env)

def getHist(sec,tensor=False):
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
        min_val = np.min(sec)
        max_val = np.max(sec)
        sec = (sec - min_val) / (max_val - min_val)
        #collapse to 1d
        histY = np.sum(sec, axis=1) #same results with mean
        histX = np.sum(sec, axis=0) #same results with mean
    
    return histY, histX

def normalize(imgc, tensor=False):
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
        nimg = (255 * nimg / nimg.max()).to(torch.uint8)
    else:
        # Ensure input is a NumPy array
        imgc = np.array(imgc, dtype=np.float32)
        # Normalize to 0-255 range
        nimg = imgc + abs(imgc.min())
        nimg = (255 * nimg / nimg.max()).astype(np.uint8)
    
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

def regFit(peaks,tensor=False):
    """
    Calculates the linear fit for a given set of peaks and the angle of inclination.

    Parameters:
    peaks (array-like or Tensor): 1D array of peak positions.
    tensor (bool): Flag indicating whether the input is a PyTorch tensor.

    Returns:
    tuple: Contains the fitted line (as an array or tensor), angle in degrees, x-values, and y-values.
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
    else:
        # Ensure input is a NumPy array
        y = np.array(peaks, dtype=np.float32)
        x = np.arange(0, len(peaks), step=1)
        # Perform linear regression using NumPy
        slope, intercept = np.polyfit(x, y, 1)
        line = slope * x + intercept
        # Convert the slope to an angle in degrees
        angle = np.arctan(slope) * (180 / np.pi)

    return line, angle, x, y

def compute_interaction_matrix(confidence_map):
    gradients = np.gradient(confidence_map.astype(float))
    grad_x, grad_y = gradients[1], gradients[0]
    
    # Get dimensions
    y, x = np.indices(confidence_map.shape)

    # Build the interaction matrix
    L = np.zeros((confidence_map.size, 6))  # For 3D motion
    for i in range(confidence_map.shape[0]):
        for j in range(confidence_map.shape[1]):
            idx = i * confidence_map.shape[1] + j
            L[idx] = [
                grad_x[i, j],   # ∇Ix
                grad_y[i, j],   # ∇Iy
                1,               # ∇Iz (assuming constant)
                y[i, j] * 1,    # y * ∇Iz
                -x[i, j] * 1,   # -x * ∇Iz
                x[i, j] * 1     # x * ∇Iy
            ]
    return L

def bandFilt(data,highcut,lowcut,fs,N,order=10):
    
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
        
        # Define filter parameters
        # order = 10  # Filter order
        # lowcut = 3e6  # Low cutoff frequency (Hz)
        # highcut = 6e6  # High cutoff frequency (Hz)
        
        # Sample spacing (inverse of the sampling frequency)
        T = 1.0 / fs
        # Compute the FFT frequency range
        frequencies = np.fft.fftfreq(N, T)
        
        #FFT of a single line
        # fourier = np.fft.fft(data[:,100])
        # plt.plot(frequencies,np.log10(np.abs(fourier)))
        
        lw,hg = lowcut/(0.5*fs),highcut/(0.5*fs)
        
        # Design Butterworth filter
        #print(lw,hg)
        b, a = butter(order, [lw, hg], btype='bandpass')
        
        # # Plot frequency response
        # w, h = freqz(b, a, worN=8000)
        # amplitude = 20 * np.log10(abs(h))
        # plt.figure()
        # plt.plot( amplitude, 'b')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Gain (dB)')
        # plt.title('Butterworth Filter Frequency Response')
        # plt.grid()
        # plt.show()
        
        #Apply the filter
        filtered_signal = filtfilt(b, a, data[:,idx])
        
        # plt.figure()
        # plt.plot(data, 'b-', label='Original Signal')
        # plt.plot(filtered_signal, 'r-', linewidth=2, label='Filtered Signal')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Amplitude')
        # plt.title('Butterworth Lowpass Filter')
        # plt.legend()
        # plt.grid()
        # plt.show()
        
        # plt.figure()
        # plt.plot(data[4100:4300], 'b-', label='Original Signal')
        # plt.plot(filtered_signal[4100:4300], 'r-', linewidth=2, label='Filtered Signal')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Amplitude')
        # plt.title('Butterworth Lowpass Filter')
        # plt.legend()
        # plt.grid()
        # plt.show()
        
        fdata.append(filtered_signal)
        
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
                    peaks.append(peaks[-1])
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
    # end = fidx[-1]

    while True:

        plt.imshow(data[:,0:strt+lineFrame+20], aspect='auto', cmap='viridis')
        plt.axvline(x=strt, color='r', linestyle='--')
        plt.axvline(x=strt+lineFrame, color='r', linestyle='--')
        plt.show()
        
        # plt.imshow(clean[:,-(end+lineFrame+20):], aspect='auto', cmap='viridis')
        # plt.axvline(x=strt, color='r', linestyle='--')
        # plt.axvline(x=strt+lineFrame, color='r', linestyle='--')
        # plt.show()
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
