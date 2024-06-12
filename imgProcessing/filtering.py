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

def envelope(data):
    env = []
    for idx in range(0,data.shape[1]):
        line = data[:,idx]
        hb = hilbert(line - line.mean())
        env.append(np.abs(hb))
    return np.array(env)

def bandFilt(data,highcut,lowcut,fs,N,order=10):
    
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
        print(lw,hg)
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

def findFrame(data,lineFrame,wind=1000):
    #Collapse height axis
    flat = np.sum(data, axis=0)
    plt.plot(flat[:wind],linewidth=1)
    plt.show()
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
        
        check = input(f'Current index was {strt}, mean: {favg}, mode: {fmode}. Enter new value or 0 to exit ')
        if check == '0' or check == '':
            break
        else:
            strt=int(check)
        
    #create frame index start array
    fidx = np.arange(strt,strt+len(flat),lineFrame)

    #Crop the correct frames -> one frame is lost 
    frames = []
    for i in range(1,len(fidx)):
        frames.append(data[:,fidx[i-1]+1:fidx[i]+1])
        
        
    return frames

def plotfft(data, fs):
    """
    Plot the Fourier transform of an array.
    
    Parameters:
    - data: array-like, the input signal.
    - fs: float, the sampling frequency.
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
    plt.plot(frequencies[:-500], magnitude[:-500])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Fourier Transform')
    plt.grid()
    plt.show()