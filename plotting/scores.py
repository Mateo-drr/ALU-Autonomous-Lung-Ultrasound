#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:27:27 2024

@author: mateo-drr
"""
import numpy as np
import matplotlib.pyplot as plt

# Define the data
s0 = [0.0907, -0.2194, 0.0709, 0.0836, 0.1102, -0.5151, 0.1621, 0.0360, -0.5901, 0.2056, 0.0742, -0.5729, -0.5981, -0.6335, -0.6839, -0.7050]
s1 = [0.2965, 0.0168, 0.2553, 0.1897, 0.1211, 0.0354, 0.1684, 0.0976, 0.2556, 0.0798, -0.6433, 0.2307, 0.2171, -0.4425, 0.1259, 0.2836, 0.1872, 0.0766, 0.1894, 0.1203, 0.2346, 0.1338]
s2 = [0.1492, 0.1715, -0.6560, -0.6971, -0.7158, -0.6486]
s3 = [0.2134, 0.0626, 0.0405, 0.0865, 0.1687, 0.0349, 0.0253, 0.0299, 0.1278, 0.0186, 0.0996, 0.0399, 0.0266, -0.5236, -0.6308, -0.6487, 0.1651, 0.0609, 0.0794, -0.5995, 0.1599, 0.1806]
s4 = [0.0502, -0.6029, 0.1259, -0.4009, -0.5009, -0.4865, -0.5063, -0.5855, 0.1513, 0.0711, -0.4534, -0.6889, 0.6996, -0.5784, -0.6522, -0.6152, -0.6935, -0.7279]
s5 = [0.2058, -0.4930, 0.1266, -0.6290, 0.3018, 0.2490, 0.3693, 0.1512, -0.6134, 0.2593, -0.6361, -0.5088, -0.6890, -0.7345, -0.2227, -0.7386]
s6 = [-0.6113, -0.5443, -0.5262, 0.0329, 0.0035, -0.5460, -0.4083, -0.5776, 0.0148, -0.5460, -0.5581, -0.4569, -0.4904, -0.4422, -0.4636, -0.5863, -0.6187]
s7 = [0.1696, 0.0149, 0.2051, -0.5152, -0.4767, -0.6429, 0.0432, 0.0965, 0.0144, 0.0087, -0.3728, -0.3626, 0.0720, 0.0424, 0.0560, -0.4159, -0.4197, 0.0654, 0.1182, 0.0640, 0.0920, 0.0336]
s8 = [0.0556, 0.1766, -0.4987, 0.0789, 0.0332, 0.0131, 0.0162, 0.0894, 0.0238, 0.0321, -0.5630, 0.0722, -0.7003, 0.0508, -0.5698, -0.6661, -0.5099, -0.6466, -0.5991]
s9 = [-0.6540, -0.6913, -0.6244, -0.6981, -0.5872]

# Combine the runs into a list
data = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]

# Find the maximum length of the runs
max_length = max(len(run) for run in data)

# Pad the shorter runs with NaN
padded_data = [run + [np.nan] * (max_length - len(run)) for run in data]

# Convert to a NumPy array
data_array = np.array(padded_data)

# Calculate the average and standard deviation, ignoring NaNs
avg = np.nanmean(data_array, axis=0)
std_dev = np.nanstd(data_array, axis=0)

# Create an x-axis for the runs
x = np.arange(max_length)

# Plot each run
for i, run in enumerate(padded_data):
    plt.plot(run, marker='o', alpha=0.6)

# Plot the average
plt.plot(avg, color='black', linewidth=2, marker='x')

# Add error bars for the average
plt.fill_between(x, avg - std_dev, avg + std_dev, color='gray', alpha=0.3)

# Add labels and title
plt.title('Runs with Average and Variance')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid()

# Show the plot
plt.show()

