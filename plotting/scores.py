#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:27:27 2024

@author: mateo-drr
"""
import numpy as np
import matplotlib.pyplot as plt

def plotAllscores(data):
    # Find the maximum length of the runs
    max_length = max(len(run) for run in data)
    
    # Pad the shorter runs with NaN
    #padded_data = [run + [np.nan] * (max_length - len(run)) for run in data]
    padded_data = [run + [min(run)] * (max_length - len(run)) for run in data]
    
    # Convert to a NumPy array
    data_array = np.array(padded_data)
    
    # Calculate the average and standard deviation, ignoring NaNs
    avg = np.nanmean(data_array, axis=0)
    std_dev = np.nanstd(data_array, axis=0)
    
    # Create an x-axis for the runs
    x = np.arange(max_length)
    plt.figure(dpi=200)
    
    # Plot each run
    for i, run in enumerate(padded_data):
        plt.plot(run, alpha=0.6)
    
    # Plot the average
    plt.plot(avg, color='black', linewidth=2)
    
    # Add error bars for the average
    plt.fill_between(x, avg - std_dev, avg + std_dev, color='gray', alpha=0.3)
    
    # Add labels and title
    # plt.title('Runs with Average and Variance')
    plt.xlabel('Iterations')
    plt.ylabel('Performance')
    plt.xlim(0)
    # plt.legend()
    plt.grid()
    
    # Show the plot
    plt.show()

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

plotAllscores(data)

s0 = [0.1571, 0.1521, 0.1718, 0.0472, -0.5029, 0.1026, 0.0343, 0.1471, -0.5490, -0.5524, 0.0175, 0.0394, -0.5833, -0.6558, -0.6375, -0.5056, -0.6652]
s1 = [-0.5168, -0.3350, -0.4617, 0.1949, 0.3396, -0.3476, 0.1186, -0.4325, 0.2324, 0.1663, 0.5771, -0.3493, -0.4021, -0.2651, 0.3209, -0.4455, -0.3490, -0.3977, -0.2776, -0.3122, -0.3169, -0.4249]
s2 = [-0.6128, -0.4998, -0.4431, -0.5696, -0.4749, -0.5349]
s3 = [-0.5679, -0.4986, -0.4643, -0.5201, -0.4557, -0.5470, -0.5016, -0.4629, -0.5998]
s4 = [-0.4644, -0.1894, -0.3249, -0.2087, -0.1166, -0.2946, -0.3934, -0.4252, 0.2574, -0.0311, -0.4610, -0.4783, -0.4465, -0.5541, -0.5369, -0.4522, -0.4315, -0.4976, -0.5951, -0.3954, -0.5980, -0.5011]
s5 = [-0.5174, -0.5020, 0.0230, -0.3998, -0.3766, -0.5372, 0.0212, -0.1867, -0.5705, -0.4477, -0.5010, -0.5762, -0.4638, -0.6018, -0.3008, -0.5857, -0.5037, -0.6088, -0.5985, -0.5444, -0.1249, -0.5002]
s6 = [-0.5338, -0.6315, -0.4631, -0.5473, -0.6303, -0.5683, -0.5003, -0.5001, -0.6030, -0.4501, -0.3784, 0.4299, -0.3505, -0.6585, -0.6232, -0.2210, -0.6243, -0.4247, -0.4426, -0.4522, -0.6258, -0.4439]
s7 = [-0.5769, -0.4514, -0.4441, -0.5521, -0.6475, -0.4646, -0.5435, -0.6067, -0.5442, -0.5951, -0.5673]
s8 = [-0.5561, -0.6494, -0.6673, -0.4620, -0.6505]
s9 = [0.0039, 0.0368, 0.1046, -0.5958, -0.3620, -0.4341, 0.0022, 0.1016, 0.0070, 0.0738, -0.5938, 0.0385, -0.6139, -0.6031, -0.6241, -0.5961, 0.1154, 0.1158, -0.4479, -0.5949, -0.5544, -0.6085]

# Combine the runs into a list
data = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]

plotAllscores(data)



#%%
# Combine the runs into a list
data = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]

# Calculate cumulative scores
cumulative_scores = [np.cumsum(np.pad(run, (0, max_length - len(run)), 'constant', constant_values=np.nan)) for run in data]

# Create an x-axis based on the maximum length
max_length = max(len(run) for run in data)
x = np.arange(max_length)

# Convert to a 2D array to compute mean and std deviation
cumulative_array = np.array(cumulative_scores)

# Calculate the mean and standard deviation ignoring NaN values
mean_cumsum = np.nanmean(cumulative_array, axis=0)
std_dev_cumsum = np.nanstd(cumulative_array, axis=0)

plt.figure(dpi=200)

# Plot the mean cumulative sum
plt.plot(x, mean_cumsum, marker='o', color='black', label='Mean Cumulative Sum', linewidth=2)

# Plot cumulative scores
for cum_score in cumulative_scores:
    plt.plot(x, cum_score, marker='o', alpha=0.6)
    
# Add shaded error bars for standard deviation
plt.fill_between(x, mean_cumsum - std_dev_cumsum, mean_cumsum + std_dev_cumsum, color='gray', alpha=0.2, label='±1 Std Dev')


# Add labels and title
# plt.title('Cumulative Scores of Runs')

plt.xlabel('Iteration')
plt.ylabel('Cumulative Score')
plt.grid()
plt.show()

# Combine the runs into a list
data = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]

# Create a box plot for each run
plt.figure(figsize=(12, 6),dpi=200)
plt.boxplot(data)

# Add labels and title
plt.xlabel('Run Index')
plt.ylabel('Score')
plt.xticks(ticks=np.arange(1, len(data) + 1), labels=np.arange(len(data)))  # Set x-ticks to match the run indices
plt.grid()

# Show the plot
plt.show()


###############################################################################
###############################################################################

s0 = [0.1571, 0.1521, 0.1718, 0.0472, -0.5029, 0.1026, 0.0343, 0.1471, -0.5490, -0.5524, 0.0175, 0.0394, -0.5833, -0.6558, -0.6375, -0.5056, -0.6652]
s1 = [-0.5168, -0.3350, -0.4617, 0.1949, 0.3396, -0.3476, 0.1186, -0.4325, 0.2324, 0.1663, 0.5771, -0.3493, -0.4021, -0.2651, 0.3209, -0.4455, -0.3490, -0.3977, -0.2776, -0.3122, -0.3169, -0.4249]
s2 = [-0.6128, -0.4998, -0.4431, -0.5696, -0.4749, -0.5349]
s3 = [-0.5679, -0.4986, -0.4643, -0.5201, -0.4557, -0.5470, -0.5016, -0.4629, -0.5998]
s4 = [-0.4644, -0.1894, -0.3249, -0.2087, -0.1166, -0.2946, -0.3934, -0.4252, 0.2574, -0.0311, -0.4610, -0.4783, -0.4465, -0.5541, -0.5369, -0.4522, -0.4315, -0.4976, -0.5951, -0.3954, -0.5980, -0.5011]
s5 = [-0.5174, -0.5020, 0.0230, -0.3998, -0.3766, -0.5372, 0.0212, -0.1867, -0.5705, -0.4477, -0.5010, -0.5762, -0.4638, -0.6018, -0.3008, -0.5857, -0.5037, -0.6088, -0.5985, -0.5444, -0.1249, -0.5002]
s6 = [-0.5338, -0.6315, -0.4631, -0.5473, -0.6303, -0.5683, -0.5003, -0.5001, -0.6030, -0.4501, -0.3784, 0.4299, -0.3505, -0.6585, -0.6232, -0.2210, -0.6243, -0.4247, -0.4426, -0.4522, -0.6258, -0.4439]
s7 = [-0.5769, -0.4514, -0.4441, -0.5521, -0.6475, -0.4646, -0.5435, -0.6067, -0.5442, -0.5951, -0.5673]
s8 = [-0.5561, -0.6494, -0.6673, -0.4620, -0.6505]
s9 = [0.0039, 0.0368, 0.1046, -0.5958, -0.3620, -0.4341, 0.0022, 0.1016, 0.0070, 0.0738, -0.5938, 0.0385, -0.6139, -0.6031, -0.6241, -0.5961, 0.1154, 0.1158, -0.4479, -0.5949, -0.5544, -0.6085]

# Combine the runs into a list
data = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]

# Calculate cumulative scores
cumulative_scores = [np.cumsum(np.pad(run, (0, max_length - len(run)), 'constant', constant_values=np.nan)) for run in data]

# Create an x-axis based on the maximum length
max_length = max(len(run) for run in data)
x = np.arange(max_length)

# Convert to a 2D array to compute mean and std deviation
cumulative_array = np.array(cumulative_scores)

# Calculate the mean and standard deviation ignoring NaN values
mean_cumsum = np.nanmean(cumulative_array, axis=0)
std_dev_cumsum = np.nanstd(cumulative_array, axis=0)

plt.figure(dpi=200)

# Plot the mean cumulative sum
plt.plot(x, mean_cumsum, marker='o', color='black', label='Mean Cumulative Sum', linewidth=2)

# Plot cumulative scores
for cum_score in cumulative_scores:
    plt.plot(x, cum_score, marker='o', alpha=0.6)
    
# Add shaded error bars for standard deviation
plt.fill_between(x, mean_cumsum - std_dev_cumsum, mean_cumsum + std_dev_cumsum, color='gray', alpha=0.2, label='±1 Std Dev')


# Add labels and title
# plt.title('Cumulative Scores of Runs')

plt.xlabel('Iteration')
plt.ylabel('Cumulative Score')
plt.grid()
plt.show()


# Combine the runs into a list
data = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]

# Create a box plot for each run
plt.figure(figsize=(12, 6),dpi=200)
plt.boxplot(data)

# Add labels and title
plt.xlabel('Run Index')
plt.ylabel('Score')
plt.xticks(ticks=np.arange(1, len(data) + 1), labels=np.arange(len(data)))  # Set x-ticks to match the run indices
plt.grid()

# Show the plot
plt.show()

