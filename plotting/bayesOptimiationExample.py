#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:58:27 2024

@author: mateo-drr
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(0)

def gaussian(x, mean, variance):
    return np.exp(-((x - mean) ** 2) / (2 * variance))

# Parameters for Gaussian
mean = 0  
variance = 10  

# Generate data
x = np.linspace(-20, 20, 100)
y = gaussian(x, mean, variance)

# Add noise
noise = 0.1 * np.random.normal(size=y.shape)
y_noisy = y + noise

# Randomly sample 7 x-values from the range -20 to 20
fixed_eval_points_x = np.random.uniform(-20, 20, 7)
eval_points_y = gaussian(fixed_eval_points_x, mean, variance) + noise[np.searchsorted(x, fixed_eval_points_x)]

# Fit the Gaussian Process model
kernel = C(1.0) * RBF(length_scale=3.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(fixed_eval_points_x.reshape(-1, 1), eval_points_y)

# Predict over a range of x-values for the GP model
x_pred = np.linspace(-20, 20, 200).reshape(-1, 1)
y_pred, sigma = gp.predict(x_pred, return_std=True)

# Plot the noisy data and GP prediction
plt.figure(dpi=300)
plt.plot(x, y_noisy, color='orange', linewidth=3, label='Noisy Data')
plt.scatter(fixed_eval_points_x, eval_points_y, color='black', edgecolor='white', s=100, zorder=5, label='Sampled Points')
plt.plot(x_pred, y_pred, color='blue', linewidth=2, label='GP Mean Prediction')
plt.fill_between(x_pred.flatten(), 
                 y_pred - 1.96 * sigma, 
                 y_pred + 1.96 * sigma, 
                 color='lightblue', alpha=0.5, label='Confidence Interval (95%)')

plt.xlabel('Degrees')
plt.ylabel('Cost')
plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1))
plt.grid()
plt.show()

# Lower Confidence Bound acquisition function
def lower_confidence_bound(x, gp, kappa=1.96):
    mu, sigma = gp.predict(x.reshape(-1, 1), return_std=True)
    lcb = mu - kappa * sigma
    return lcb

# Predict over a range of x-values for the acquisition function
x_acq = np.linspace(-20, 20, 200)
lcb = lower_confidence_bound(x_acq, gp)

# Plot the Lower Confidence Bound
plt.figure(dpi=300)
plt.plot(x_acq, lcb, color='green', linewidth=2)

plt.xlabel('Degrees')
plt.ylabel('Lower Confidence Bound')
plt.grid()
plt.show()
