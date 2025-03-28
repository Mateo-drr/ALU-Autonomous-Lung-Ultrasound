import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-1, 1, 41)
sigma = 0.3  # Adjust sigma for the desired smoothness
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
x_values = list(range(-20, 21))
plt.plot(x_values, gaussian_array, color='#ff7f0e')
plt.axvline(-3.25, linestyle='--')
plt.axvline(3.25, linestyle='--')
plt.axhline(0.86, linestyle='--')
plt.xlabel('Degrees')
plt.ylabel('Cost')
plt.grid()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the grid for the rectangular plane with the original range for x and y
x = np.linspace(-5, 5, 100)
y = np.linspace(0, 40, 100)
X, Y = np.meshgrid(x, y)

# Define the Gaussian function along one axis (e.g., X)
Z = np.exp(-0.5*((X/1.3)**2))

# Plotting
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Set custom ticks for the x-axis to display from -20 to 20
ax.set_xticks(np.linspace(-5, 5, 5))  # Keep the original data range for x
ax.set_xticklabels(np.linspace(-20, 20, 5))  # Set the labels to go from -20
ax.set_zticks(np.linspace(0, 1, 5))  # Keep the original data range for x
ax.set_zticklabels(np.linspace(0, -1, 5))  # Set the labels to go from -20 to 20

# Labels and title
ax.set_xlabel('Degrees')
ax.set_ylabel('Grid columns')
ax.set_zlabel('Cost')
#ax.set_title('3D Gaussian Ridge')
ax.set_box_aspect(None, zoom=0.9)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.filters import laplace
import byble as byb



# Assuming 'byb' is your envelope calculation method and 'image_62' is your image
image_41 = justimgs[41][0]
# Calculate envelope image
envelope_img = byb.envelope(image_41)

# Apply log scaling on envelope image
log_envelope_img = 20 * np.log10(np.abs(envelope_img) + 1)

# Calculate Laplacian of the raw image and log scale it
laplacian_raw = laplace(image_41)
log_laplacian_raw = 20 * np.log10(np.abs(laplacian_raw) + 1)

# Calculate Laplacian of the envelope image and apply log scaling to it
laplacian_env = laplace(envelope_img)
log_laplacian_env = 20 * np.log10(np.abs(laplacian_env) + 1)

# Calculate Laplacian of the envelope + log image and log scale it
laplacian_env_log = laplace(log_envelope_img)
log_laplacian_env_log = 20 * np.log10(np.abs(laplacian_env_log) + 1)

# Find global min and max across all images
vmin = min(log_envelope_img.min(), log_laplacian_raw.min(), log_laplacian_env.min(),
           log_laplacian_env_log.min())
vmax = max(log_envelope_img.max(), log_laplacian_raw.max(), log_laplacian_env.max(),
           log_laplacian_env_log.max())

# Plotting all four images
fig, ax = plt.subplots(1, 4, figsize=(24, 6))

# Plot envelope image with log scaling
im0 = ax[0].imshow(log_envelope_img, aspect='auto', vmin=vmin, vmax=vmax)
ax[0].set_title("(A) US image: Envelope + Log Scaling", fontsize=18, pad=20)
#ax[0].axis('off')  # Remove axis
ax[0].tick_params(axis='both', labelsize=16)
ax[0].set_xlabel("Width [px]", fontsize=16)
ax[0].set_ylabel("Depth [px]", fontsize=16)

# Plot Laplacian of raw image with log scaling
im1= ax[1].imshow(log_laplacian_raw, aspect='auto', vmin=vmin, vmax=vmax)
ax[1].set_title("(B) Laplacian of Filtered RF data", fontsize=18, pad=20)
ax[1].set_yticks([])
ax[1].set_xlabel("Width [px]", fontsize=16)
ax[1].tick_params(axis='both', labelsize=16)
#ax[1].axis('off')  # Remove axis

# Plot Laplacian of envelope image with log scaling
im2 = ax[2].imshow(log_laplacian_env, aspect='auto', vmin=vmin, vmax=vmax)
ax[2].set_title("(C) Laplacian of Envelope", fontsize=18, pad=20)
ax[2].set_xlabel("Width [px]", fontsize=16)
ax[2].tick_params(axis='both', labelsize=16)
ax[2].set_yticks([])
#ax[2].axis('off')  # Remove axis

# Plot Laplacian of envelope + log image with log scaling
im3 = ax[3].imshow(log_laplacian_env_log, aspect='auto', vmin=vmin, vmax=vmax)
ax[3].set_title("(D) Laplacian of Envelope + Log Scaling", fontsize=18, pad=20)
ax[3].set_xlabel("Width [px]", fontsize=16)
ax[3].tick_params(axis='both', labelsize=16)
ax[3].set_yticks([])
#ax[3].axis('off')  # Remove axis

plt.subplots_adjust(right=0.85)  # Adjust this to your liking

c0 = fig.colorbar(im0, ax=ax[0])
c1 = fig.colorbar(im1, ax=ax[1])
c2 = fig.colorbar(im2, ax=ax[2])
c3 = fig.colorbar(im3, ax=ax[3])
c3.set_label("Intensity [dB]", fontsize=16)


c0.ax.tick_params(labelsize=16)
c1.ax.tick_params(labelsize=16)
c2.ax.tick_params(labelsize=16)
c3.ax.tick_params(labelsize=16)


plt.tight_layout()
plt.show()

# Assuming 'byb' is your envelope calculation method and 'image_62' is your image
image_41 = justimgs[62][0]
# Calculate envelope image
envelope_img = byb.envelope(image_41)

# Apply log scaling on envelope image
log_envelope_img = 20 * np.log10(np.abs(envelope_img) + 1)

# Calculate Laplacian of the raw image and log scale it
laplacian_raw = laplace(image_41)
log_laplacian_raw = 20 * np.log10(np.abs(laplacian_raw) + 1)

# Calculate Laplacian of the envelope image and apply log scaling to it
laplacian_env = laplace(envelope_img)
log_laplacian_env = 20 * np.log10(np.abs(laplacian_env) + 1)

# Calculate Laplacian of the envelope + log image and log scale it
laplacian_env_log = laplace(log_envelope_img)
log_laplacian_env_log = 20 * np.log10(np.abs(laplacian_env_log) + 1)

# Find global min and max across all images
vmin = min(log_envelope_img.min(), log_laplacian_raw.min(), log_laplacian_env.min(),
           log_laplacian_env_log.min())
vmax = max(log_envelope_img.max(), log_laplacian_raw.max(), log_laplacian_env.max(),
           log_laplacian_env_log.max())

# Plotting all four images
fig, ax = plt.subplots(1, 4, figsize=(24, 6))

# Plot envelope image with log scaling
im0 = ax[0].imshow(log_envelope_img, aspect='auto', vmin=vmin, vmax=vmax)
ax[0].set_title("(A) US image: Envelope + Log Scaling", fontsize=18, pad=20)
#ax[0].axis('off')  # Remove axis
ax[0].tick_params(axis='both', labelsize=16)
ax[0].set_xlabel("Width [px]", fontsize=16)
ax[0].set_ylabel("Depth [px]", fontsize=16)

# Plot Laplacian of raw image with log scaling
im1= ax[1].imshow(log_laplacian_raw, aspect='auto', vmin=vmin, vmax=vmax)
ax[1].set_title("(B) Laplacian of Filtered RF data", fontsize=18, pad=20)
ax[1].set_yticks([])
ax[1].set_xlabel("Width [px]", fontsize=16)
ax[1].tick_params(axis='both', labelsize=16)
#ax[1].axis('off')  # Remove axis

# Plot Laplacian of envelope image with log scaling
im2 = ax[2].imshow(log_laplacian_env, aspect='auto', vmin=vmin, vmax=vmax)
ax[2].set_title("(C) Laplacian of Envelope", fontsize=18, pad=20)
ax[2].set_xlabel("Width [px]", fontsize=16)
ax[2].tick_params(axis='both', labelsize=16)
ax[2].set_yticks([])
#ax[2].axis('off')  # Remove axis

# Plot Laplacian of envelope + log image with log scaling
im3 = ax[3].imshow(log_laplacian_env_log, aspect='auto', vmin=vmin, vmax=vmax)
ax[3].set_title("(D) Laplacian of Envelope + Log Scaling", fontsize=18, pad=20)
ax[3].set_xlabel("Width [px]", fontsize=16)
ax[3].tick_params(axis='both', labelsize=16)
ax[3].set_yticks([])
#ax[3].axis('off')  # Remove axis

plt.subplots_adjust(right=0.85)  # Adjust this to your liking

c0 = fig.colorbar(im0, ax=ax[0])
c1 = fig.colorbar(im1, ax=ax[1])
c2 = fig.colorbar(im2, ax=ax[2])
c3 = fig.colorbar(im3, ax=ax[3])
c3.set_label("Intensity [dB]", fontsize=16)

c0.ax.tick_params(labelsize=16)
c1.ax.tick_params(labelsize=16)
c2.ax.tick_params(labelsize=16)
c3.ax.tick_params(labelsize=16)


plt.tight_layout()
plt.show()

#%%

step = 100
i, j = 1800, 3500
# i, j = 2000, 2800
log=False
hilb=True

# Step 1: Extract and process images (replace with your actual image data)
img82 = justimgs[82][0][i:j]
img102 = justimgs[102][0][i:j]
img122 = justimgs[122][0][i:j]

raw82=laplace(img82)
raw102=laplace(img102)
raw122=laplace(img122)

if hilb:
    raw82=laplace(byb.envelope(img82))
    raw102=laplace(byb.envelope(img102))
    raw122=laplace(byb.envelope(img122))
    
    if log:
        raw82=laplace(byb.logS(byb.envelope(img82)))
        raw102=laplace(byb.logS(byb.envelope(img102)))
        raw122=laplace(byb.logS(byb.envelope(img122)))


variance82=np.var(raw82)
variance102=np.var(raw102)
variance122=np.var(raw122)

print(f"Variance of img82: {variance82:.4f}")
print(f"Variance of img102: {variance102:.4f}")
print(f"Variance of img122: {variance122:.4f}")

print(f"Variance of img82: {variance82:.2e}")
print(f"Variance of img102: {variance102:.2e}")
print(f"Variance of img122: {variance122:.2e}")




#%%

import matplotlib.pyplot as plt
import numpy as np

# Data from the runs
data = {
    'x': [-22.6943, -23.4069, -22.9048, -22.4599, -23.9795, -22.7713, -22.3733, -21.9764, -21.2108, -22.5592],
    'y': [27.9227, 27.8747, 27.9391, 27.5971, 28.2443, 27.9483, 27.9361, 27.9710, 27.6780, 27.8843],
    'z': [80.9864, 80.6538, 80.8416, 80.7635, 80.8602, 80.8466, 80.9188, 80.9020, 79.9439, 79.9830],
    'r': [1.1411, -0.2701, -0.0687, 21.3344, -2.1314, -2.0378, -1.1184, 6.3373, 0.5188, -0.9248],
    'p': [-4.7827, -19.6109, 17.3122, 7.5390, -3.9621, 2.3057, 1.0709, -18.0003, -10.4953, 9.8994],
    'yaw': [10.8914, -0.2422, 0.3055, 12.3669, -17.9696, 87.4065, 92.9994, 70.2158, 75.5142, 70.4177],
    'score': [-0.6652, -0.5168, -0.6128, -0.5998, -0.5980, -0.6088, -0.6585, -0.6475, -0.6673, -0.6241],
}

best = {
    "x": [-22.6943, -23.4069, -22.6773, -22.3200, -24.0652,
        -22.2640, -22.2356, -21.5767, -22.5753, -22.7040],
    "y": [27.9227, 27.8747, 27.9491, 27.6171, 28.2443,
        27.9283, 27.9505, 27.9710, 27.6780, 27.8843],
    "z": [80.9864, 80.6538, 80.8416, 80.7635, 80.8602,
        80.8466, 80.9188, 80.9020, 79.9393, 79.9830],
    "r": [1.1411, -0.2701, 0.1990, 6.3128, -2.4568,
        -2.0378, -1.1184, -0.4784, -1.0711, -0.9248],
    "p": [-4.7827, -19.6109, 15.2419, 8.4278, 0.0686,
        1.8078, 0.2972, -6.0367, -9.2489, 0.2458],
    "yaw": [10.8914, -0.2422, 0.8147, -0.9647, -19.2888,
        91.7755, 100.1870, 107.6207, 84.3379, 107.8478],
    "score": [-0.6652, -0.5168, -0.5696, -0.4986, -0.5369,
        -0.5037, -0.4501, -0.5435, -0.6494, -0.5544]
}

new = {
    "x": [-22.0331, -23.4493, -22.6494, -22.3200, -23.6672, -23.2488, -23.4050, -21.5452, -22.5753, -22.9237],
    "y": [27.9227, 27.8847, 27.9291, 27.6171, 28.2443, 27.9483, 27.9361, 27.9710, 27.6780, 27.8843],
    "z": [80.9864, 80.6538, 80.8416, 80.7635, 80.8602, 80.8466, 80.9188, 80.9020, 79.9393, 79.9830],
    "r": [3.5639, -1.1208, 8.0742, 6.3128, -18.7470, -17.5509, -1.1184, 4.7469, -1.0711, -2.5254],
    "p": [-10.2495, -20.3572, 7.8034, 8.4278, 7.0080, 1.6012, 1.3095, -17.5219, -9.2489, 1.3351],
    "yaw": [12.8668, -0.4298, 7.9732, -0.9647, -2.9782, 87.7268, 70.7245, 96.2415, 84.3379, 94.4198],
    "score": [-0.5490, -0.3122, -0.4431, -0.4986, -0.4610, -0.3008, -0.6258, -0.5673, -0.6494, 0.0368]
}



# Convert to numpy arrays for easier calculations
r_vals = np.array(data['r'])
p_vals = np.array(data['p'])
yaw_vals = np.array(data['yaw'])
scores = np.array(data['score'])

# Calculate mean and standard deviation for each angle
mean_r, std_r = np.mean(r_vals), np.std(r_vals)
mean_p, std_p = np.mean(p_vals), np.std(p_vals)
mean_yaw, std_yaw = np.mean(yaw_vals), np.std(yaw_vals)

# Create plots
fig, axs = plt.subplots(3, 1, figsize=(10, 12), dpi=300)

# Roll vs Score
axs[0].scatter(r_vals, scores, color='b', label='Data Points')
axs[0].errorbar(mean_r, np.mean(scores), xerr=std_r, fmt='o', color='black', label='Mean ± Std')
axs[0].set_title('Roll (r) vs Score')
axs[0].set_xlabel('Roll (r)')
axs[0].set_ylabel('Score')
axs[0].grid()
axs[0].legend()

# Pitch vs Score
axs[1].scatter(p_vals, scores, color='g', label='Data Points')
axs[1].errorbar(mean_p, np.mean(scores), xerr=std_p, fmt='o', color='black', label='Mean ± Std')
axs[1].set_title('Pitch (p) vs Score')
axs[1].set_xlabel('Pitch (p)')
axs[1].set_ylabel('Score')
axs[1].grid()
axs[1].legend()

# Yaw vs Score
axs[2].scatter(yaw_vals, scores, color='r', label='Data Points')
axs[2].errorbar(mean_yaw, np.mean(scores), xerr=std_yaw, fmt='o', color='black', label='Mean ± Std')
axs[2].set_title('Yaw (y) vs Score')
axs[2].set_xlabel('Yaw (y)')
axs[2].set_ylabel('Score')
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()

#%%###############################################################################
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Assuming img82, img102, and img122 are your images
img82, img102, img122 = justimgs[82][0], justimgs[102][0], justimgs[122][0]

# Compute the images and calculate global min and max
images = [20 * np.log10(np.abs(byb.envelope(img)) + 1) for img in [img82, img102, img122]]
global_min = min(img.min() for img in images)
global_max = max(img.max() for img in images)

# Create a gridspec layout
fig = plt.figure(figsize=(15, 5), dpi=200)
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.1], wspace=0.05)  # Add extra column for colorbar

# Display each image
ang = [-20, 0, 20]
axs = [fig.add_subplot(gs[i]) for i in range(3)]  # Create three subplots
for i, (ax, img, angle) in enumerate(zip(axs, images, ang)):
    img_plot = ax.imshow(img, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
    ax.axvline(x=128//2, color='red', linestyle='--')  # Red line at x = 128//2
    
    if i == 0:
        ax.yaxis.set_visible(True)
        ax.set_ylabel('Depth [px]', fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
    else:
        ax.yaxis.set_visible(False)
    ax.set_title(f"{angle}", fontsize=16)
    # ax.set_xticks([])
    ax.set_xlabel("Width [px]", fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xticks(np.arange(0,129, 30))

# Add a shared colorbar to the rightmost position
cbar_ax = fig.add_subplot(gs[-1])  # Allocate the last slot to the colorbar
cbar = fig.colorbar(img_plot, cax=cbar_ax)
cbar.set_label('Intensity [dB]', fontsize=18, labelpad=10)
cbar.ax.tick_params(labelsize=16)

# Add a single x-axis label for the whole figure
plt.figtext(0.5, 0.98, 'Degrees', ha='center', va='center', fontsize=18)  # Centered label
plt.show()

# Plot the column of each image at the position of the red line in subplots
line_pos = 128 // 2
line82 = img82[:, line_pos]
line102 = img102[:, line_pos]
line122 = img122[:, line_pos]

# Create a new figure for the line plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Plot each line in separate subplots
axs[0].plot(line82, color='blue')
axs[1].plot(line102, color='orange')
axs[2].plot(line122, color='green')

# Set labels and titles for each subplot
i=0
for ax in axs:
    ax.set_xlabel('Depth [px]', fontsize=18)
    if i == 0:
        ax.set_ylabel('Amplitude', fontsize=18)


    ax.grid()
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick font size
    ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick font size
    #ax.legend()
    i+=1

# Set the title for the overall figure
#plt.suptitle('Amplitude at Depth 128//2', fontsize=16)

plt.tight_layout()
plt.show()

# Now create another subplot for the processed lines using byb.envelope
line82_processed = byb.envelope(img82)[:, line_pos]
line102_processed = byb.envelope(img102)[:, line_pos]
line122_processed = byb.envelope(img122)[:, line_pos]

# Create a new figure for the processed line plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Plot each processed line in separate subplots
axs[0].plot(line82_processed, color='blue')
axs[1].plot(line102_processed, color='orange')
axs[2].plot(line122_processed, color='green')

# Set labels and titles for each subplot
i=0
for ax in axs:
    ax.set_xlabel('Depth [px]', fontsize=18)
    if i == 0:
        ax.set_ylabel('Envelope Amplitude', fontsize=18)


    ax.grid()
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick font size
    ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick font size
    #ax.legend()
    i+=1

# Set the title for the overall figure
#plt.suptitle('Processed Amplitude at Depth 128//2', fontsize=16)

plt.tight_layout()
plt.show()

# Now create another subplot for the lines after byb.envelope and after 20 * log10(abs(...)+1)
line82_final = 20 * np.log10(np.abs(byb.envelope(img82)) + 1)[:, line_pos]
line102_final = 20 * np.log10(np.abs(byb.envelope(img102)) + 1)[:, line_pos]
line122_final = 20 * np.log10(np.abs(byb.envelope(img122)) + 1)[:, line_pos]

# Create a new figure for the final processed line plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Plot each final processed line in separate subplots
axs[0].plot(line82_final, color='blue')
axs[1].plot(line102_final, color='orange')
axs[2].plot(line122_final, color='green')

# Set labels and titles for each subplot
i=0
for ax in axs:
    ax.set_xlabel('Depth [px]', fontsize=18)
    if i == 0:
        ax.set_ylabel('Log Amplitude [dB]', fontsize=18)


    ax.grid()
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick font size
    ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick font size
    #ax.legend()
    i+=1

# Set the title for the overall figure
#plt.suptitle('Final Processed Amplitude at Depth 128//2', fontsize=16)

plt.tight_layout()
plt.show()


#%%#############################
#THIS PLOT IS FOR THE MEAN OF THE LINES, THE ONE ABOVE IS FOR A SINGLE LINE
import matplotlib.pyplot as plt
import numpy as np

# Assuming img82, img102, and img122 are your images
img82, img102, img122 = justimgs[82][0], justimgs[102][0], justimgs[122][0]

# Calculate the mean along axis 1 for each image
mean_img82 = np.mean(img82, axis=1)
mean_img102 = np.mean(img102, axis=1)
mean_img122 = np.mean(img122, axis=1)

# Create a subplot for the raw means
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Plot each raw mean in separate subplots
axs[0].plot(mean_img82, color='blue')
axs[1].plot(mean_img102, color='orange')
axs[2].plot(mean_img122, color='green')

# Set labels and titles for each subplot
i=0
for ax in axs:
    ax.set_xlabel('Depth [px]', fontsize=18)
    if i == 0:
        ax.set_ylabel('Mean Amplitude', fontsize=18)


    ax.grid()
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick font size
    ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick font size
    #ax.legend()
    i+=1

#plt.suptitle('Raw Mean Amplitude along Axis 1', fontsize=16)
plt.tight_layout()
plt.show()

# Calculate the mean after applying byb.envelope
mean_img82_envelope = np.mean(byb.envelope(img82), axis=1)
mean_img102_envelope = np.mean(byb.envelope(img102), axis=1)
mean_img122_envelope = np.mean(byb.envelope(img122), axis=1)

# Create a subplot for the means after byb.envelope
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Plot each mean after envelope in separate subplots
axs[0].plot(mean_img82_envelope, color='blue')
axs[1].plot(mean_img102_envelope, color='orange')
axs[2].plot(mean_img122_envelope, color='green')

# Set labels and titles for each subplot
i=0
for ax in axs:
    ax.set_xlabel('Depth [px]', fontsize=18)
    if i == 0:
        ax.set_ylabel('Mean Envelope Amplitude', fontsize=18)


    ax.grid()
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick font size
    ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick font size
    #ax.legend()
    i+=1

#plt.suptitle('Mean Amplitude after byb.envelope', fontsize=16)
plt.tight_layout()
plt.show()

# Calculate the mean after applying 20 * log10(np.abs(...) + 1)
mean_img82_final = np.mean(20 * np.log10(np.abs(byb.envelope(img82)) + 1), axis=1)
mean_img102_final = np.mean(20 * np.log10(np.abs(byb.envelope(img102)) + 1), axis=1)
mean_img122_final = np.mean(20 * np.log10(np.abs(byb.envelope(img122)) + 1), axis=1)

# Create a subplot for the means after byb.envelope + log
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Plot each mean after envelope in separate subplots
axs[0].plot(mean_img82_final, color='blue')
axs[1].plot(mean_img102_final, color='orange')
axs[2].plot(mean_img122_final, color='green')

# Set labels and titles for each subplot
i=0
for ax in axs:
    ax.set_xlabel('Depth [px]', fontsize=18)
    if i == 0:
        ax.set_ylabel('Mean Log Amplitude [dB]', fontsize=18)


    ax.grid()
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick font size
    ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick font size
    #ax.legend()
    i+=1


#plt.suptitle('Mean Amplitude after byb.envelope + 20log10(abs(...) + 1)', fontsize=16)
plt.tight_layout()
plt.show()

#%%#######################

import matplotlib.pyplot as plt
import numpy as np

# strt,end = 2000,2800
strt,end = 1800,3500
# Assuming img82, img102, and img122 are your images
img82, img102, img122 = justimgs[82][0], justimgs[102][0], justimgs[122][0]
#img82, img102, img122 = img82[2000:2800], img102[2000:2800], img122[2000:2800]
img82, img102, img122 = img82[strt:end], img102[strt:end], img122[strt:end]
# Calculate the mean along axis 1 for each image
mean_img82 = np.mean(img82, axis=1)
mean_img102 = np.mean(img102, axis=1)
mean_img122 = np.mean(img122, axis=1)

# Create a subplot for the raw means
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Plot each raw mean in separate subplots
axs[0].plot(mean_img82, color='blue')
axs[1].plot(mean_img102, color='orange')
axs[2].plot(mean_img122, color='green')

# Set labels and titles for each subplot
i=0
for ax in axs:
    ax.set_xlabel('Depth [px]', fontsize=18)
    if i == 0:
        ax.set_ylabel('Mean Amplitude', fontsize=18)


    ax.grid()
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick font size
    ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick font size
    #ax.legend()
    i+=1

#plt.suptitle('Raw Mean Amplitude along Axis 1', fontsize=16)
plt.tight_layout()
plt.show()

# Calculate the mean after applying byb.envelope
mean_img82_envelope = np.mean(byb.envelope(img82), axis=1)
mean_img102_envelope = np.mean(byb.envelope(img102), axis=1)
mean_img122_envelope = np.mean(byb.envelope(img122), axis=1)

# Create a subplot for the means after byb.envelope
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Plot each mean after envelope in separate subplots
axs[0].plot(mean_img82_envelope, color='blue')
axs[1].plot(mean_img102_envelope, color='orange')
axs[2].plot(mean_img122_envelope, color='green')

i=0
for ax in axs:
    ax.set_xlabel('Depth [px]', fontsize=18)
    if i == 0:
        ax.set_ylabel('Mean Envelope Amplitude', fontsize=18)
    ax.grid()
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick font size
    ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick font size
    #ax.legend()
    i+=1

#plt.suptitle('Mean Amplitude after byb.envelope', fontsize=16)
plt.tight_layout()
plt.show()

# Calculate the mean after applying 20 * log10(np.abs(...) + 1)
mean_img82_final = np.mean(20 * np.log10(np.abs(byb.envelope(img82)) + 1), axis=1)
mean_img102_final = np.mean(20 * np.log10(np.abs(byb.envelope(img102)) + 1), axis=1)
mean_img122_final = np.mean(20 * np.log10(np.abs(byb.envelope(img122)) + 1), axis=1)

# Create a subplot for the means after byb.envelope + log
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# Plot each final mean in separate subplots
axs[0].plot(mean_img82_final, color='blue')
axs[1].plot(mean_img102_final, color='orange')
axs[2].plot(mean_img122_final, color='green')

i=0
for ax in axs:
    ax.set_xlabel('Depth [px]', fontsize=18)
    if i == 0:
        ax.set_ylabel('Mean Log Amplitude [dB]', fontsize=18)
    ax.grid()
    # Increase the font size of the tick labels on both axes
    ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick font size
    ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick font size
    #ax.legend()
    i+=1

#plt.suptitle('Mean Amplitude after byb.envelope + 20log10(abs(...) + 1)', fontsize=16)
plt.tight_layout()
plt.show()

#%%########################## IMAGES CROP ZONE

# Assuming img82, img102, and img122 are your images
img82, img102, img122 = justimgs[82][0], justimgs[102][0], justimgs[122][0]

# Compute the images and calculate global min and max
images = [20 * np.log10(np.abs(byb.envelope(img)) + 1) for img in [img82, img102, img122]]
global_min = min(img.min() for img in images)
global_max = max(img.max() for img in images)

# Create a gridspec layout for the images and colorbar
fig = plt.figure(figsize=(15, 5), dpi=200)
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.1], wspace=0.05)  # Reserve space for colorbar

# Display each image with shared color limits
ang = [-20, 0, 20]
axs = [fig.add_subplot(gs[i]) for i in range(3)]
for i, (ax, img, angle) in enumerate(zip(axs, images, ang)):
    img_plot = ax.imshow(img, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
    ax.axhline(y=1800, color='red', linestyle='--', linewidth=2.5)  # Red horizontal line
    ax.axhline(y=3500, color='blue', linestyle='--', linewidth=2.5)  # Blue horizontal line

    if i == 0:
        ax.yaxis.set_visible(True)
        ax.set_ylabel('Depth [px]', fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
    else:
        ax.yaxis.set_visible(False)

    # ax.set_xticks([])
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel('Width [px]', fontsize=18)
    ax.set_title(f"{angle}", fontsize=16)
    ax.set_xticks(np.arange(0, 129, 30))

# Add a shared colorbar to the right
cbar_ax = fig.add_subplot(gs[-1])  # Last slot for the colorbar
cbar = fig.colorbar(img_plot, cax=cbar_ax)
cbar.set_label('Intensity [dB]', fontsize=18, labelpad=10)
cbar.ax.tick_params(labelsize=16)

# Add a single x-axis label for the whole figure
plt.figtext(0.495, 1, 'Degrees', ha='center', va='center', fontsize=18)

plt.show()


#CROP FOR NO MEAT IMAGES IS 1800 3500 Aug0

#%%###############################
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

step = 100
# i, j = 1800, 3500
i, j = 2000, 2800
log=True

# Step 1: Extract and process images (replace with your actual image data)
img82 = justimgs[82][0][i:j]
img102 = justimgs[102][0][i:j]
img122 = justimgs[122][0][i:j]

# Step 2: Calculate the mean envelopes for each image
if log:
    mean_img82_envelope = np.mean(byb.logS(byb.envelope(img82)), axis=1)
    mean_img102_envelope = np.mean(byb.logS(byb.envelope(img102)), axis=1)
    mean_img122_envelope = np.mean(byb.logS(byb.envelope(img122)), axis=1)
else:
    mean_img82_envelope = np.mean(byb.envelope(img82), axis=1)
    mean_img102_envelope = np.mean(byb.envelope(img102), axis=1)
    mean_img122_envelope = np.mean(byb.envelope(img122), axis=1)

mean_img82_envelope = resize(mean_img82_envelope, [800], anti_aliasing=True)
mean_img102_envelope = resize(mean_img102_envelope, [800], anti_aliasing=True)
mean_img122_envelope = resize(mean_img122_envelope, [800], anti_aliasing=True)

# Step 3: Normalize each envelope to create probability distributions
probabilities_82 = mean_img82_envelope / np.sum(mean_img82_envelope)
probabilities_102 = mean_img102_envelope / np.sum(mean_img102_envelope)
probabilities_122 = mean_img122_envelope / np.sum(mean_img122_envelope)

# Create corresponding x values for all images
x82 = np.arange(len(probabilities_82))
x102 = np.arange(len(probabilities_102))
x122 = np.arange(len(probabilities_122))

# Calculate the Mean and Variance for each image
mean82 = np.sum(x82 * probabilities_82)
variance82 = np.sum((x82 - mean82)**2 * probabilities_82)

mean102 = np.sum(x102 * probabilities_102)
variance102 = np.sum((x102 - mean102)**2 * probabilities_102)

mean122 = np.sum(x122 * probabilities_122)
variance122 = np.sum((x122 - mean122)**2 * probabilities_122)

# Step 4: Plot the distributions in side-by-side subplots
plt.figure(figsize=(24, 6), dpi=200)  # Increased width from 18 to 24

# Subplot 1: img82
plt.subplot(1, 3, 1)
plt.plot(x82, probabilities_82, marker='o', linestyle='-', color='blue', markersize=5, label=f'Probability Distribution (-20º)')
plt.xticks(np.arange(0, len(x82), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=mean82, color='red', linestyle='--', label=f'Mean (μ) = {mean82:.2f}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Subplot 2: img102
plt.subplot(1, 3, 2)
plt.plot(x102, probabilities_102, marker='o', linestyle='-', color='orange', markersize=5, label=f'Probability Distribution (0º)')
plt.xticks(np.arange(0, len(x102), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
#plt.ylabel('Probability', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=mean102, color='red', linestyle='--', label=f'Mean (μ) = {mean102:.2f}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Subplot 3: img122
plt.subplot(1, 3, 3)
plt.plot(x122, probabilities_122, marker='o', linestyle='-', color='green', markersize=5, label=f'Probability Distribution (+20º)')
plt.xticks(np.arange(0, len(x122), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
#plt.ylabel('Probability', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=mean122, color='red', linestyle='--', label=f'Mean (μ) = {mean122:.2f}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Adjust layout and display the side-by-side plots
plt.tight_layout()
plt.show()

print(f"Variance of img82: {variance82:.4f}")
print(f"Variance of img102: {variance102:.4f}")
print(f"Variance of img122: {variance122:.4f}")

#%%############################### np var
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

step = 100
# i, j = 1800, 3500
i, j = 2000, 2800
log=False
maxnorm=False
minmax=True
cropAfter=True #use only with minmax

# Step 1: Extract and process images (replace with your actual image data)
if cropAfter:
    img82 = justimgs[82][0]
    img102 = justimgs[102][0]
    img122 = justimgs[122][0]
else:
    img82 = justimgs[82][0][i:j]
    img102 = justimgs[102][0][i:j]
    img122 = justimgs[122][0][i:j]

# Step 2: Calculate the mean envelopes for each image
if maxnorm:
    logim82=byb.logS(byb.envelope(img82))
    logim102=byb.logS(byb.envelope(img102))
    logim122=byb.logS(byb.envelope(img122))
    logim82=logim82-np.max(logim82)
    logim102=logim102-np.max(logim102)
    logim122=logim122-np.max(logim122)
    mean_img82_envelope = np.mean(logim82, axis=1)
    mean_img102_envelope = np.mean(logim102, axis=1)
    mean_img122_envelope = np.mean(logim122, axis=1)
else:
    if minmax:
        him82=byb.envelope(img82)
        him102=byb.envelope(img102)
        him122=byb.envelope(img122)
        for k in range(him82.shape[1]):
            line = him82[:,k]
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            him82[:,k] = line 
        for k in range(him102.shape[1]):
            line = him102[:,k]
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            him102[:,k] = line 
        for k in range(him122.shape[1]):
            line = him122[:,k]
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            him122[:,k] = line 
        if cropAfter:
            him82 = him82[i:j]
            him102 = him102[i:j]
            him122 = him122[i:j]
        mean_img82_envelope = np.mean(him82, axis=1)
        mean_img102_envelope = np.mean(him102, axis=1)
        mean_img122_envelope = np.mean(him122, axis=1)
    else:
        if log:
            mean_img82_envelope = np.mean(byb.logS(byb.envelope(img82)), axis=1)
            mean_img102_envelope = np.mean(byb.logS(byb.envelope(img102)), axis=1)
            mean_img122_envelope = np.mean(byb.logS(byb.envelope(img122)), axis=1)
        else:
            mean_img82_envelope = np.mean(byb.envelope(img82), axis=1)
            mean_img102_envelope = np.mean(byb.envelope(img102), axis=1)
            mean_img122_envelope = np.mean(byb.envelope(img122), axis=1)

mean_img82_envelope = resize(mean_img82_envelope, [800], anti_aliasing=True)
mean_img102_envelope = resize(mean_img102_envelope, [800], anti_aliasing=True)
mean_img122_envelope = resize(mean_img122_envelope, [800], anti_aliasing=True)

# Create corresponding x values for all images
x82 = np.arange(len(mean_img82_envelope))
x102 = np.arange(len(mean_img102_envelope))
x122 = np.arange(len(mean_img122_envelope))

#mean values
mean82=np.mean(mean_img82_envelope)
mean102=np.mean(mean_img102_envelope)
mean122=np.mean(mean_img122_envelope)

#var vals
variance82=np.var(mean_img82_envelope)
variance102=np.var(mean_img102_envelope)
variance122=np.var(mean_img122_envelope)

# Step 4: Plot the distributions in side-by-side subplots
plt.figure(figsize=(24, 6), dpi=200)  # Increased width from 18 to 24

# Subplot 1: img82
plt.subplot(1, 3, 1)
plt.plot(x82, mean_img82_envelope, marker='o', linestyle='-', color='blue', markersize=5, label=f'Mean Lines (-20º)')
plt.xticks(np.arange(0, len(x82), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
plt.ylabel('Min-Max Norm./L Mean Envelope Amplitude', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=mean82, color='red', linestyle='--', label=f'Mean (μ) = {mean82:.2f}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Subplot 2: img102
plt.subplot(1, 3, 2)
plt.plot(x102, mean_img102_envelope, marker='o', linestyle='-', color='orange', markersize=5, label=f'Mean Lines (0º)')
plt.xticks(np.arange(0, len(x102), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
#plt.ylabel('Probability', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=mean102, color='red', linestyle='--', label=f'Mean (μ) = {mean102:.2f}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Subplot 3: img122
plt.subplot(1, 3, 3)
plt.plot(x122, mean_img122_envelope, marker='o', linestyle='-', color='green', markersize=5, label=f'Mean Lines (+20º)')
plt.xticks(np.arange(0, len(x122), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
#plt.ylabel('Probability', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=mean122, color='red', linestyle='--', label=f'Mean (μ) = {mean122:.2f}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Adjust layout and display the side-by-side plots
plt.tight_layout()
plt.show()

print(f"Variance of img82: {variance82:.4f}")
print(f"Variance of img102: {variance102:.4f}")
print(f"Variance of img122: {variance122:.4f}")

print(f"Variance of img82: {variance82:.2e}")
print(f"Variance of img102: {variance102:.2e}")
print(f"Variance of img122: {variance122:.2e}")

print(f"mean of img82: {mean82:.2e}")
print(f"mean of img102: {mean102:.2e}")
print(f"mean of img122: {mean122:.2e}")

#%%######################################################
#DISTRIBUTION FOR CMAP
import numpy as np
import matplotlib.pyplot as plt

step=100
# i,j=1800,3500
i, j = 2000, 2800

# Step 1: Extract and process images (replace with your actual image data)
img82 = justimgs[82][1]
img102 = justimgs[102][1]
img122 = justimgs[122][1]

line82 = np.mean(img82[i:j],axis=1)
line102 = np.mean(img102[i:j],axis=1)
line122 = np.mean(img122[i:j],axis=1)

# Step 2: Calculate the mean envelopes for each image
'''
mean_img82_envelope = abs(np.diff(line82))
mean_img102_envelope = abs(np.diff(line102))
mean_img122_envelope = abs(np.diff(line122))
#'''

#'''
mean_img82_envelope = abs(savgol_filter(line82, window_length=len(line82)//16,
                                  polyorder=2, deriv=1))
mean_img102_envelope = abs(savgol_filter(line102, window_length=len(line102)//16,
                                  polyorder=2, deriv=1))
mean_img122_envelope = abs(savgol_filter(line122, window_length=len(line122)//16,
                                  polyorder=2, deriv=1))
#'''

mean_img82_envelope = resize(mean_img82_envelope, [800], anti_aliasing=True)
mean_img102_envelope = resize(mean_img102_envelope, [800], anti_aliasing=True)
mean_img122_envelope = resize(mean_img122_envelope, [800], anti_aliasing=True)

# Step 3: Normalize each envelope to create probability distributions
probabilities_82 = mean_img82_envelope / np.sum(mean_img82_envelope)
probabilities_102 = mean_img102_envelope / np.sum(mean_img102_envelope)
probabilities_122 = mean_img122_envelope / np.sum(mean_img122_envelope)

# Create corresponding x values for all images
x82 = np.arange(len(probabilities_82))
x102 = np.arange(len(probabilities_102))
x122 = np.arange(len(probabilities_122))

# Calculate the Mean and Variance for each image
mean82 = np.sum(x82 * probabilities_82)
variance82 = np.sum((x82 - mean82)**2 * probabilities_82)

mean102 = np.sum(x102 * probabilities_102)
variance102 = np.sum((x102 - mean102)**2 * probabilities_102)

mean122 = np.sum(x122 * probabilities_122)
variance122 = np.sum((x122 - mean122)**2 * probabilities_122)

# Step 4: Plot the distributions in side-by-side subplots
plt.figure(figsize=(24, 6), dpi=200)  # Increased width from 18 to 24

# Subplot 1: img82
plt.subplot(1, 3, 1)
plt.plot(x82, probabilities_82, marker='o', linestyle='-', color='blue', markersize=5, label='Probability Distribution (-20º)')
plt.xticks(np.arange(0, len(x82), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=mean82, color='red', linestyle='--', label=f'Mean (μ) = {mean82:.2f}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Subplot 2: img102
plt.subplot(1, 3, 2)
plt.plot(x102, probabilities_102, marker='o', linestyle='-', color='orange', markersize=5, label='Probability Distribution (0º)')
plt.xticks(np.arange(0, len(x102), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=mean102, color='red', linestyle='--', label=f'Mean (μ) = {mean102:.2f}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Subplot 3: img122
plt.subplot(1, 3, 3)
plt.plot(x122, probabilities_122, marker='o', linestyle='-', color='green', markersize=5, label='Probability Distribution (+20º)')
plt.xticks(np.arange(0, len(x122), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(x=mean122, color='red', linestyle='--', label=f'Mean (μ) = {mean122:.2f}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Adjust layout and display the side-by-side plots
plt.tight_layout()
plt.show()

print(f"Variance of img82: {variance82:.4f}")
print(f"Variance of img102: {variance102:.4f}")
print(f"Variance of img122: {variance122:.4f}")

#%%######################################################
#np var and mean
import numpy as np
import matplotlib.pyplot as plt

step=100
# i,j=1800,3500
i, j = 2000, 2800

# Step 1: Extract and process images (replace with your actual image data)
img82 = justimgs[82][1]
img102 = justimgs[102][1]
img122 = justimgs[122][1]

line82 = np.mean(img82[i:j],axis=1)
line102 = np.mean(img102[i:j],axis=1)
line122 = np.mean(img122[i:j],axis=1)

# Step 2: Calculate the mean envelopes for each image
'''
mean_img82_envelope = abs(np.diff(line82))
mean_img102_envelope = abs(np.diff(line102))
mean_img122_envelope = abs(np.diff(line122))
#'''

#'''
mean_img82_envelope = abs(savgol_filter(line82, window_length=len(line82)//16,
                                  polyorder=2, deriv=1))
mean_img102_envelope = abs(savgol_filter(line102, window_length=len(line102)//16,
                                  polyorder=2, deriv=1))
mean_img122_envelope = abs(savgol_filter(line122, window_length=len(line122)//16,
                                  polyorder=2, deriv=1))
#'''

mean_img82_envelope = resize(mean_img82_envelope, [800], anti_aliasing=True)
mean_img102_envelope = resize(mean_img102_envelope, [800], anti_aliasing=True)
mean_img122_envelope = resize(mean_img122_envelope, [800], anti_aliasing=True)

# Create corresponding x values for all images
x82 = np.arange(len(mean_img82_envelope))
x102 = np.arange(len(mean_img102_envelope))
x122 = np.arange(len(mean_img122_envelope))

#mean values
mean82=np.mean(mean_img82_envelope)
mean102=np.mean(mean_img102_envelope)
mean122=np.mean(mean_img122_envelope)

#var vals
variance82=np.var(mean_img82_envelope)
variance102=np.var(mean_img102_envelope)
variance122=np.var(mean_img122_envelope)


# Step 4: Plot the distributions in side-by-side subplots
plt.figure(figsize=(24, 6), dpi=200)  # Increased width from 18 to 24

# Subplot 1: img82
plt.subplot(1, 3, 1)
plt.plot(x82, mean_img82_envelope, marker='o', linestyle='-', color='blue', markersize=5, label='Mean Columns (-20º)')
plt.xticks(np.arange(0, len(x82), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
plt.ylabel('Derivative of Average Confidence', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=mean82, color='red', linestyle='--', label=f'Mean (μ) = {mean82:.2e}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Subplot 2: img102
plt.subplot(1, 3, 2)
plt.plot(x102, mean_img102_envelope, marker='o', linestyle='-', color='orange', markersize=5, label='Mean Columns (0º)')
plt.xticks(np.arange(0, len(x102), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=mean102, color='red', linestyle='--', label=f'Mean (μ) = {mean102:.2e}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Subplot 3: img122
plt.subplot(1, 3, 3)
plt.plot(x122, mean_img122_envelope, marker='o', linestyle='-', color='green', markersize=5, label='Mean Columns (+20º)')
plt.xticks(np.arange(0, len(x122), step=step), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Index', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=mean122, color='red', linestyle='--', label=f'Mean (μ) = {mean122:.2e}')
plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Adjust layout and display the side-by-side plots
plt.tight_layout()
plt.show()

print(f"Variance of img82: {variance82:.2e}")
print(f"Variance of img102: {variance102:.2e}")
print(f"Variance of img122: {variance122:.2e}")

#%% plots of the effects of min max per line (w envelope and no log)

# Assuming img82, img102, and img122 are your images
img82, img102, img122 = justimgs[82][0], justimgs[102][0], justimgs[122][0]
img82, img102, img122 = byb.envelope(img82),byb.envelope(img102),byb.envelope(img122)

'''
global min-max
'''

img82 = (img82 - np.min(img82)) / (np.max(img82) - np.min(img82))
img102 = (img102 - np.min(img102)) / (np.max(img102) - np.min(img102))
img122 = (img122 - np.min(img122)) / (np.max(img122) - np.min(img122))

'''
global min-max crop zone
'''
strt,end=2000,2800

img82 = (img82 - np.min(img82[strt:end])) / (np.max(img82[strt:end]) - np.min(img82[strt:end]))
img102 = (img102 - np.min(img102[strt:end])) / (np.max(img102[strt:end]) - np.min(img102[strt:end]))
img122 = (img122 - np.min(img122[strt:end])) / (np.max(img122[strt:end]) - np.min(img122[strt:end]))

'''
per-line min-max uncropped
'''
img82 = byb.perlineMinMax(img82)
img102 = byb.perlineMinMax(img102)
img122 = byb.perlineMinMax(img122)

'''
per-line min-max cropped
'''
img82 = byb.perlineMinMax(img82, cropidxs = [strt,end])
img102 = byb.perlineMinMax(img102, cropidxs = [strt,end])
img122 = byb.perlineMinMax(img122, cropidxs = [strt,end])

# Compute the images and calculate global min and max
images = [img82, img102, img122]
global_min = min(img.min() for img in images)
global_max = max(img.max() for img in images)

# Create a gridspec layout for the images and colorbar
fig = plt.figure(figsize=(15, 5), dpi=200)
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.1], wspace=0.05)  # Reserve space for colorbar

# Display each image with shared color limits
ang = [-20, 0, 20]
axs = [fig.add_subplot(gs[i]) for i in range(3)]
for i, (ax, img, angle) in enumerate(zip(axs, images, ang)):
    img_plot = ax.imshow(img, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
    # ax.axhline(y=1800, color='red', linestyle='--', linewidth=2.5)  # Red horizontal line
    # ax.axhline(y=3500, color='blue', linestyle='--', linewidth=2.5)  # Blue horizontal line

    if i == 0:
        ax.yaxis.set_visible(True)
        ax.set_ylabel('Depth [px]', fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
    else:
        ax.yaxis.set_visible(False)

    # ax.set_xticks([])
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel('Width [px]', fontsize=18)
    ax.set_title(f"{angle}", fontsize=16)
    ax.set_xticks(np.arange(0, 129, 30))

# Add a shared colorbar to the right
cbar_ax = fig.add_subplot(gs[-1])  # Last slot for the colorbar
cbar = fig.colorbar(img_plot, cax=cbar_ax)
cbar.set_label('Envelope Amplitude', fontsize=18, labelpad=10)
cbar.ax.tick_params(labelsize=16)

# Add a single x-axis label for the whole figure
plt.figtext(0.495, 1, 'Degrees', ha='center', va='center', fontsize=18)

plt.show()



#%%##############################################################################

#OTHER STUFF

# Define the additional data for each method
additional_data = [
    {"method": "Hilbert → Crop → Mean", "train_min": 164, "train_max": 1137, "test_min": 0, "test_max": 59},
    {"method": "Hilbert → Log → Crop → Mean", "train_min": 41, "train_max": 57, "test_min": 2, "test_max": 31},
    {"method": "Hilbert → Log → Crop → Log norm. → Mean", "train_min": -42, "train_max": -17, "test_min": -37, "test_max": -13},
    {"method": "Crop → Laplace → Variance", "train_min": 18844, "train_max": 1691251, "test_min": 0, "test_max": 1634},
    {"method": "Hilbert → Crop → Laplace → Variance", "train_min": 9042, "train_max": 376030, "test_min": 0, "test_max": 2725},
    {"method": "Hilbert → Log → Crop → Laplace → Variance", "train_min": 30, "train_max": 66, "test_min": 4, "test_max": 64},
]

# Combine both sets of data
all_data = data + additional_data  # assuming `data` is defined from previous code

# Calculate percentage errors for all data
results = []
for d in all_data:
    method = d["method"]
    train_min = d["train_min"]
    train_max = d["train_max"]
    test_min = d["test_min"]
    test_max = d["test_max"]

    # Percentage error calculations
    min_error = ((test_min - train_min) / train_min) * 100 if train_min != 0 else None
    max_error = ((test_max - train_max) / train_max) * 100 if train_max != 0 else None

    results.append({
        "method": method,
        "train_range": (train_min, train_max),
        "test_range": (test_min, test_max),
        "min_error_percent": min_error,
        "max_error_percent": max_error
    })

# Print the results
for result in results:
    print(f"Method: {result['method']}")
    print(f"  Train Range: {result['train_range']}")
    print(f"  Test Range: {result['test_range']}")
    if result['min_error_percent'] is not None:
        print(f"  Min Error: {result['min_error_percent']:.2f}%")
    else:
        print("  Min Error: Undefined (Train Min is 0)")
    if result['max_error_percent'] is not None:
        print(f"  Max Error: {result['max_error_percent']:.2f}%")
    else:
        print("  Max Error: Undefined (Train Max is 0)")
    print()

#%%
'''
WAve PLOTS
'''
import numpy as np
import matplotlib.pyplot as plt

# Time axis
t = np.linspace(0, 10, 1000)

# Define an envelope function that increases and then decreases
def envelope(t, start, end):
    center = (start + end) / 2
    width = (end - start) / 2
    return np.exp(-((t - center) ** 2) / (2 * (width / 2) ** 2))

# Define wavy pulses with increasing and decreasing amplitude
# Non-overlapping pulses (closer together)
pulse1_no_overlap = envelope(t, 1, 2) * np.sin(2 * np.pi * 5 * t) * (t >= 1) * (t < 2)
pulse2_no_overlap = envelope(t, 2, 3) * np.sin(2 * np.pi * 5 * (t - 1)) * (t >= 2) * (t < 3)
pulse3_no_overlap = envelope(t, 3, 4) * np.sin(2 * np.pi * 5 * (t - 2)) * (t >= 3) * (t < 4)

# Overlapping pulses (more overlap)
pulse1_overlap = envelope(t, 1, 3) * np.sin(2 * np.pi * 5 * t) * (t >= 1) * (t < 3)
pulse2_overlap = envelope(t, 1.5, 3.5) * np.sin(2 * np.pi * 5 * (t - 0.5)) * (t >= 1.5) * (t < 3.5)
pulse3_overlap = envelope(t, 2, 4) * np.sin(2 * np.pi * 5 * (t - 1)) * (t >= 2) * (t < 4)

hilb1 = byb.hilb(pulse1_no_overlap)
hilb2 = byb.hilb(pulse2_no_overlap)
hilb3 = byb.hilb(pulse3_no_overlap)
hilb = np.mean([hilb1,hilb2,hilb3],axis=0)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot non-overlapping pulses (left subplot)
axs[0].plot(t, pulse1_no_overlap, label='RF data 1', color='blue')
axs[0].plot(t, pulse2_no_overlap, label='RF data 2', color='orange')
axs[0].plot(t, pulse3_no_overlap, label='RF data 3', color='green')
axs[0].plot(t, hilb, label='Mean Envelope', color='red', linestyle='--')
axs[0].set_xlim(0, 5)
axs[0].set_xlabel("Depth [mm]", fontsize=18)
axs[0].set_ylabel("Amplitude", fontsize=18)
axs[0].set_title("(A)", fontsize=18)
axs[0].grid(True)
axs[0].tick_params(axis='both', labelsize=16)

hilb1 = byb.hilb(pulse1_overlap)
hilb2 = byb.hilb(pulse2_overlap)
hilb3 = byb.hilb(pulse3_overlap)
hilb = np.mean([hilb1,hilb2,hilb3],axis=0)

# Plot overlapping pulses (right subplot)
axs[1].plot(t, pulse1_overlap, label='RF data 1', color='blue')
axs[1].plot(t, pulse2_overlap, label='RF data 2', color='orange')
axs[1].plot(t, pulse3_overlap, label='RF data 3', color='green')
axs[1].plot(t, hilb, label='Mean Envelope', color='red', linestyle='--')
axs[1].set_xlim(0, 5)
axs[1].set_xlabel("Depth [mm]", fontsize=18)
axs[1].set_title("(B)", fontsize=18)
axs[1].grid(True)
axs[1].tick_params(axis='both', labelsize=16)

# Add legend only to the right subplot
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
plt.tight_layout()
plt.show()












#%%

'''
FAKE US PLOTS
'''


import numpy as np
import matplotlib.pyplot as plt
import cv2  # Import OpenCV

# Create a black background image with a thicker inclined white line
height, width = 512, 512
image = np.zeros((height, width), dtype=np.uint8)
line_thickness = 10  # Thickness of the line
center_y = height // 2  # Center line of the image
ang = 5

# Define start and end points for the inclined line
start_point = (0, center_y - (width // (2 * ang)))
end_point = (width, center_y + (width // (2 * ang)))

# Draw the line with anti-aliasing
cv2.line(image, start_point, end_point, color=255, thickness=line_thickness, lineType=cv2.LINE_AA)

# Calculate the mean of all columns (axis=1)
mean_values = np.mean(image, axis=1)

# Create the figure and axes with a smaller width
fig, (ax, ax_mean) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), gridspec_kw={'width_ratios': [1, 0.5]})

# Plot the image with the inclined white line
ax.imshow(image, cmap='gray', aspect='auto')
ax.set_title('US image with Smooth Line')
ax.set_xlabel('Width')  # X-axis label
ax.set_ylabel('Depth []')  # Y-axis label

# Plot the mean of the columns
ax_mean.plot(mean_values, np.arange(len(mean_values)), color='blue')
ax_mean.set_title('Mean of All Columns')
ax_mean.set_xlabel('Intensity')  # X-axis label for the mean plot
ax_mean.set_ylabel('Depth')      # Y-axis label for the mean plot
ax_mean.grid(True)
ax_mean.set_ylim(len(mean_values) - 1, 0)  # Flip the y-axis

# Adjust layout for more space between the image and the mean plot
plt.subplots_adjust(wspace=1)  # Increase the space between subplots

# Display the plot
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import cv2  # Import OpenCV

# Create a black background image with a thicker inclined white line
height, width = 512, 512
image = np.zeros((height, width), dtype=np.uint8)
line_thickness = 10  # Thickness of the line
center_y = height // 2  # Center line of the image
ang = 5

# Define start and end points for the inclined line
start_point = (0, center_y - (width // (2 * ang)))
end_point = (width, center_y + (width // (2 * ang)))

# Draw the line with anti-aliasing
cv2.line(image, start_point, end_point, color=255, thickness=line_thickness, lineType=cv2.LINE_AA)

# Calculate the mean of all columns (axis=1)
mean_values = np.mean(image, axis=1)

# Create an image from the mean values
mean_image = np.zeros((height, 1), dtype=np.uint8)  # Create an empty image
mean_image[:, 0] = mean_values  # Fill the single column with mean values
mean_image = cv2.resize(mean_image, (50, height))  # Resize to match original image dimensions

# Create the figure and axes for a single row of subplots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

# Plot the main image with the inclined white line
axs[0].imshow(image, cmap='gray', aspect='auto')
axs[0].set_title('US Image with Smooth Line')
axs[0].set_xlabel('Width')
axs[0].set_ylabel('Depth')

# Plot the mean image
axs[1].imshow(mean_image, cmap='gray')
axs[1].set_title('Mean Values Image')
axs[1].axis('off')  # Hide axes

# Plot the mean values as a line plot
axs[2].plot(mean_values, np.arange(len(mean_values)), color='blue')
axs[2].set_title('Mean of All Columns')
axs[2].set_xlabel('Intensity')
axs[2].set_ylabel('Depth')
axs[2].grid(True)
axs[2].set_ylim(len(mean_values) - 1, 0)  # Flip the y-axis

# Adjust layout for better spacing
plt.subplots_adjust(wspace=0.5)  # Increase the space between subplots

# Display the final plot
plt.show()

################################################################################
#Goal plot

import numpy as np
import matplotlib.pyplot as plt

# Original data
x = np.linspace(-1, 1, 41)
sigma = 0.3  # Adjust sigma for desired smoothness
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)

# Plot the Gaussian curve
plt.figure(dpi=200)
plt.plot(x, gaussian_array, label='Gaussian Curve', color='orange')

plt.xlabel('Degrees')
plt.ylabel('Cost')
plt.grid(True)

# Set x-axis limits and labels with a step of 5
plt.xticks(np.linspace(-1, 1, 9), labels=np.arange(-20, 25, 5))

plt.show()

#%% average rotation values
#gold target
roll = [1.1411, -0.2701, 0.1990, 6.3128,-2.4568,-2.0378,-1.1184, -0.4436,-1.0711, -2.5254]
pit = [-4.7827,  -19.6109, 15.2419, 8.4278, 0.0686,2.3057, 1.0709,-6.8726,-9.2489,1.3351]
yaw = [10.8914, -0.2422, 0.8147 ,-0.9647,-19.2888, 87.4065, 92.9994,  103.1017,84.3379,94.4198]
print(np.mean(np.abs(roll)), np.max(roll), np.min(roll), np.std(np.abs(roll)))
print(np.mean(np.abs(pit)), np.max(pit), np.min(pit), np.std(np.abs(pit)))
print(np.mean(np.abs(yaw)), np.max(yaw), np.min(yaw), np.std(np.abs(yaw)))

#prelim testing
roll = [1.1411, -0.2701, 0.1990,21.3344,-2.1314,-2.0378,-1.1184,6.3373, 0.5188, -0.9248]
pit = [-4.7827,  -19.6109, 15.2419,7.5390,-3.9621,2.3057,1.0709,-18.0003,-10.4953,9.8994]
yaw = [10.8914, -0.2422, 0.8147 ,12.3669,-17.9696,87.4065,92.9994,70.2158,75.5142,70.4177]
print(np.mean(np.abs(roll)), np.max(roll), np.min(roll), np.std(np.abs(roll)))
print(np.mean(np.abs(pit)), np.max(pit), np.min(pit), np.std(np.abs(pit)))
print(np.mean(np.abs(yaw)), np.max(yaw), np.min(yaw), np.std(np.abs(yaw))) 

#sim testing
roll = [2.2820, -0.2701,0.1990,6.3128,-19.8913,  -7.0659,-1.1184,-0.4784,-1.0711,-2.5254]
pit = [ -5.7832,-19.6109,15.2419,8.4278,-0.1929,5.7144,18.1139,-6.7669, -9.2489,1.3351]
yaw = [16.9680,-0.2422,0.8147,-0.9647,0.6202,76.5682,109.1211,85.3367,84.3379,94.4198]
print(np.mean(np.abs(roll)), np.max(roll), np.min(roll), np.std(np.abs(roll)))
print(np.mean(np.abs(pit)), np.max(pit), np.min(pit), np.std(np.abs(pit)))
print(np.mean(np.abs(yaw)), np.max(yaw), np.min(yaw), np.std(np.abs(yaw)))   

#%%###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define the path and best indices
path = Path('C:/Users/Mateo-drr/Documents/data/dataMeat/')
best = [16, 0, 3, 1, 14, 17, 13, 7, 1, 1] #gt
best = [16, 0, 0, 8, 20, 17, 13, 4, 2, 14] #test1
# best = [6, 0, 3, 1, 5, 4, 15, 8, 1, 1] #simtest final

# '''
# Define the path and best indices
#path = Path('C:/Users/Mateo-drr/Documents/data/dataChest/')
# best = [15, 10, 4, 15, 17, 15, 16, 5, 12, 3] #test1
# best = [2, 2, 3, 11, 3, 14, 0, 21, 17, 0] #simtest final
# best = [4, 2, 3, 11, 3, 14, 0, 21, 17, 0] #3feat
# best = [7, 17, 2, 6, 5, 5, 0, 21, 10, 1] #2feat

#best = [11, 5, 5, 19, 17, 1, 3, 15, 14, 2]
# best = [11, 5, 5, 19, 0, 1, 1, 15, 14, 2]
# '''
#best = [11, 5, 4, 0, 17, 1, 15, 10, 14, 2]
#best= [11, 5, 5, 0, 17, 1, 11, 10, 0, 2]
#best = [4, 2, 4, 6, 1, 5, 10, 21, 4, 3]
#best =[11, 5, 5, 3, 17, 1, 3, 9, 3, 2]
#best = [0, 14, 3, 15, 0, 15, 13, 18, 3, 4]
#best = [0, 5, 4, 0, 17, 1, 15, 9, 3, 2]
#best = [0, 5, 5, 19, 0, 1, 1, 15, 14, 2]

# Get all subdirectories in the path
folders = [f for f in path.iterdir() if f.is_dir()]

# Initialize the figure for subplots
fig, axs = plt.subplots(1, 10, figsize=(10, 4),dpi=200)  # Adjust figsize for better visualization

# Iterate through the folders and best indices
for i, (folder, idx) in enumerate(zip(folders, best)):
    # Construct the file path for the current image
    file_path = folder / 'runData' / f'USRaw{idx}.npy'

    # Check if the file exists
    if file_path.exists():
        # Load the image and process it
        img = np.load(file_path)
        # axs[i].imshow(20 * np.log10(byb.envelope(img)+1))  # Use a grayscale colormap for US images
        axs[i].imshow(20 * np.log10(np.abs(img)+1))
        #axs[i].set_title(f'{folder.name}')
        axs[i].axis('off')  # Turn off axis
    else:
        axs[i].set_title('Missing')
        axs[i].axis('off')

# Display the plot
plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Initialize the figure for subplots (add one extra subplot for colorbar)
fig, axs = plt.subplots(1, 11, figsize=(11, 4), dpi=200)  # Adjust the number of subplots to 11

# To collect the images and min/max values
images = []
all_min, all_max = float('inf'), float('-inf')

# First loop: Load the images and find global min and max
for i, (folder, idx) in enumerate(zip(folders, best)):
    # Construct the file path for the current image
    file_path = folder / 'runData' / f'USRaw{idx}.npy'

    # Check if the file exists
    if file_path.exists():
        # Load the image
        img = np.load(file_path)
        
        # Store the image in the images list
        images.append(img)
        
        # Update the global min and max values
        img = byb.logS(np.abs(img))
        all_min = min(all_min, np.min(img))
        all_max = max(all_max, np.max(img))
    else:
        images.append(None)

# Second loop: Plot the images with the same color scale
for i, (folder, idx) in enumerate(zip(folders, best)):
    if images[i] is not None:
        # Plot the image with global vmin and vmax for consistent color scaling
        im = axs[i].imshow(byb.logS(np.abs(images[i])), vmin=all_min, vmax=all_max)
        
        # Set x-axis on all subplots
        if i == 0:  # Add y-axis label only for the first subplot
            axs[i].set_ylabel('Depth [px]', fontsize=12)
        else:
            axs[i].yaxis.set_ticks([]) 
            
        # axs[i].set_xlabel('Width [px]')
        
        # Turn off axes ticks
        axs[i].axis('on')
        axs[i].set_xticks(range(0, 128, 50))
        axs[i].set_title(f'({i})')
    else:
        # axs[i].set_title('Missing')
        axs[i].axis('off')

# Create the global colorbar subplot at the last position
cbar_ax = axs[-1]
cbar_ax.axis('off')  # Hide axes for the colorbar subplot

# cbar_ax.set_position([1.02, 0.15, 0.02, 0.7])
# Create a global colorbar using the min and max values from all images
cbar = fig.colorbar(im, ax=cbar_ax, orientation='vertical')
cbar.set_label('Intensity [dB]', rotation=270, labelpad=15, fontsize=12)
fig.text(0.5, 0.01, 'Width [px]', ha='center', va='center', fontsize=12)
# Display the plot
plt.subplots_adjust(wspace=0.1, hspace=0.02)
plt.show()



#%%###############################################################################
#SIM TESTING
# Assuming `allres` is the array containing the test results
# Shape: (10, 4, 100) -> 10 runs, 4 search spaces, 100 steps
num_runs, num_spaces, num_steps = allres.shape

# Calculate the average learning curve for each search space
space_averages = allres.mean(axis=0)  # Shape: (4, 100)

# Calculate the global average learning curve across all search spaces
global_average = allres.mean(axis=(0, 1))  # Shape: (100,)

# Plotting
plt.figure(figsize=(10, 6))

# Plot each search space's average learning curve
for i in range(num_spaces):
    plt.plot(space_averages[i], label=f"Search Space {i+1}")

# Plot the global average
plt.plot(global_average, label="Global Average", linestyle='--', linewidth=2, color='black')

# Add plot details
plt.title("Average Learning Curves Across Search Spaces", fontsize=14)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Performance", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the middle row index
middle_row = 20

# Calculate the distances to the middle row for all runs and search spaces
# postest.shape = (10, 4, 100, 2) -> 10 runs, 4 search spaces, 100 steps, 2 (row, col)
row_positions = postest[..., 0]  # Extract the row positions, shape: (10, 4, 100)
distances = np.abs(row_positions - middle_row)  # Calculate the distance to the middle row

# Calculate the average distance for each search space at each step
space_distances_avg = distances.mean(axis=0)  # Shape: (4, 100)

# Calculate the global average distance across all search spaces
global_distances_avg = distances.mean(axis=(0, 1))  # Shape: (100,)

# Plotting
plt.figure(figsize=(10, 6))

# Plot each search space's average distance curve
for i in range(space_distances_avg.shape[0]):
    plt.plot(space_distances_avg[i], label=f"Search Space {i+1}")

# Plot the global average distance curve
plt.plot(global_distances_avg, label="Global Average", linestyle='--', linewidth=2, color='black')

# Add plot details
plt.title("Average Distance to Middle Row (Index 20) Across Iteration Steps", fontsize=14)
plt.xlabel("Iteration Steps", fontsize=12)
plt.ylabel("Distance to Middle Row", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()

#heatmap not log
import numpy as np
import matplotlib.pyplot as plt

# Dimensions of the search space
grid_size = (41, 41)

# Initialize arrays to accumulate counts for each search space and global average
search_space_heatmaps = np.zeros((4, *grid_size))  # Shape: (4, 41, 41)
global_heatmap = np.zeros(grid_size)  # Shape: (41, 41)

# Accumulate tested positions for each search space
for run in range(postest.shape[0]):  # Iterate over runs
    for space in range(postest.shape[1]):  # Iterate over search spaces
        for step in range(postest.shape[2]):  # Iterate over iteration steps
            row, col = postest[run, space, step]  # Extract row and col
            search_space_heatmaps[space, int(row), int(col)] += 1
            global_heatmap[int(row), int(col)] += 1

# Normalize the heatmaps by the number of runs
search_space_heatmaps /= postest.shape[0]  # Average per search space
global_heatmap /= (postest.shape[0] * postest.shape[1])  # Global average across all search spaces

# Find the global maximum value across all heatmaps for consistent scaling
max_value = max(global_heatmap.max(), search_space_heatmaps.max())

# Plot heatmaps
fig, axes = plt.subplots(1, 5, figsize=(20, 5), constrained_layout=True)

# Plot each search space heatmap
for i in range(4):
    ax = axes[i]
    im = ax.imshow(search_space_heatmaps[i], cmap="hot", interpolation="nearest", vmin=0, vmax=max_value)
    ax.set_title(f"Search Space {i+1}")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    plt.colorbar(im, ax=ax)

# Plot global heatmap
im = axes[4].imshow(global_heatmap, cmap="hot", interpolation="nearest", vmin=0, vmax=max_value)
axes[4].set_title("Global Average")
axes[4].set_xlabel("Columns")
axes[4].set_ylabel("Rows")
plt.colorbar(im, ax=axes[4])

# Add a global title
fig.suptitle("Average Tested Positions Heatmaps (Normalized Scale)", fontsize=16)

# Show the plot
plt.show()

#heatmap w log
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Dimensions of the search space
grid_size = (41, 41)

# Initialize arrays to accumulate counts for each search space and global average
search_space_heatmaps = np.zeros((4, *grid_size))  # Shape: (4, 41, 41)
global_heatmap = np.zeros(grid_size)  # Shape: (41, 41)

# Accumulate tested positions for each search space
for run in range(postest.shape[0]):  # Iterate over runs
    for space in range(postest.shape[1]):  # Iterate over search spaces
        for step in range(postest.shape[2]):  # Iterate over iteration steps
            row, col = postest[run, space, step]  # Extract row and col
            search_space_heatmaps[space, int(row), int(col)] += 1
            global_heatmap[int(row), int(col)] += 1

# Normalize the heatmaps by the number of runs
search_space_heatmaps /= postest.shape[0]  # Average per search space
global_heatmap /= (postest.shape[0] * postest.shape[1])  # Global average across all search spaces

# Add a small offset to avoid issues with log(0)
search_space_heatmaps += 1e-3
global_heatmap += 1e-3

# Determine the global min and max for the color scale
vmin = min(np.min(search_space_heatmaps), np.min(global_heatmap))
vmax = max(np.max(search_space_heatmaps), np.max(global_heatmap))

# Plot heatmaps with logarithmic scaling
fig, axes = plt.subplots(1, 5, figsize=(20, 5), constrained_layout=True)

# Plot each search space heatmap
for i in range(4):
    ax = axes[i]
    im = ax.imshow(search_space_heatmaps[i], cmap="hot", interpolation="nearest", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title(f"Search Space {i+1}")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    plt.colorbar(im, ax=ax)

# Plot global heatmap
im = axes[4].imshow(global_heatmap, cmap="hot", interpolation="nearest", norm=LogNorm(vmin=vmin, vmax=vmax))
axes[4].set_title("Global Average")
axes[4].set_xlabel("Columns")
axes[4].set_ylabel("Rows")
plt.colorbar(im, ax=axes[4])

# Add a global title
fig.suptitle("Average Tested Positions Heatmaps (Logarithmic Scale, Consistent Range)", fontsize=16)

# Show the plot
plt.show()



################################################################################



#%%


import numpy as np
import matplotlib.pyplot as plt
import cv2  # Import OpenCV
import os

# Set parameters for the image
height, width = 512, 512
line_thickness = 10  # Thickness of the line
num_frames = 40  # Number of frames for the line rotation
output_dir = 'C:/Users/Mateo-drr/Documents/data/figures/fakelines'
os.makedirs(output_dir, exist_ok=True)

# Adjust the maximum inclination for the line
max_incline = -70  # Maximum incline in pixels

# Loop over each frame to create an image
for i in range(num_frames):
    # Create a black background image
    image = np.zeros((height, width), dtype=np.uint8)

    # Calculate the line's start and end points based on the frame index
    if i < num_frames // 2:
        # First half: line starts inclined and becomes flat
        start_point = (0, height // 2 - (max_incline - (i * max_incline // (num_frames // 2))))  # Higher left
        end_point = (width, height // 2 + (max_incline - (i * max_incline // (num_frames // 2))))  # Higher right
    else:
        # Second half: line becomes inclined again in the opposite direction
        start_point = (0, height // 2 + ((i - num_frames // 2) * max_incline // (num_frames // 2)))  # Lower left
        end_point = (width, height // 2 - ((i - num_frames // 2) * max_incline // (num_frames // 2)))  # Lower right

    # Draw the line with anti-aliasing
    cv2.line(image, start_point, end_point, color=255, thickness=line_thickness, lineType=cv2.LINE_AA)


    image = image/255
    # Calculate the mean of all columns (axis=1)
    mean_values = np.mean(image, axis=1)

    # Create an image from the mean values
    mean_image = np.zeros((height, 1), dtype=np.float32)  # Create an empty image
    mean_image[:, 0] = mean_values  # Fill the single column with mean values
    mean_image = cv2.resize(mean_image, (50, height))  # Resize to match original image dimensions

    # Create the figure and axes for a single row of subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), dpi=200)

    # Plot the main image with the inclined white line
    axs[0].imshow(image, cmap='gray', aspect='auto')
    axs[0].set_title('US image')
    axs[0].set_xlabel('Width [px]')
    axs[0].set_ylabel('Depth [px]')

    # Plot the mean image
    axs[1].imshow(mean_image, cmap='gray')
    axs[1].set_title('Mean columns')
    axs[1].axis('off')  # Hide axes

    # Plot the mean values as a line plot
    axs[2].plot(mean_values, np.arange(len(mean_values)), color='blue')
    axs[2].set_title('Plot of the mean columns')
    axs[2].set_xlabel('Intensity')
    axs[2].set_ylabel('Depth [px]')
    axs[2].grid(True)
    axs[2].set_ylim(len(mean_values) - 1, 0)  # Flip the y-axis

    # Adjust layout for better spacing
    plt.subplots_adjust(wspace=0.5)  # Increase the space between subplots

    # Save the figure as an image file
    plt.savefig(os.path.join(output_dir, f'rotating_line_{i:03d}.png'), bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

print(f'Images saved in the directory: {output_dir}')

#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Set parameters for the image
height, width = 512, 512
num_frames = 40  # Number of frames for the line transformation
output_dir = 'C:/Users/Mateo-drr/Documents/data/figures/fakelinesrot'
os.makedirs(output_dir, exist_ok=True)

# Line properties
max_thickness = 40  # Maximum thickness of the line
min_thickness = 10   # Minimum thickness of the line
low_intensity = 0.1  # Starting intensity as a fraction (e.g., 0.2 for 20%)
high_intensity = 1.0  # Max intensity as a fraction (e.g., 1.0 for 100%)
max_blur_kernel = 30  # Maximum kernel size for Gaussian blur

# Loop over each frame to create an image
for i in range(num_frames):
    # Create a black background image
    image = np.zeros((height, width), dtype=np.uint8)

    # Calculate the line's thickness and intensity factor based on the frame index
    if i < num_frames // 2:
        # First half: line starts thick and low intensity, gets thinner and brighter
        thickness = max_thickness - (i * (max_thickness - min_thickness) // (num_frames // 2))
        intensity_factor = low_intensity + (i * (high_intensity - low_intensity) / (num_frames // 2))
    else:
        # Second half: line becomes thicker and lower in intensity again
        thickness = min_thickness + ((i - num_frames // 2) * (max_thickness - min_thickness) // (num_frames // 2))
        intensity_factor = high_intensity - ((i - num_frames // 2) * (high_intensity - low_intensity) / (num_frames // 2))

    # Draw the line at maximum intensity
    start_point = (0, height // 2)
    end_point = (width, height // 2)
    cv2.line(image, start_point, end_point, color=255, thickness=thickness, lineType=cv2.LINE_AA)

    # Apply intensity scaling by multiplying the entire image by the intensity factor
    image = (image * intensity_factor).astype(np.uint8)

    # Apply Gaussian blur based on the intensity factor
    # Calculate the blur kernel size (lower intensity = larger blur)
    blur_kernel_size = int(max_blur_kernel * (1 - intensity_factor))
    # Ensure kernel size is odd and at least 1
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    if blur_kernel_size > 1:
        image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)


    image = image/255
    # Calculate the mean of all columns (axis=1)
    mean_values = np.mean(image, axis=1)

    # Create an image from the mean values
    mean_image = np.zeros((height, 1), dtype=np.float32)  # Create an empty image
    mean_image[:, 0] = mean_values  # Fill the single column with mean values
    mean_image = cv2.resize(mean_image, (50, height))  # Resize to match original image dimensions

    # Create the figure and axes for a single row of subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), dpi=200)

    # Plot the main image with the line, fixing the intensity range
    axs[0].imshow(image, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axs[0].set_title('US image')
    axs[0].set_xlabel('Width [px]')
    axs[0].set_ylabel('Depth [px]')

    # Plot the mean image, also fixing the intensity range
    axs[1].imshow(mean_image, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Mean columns')
    axs[1].axis('off')  # Hide axes

    # Plot the mean values as a line plot
    axs[2].plot(mean_values, np.arange(len(mean_values)), color='blue')
    axs[2].set_title('Plot of the mean columns')
    axs[2].set_xlabel('Intensity')
    axs[2].set_ylabel('Depth [px]')
    axs[2].grid(True)
    axs[2].set_ylim(len(mean_values) - 1, 0)  # Flip the y-axis

    # Adjust layout for better spacing
    plt.subplots_adjust(wspace=0.5)  # Increase the space between subplots

    # Save the figure as an image file
    plt.savefig(os.path.join(output_dir, f'line_transform_{i:03d}.png'), bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

print(f'Images saved in the directory: {output_dir}')

#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.signal import savgol_filter

# Set parameters for the image
height, width = 512, 512
num_frames = 40  # Number of frames for the line transformation
output_dir = 'C:/Users/Mateo-drr/Documents/data/figures/fakecmap'
os.makedirs(output_dir, exist_ok=True)

# Line properties
max_thickness = 40  # Maximum thickness of the line
min_thickness = 10   # Minimum thickness of the line
low_intensity = 0.1  # Starting intensity as a fraction (e.g., 0.2 for 20%)
high_intensity = 1.0  # Max intensity as a fraction (e.g., 1.0 for 100%)
max_blur_kernel = 100  # Maximum kernel size for Gaussian blur

# Loop over each frame to create an image
for i in range(num_frames):
    # Create a black background image
    image = np.zeros((height, width), dtype=np.uint8)

    # Calculate the line's thickness and intensity factor based on the frame index
    if i < num_frames // 2:
        # First half: line starts thick and low intensity, gets thinner and brighter
        thickness = max_thickness - (i * (max_thickness - min_thickness) // (num_frames // 2))
        intensity_factor = low_intensity + (i * (high_intensity - low_intensity) / (num_frames // 2))
    else:
        # Second half: line becomes thicker and lower in intensity again
        thickness = min_thickness + ((i - num_frames // 2) * (max_thickness - min_thickness) // (num_frames // 2))
        intensity_factor = high_intensity - ((i - num_frames // 2) * (high_intensity - low_intensity) / (num_frames // 2))

    # Draw the line at maximum intensity
    start_point = (0, height // 2)
    end_point = (width, height // 2)
    cv2.line(image, start_point, end_point, color=255, thickness=thickness, lineType=cv2.LINE_AA)

    # Apply intensity scaling by multiplying the entire image by the intensity factor
    image = (image * intensity_factor).astype(np.uint8)

    # Apply Gaussian blur based on the intensity factor
    # Calculate the blur kernel size (lower intensity = larger blur)
    blur_kernel_size = int(max_blur_kernel * (1 - intensity_factor))
    # Ensure kernel size is odd and at least 1
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    if blur_kernel_size > 1:
        image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

    # Set all pixels above the line to the same intensity as the line
    image[:height // 2, :] = image[height // 2, :]
    
    image = image/255

    # Calculate the mean of all columns (axis=1)
    mean_values = np.mean(image, axis=1)

    # Calculate the derivative of the mean values
    derivative_values = abs(savgol_filter(mean_values, window_length=len(mean_values)//16,polyorder=2, deriv=1))

    # Create an image from the mean values
    mean_image = np.zeros((height, 1), dtype=np.float32)  # Create an empty image
    mean_image[:, 0] = mean_values  # Fill the single column with mean values
    mean_image = cv2.resize(mean_image, (50, height))  # Resize to match original image dimensions

    # Create the figure and axes for a single row of subplots
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 5), dpi=200)

    # Plot the main image with the line, fixing the intensity range
    axs[0].imshow(image, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axs[0].set_title('Confidence Map')
    axs[0].set_xlabel('Width [px]')
    axs[0].set_ylabel('Depth [px]')

    # Plot the mean image, also fixing the intensity range
    axs[1].imshow(mean_image, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Mean columns')
    axs[1].axis('off')  # Hide axes

    # Plot the mean values as a line plot
    axs[2].plot(mean_values, np.arange(len(mean_values)), color='blue')
    axs[2].set_title('Plot of the mean columns')
    axs[2].set_xlabel('Mean Confidence')
    axs[2].set_ylabel('Depth [px]')
    axs[2].grid(True)
    axs[2].set_ylim(len(mean_values) - 1, 0)  # Flip the y-axis

    # Plot the derivative of the mean values
    axs[3].plot(derivative_values, np.arange(0, len(mean_values)), color='green')
    axs[3].set_title('Derivative of mean columns')
    axs[3].set_xlabel('Mean Confidence Change')
    axs[3].set_ylabel('Depth [px]')
    axs[3].grid(True)
    axs[3].set_ylim(len(mean_values) - 1, 1)  # Flip the y-axis to match depth

    # Adjust layout for better spacing
    plt.subplots_adjust(wspace=1.2)  # Increase the space between subplots

    # Save the figure as an image file
    plt.savefig(os.path.join(output_dir, f'line_transform_{i:03d}.png'), bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

print(f'Images saved in the directory: {output_dir}')


#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.signal import savgol_filter

# Set parameters for the image
height, width = 512, 512
num_frames = 40  # Number of frames for the line transformation
output_dir = 'C:/Users/Mateo-drr/Documents/data/figures/fakecmaprot'
os.makedirs(output_dir, exist_ok=True)

# Line properties
line_thickness = 10  # Thickness of the line
max_incline = -70  # Maximum incline in pixels

# Loop over each frame to create an image
for i in range(num_frames):
    # Create a black background image
    image = np.zeros((height, width), dtype=np.uint8)

    # Calculate the line's start and end points based on the frame index
    if i < num_frames // 2:
        # First half: line starts inclined and becomes flat
        start_point = (0, height // 2 - (max_incline - (i * max_incline // (num_frames // 2))))  # Higher left
        end_point = (width, height // 2 + (max_incline - (i * max_incline // (num_frames // 2))))  # Higher right
    else:
        # Second half: line becomes inclined again in the opposite direction
        start_point = (0, height // 2 + ((i - num_frames // 2) * max_incline // (num_frames // 2)))  # Lower left
        end_point = (width, height // 2 - ((i - num_frames // 2) * max_incline // (num_frames // 2)))  # Lower right

    # Draw the line with anti-aliasing
    cv2.line(image, start_point, end_point, color=255, thickness=line_thickness, lineType=cv2.LINE_AA)

    # Set all pixels above the line to the same intensity as the line
    for y in range(image.shape[1]):  # Iterate over rows (depth)
        indices = np.where(image[:,y] == 255)[0]  # Find indices where the line exists in the current row
        if indices.size > 0:  # Ensure there is a line in this row
            image[:indices[-1],y] = 255  # Set all pixels before the last line pixel to 1

    image = image/255

    # Calculate the mean of all columns (axis=1)
    mean_values = np.mean(image, axis=1)

    # Calculate the derivative of the mean values
    derivative_values = abs(savgol_filter(mean_values, window_length=len(mean_values)//16, polyorder=2, deriv=1))

    # Create an image from the mean values
    mean_image = np.zeros((height, 1), dtype=np.float32)  # Create an empty image
    mean_image[:, 0] = mean_values  # Fill the single column with mean values
    mean_image = cv2.resize(mean_image, (50, height))  # Resize to match original image dimensions

    # Create the figure and axes for a single row of subplots
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 5), dpi=200)

    # Plot the main image with the rotating line
    axs[0].imshow(image, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axs[0].set_title('Confidence Map')
    axs[0].set_xlabel('Width [px]')
    axs[0].set_ylabel('Depth [px]')

    # Plot the mean image
    axs[1].imshow(mean_image, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Mean columns')
    axs[1].axis('off')  # Hide axes

    # Plot the mean values as a line plot
    axs[2].plot(mean_values, np.arange(len(mean_values)), color='blue')
    axs[2].set_title('Plot of the mean columns')
    axs[2].set_xlabel('Mean Confidence')
    axs[2].set_ylabel('Depth [px]')
    axs[2].grid(True)
    axs[2].set_ylim(len(mean_values) - 1, 0)  # Flip the y-axis

    # Plot the derivative of the mean values
    axs[3].plot(derivative_values, np.arange(0, len(mean_values)), color='green')
    axs[3].set_title('Derivative of mean columns')
    axs[3].set_xlabel('Mean Confidence Change')
    axs[3].grid(True)
    axs[3].set_ylim(len(mean_values) - 1, 1)  # Flip the y-axis to match depth

    # Adjust layout for better spacing
    plt.subplots_adjust(wspace=1.2)  # Increase the space between subplots

    # Save the figure as an image file
    plt.savefig(os.path.join(output_dir, f'line_rotation_{i:03d}.png'), bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

print(f'Images saved in the directory: {output_dir}')

#%% presentation plots
import numpy as np
import matplotlib.pyplot as plt

# Pulse parameters
length = 6200  # X length
amplitude = 30  # Amplitude

# Create x-axis
x = np.linspace(0, length, length)

# Create trapezoid pulse
pulse = np.zeros_like(x)
start = length * 0.2  # Start of rise
rise_end = length * 0.3  # End of rise
fall_start = length * 0.7  # Start of fall
fall_end = length * 0.8  # End of fall

pulse[(x >= start) & (x < rise_end)] = np.linspace(0, amplitude, int(length * 0.1))
pulse[(x >= rise_end) & (x < fall_start)] = amplitude
pulse[(x >= fall_start) & (x < fall_end)] = np.linspace(amplitude, 0, int(length * 0.1))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x, pulse)
plt.title('Mean Columns', fontsize=18)
plt.xlabel('Depth [px]', fontsize=16)
plt.ylabel('Intensity [dB]', fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

