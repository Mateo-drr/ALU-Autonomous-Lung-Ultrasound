# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:28:46 2024

@author: Mateo-drr
"""

import byble as byb
from pathlib import Path
import numpy as np
from confidenceMap import confidenceMap
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import laplace
from skopt.space import Real, Integer
from tqdm import tqdm

# Get the base directory
current_dir = Path(__file__).resolve().parent.parent.parent
import sys
base = current_dir / 'ALU---Autonomous-Lung-Ultrasound'
sys.path.append(base.as_posix())
from bayesianOp import ManualGPMinimize
import pickle
import random

loadData=False

if __name__ == "__main__":
    if loadData:
        #'''
        # PARAMS
        date = '01Aug6'
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
            conf = byb.loadConf(datapath, confname)
            all_conf.append(conf)
            
            # Organize the data as [coord, q rot, id]
            positions = []
            for i, coord in enumerate(conf['tcoord']):
                positions.append(coord + conf['quater'][i] + [i])
            
            all_positions.append(np.array(positions))
        
        # If you need to concatenate positions or other data across ptpyes, you can do so here
        allmove = np.concatenate(all_positions, axis=0)
        
        alldat = []
        
        datapath = all_filenames[0][0]
        fileNames = all_filenames[0][1]
        for pos,x in enumerate(allmove):
            
            if pos%82 == 0:
                datapath = all_filenames[pos//82][0]
                fileNames = all_filenames[pos//82][1]
            
            img = byb.loadImg(fileNames, int(x[-1]), datapath)#[100:]
            cmap = confidenceMap(img,rsize=True)
            cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)#[strt:end]
            
            alldat.append((img,cmap))
            
        
    '''
    Split the data in the 8 paths 
    '''
    # Split the data into paths
    paths = np.array(np.split(np.array(alldat), 8, axis=0))  # Shape: (8, 41, 2, h, w)
    coords = np.array(np.split(allmove, 8, axis=0))
    
    #COORDINATES
    #xmove,ymove
    cloo,clin,cfin,cfoo,rfin,rfoo,rloo,rlin = coords #split in the correct 
    cloofake = np.reshape(np.tile(cloo, (41, 1)), (41, 41, 8))
    clinfake = np.reshape(np.tile(clin, (41, 1)), (41, 41, 8))
    cfinfake = np.reshape(np.tile(cfin, (41, 1)), (41, 41, 8))
    cfoofake = np.reshape(np.tile(cfoo, (41, 1)), (41, 41, 8))
    rfinfake = np.reshape(np.tile(rfin, (41, 1)), (41, 41, 8))
    rfoofake = np.reshape(np.tile(rfoo, (41, 1)), (41, 41, 8))
    rloofake = np.reshape(np.tile(rloo, (41, 1)), (41, 41, 8))
    rlinfake = np.reshape(np.tile(rlin, (41, 1)), (41, 41, 8))
    
    #Basically I copy the y position to all the elements in the first set
    #Effectively shifting all the acquisition cordinates in y (same for x)
    for i,pos in enumerate(clin):
        cloofake[i, :, 1] = pos[1]
    for i,pos in enumerate(cloo):
        clinfake[i, :, 0] = pos[0]
        
    for i,pos in enumerate(cfoo):
        cfinfake[i, :, 1] = pos[1]
    for i,pos in enumerate(cfin):
        cfoofake[i, :, 0] = pos[0]
        
    for i,pos in enumerate(cfoo):
        rfinfake[i, :, 1] = pos[1]
    for i,pos in enumerate(cfin):
        rfoofake[i, :, 0] = pos[0]
        
    for i,pos in enumerate(clin):
        rloofake[i, :, 1] = pos[1]
    for i,pos in enumerate(cloo):
        rlinfake[i, :, 0] = pos[0]
    
    
    #IMAGES
    # Separate curve and rotation data
    curve = paths[:4]  # Shape: (4, 41, 2, h, w)
    rot = paths[4:]    # Shape: (4, 41, 2, h, w)
    
    # Define in-plane and out-of-plane images (2,41,2,h,w)
    ooplane = np.stack([curve[0], curve[-1]])  # Combine first and last curves
    inplane = curve[1:3]  # Select the second and third curves
    
    ooplaneR = np.stack([rot[0], rot[-1]])  # Combine first and last curves
    inplaneR = rot[1:3]  # Select the second and third curves
    
    # Create search space
    searchSpace = np.empty((41, 41), dtype=object)  # Use 'object' to store images
    
    # Create spaceA
    spaceA = searchSpace.copy()
    for i in range(41):
        for j in range(41):
            # Randomly select an image from either ooplane[1] or ooplane[2] at index [i][0]
            rdm = np.random.randint(0,2)
            random_choice =ooplane[rdm, i, :]  # Select from ooplane
            spaceA[i, j] = random_choice
    
    # Create spaceB
    spaceB = searchSpace.copy()
    for i in range(41):
        for j in range(41):
            # Randomly select an image from either inplane[0] or inplane[-1] at index [i][0]
            rdm = np.random.randint(0,2)
            random_choice = inplane[rdm, i, :]
            spaceB[i, j] = random_choice
    
    # Create spaceC
    spaceC = searchSpace.copy()
    for i in range(41):
        for j in range(41):
            # Randomly select an image from either ooplane[1] or ooplane[2] at index [i][0]
            rdm = np.random.randint(0,2)
            random_choice =ooplaneR[rdm, i, :]  # Select from ooplane
            spaceC[i, j] = random_choice
    
    # Create spaceD
    spaceD = searchSpace.copy()
    for i in range(41):
        for j in range(41):
            # Randomly select an image from either inplane[0] or inplane[-1] at index [i][0]
            rdm = np.random.randint(0,2)
            random_choice = inplaneR[rdm, i, :]
            spaceD[i, j] = random_choice
    
    '''
    Cost Function
    '''
    
    def costfunc(w,cmap,scaler):
        strt,end=2000,2800
        rsize=False
        coords=None
        
        # #Hilbert → Crop → MinMax Line → Mean Lines → Mean
        # himg = byb.envelope(img)
        # crop = himg[strt:end, :]
        # imgc = crop.copy()
        # for k in range(crop.shape[1]):
        #     line = crop[:,k]
        #     min_val = np.min(line)
        #     max_val = np.max(line)
        #     line = (line - min_val) / (max_val - min_val)
        #     imgc[:,k] = line 
        # lineMean = np.mean(imgc,axis=1)
        # if rsize:
        #     lineMean = resize(lineMean, [800], anti_aliasing=True)
        # r13 = np.mean(lineMean)
        
        # #Confidence Map → Crop → Mean Lines → Deriv. → Abs → Prob. → Variance
        # if coords is not None:
        #     crop = cmap[strt-10:end]
        # else:
        #     crop = cmap[strt:end]
        # lineMean = np.mean(crop,axis=1)
        # deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//16,
        #                             polyorder=2, deriv=1))
        # t=deriv
        # if rsize:
        #     t = resize(t, [800], anti_aliasing=True)
        # probs = t/np.sum(t)
        # x = np.arange(len(probs))
        # avg = np.sum(x*probs)
        # var = np.sum((x - avg)**2 * probs)
        # r4 = var
    
        # #Hilbert → Log → Crop → Log norm. → Mean
        # himg = byb.envelope(img)
        # loghimg = 20*np.log10(himg+1)
        # crop = loghimg[strt:end, :]
        # if rsize:
        #     crop = resize(crop, [800,129], anti_aliasing=True)
        # # print(crop.shape)
        # crop = crop - crop.max()
        # r9 = crop.mean()
    
        # #Hilbert → Log → Crop → Laplace → Variance
        # himg = byb.envelope(img)
        # loghimg = 20*np.log10(himg+1)
        # crop = loghimg[strt:end, :]
        # if rsize:
        #     crop = resize(crop, [800,129], anti_aliasing=True)
        # # print(crop.shape)
        # lap = laplace(crop)
        # r12 = lap.var()
    
        # ########################
    
        # features = [r13,r4,r9,r12]
        # scaled = scaler.transform([features])[0]
        # w = [-0.38615809, -0.13791214, -0.00528005, -0.02866812]
        # b = 0.2903600049096854
        
        #Hilbert → Crop → Mean Lines → Prob. → Variance
        himg = byb.envelope(w)
        crop = himg[strt:end, :]
        t = np.mean(crop,axis=1)
        if rsize:
            t = resize(t, [800], anti_aliasing=True, order=5)
        probs = t/np.sum(t)
        x = np.arange(len(probs))
        avg = np.sum(x*probs)
        var = np.sum((x - avg)**2 * probs)
        r0 = var
        
        #Confidence Map → Crop → Mean Lines → Deriv. → Abs → Prob. → Variance
        crop = cmap[strt:end]
        lineMean = np.mean(crop,axis=1)
        if rsize:
            lineMean = resize(lineMean, [800], anti_aliasing=True)
        deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//16,
                                    polyorder=2, deriv=1))
        t=deriv
        # if rsize:
        #     t = resize(t, [800], anti_aliasing=True, order=5)
        probs = t/np.sum(t)
        x = np.arange(len(probs))
        avg = np.sum(x*probs)
        var = np.sum((x - avg)**2 * probs)
        r6 = var
        
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
            imgc = resize(imgc, [800,129], anti_aliasing=True, order=5)
        r12 = imgc.mean()
        
        #Hilbert → Log → Crop → Laplace → Variance
        himg = byb.envelope(w)
        loghimg = 20*np.log10(himg+1)
        crop = loghimg[strt:end, :]
        if rsize:
            crop = resize(crop, [800,129], anti_aliasing=True, order=5)
        # print(crop.shape)
        lap = laplace(crop)
        r16 = lap.var()
        
        features = [r0,r6,r12,r16]
        scaled = scaler.transform([features])[0]
        w = [0.43713773, -0.56152499, -0.90599132, -0.14218985]
        b = 0.9528786769862885
    
        weighted = np.array(scaled) * np.array(w)
        
        cost = np.sum(weighted) + b
    
        return -cost
    
    '''
    load scaler
    '''
    #load the scaler
    with open(current_dir / 'data' / 'scalers' / 'final2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    '''
    bayes op
    '''
    def runLUS(cloofake, spaceA, scaler, plot=True):
        lim=1e-16
        space = [
            Real(np.min(cloofake[:, :, 0])-lim,
                 np.max(cloofake[:, :, 0])+lim, name='x0'),
            
            Real(np.min(cloofake[:, :, 1])-lim,
                 np.max(cloofake[:, :, 1])+lim, name='x1'),
            
            Real(np.min(cloofake[:, :, 2])-lim
                 , np.max(cloofake[:, :, 2])+lim, name='x2'),
            
            Real(np.min(cloofake[:, :, 3])-lim,
                 np.max(cloofake[:, :, 3])+lim, name='x3'),
            
            Real(np.min(cloofake[:, :, 4])-lim,
                 np.max(cloofake[:, :, 4])+lim, name='x4'),
            
            Real(np.min(cloofake[:, :, 5])-lim,
                 np.max(cloofake[:, :, 5])+lim, name='x5'),
            
            Real(np.min(cloofake[:, :, 6])-lim,
                 np.max(cloofake[:, :, 6])+lim, name='x6')
        ]
        
        optim = ManualGPMinimize(costfunc,
                                space,
                                n_initial_points=3, #points before optimizing
                                n_restarts_optimizer=2,
                                n_jobs=-1, #num cores to run optim
                                verbose=True,
                                n_points = 10000, # number of predicted points, from which one is taken 
                                )
        
        # Choose a random item
        # Define row and column indices for the edges
        edge_rows = [0, cloofake.shape[0] - 1]  # First and last rows
        edge_cols = [0, cloofake.shape[1] - 1]  # First and last columns
        
        # Choose a random edge row or column
        if np.random.rand() > 0.5:
            # Random row on the edge, with any column
            row = np.random.choice(edge_rows)
            col = np.random.randint(0, cloofake.shape[1])
        else:
            # Random column on the edge, with any row
            row = np.random.randint(0, cloofake.shape[0])
            col = np.random.choice(edge_cols)
        startingPoint = cloofake[row, col]
        
        pos = byb.findClosestPosition(startingPoint[-1],cloofake,cloofake)
        
        # print('init pos', pos)
        
        res = []
        tested_positions = []  # To store the tested positions
        past_moves=[]
        # for i in tqdm(range(100), desc="Processing", unit="iteration"):
        for i in range(100):    
            # row,col = np.where((cloofake == pos).all(axis=-1))
            # Calculate Euclidean distances
            distances = np.linalg.norm(cloofake[:,:,:-1] - pos[:-1], axis=-1)
            
            # Find the indices of the minimum distance
            row, col = np.unravel_index(np.argmin(distances), distances.shape)
            
            #Get the image and cmap in that position
            img,cmap = spaceA[row,col]
            
            #add speckle noise
            if np.random.rand() > 0.5:
                level = 0.01 + (0.2 - 0.01) * np.random.rand()
                noise = np.random.normal(0, level, img.shape).astype(np.float32)
                # print(img.max(),img.min())
                img = img + img * noise
                # print(img.max(),img.min())
                # Calculate power of the signal
                signal_power = np.mean(img ** 2)
                # Calculate power of the noise
                noise_power = np.mean((img * noise) ** 2)
                # print(10 * np.log10(signal_power / noise_power))
            
            cost = costfunc(img,cmap,scaler) #check score of last suggested position
            optim.update(pos[:-1].tolist(), cost)  # Update the optimizer
            
            res.append(cost)
        
            nextMove = optim.step() #get new move
            # print('suggested next',nextMove)
            pos = byb.findClosestPosition(nextMove,cloofake,cloofake)
            # print('actual next',pos)
            # while True:
            #     nextMove = optim.step() #get new move
            #     pos = byb.findClosestPosition(nextMove,cloofake,cloofake)
            #     pos = pos.tolist()
            #     if len(past_moves) == 0 or pos[:-1] not in past_moves:
            #         break
            #     print(nextMove)
            # past_moves.append(pos[:-1])
            # pos = np.array(pos)
            
            # Store the tested position
            tested_positions.append((row, col))
            
            if plot:
                # Plot the current image and the grid of positions
                plt.figure(figsize=(12, 6))  # Set the figure size
            
                # Subplot for the current image
                plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
                plt.imshow(20 * np.log10(byb.envelope(img)+1), aspect='auto')  # Assuming the image is grayscale; adjust as needed
                plt.title(f'Current Image at Position (Row: {row}, Col: {col})', fontsize=16)
                plt.axis('off')  # Hide axis
            
                # Subplot for the grid of positions
                plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
                grid_color = np.zeros(cloofake.shape[:2])  # Create a grid filled with zeros
            
                # Mark tested positions
                for r, c in tested_positions:
                    grid_color[r, c] = 1  # Mark past tested positions in red
            
                # Mark the current position
                grid_color[row, col] = 2  # Mark current position in blue
            
                # Create a color map
                color_map = plt.cm.colors.ListedColormap(['white', 'red', 'blue'])  # Define colors for the grid
                plt.imshow(grid_color, cmap=color_map, aspect='auto')  # Display the grid with color mapping
                plt.title('Position Grid', fontsize=16)
                plt.axis('on')  # Show axis
            
                plt.grid(False)  # Disable default grid lines
            
                # Optional: add grid lines for better visibility
                for r in range(cloofake.shape[0]):
                    plt.axhline(r - 0.5, color='black', linewidth=0.5, linestyle='--')  # Horizontal grid lines
                for c in range(cloofake.shape[1]):
                    plt.axvline(c - 0.5, color='black', linewidth=0.5, linestyle='--')  # Vertical grid lines
            
                plt.legend(['Tested Position', 'Current Position'], loc='upper right')  # Add a legend
                plt.tight_layout()  # Adjust layout to fit elements
                plt.show()  # Display the plots
        
        res = np.array(res)
        
        if plot:
            # Create a plot for the first column of `res`
            plt.figure(figsize=(10, 6))  # Set the figure size
            plt.plot(res, marker='o', linestyle='-', color='blue', markersize=5)  # Use markers and a line
            plt.title('Learning curve', fontsize=16)  # Add a title
            plt.xlabel('Index', fontsize=14)  # Add x-axis label
            plt.ylabel('Value', fontsize=14)  # Add y-axis label
            plt.grid(True)  # Add a grid for better readability
            plt.axhline(0, color='red', linestyle='--', label='Zero Line')  # Add a reference line at y=0
            plt.legend()  # Add a legend
            plt.tight_layout()  # Adjust layout to fit elements
            plt.show()  # Display the plot
            
        return res, tested_positions
    
    
    # for i in range(100):
    allres = []
    alltpos = []
    
    # import multiprocessing
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for i in tqdm(range(100), desc="Processing", unit="iteration"):
        # tempA = pool.apply_async(runLUS, args=(cloofake, spaceA, scaler, False))
        # tempB = pool.apply_async(runLUS, args=(cloofake, spaceB, scaler, False))
        # tempC = pool.apply_async(runLUS, args=(cloofake, spaceC, scaler, False))
        # tempD = pool.apply_async(runLUS, args=(cloofake, spaceD, scaler, False))
        
        
        #rot+trans
        fakecords = random.choice([cloofake, cfoofake])
        resA, tested_positionsA = runLUS(fakecords, spaceA, scaler, plot=False)
        fakecords = random.choice([clinfake, cfinfake])
        resB, tested_positionsB = runLUS(fakecords, spaceB, scaler, plot=False)
        
        #only rot
        fakecords = random.choice([rloofake, rfoofake])
        resC, tested_positionsC = runLUS(fakecords, spaceC, scaler, plot=False)
        fakecords = random.choice([rlinfake, rfinfake])
        resD, tested_positionsD = runLUS(fakecords, spaceD, scaler, plot=False)
        
        # resA, tested_positionsA = tempA.get()
        # resB, tested_positionsB = tempB.get()
        # resC, tested_positionsC = tempC.get()
        # resD, tested_positionsD = tempD.get()
        
        allres.append((resA,resB,resC,resD))
        alltpos.append((tested_positionsA,tested_positionsB,tested_positionsC,tested_positionsD))

allres = np.array(allres)
postest = np.array(alltpos)

# Shape: (10, 4, 100) -> 10 runs, 4 search spaces, 100 steps
num_runs, num_spaces, num_steps = allres.shape

# Calculate the average learning curve for each search space
space_averages = allres.mean(axis=0)  # Shape: (4, 100)

# Calculate the global average learning curve across all search spaces
global_average = allres.mean(axis=(0, 1))  # Shape: (100,)

# Calculate the global standard deviation across all search spaces
global_std = allres.std(axis=(0, 1))  # Shape: (100,)

# Plotting
plt.figure(figsize=(10, 6))

# Plot each search space's average learning curve
for i in range(num_spaces):
    plt.plot(space_averages[i], label=f"Search Space {i+1}")

# Plot the global average
plt.plot(global_average, label="Global Average", linestyle='--', linewidth=2, color='black')

# Add a shaded region for standard deviation
plt.fill_between(
    range(num_steps), 
    global_average - global_std, 
    global_average + global_std, 
    color='black', 
    alpha=0.2, 
    label="Global Std Dev"
)

# Add plot details
plt.title("Average Learning Curves Across Search Spaces", fontsize=14)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Performance", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()

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

# Calculate the global standard deviation of distances across all search spaces
global_distances_std = distances.std(axis=(0, 1))  # Shape: (100,)

# Plotting
plt.figure(figsize=(10, 6))

# Plot each search space's average distance curve
for i in range(space_distances_avg.shape[0]):
    plt.plot(space_distances_avg[i], label=f"Search Space {i+1}")

# Plot the global average distance curve
plt.plot(global_distances_avg, label="Global Average", linestyle='--', linewidth=2, color='black')

# Calculate clamped bounds for the shaded region
lower_bound = np.maximum(global_distances_avg - global_distances_std, 0)
upper_bound = global_distances_avg + global_distances_std

# Plot the shaded region
plt.fill_between(
    range(global_distances_avg.shape[0]), 
    lower_bound, 
    upper_bound, 
    color='black', 
    alpha=0.2, 
    label="Global Std Dev"
)

# Add plot details
plt.title("Average Distance to Middle Row (Index 20) Across Iteration Steps", fontsize=14)
plt.xlabel("Iteration Steps", fontsize=12)
plt.ylabel("Distance to Middle Row", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()

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
fig.suptitle("Average Tested Positions Heatmaps", fontsize=16)

# Show the plot
plt.show()

#heatmap w log
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
fig.suptitle("Average Tested Positions Heatmaps (Logarithmic Scale)", fontsize=16)

# Show the plot
plt.show()
 