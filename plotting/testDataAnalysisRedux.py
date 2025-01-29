# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:09:56 2024

@author: Mateo-drr
"""


from pathlib import Path
import numpy as np
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
current_dir = Path(__file__).resolve().parent.parent.parent
import sys
base = current_dir / 'ALU---Autonomous-Lung-Ultrasound'
sys.path.append(base.as_posix())
base = base / 'imgProcessing'
sys.path.append(base.as_posix())
import byble as byb
from confidenceMap import confidenceMap

loadData=True

if loadData:
    current_dir = Path(__file__).resolve().parent.parent.parent
    alldat,alldat0,filtRF,_ = byb.loadAllData(current_dir)


#%%

###############################################################################
'''
Theoretical score for positions
'''
###############################################################################
# '''
x = np.linspace(-1, 1, 41)
sigma = 0.3  # Adjust sigma for the desired smoothness
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)

idealScores = []
for run in filtRF:
    temp = []
    for point in run[-1]:
        temp.append([gaussian_array[min(round(point + 20), 40)], min(round(point + 20), 40), point])
    idealScores.append(temp)

for run in idealScores:
    run = np.array(run)
    rmax = np.max(run[:,0])
    print(np.where(run[:,0] == rmax))
    # print(np.where(run[:,-1] == np.min(run[:,-1])))
    print(run[:,-1])
    
gg=[16, 0, 3, 1, 14, 17, 13, 7, 1, 1]

# Plotting
plt.figure(dpi=200)
plt.plot(gaussian_array[:])

# Overlay Ideal Scores as Points
for i, scores in enumerate(idealScores):
    scores = np.array(scores)
    
    best = scores[gg[i],0]
    
    best = np.max(scores[:, 0])
    # step = np.argmax(scores[:, 0])
    idx = scores[gg[i], 1]
    print(best,idx)

    # Jittering for overlapping points
    jitter_strength = 1.5  # You can adjust this value
    jittered_idx = idx + 1*i

    plt.scatter(jittered_idx, best)
    plt.text(jittered_idx, best, f'r{i}-'+str(gg[i]), fontsize=8, ha='center',
             va='bottom' if i%3==0 else 'top')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

crops=[[100,-250],
       [100,-250],
       [100,-150],
       [125,-200],
       [75,-275],
       [75,-250],
       [75,-250],
       [75,-250],
       [175,-150],
       [175,-150]]
#for chest
# crops = [
#     [20, 180],
#     [70, 180],
#     [50, 150],
#     [30, 180],
#     [40, 150],
#     [30, 130],
#     [25, 120],
#     [50, 200],
#     [40, 160],
#     [60, 120],
# ]

test_ds = []
coords = []
for i,run in enumerate(filtRF):
    temp = []
    for img in run[0]:
        img = np.array(img)
        cmap = confidenceMap(img[10:], rsize=False)
        cmap = resize(cmap, img.shape, anti_aliasing=True)
        # cmapbig = confidenceMap(imgbig[10*12:], rsize=True)
        # cmapbig = resize(cmap, (6292-10,129), anti_aliasing=True)
        #test_ds.append((img,cmap, imgbig,cmapbig))
        temp.append((img,cmap))
        coords.append((crops[i]))
    test_ds.append(temp)

#plot images and cmap

# for img in test_ds:
#     img, cmap = img
#     # Set up the figure and axes
#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 3 columns
#     # Display the first image
#     ax[0].imshow(20*np.log10(byb.envelope(img)+1), aspect='auto')
#     ax[0].set_title("Image")
#     # Display the second image
#     ax[1].imshow(cmap, aspect='auto')
#     ax[1].set_title("Cmap")
#     # Display the third image
#     # ax[2].imshow(cmapbig, aspect='auto')
#     # ax[2].set_title("CmapBig")
#     # Remove axes for clarity
#     for a in ax:
#         a.axis('off')
#     plt.tight_layout()
#     plt.show()
    
#%% Test features on test images

i=0
res=[]
for run in test_ds:
    temp=[]
    for data in run:
        w,cmap = data
        strt,end = coords[i]
        rsize=True
        
        sfactor = 6292/512
        strt = round(strt*sfactor)
        end = round((512-end)*sfactor)
        imgog = w.copy()
        w = resize(w, (6292,129), anti_aliasing=True, order=5)
        cmapog= cmap.copy()
        # cmap = confidenceMap(w, rsize=True)
        cmap = resize(cmap, (6292,129), anti_aliasing=True, order=5)
        
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

        i+=1
        
        #PENALTY
        confC = np.sum(cmapog < 0.85)
        confRT = confC / cmapog.size
        
        if confRT <= 0.9:
            temp.append((r0,r6,r12,r16))
        else:
            temp.append([0,])
    res.append(temp)

bestidx=[]
for run in res:
    maxc = -100
    besti=0
    print(len(run))
    for i,feat in enumerate(run):
        
        #f0,6,12,16
        w = [0.43713773, -0.56152499, -0.90599132, -0.14218985]
        b = 0.9528786769862885
        #f0,6,12
        # w = [ 0.43177183 -0.60373572 -0.93485367]
        # b = 0.9200032405081735
        
        #inverted
        #f0,6,12,16
        # w = [-0.43713773,  0.56152499,  0.90599132,  0.14218985]
        # b = 0.047121323013711414
        #f0,6,12
        # w = [-0.43177183,  0.60373572,  0.93485367]
        # b = 0.07999675949182661
        # w = [0.4547998,  0.67371348]
        # b = 0.06914889853489137
        
        if len(feat) == 1:
            cost = -100
        else:
            sfeat = robust_scaler.transform([feat])[0]
            
            wfeat = np.array(sfeat) * np.array(w)
            
            cost = wfeat.sum() + b
        
        if cost >= maxc:
            maxc = cost
            besti=i
            print(i, feat, f'{maxc:.2f}')
            print(i, sfeat, f'{maxc:.2f}')
            print(i, wfeat, f'{maxc:.2f}')
    print(besti, f'{maxc:.2f}\n')
    bestidx.append(besti)

print(bestidx)
#%%
# '''
###############################################################################
'''
Feature Calculation
'''
###############################################################################

def varlines(img,strt,end,rsize=False):
    himg = byb.envelope(img)
    loghimg = 20*np.log10(himg+1)
    crop = loghimg[strt:end, :]
    lineMean = np.mean(crop,axis=1)
    
    if rsize:
        lineMean = resize(lineMean, [800], anti_aliasing=True)
    
    var = np.var(lineMean)
    return var 
    
def cmapderiv(cmap,strt,end,rsize=False):
    crop = cmap[strt:end]
    lineMean = np.mean(crop,axis=1)
    
    if rsize:
        lineMean = resize(lineMean, [800], anti_aliasing=True)
    
    deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//16, polyorder=2, deriv=1))
    probs = deriv/np.sum(deriv)
    x = np.arange(len(probs))
    avg = np.sum(x*probs)
    var = np.sum((x - avg)**2 * probs)
    return var

def intencity(img,strt,end,rsize=False):
    himg = byb.envelope(img)
    crop = himg[strt:end, :]
    
    if rsize:
        crop = resize(crop, [800], anti_aliasing=True)
    
    p25, p75 = np.percentile(crop, [25, 75])
    lum = np.mean((crop-p25)/(p75-p25))
    
    return lum

def lapvar(img,strt,end,rsize=False):
    himg = byb.envelope(img)
    loghimg = 20*np.log10(himg+1)
    crop = loghimg[strt:end, :]
    
    if rsize:
        crop = resize(crop, [800], anti_aliasing=True)
    
    lap = laplace(crop)
    var = np.var(lap)
    return var

def calculate_metrics(data,strt=0,end=0,rsize=False,coords=None):
    cost = []
    
    for idx,pos in enumerate(data):
        metrics = []
        img, cmap = pos
        
        if coords is not None:
            strt,end = coords[idx]
        
        metrics.append(varlines(img, strt, end, rsize=rsize))
        
        metrics.append(-cmapderiv(cmap, strt, end, rsize=rsize))
        
        metrics.append(intencity(img, strt, end, rsize=rsize))
        
        metrics.append(-lapvar(img, strt, end, rsize=rsize))
        
        cost.append(metrics)
    
    return cost

cost = calculate_metrics(alldat,2000,2800)
cost0 = calculate_metrics(alldat0,1800,3500, rsize=True)

labels = [
    'Variance of joint lines',
    'Variance of conf deriv',
    'Mean hilb intensity', 
    'Variance laplacian hilb+log',
]

#%% var mean graphs of both with and without meat in one

# Define function for processing each dataset
def process_data(dataset):
    # justimgs = np.array(dataset) #use this for angles load both cmap and img
    justimgs = np.array([(img[0]) for img in dataset])
    paths = np.array(justimgs)
    paths = np.array(np.split(paths, 8, axis=0))

    all_means = []
    all_variances = []
    for path in paths:
        qwer = []
        for w in path:
            strt,end=2000,2800
            rsize=False
            coords=None
                        
            # himg = byb.envelope(w[0])
            # loghimg = 20*np.log10(himg+1)
            # crop = loghimg[strt:end, :]
            # nimg = byb.normalize(crop)
            # peaks = byb.findPeaks(nimg)
            # l1, ang1, _, _, r21, rmse1 = byb.regFit(peaks)
            
            # crop = w[1][strt:end]
            # cxhist = np.sum(crop,axis=0)
            # l2, ang2, _, _, r22, rmse2 = byb.regFit(cxhist)
            
            # # r0=abs(ang1)
            # r0=abs(ang2)
            # r0=(abs(ang1)+abs(ang2))/2
            # r0=abs(r21)
            # r0=abs(r22)
            # r0=(abs(r22) + abs(r21))/2
            # r0=abs(rmse1)
            # r0=abs(rmse2)
            # r0=(abs(rmse2) + abs(rmse1))/2

            
            # himg = byb.envelope(w)
            # loghimg = 20*np.log10(himg+1)[strt:end, :]
            # crop = ((loghimg-loghimg.min())/(loghimg.max()-loghimg.min()))
            # if rsize:
            #     crop = resize(crop, [800,129], anti_aliasing=True)
            # # print(crop.shape)
            # lap = laplace(crop)
            # r0 = lap.var()
            
            # himg = byb.envelope(w)
            # crop = himg
            # img = crop
            # imgc = crop.copy()
            # for k in range(img.shape[1]):
            #     line = img[:,k]
            #     min_val = np.min(line)
            #     max_val = np.max(line)
            #     line = (line - min_val) / (max_val - min_val)
            #     imgc[:,k] = line 
            # imgc=imgc[strt:end, :]
            # if rsize:
            #     imgc = resize(imgc, [800,129], anti_aliasing=True)
            # r0 = imgc.mean()
            
            # #Hilbert → Log → Crop → Laplace → Variance
            # himg = byb.envelope(w)
            # loghimg = 20*np.log10(himg+1)
            # crop = loghimg[strt:end, :]
            # if rsize:
            #     crop = resize(crop, [800,129], anti_aliasing=True)
            # # print(crop.shape)
            # lap = laplace(crop)
            # r0 = lap.var()
            
            #Hilbert → Crop → MinMax Line → mean
            himg = byb.envelope(w)
            crop = himg
            img = crop
            imgc = crop.copy()
            for k in range(img.shape[1]):
                line = img[:,k]
                min_val = np.min(line)
                max_val = np.max(line)
                line = (line - min_val) / (max_val - min_val)
                imgc[:,k] = line 
            imgc=imgc[strt:end, :]
            if rsize:
                imgc = resize(imgc, [800,129], anti_aliasing=True)
            r0 = imgc.mean()
            
            qwer.append([r0, r0])

        # Extract means and variances for each path
        means = [item[0] for item in qwer]
        variances = [item[1] for item in qwer]
        all_means.append(means)
        all_variances.append(variances)

    all_means = np.array(all_means)
    all_variances = np.array(all_variances)

    # Calculate average means and variance among paths
    avg_means = all_means.mean(axis=0)
    variance_among_paths = np.var(all_means, axis=0)
    avg_variance_number = np.mean(variance_among_paths)
    cv_per_angle = 100 * np.std(all_means, axis=0) / avg_means

    return avg_means, variance_among_paths, avg_variance_number, cv_per_angle, all_means, all_variances

# Process each dataset
avg_means_0, var_paths_0, avg_var_0, cv_angle_0, means_0, variances_0 = process_data(alldat0)
avg_means_all, var_paths_all, avg_var_all, cv_angle_all, means_all, variances_all = process_data(alldat)

# Define x values and adjust to match the length of avg_means
x_values = list(range(-20, 21))
extended_means_0 = avg_means_0[:len(x_values)]
extended_means_all = avg_means_all[:len(x_values)]

# Plotting with both datasets in the same plot
name = 'Predicted Angle'
xlbl = f'Degrees\nCoefficient of variation (alldat0): {cv_angle_0.mean():.2f}%, (alldat): {cv_angle_all.mean():.2f}%'

# Errorbar plot with averages and variances for alldat0 and alldat
plt.figure(figsize=(10, 6), dpi=200)
plt.errorbar(x_values, extended_means_0, yerr=var_paths_0**0.5, fmt='-o', label='alldat0', ecolor='blue', capsize=5, capthick=2)
plt.errorbar(x_values, extended_means_all, yerr=var_paths_all**0.5, fmt='-o', label='alldat', ecolor='red', capsize=5, capthick=2)
plt.xlabel(xlbl, fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel(name, fontsize=18)
plt.legend(fontsize=16, loc='upper right')
plt.grid(True)
plt.show()

# Plotting means with average standard deviation on x-axis for both datasets
plt.figure(figsize=(10, 6), dpi=200)
# plt.title('(C)', fontsize=18)
plt.plot(x_values, extended_means_0, '-o', label='without meat')
plt.plot(x_values, extended_means_all, '-o', label='with meat')
plt.xlabel('Degrees', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel(name, fontsize=18)
# plt.ylim(0,50)
# plt.ylim(-5,190)
plt.legend(fontsize=16)
plt.grid(True)
plt.show()


# Iterate to find the minimum and maximum of all means
plegend=['pW along mL', 'pL along mW',
         'pL along mL', 'pW along mW',
         'pL along mL', 'pW along mW',
         'pW along mL', 'pL along mW',]
min_value = float('inf')
max_value = float('-inf')
for i in range(8):
    min_value = min(min_value, np.min(means_0[i][:len(x_values)]), np.min(means_all[i][:len(x_values)]))
    max_value = max(max_value, np.max(means_0[i][:len(x_values)]), np.max(means_all[i][:len(x_values)]))

# Subplot with individual paths for alldat0 and alldat
fig, axes = plt.subplots(4, 2, figsize=(15, 15), dpi=200)
axes = axes.ravel()
for i in range(8):
    std_devs_0 = np.sqrt(variances_0[i][:len(x_values)])
    std_devs_all = np.sqrt(variances_all[i][:len(x_values)])

    axes[i].plot(x_values, means_0[i][:len(x_values)], '-o', label='without meat')
    axes[i].plot(x_values, means_all[i][:len(x_values)], '-o', label='with meat')
    axes[i].set_title(f'Path {i+1}: {plegend[i]}')
    axes[i].set_xlabel('Degrees')
    axes[i].set_ylabel('Predicted Angle')
    axes[i].grid(True)
    # Set y-axis limit to the global min and max for consistent scaling
    axes[i].set_ylim(min_value-5, max_value+5)
    
# Adjust layout and show single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 0.95))
plt.tight_layout()
plt.show()

'''
Mean Var graphs
'''
#%%

datapath = Path(__file__).resolve().parent.parent / 'imgProcessing' / 'ml' / 'lines'
fileNames = [f.name for f in datapath.iterdir() if f.is_file()]

lines = []
for i in range(0, 4):
    btm = np.load(datapath / fileNames[i])
    top = np.load(datapath / fileNames[i+4])
    lines.append([top, btm])

lbls = np.concatenate(np.transpose(lines, (0, 2, 1)))
lbls = np.array(np.split(lbls, 8, axis=0))

# Group the data into 8 paths with 41 images each
justimgs = np.array([(img[0]) for img in alldat])
paths = np.array(justimgs)
paths = np.array(np.split(paths, 8, axis=0))

all_means = []
all_variances = []
arrays = []
for i, path in enumerate(paths):
    qwer = []
    hists = []
    for j, w in enumerate(path):
        
        strt,end = 2000,2800#lbls[i,j]
        rsize=True
        coords=None
        
        himg = byb.envelope(w)
        loghimg = 20*np.log10(himg+1)
        crop = loghimg[strt:end, :]
        img = crop
        imgc = crop.copy()
        for k in range(img.shape[1]):
            line = img[:,k]
            min_val = np.min(line)
            max_val = np.max(line)
            line = line- max_val#(line-min_val)/(max_val-min_val)
            imgc[:,k] = line 
        imgc=imgc
        if rsize:
            imgc = resize(imgc, [800,129], anti_aliasing=True)
        r0 = imgc.mean(axis=1).mean()
        
        avg=r0
        qwer.append([avg, avg])
        # hists.append(t)
    data = qwer
    means = [item[0] for item in data]
    variances = [item[1] for item in data]
    all_means.append(means)
    all_variances.append(variances)

# plt.figure(dpi=200)
# for k in hists:
#     plt.plot(k)
# plt.show()

all_means = np.array(all_means)
all_variances = np.array(all_variances)

avg_means = all_means.mean(axis=0) #means among paths
x_values = list(range(-20, 21))
extended_means = avg_means[:len(x_values)]

all_meansn = (all_means - all_means.min())/(all_means.max() - all_means.min())
avg_means = all_meansn.mean(axis=0) #means among paths

variance_among_paths = np.var(all_means, axis=0)
avg_variance_number = np.mean(variance_among_paths)

cv_per_angle = 100*np.std(all_meansn, axis=0)/avg_means
print(cv_per_angle.mean())

colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
name='Mean'
xlbl = 'Degrees'#f'Degrees\nCoefficient of variation: {cv_per_angle.mean():.2f}%'
plt.figure(figsize=(10, 6), dpi=200)
plt.title('(B)', fontsize=18)
plt.errorbar(x_values, extended_means, yerr=variance_among_paths**0.5, fmt='-o', ecolor='r',capsize=5, capthick=2, color=colors[1])
plt.xlabel(xlbl, fontsize=18)
# plt.ylabel(name, fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(-40,-7.5)
# plt.legend(fontsize=16)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6), dpi=200)
plt.plot(x_values, extended_means, '-o' )
plt.xlabel(f'Degrees\nAverage STD: {avg_variance_number**0.5}', fontsize=18)
plt.ylabel(name, fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(15, 15), dpi=200) 
axes = axes.ravel()
for i, (means, variances) in enumerate(zip(all_means, all_variances)):
    std_devs = np.sqrt(variances[:len(x_values)])
    axes[i].plot(x_values, means[:len(x_values)], '-o')
    axes[i].set_title(f'Path {i+1}')
    axes[i].set_xlabel('Degrees')
    axes[i].set_ylabel('Mean Value')
    axes[i].grid(True)
plt.tight_layout()
plt.show()

#%%
def allcombs(paths,strt=2000,end=2800,rsize=False,coords=None):
    all_scores = []
    for i, path in enumerate(paths):
        scores = []
        for j, w in enumerate(path):
            
            w,cmap = w
            
            if coords is not None:
                strt,end = coords[i]
            
            '''
            Joint lines
            '''
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
            #Hilbert → Log → Crop → Mean Lines → Prob. → Variance    
            # himg = byb.envelope(w)
            # loghimg = 20*np.log10(himg+1)
            # crop = loghimg[strt:end, :]
            # t = np.mean(crop,axis=1)
            # if rsize:
            #     t = resize(t, [800], anti_aliasing=True)
            # probs = t/np.sum(t)
            # x = np.arange(len(probs))
            # avg = np.sum(x*probs)
            # var = np.sum((x - avg)**2 * probs)
            # r1 = var  
            #Hilbert → Crop → Mean Lines → Variance
            himg = byb.envelope(w)
            crop = himg[strt:end, :]
            lineMean = np.mean(crop,axis=1)
            if rsize:
                lineMean = resize(lineMean, [800], anti_aliasing=True)
            r2 = lineMean.var()
            #Hilbert → Log → Crop → Mean Lines → Variance
            himg = byb.envelope(w)
            loghimg = 20*np.log10(himg+1)
            crop = loghimg[strt:end, :]
            lineMean = np.mean(crop,axis=1)
            if rsize:
                lineMean = resize(lineMean, [800], anti_aliasing=True)
            r3 = lineMean.var()
            #Hilbert → MinMax Line → Crop → Mean Lines → var
            himg = byb.envelope(w)
            crop = himg
            img = crop
            imgc = crop.copy()
            for k in range(img.shape[1]):
                line = img[:,k]
                min_val = np.min(line)
                max_val = np.max(line)
                line = (line - min_val) / (max_val - min_val)
                imgc[:,k] = line 
            lineMean = np.mean(imgc[strt:end, :],axis=1)
            if rsize:
                lineMean = resize(lineMean, [800], anti_aliasing=True)
            r4 = np.var(lineMean)
            #Hilbert → Log → Crop → MinMax Line → Mean Lines → var
            # himg = byb.envelope(w)
            # loghimg = 20*np.log10(himg+1)
            # crop = loghimg[strt:end, :]
            # img = crop
            # imgc = crop.copy()
            # for k in range(img.shape[1]):
            #     line = img[:,k]
            #     min_val = np.min(line)
            #     max_val = np.max(line)
            #     line = (line - min_val) / (max_val - min_val)
            #     imgc[:,k] = line 
            # lineMean = np.mean(imgc,axis=1)
            # if rsize:
            #     lineMean = resize(lineMean, [800], anti_aliasing=True)
            # r5 = np.var(lineMean)
            
            '''
            Confidence map
            '''
            #Confidence Map → Crop → Mean Lines → Deriv. → Abs → Prob. → Variance
            if coords is not None:
                crop = cmap[strt-10:end]
            else:
                crop = cmap[strt:end]
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
            r6 = var
            #Confidence Map → Crop → Mean Lines → Deriv. → Abs → Mean
            if coords is not None:
                crop = cmap[strt-10:end]
            else:
                crop = cmap[strt:end]
            lineMean = np.mean(crop,axis=1)
            deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//16,
                                        polyorder=2, deriv=1))
            if rsize:
                deriv = resize(deriv, [800], anti_aliasing=True)
            r7 = deriv.mean()
            #Confidence Map → Crop → Mean Lines → Deriv. → Abs → Variance
            if coords is not None:
                crop = cmap[strt-10:end]
            else:
                crop = cmap[strt:end]
            lineMean = np.mean(crop,axis=1)
            deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//16,
                                        polyorder=2, deriv=1))
            if rsize:
                deriv = resize(deriv, [800], anti_aliasing=True)
            r8 = deriv.var()
            
            '''
            Intensity
            '''
            #Hilbert → Crop → Mean
            himg = byb.envelope(w)
            crop = himg[strt:end, :]
            if rsize:
                crop = resize(crop, [800,129], anti_aliasing=True)
            r9 = crop.mean()
            #Hilbert → Log → Crop → Mean
            himg = byb.envelope(w)
            loghimg = 20*np.log10(himg+1)
            crop = loghimg[strt:end, :]
            if rsize:
                crop = resize(crop, [800,129], anti_aliasing=True)
            r10 = crop.mean()
            #Hilbert → Log → Crop → Log norm. → Mean
            himg = byb.envelope(w)
            loghimg = 20*np.log10(himg+1)
            crop = loghimg[strt:end, :]
            if rsize:
                crop = resize(crop, [800,129], anti_aliasing=True)
            crop = crop - crop.max()
            r11 = crop.mean()
            # himg = byb.envelope(w)
            # loghimg = 20*np.log10(himg+1)
            # crop = loghimg[strt:end, :]
            # img = crop
            # imgc = crop.copy()
            # for k in range(img.shape[1]):
            #     line = img[:,k]
            #     min_val = np.min(line)
            #     max_val = np.max(line)
            #     line = line - max_val
            #     imgc[:,k] = line 
            # if rsize:
            #     imgc = resize(imgc, [800,129], anti_aliasing=True)
            # r11 = imgc.mean()
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
            r12 = imgc.mean()
            #Hilbert → Log → Crop → MinMax Line → mean
            # himg = byb.envelope(w)
            # loghimg = 20*np.log10(himg+1)
            # crop = loghimg[strt:end, :]
            # img = crop
            # imgc = crop.copy()
            # for k in range(img.shape[1]):
            #     line = img[:,k]
            #     min_val = np.min(line)
            #     max_val = np.max(line)
            #     line = (line - min_val) / (max_val - min_val)
            #     imgc[:,k] = line 
            # if rsize:
            #     imgc = resize(imgc, [800,129], anti_aliasing=True)
            # r13 = imgc.mean()
            '''
            Laplace
            '''
            #Crop → Laplace → Variance
            crop = w[strt:end,:]
            if rsize:
                crop = resize(crop, [800,129], anti_aliasing=True)
            # print(crop.shape)
            lap = laplace(crop)
            r14 = lap.var()
            #Hilbert → Crop → Laplace → Variance
            himg = byb.envelope(w)
            crop = himg[strt:end, :]
            if rsize:
                crop = resize(crop, [800,129], anti_aliasing=True)
            # print(crop.shape)
            lap = laplace(crop)
            r15 = lap.var()
            #Hilbert → Log → Crop → Laplace → Variance
            himg = byb.envelope(w)
            loghimg = 20*np.log10(himg+1)
            crop = loghimg[strt:end, :]
            if rsize:
                crop = resize(crop, [800,129], anti_aliasing=True)
            # print(crop.shape)
            lap = laplace(crop)
            r16 = lap.var()
            
            '''
            angle
            '''
            # himg = byb.envelope(w)
            # loghimg = 20*np.log10(himg+1)
            # crop = loghimg[strt:end, :]
            # nimg = byb.normalize(crop)
            # peaks = byb.findPeaks(nimg)
            # l1, ang1, _, _, r21, rmse1 = byb.regFit(peaks)
            
            # crop = cmap[strt:end]
            # cxhist = np.sum(crop,axis=0)
            # l2, ang2, _, _, r22, rmse2 = byb.regFit(cxhist)
            
            '''
            lap avg
            '''
            # crop = (w-w.min())/(w.max()-w.min())
            # crop = crop[strt:end, :]
            # if rsize:
            #     crop = resize(crop, [800,129], anti_aliasing=True)
            # # print(crop.shape)
            # lap = laplace(crop)
            # ravg = lap.var()
            '''
            minmax post crop
            '''
            #Hilbert → Crop → MinMax Line → Mean Lines → var
            # himg = byb.envelope(w)
            # crop = himg[strt:end, :]
            # img = crop
            # imgc = crop.copy()
            # for k in range(img.shape[1]):
            #     line = img[:,k]
            #     min_val = np.min(line)
            #     max_val = np.max(line)
            #     line = (line - min_val) / (max_val - min_val)
            #     imgc[:,k] = line 
            # lineMean = np.mean(imgc,axis=1)
            # if rsize:
            #     lineMean = resize(lineMean, [800], anti_aliasing=True)
            # ravg = np.var(lineMean)

                    
            scores.append([r0,r2,r3,r4,r6,r7,r8,r9,r10,r11,r12,r14,r15,r16])
        
        all_scores.append(scores)
    return all_scores

# Group the data into 8 paths with 41 images each
justimgs = np.array(alldat)
paths = np.array(justimgs)
paths = np.array(np.split(paths, 8, axis=0))

all_scores = allcombs(paths,strt=2000,end=2800,rsize=False)
# all_scores = allcombs(paths,strt=1800,end=3500,rsize=True) 

# Convert all_scores to a NumPy array for easier manipulation
all_scores = np.array(all_scores)  # Shape: (8, 41, 13) for 8 paths, 41 images, 13 features

# Separate scores by feature
feature_scores = np.transpose(all_scores, (2, 0, 1))  # Shape: (13, 8, 41)

# Calculate metrics for each feature
feature_metrics = {}
for i, feature_data in enumerate(feature_scores):
    feature_name = f"$f_{i}$"
    
    # feature_data = (feature_data - feature_data.min())/(feature_data.max() - feature_data.min())
    
    # Flatten the data across paths
    flattened_data = feature_data.flatten() #(8,41) --> (328)
    
    mean_among_paths = np.mean(feature_data, axis=0)
    std_among_paths = np.std(feature_data,axis=0)
    
    # Calculate metrics
    # mean_val = flattened_data.mean()
    mean_val = np.mean(mean_among_paths)
    # std_dev_val = flattened_data.std()
    std_dev_val = np.mean(std_among_paths)
    
    cv_val = np.mean(100 * std_among_paths/mean_among_paths)
    
    # iqr_val = np.percentile(flattened_data, 75) - np.percentile(flattened_data, 25)
    iqr_val = np.mean(np.percentile(feature_data, 75, axis=0) - np.percentile(feature_data, 25, axis=0))
    
    # mad_val = np.median(np.abs(flattened_data - np.median(flattened_data)))
    mad_val = np.mean(np.median(np.abs(feature_data - np.median(feature_data, axis=0, keepdims=True)),axis=0))
    
    range_val = flattened_data.max() - flattened_data.min()
    
    # percentile_range_val = np.percentile(flattened_data, 90) - np.percentile(flattened_data, 10)
    percentile_range_val = np.mean(np.percentile(feature_data, 90, axis=0) - np.percentile(feature_data, 10, axis=0))
    
    # skewness_val = skew(flattened_data)
    skewness_val = np.mean(skew(feature_data, axis=0))
    
    # kurtosis_val = kurtosis(flattened_data)
    kurtosis_val = np.mean(kurtosis(feature_data, axis=0))
    
    # mean_to_range_ratio = mean_val / range_val 
    mean_to_range_ratio = np.mean(mean_among_paths / range_val )
    
    # snr_val = mean_val / std_dev_val 
    snr_val = np.mean(mean_among_paths / std_among_paths)
    
    minv,maxv = flattened_data.min(),flattened_data.max()


    
    # Store metrics in a dictionary
    feature_metrics[feature_name] = {
        #"mean": mean_val,
        "std_dev": std_dev_val,
        "CV (%)": cv_val,
        "iqr": iqr_val,
        "mad": mad_val,
        "range": range_val,
        "percentile_range": percentile_range_val,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "mean_to_range_ratio": mean_to_range_ratio,
        "snr": snr_val,
        'min':minv,
        'max':maxv
    }

# for feature_name, metrics in feature_metrics.items():
#     print(f"{feature_name}:")
#     for metric_name, value in metrics.items():
#         print(f"  {metric_name}: {value:.2f}")
#     print()  # Blank line for better readability between features

# for feature_name, metrics in feature_metrics.items():
#     print(f"{feature_name}:")
#     for metric_name, value in metrics.items():
#         # Adjust format based on value size: scientific notation for values < 0.01 or > 1e6
#         if abs(value) < 0.01 or abs(value) > 1e6:
#             print(f"  {metric_name}: {value:.2e}")  # Scientific notation
#         else:
#             print(f"  {metric_name}: {value:.2f}")  # Standard decimal notation
#     print() 
    
    
    
df = pd.DataFrame(feature_metrics).T  # Transpose to make features the rows
# Format all values in scientific notation
def custom_sci_format(x):
    formatted = f"{x:.2e}".replace("e+0", "e").replace("e-0", "e-").replace("e+","e").replace("e-","e-")
    formatted = f"{x:.2f}"
    return formatted

# Apply scientific formatting to all values
formatted_df = df.applymap(custom_sci_format)

# Convert to LaTeX with the desired format
latex_table = formatted_df.to_latex(index=True, header=True, column_format='|c' * (len(df.columns) + 1) + '|', escape=False)
print(latex_table)

#%% Test data metrics
crops=[[100,-250],
       [100,-250],
       [100,-150],
       [125,-200],
       [75,-275],
       [75,-250],
       [75,-250],
       [75,-250],
       [175,-150],
       [175,-150]]
#for chest
# crops = [
#     [20, 180],
#     [70, 180],
#     [50, 150],
#     [30, 180],
#     [40, 150],
#     [30, 130],
#     [25, 120],
#     [50, 200],
#     [40, 160],
#     [60, 120],
# ]
test_ds = []
coords = []
for i,run in enumerate(filtRF):
    temp = []
    for img in run[0]:
        img = np.array(img)
        cmap = confidenceMap(img[10:], rsize=False)
        cmap = resize(cmap, img.shape, anti_aliasing=True)
        # cmapbig = confidenceMap(imgbig[10*12:], rsize=True)
        # cmapbig = resize(cmap, (6292-10,129), anti_aliasing=True)
        #test_ds.append((img,cmap, imgbig,cmapbig))
        temp.append((img,cmap))
        coords.append((crops[i]))
    test_ds.append(temp)

t_scores = allcombs(test_ds,rsize=True,coords=coords)

flat_t = []
for run in t_scores:
    for score in run:
        flat_t.append(score)
flat_t = np.array(flat_t)
        
metrics = {}

for i in range(flat_t.shape[1]):
    method_data = flat_t[:, i]
    
    # Calculate metrics
    mean_val = np.mean(method_data)
    std_val = np.std(method_data)
    cv_val = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    iqr_val = np.percentile(method_data, 75) - np.percentile(method_data, 25)
    mad_val = np.median(np.abs(method_data - np.median(method_data)))
    range_val = np.max(method_data) - np.min(method_data)
    perc_range_val = np.percentile(method_data, 90) - np.percentile(method_data, 10)
    skewness_val = skew(method_data)
    kurtosis_val = kurtosis(method_data)
    mean_to_range_ratio = mean_val / range_val if range_val != 0 else np.nan
    snr_val = mean_val / std_val if std_val != 0 else np.nan
    minv,maxv = method_data.min(),method_data.max()

    # Store in dictionary for each method
    metrics[f"$f_{i}$"] = {
        # "mean": mean_val,
        "std_dev": std_val,
        "CV (%)": cv_val,
        "IQR": iqr_val,
        "MAD": mad_val,
        "range": range_val,
        "percentile_range": perc_range_val,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "mean/range": mean_to_range_ratio,
        "SNR": snr_val,
        'min':minv,
        'max':maxv
    }

# Display metrics for each method
for method, values in metrics.items():
    print(f"{method}:")
    for metric, value in values.items():
        print(f"  {metric}: {value}")
    print()
    
df = pd.DataFrame(metrics).T  # Transpose to make features the rows
# Format all values in scientific notation
def custom_sci_format(x):
    formatted = f"{x:.2e}".replace("e+0", "e").replace("e-0", "e-").replace("e+","e").replace("e-","e-")
    return formatted

# Apply scientific formatting to all values
formatted_df = df#df.applymap(custom_sci_format)

# Convert to LaTeX with the desired format
latex_table = formatted_df.to_latex(index=True, header=True, column_format='|c' * (len(df.columns) + 1) + '|', escape=False)
print(latex_table)

#%%
'''
Scalers Independent Y axis
'''
reshaped_data = all_scores.reshape(328, all_scores.shape[-1])

scalers = {
    'Min-Max S.': MinMaxScaler(),
    'Quantile T.': QuantileTransformer(),
    'Robust S.': RobustScaler(),
    'Standard S.': StandardScaler(),
    # 'MaxAbsScaler': MaxAbsScaler(),
    'Power T.': PowerTransformer(),
    # 'Normalizer': Normalizer()
}

# Feature groups for subplots
feature_groups = {
    'Joint Lines': [0, 1, 2, 3],
    'Conf. Map deriv.': [4,5,6],
    'Mean Intensity': [7,8,9,10],
    'Lap. Variance': [11,12,13]
}

# Gaussian goal array
x = np.linspace(-1, 1, 41)
sigma = 0.3
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
goal = np.tile(gaussian_array, 8)  # Shape (328,)

# Dictionary to store MSE values for analysis, scalers, and models
mse_results = {feature_idx: {} for group in feature_groups for feature_idx in feature_groups[group]}
scaler_dict = {feature_idx: {} for group in feature_groups for feature_idx in feature_groups[group]}
model_dict = {feature_idx: {} for group in feature_groups for feature_idx in feature_groups[group]}

# Plot each feature group across scalers
for group_name, features in feature_groups.items():
    fig, axs = plt.subplots(len(features), 1, figsize=(10, len(features)*4), dpi=300)
    # fig.suptitle(f'Scaled and Fitted Features for {group_name}', fontsize=16)
    fig.suptitle('Train Data', fontsize=18, x=0.4)
    for i, feature_idx in enumerate(features):
        ax = axs[i]
        ax.plot(goal, linestyle='--', color='black', label='Gaussian Goal')  # Gaussian target line

        # Scale, fit, and plot for each scaler
        for scaler_name, scaler in scalers.items():
            # Fit and transform data using the scaler
            scaled_data = scaler.fit_transform(reshaped_data)
            scaler_dict[feature_idx][scaler_name] = scaler  # Store the trained scaler

            # Extract and reshape the feature
            feature = scaled_data[:, feature_idx].reshape(-1, 1)

            # Fit the linear model to match the Gaussian goal
            model = LinearRegression()
            model.fit(feature, goal)
            fitted_feature = model.predict(feature)
            model_dict[feature_idx][scaler_name] = model  # Store the trained model

            # Calculate MSE
            mse = mean_absolute_error(fitted_feature, goal)

            # Store MSE for analysis
            mse_results[feature_idx][scaler_name] = mse

            # Plot the fitted feature with MSE in legend
            ax.plot(fitted_feature, label=f'{scaler_name} (MAE: {mse:.3f})')

        # ax.set_title(f'Feature {features[i]}')
        ax.set_xlabel('Data Points', fontsize=18)
        ax.set_ylabel(f'Scaled Values of f{features[i]}', fontsize=18)

        # Place legend outside each subplot on the right
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Scalers", fontsize=16, title_fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout(rect=[0, 0, 0.80, 0.98])  # Adjust layout to make space for the legend
    plt.show()


# Print MSE values for analysis, organized by feature
print("MSE Values by Feature and Scaler:")
for feature_idx in mse_results.keys():
    print(f"\nFeature {feature_idx}:")
    for scaler_name, mse in mse_results[feature_idx].items():
        print(f"  {scaler_name}: MSE = {mse:.4f}")



# Dictionary to store the predictions for analysis
test_predictions = {feature_idx: {} for feature_idx in range(flat_t.shape[1])}

# Run each feature of the test data through the trained scalers and models
for feature_idx in range(flat_t.shape[1]):
    for scaler_name, scaler in scaler_dict[feature_idx].items():
        # Scale the test data for the specific feature
        scaled_test_data = scaler.transform(flat_t)  # Transform test data using the trained scaler
        test_feature = scaled_test_data[:, feature_idx].reshape(-1, 1)

        # Retrieve the corresponding model and make predictions
        model = model_dict[feature_idx][scaler_name]
        predicted_feature = model.predict(test_feature)

        # Store predictions for analysis
        if scaler_name not in test_predictions[feature_idx]:
            test_predictions[feature_idx][scaler_name] = predicted_feature
        else:
            test_predictions[feature_idx][scaler_name].append(predicted_feature)
    
colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
# Plot each feature group with boxplots and independent scales for each scaler
for group_name, features in feature_groups.items():
    num_features = len(features)
    num_scalers = len(test_predictions[features[0]])  # Number of scalers
    fig, axs = plt.subplots(num_features, num_scalers, figsize=(10, num_features * 4), dpi=300)
    
    fig.suptitle('Test Data', fontsize=18)

    for i, feature_idx in enumerate(features):
        scaler_names = list(test_predictions[feature_idx].keys())

        for j, scaler_name in enumerate(scaler_names):
            # Handle subplots dynamically
            ax = axs[i, j] if num_features > 1 else axs[j]  # Adjust for single row case

            # Retrieve predictions for the specific scaler
            predicted_feature = test_predictions[feature_idx][scaler_name]

            # Plot boxplot for the current feature and scaler
            box = ax.boxplot(predicted_feature, patch_artist=True)
            # Customize colors for the boxplot
            for patch in box['boxes']:
                patch.set_facecolor(colors[j])  # Color for the box
                patch.set_edgecolor('black')  # Color for the box edges
                patch.set_linewidth(1.5)  # Box edge width
                
            for median in box['medians']:
                median.set(color='black', linewidth=2)  # Color for the median line
                
            ax.set_xlabel(f'{scaler_name}', fontsize=18)
            if j == 0:
                ax.set_ylabel(f'Scaled Value of f{features[i]}', fontsize=18)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xticks([])
            ax.tick_params(axis='both', labelsize=16)

    # Adjust layout to make space for titles and avoid overlap
    plt.tight_layout()  # Ensure everything fits within the figure
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make space for the legend
    plt.show()


#%% PLOTS WITH EQUAL Y AXIS PER FEAT
# run previous cell first!

reshaped_data = all_scores.reshape(328, all_scores.shape[-1])

scalers = {
    'Min-Max S.': MinMaxScaler(),
    'Quantile T.': QuantileTransformer(),
    'Robust S.': RobustScaler(),
    'Standard S.': StandardScaler(),
    # 'MaxAbsScaler': MaxAbsScaler(),
    'Power T.': PowerTransformer(),
    # 'Normalizer': Normalizer()
}

# Feature groups for subplots
feature_groups = {
    'Joint Lines': [0, 1, 2, 3],
    'Conf. Map deriv.': [4,5,6],
    'Mean Intensity': [7,8,9,10],
    'Lap. Variance': [11,12,13]
}

# Gaussian goal array
x = np.linspace(-1, 1, 41)
sigma = 0.3
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
goal = np.tile(gaussian_array, 8)  # Shape (328,)

# Dictionary to store MSE values for analysis, scalers, and models
mse_results = {feature_idx: {} for group in feature_groups for feature_idx in feature_groups[group]}
scaler_dict = {feature_idx: {} for group in feature_groups for feature_idx in feature_groups[group]}
model_dict = {feature_idx: {} for group in feature_groups for feature_idx in feature_groups[group]}

# Plot each feature with its own Train and Test data
for group_name, features in feature_groups.items():
    num_features = len(features)
    fig, axs = plt.subplots(num_features, 2, figsize=(17, num_features * 4), dpi=300, 
                            gridspec_kw={'width_ratios': [2, 1]})
    
    # fig.suptitle(f'{group_name}: Train and Test Data', fontsize=18, x=0.5)

    for i, feature_idx in enumerate(features):
        # Get axes for Train and Test data
        ax_train, ax_test = axs[i] if num_features > 1 else axs

        # ----- Compute per-feature y-axis limits -----
        feature_min, feature_max = float('inf'), float('-inf')

        # Include Gaussian goal range in y-axis limits
        goal_min, goal_max = np.min(goal), np.max(goal)
        feature_min = min(feature_min, goal_min)
        feature_max = max(feature_max, goal_max)

        # Compute limits from Train data
        for scaler_name, scaler in scalers.items():
            # Scale and transform Train data
            scaled_train_data = scaler.fit_transform(reshaped_data)
            feature_train = scaled_train_data[:, feature_idx].reshape(-1, 1)

            # Fit the model and predict Train feature
            model = LinearRegression()
            model.fit(feature_train, goal)
            fitted_train_feature = model.predict(feature_train)

            # Update min and max with Train data
            feature_min = min(feature_min, np.min(fitted_train_feature))
            feature_max = max(feature_max, np.max(fitted_train_feature))

        # Compute limits from Test data
        scaler_names = list(test_predictions[feature_idx].keys())
        for scaler_name in scaler_names:
            predicted_test_feature = test_predictions[feature_idx][scaler_name]

            # Update min and max with Test data
            feature_min = min(feature_min, np.min(predicted_test_feature))
            feature_max = max(feature_max, np.max(predicted_test_feature))

        # Add margin to y-axis limits
        y_margin = 0.05 * (feature_max - feature_min)
        feature_min -= y_margin
        feature_max += y_margin

        # ----- Train Data (line plots) -----
        ax_train.plot(goal, linestyle='--', color='black', label='Gaussian Goal')  # Gaussian target line

        for scaler_name, scaler in scalers.items():
            # Scale and transform Train data
            scaled_train_data = scaler.fit_transform(reshaped_data)
            feature_train = scaled_train_data[:, feature_idx].reshape(-1, 1)

            # Fit the model and predict Train feature
            model = LinearRegression()
            model.fit(feature_train, goal)
            fitted_train_feature = model.predict(feature_train)

            # Plot the fitted Train feature
            mse = mean_absolute_error(fitted_train_feature, goal)
            ax_train.plot(fitted_train_feature, label=f'{scaler_name} (MAE: {mse:.3f})')

        ax_train.set_title(f'Train Data: Feature f{feature_idx}', fontsize=16)
        ax_train.set_xlabel('Data Points', fontsize=14)
        ax_train.set_ylabel('Scaled Values', fontsize=14)
        ax_train.grid(True, linestyle='--', alpha=0.6)
        ax_train.set_ylim(feature_min, feature_max)  # Apply per-feature y-axis limits

        # Move legend outside the Train plot
        ax_train.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title="Scalers", title_fontsize=14)

        # ----- Test Data (boxplots) -----
        box_data = [test_predictions[feature_idx][scaler_name] for scaler_name in scaler_names]

        # Boxplot with aligned y-axis
        box = ax_test.boxplot(box_data, patch_artist=True, labels=scaler_names)
        for j, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors[j % len(colors)])
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

        for median in box['medians']:
            median.set(color='black', linewidth=2)

        ax_test.set_title(f'Test Data: Feature f{feature_idx}', fontsize=16)
        ax_test.set_xlabel('Scalers', fontsize=14)
        if i == 0:
            ax_test.set_ylabel('Scaled Values', fontsize=14)
        ax_test.grid(True, linestyle='--', alpha=0.6)
        ax_test.set_ylim(feature_min, feature_max)  # Apply per-feature y-axis limits
        ax_test.tick_params(axis='x', labelrotation=45)

    # Adjust layout to leave space for legends
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])  # Ensure space on the right for legends
    plt.show()



#%%
# Plot Feature 9 from Group 1 across different scalers, each in its own figure
feature_idx = 0  # Feature 9

# Plot each scaler's predictions for Feature 9 in the test data
for scaler_name in test_predictions[feature_idx].keys():
    # Create a new figure for each scaler
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Retrieve predictions for the specific scaler
    predicted_feature = test_predictions[feature_idx][scaler_name]

    # Plot the predicted feature
    ax.plot(predicted_feature, label=scaler_name)

    # Set title, labels, and legend
    ax.set_title(f'Feature {feature_idx} - Test Data with {scaler_name}')
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Transformed Value')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Scaler")

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for the legend
    plt.show()




#%%
# Define scalers, their names, associated colors, and line styles in a dictionary
scalers_info = {
    'MinMaxScaler': {'scaler': MinMaxScaler(), 'color': 'steelblue', 'linestyle': '--'},
    'QuantileTransformer': {'scaler': QuantileTransformer(), 'color': 'orange', 'linestyle': '-.'},
    'RobustScaler': {'scaler': RobustScaler(), 'color': 'green', 'linestyle': ':'}
}

# Define best features
best_features = {
    'Joint Lines': 0,
    'Conf. Map deriv.': 4,
    'Mean Abs. Intens.': 10,
    'Lap. Variance': 13
}

# Gaussian goal array
x = np.linspace(-1, 1, 41)
sigma = 0.3
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
goal = np.tile(gaussian_array, 8)  # Shape (328,)

# Prepare to plot
fig, axs = plt.subplots(len(best_features), 2, figsize=(20, 15), dpi=300)
# fig.suptitle('Best Features: Training vs Test Data', fontsize=16)

# Iterate over best features
for i, (group_name, feature_idx) in enumerate(best_features.items()):
    # Iterate over the scalers in the dictionary
    axs[i, 0].plot(goal, linestyle='--', color='black', label='Gaussian Goal')  # Gaussian target line
    for scaler_name, scaler_info in scalers_info.items():
        sca = scaler_info['scaler']
        color = scaler_info['color']
        linestyle = scaler_info['linestyle']
        
        # Scale the training data using the scaler
        scaled_train_data = sca.fit_transform(reshaped_data)
        
        # Extract and reshape the training feature
        train_feature = scaled_train_data[:, feature_idx].reshape(-1, 1)
    
        # Fit the linear model to match the Gaussian goal
        model = LinearRegression()
        model.fit(train_feature, goal)
        fitted_train_feature = model.predict(train_feature)
        
        mse = mean_absolute_error(fitted_train_feature, goal)

        # Plot training data with goal on the left
        axs[i, 0].plot(fitted_train_feature, color=color, linestyle=linestyle, label=f'{scaler_name} (MAE: {mse:.4f})')
        if i==0:
            axs[i, 0].set_title('Train Data',fontsize=18)
        axs[i, 0].set_xlabel('Data Points',fontsize=18)
        axs[i, 0].set_ylabel(f'Scaled Value of f{feature_idx}',fontsize=18)
        axs[i,0].tick_params(axis='both', labelsize=16)
    
        # Scale the test data using the fitted scaler
        scaled_test_data = sca.transform(flat_t)
    
        # Extract and reshape the test feature
        test_feature = scaled_test_data[:, feature_idx].reshape(-1, 1)
    
        # Retrieve the model and make predictions on test data
        predicted_test_feature = model.predict(test_feature)
    
        # Plot test data on the right
        axs[i, 1].plot(predicted_test_feature, label=scaler_name, color=color, linestyle=linestyle)
        if i==0:
            axs[i, 1].set_title('Test Data',fontsize=18)
        axs[i, 1].set_xlabel('Data Points',fontsize=18)
        axs[i, 1].set_ylabel(f'Scaled Value of f{feature_idx}',fontsize=18)
        axs[i,1].tick_params(axis='both', labelsize=16)
        
    axs[i, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Scalers", fontsize=16, title_fontsize=16)
    # axs[i, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Scalers", fontsize=16, title_fontsize=16)
    
    
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
plt.show()

#%% Average weitght among all combinations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from itertools import combinations


reshaped_data = all_scores.reshape(328, all_scores.shape[-1])
reshaped_data = reshaped_data[:, list(best_features.values())]

goal = np.tile(np.exp(-0.5 * (np.linspace(-1, 1, 41) / 0.3) ** 2), 8)  # Gaussian goal

# Define best features and index mapping
best_features = {
    'f0': 0,
    'f4': 4,
    'f10': 10,
    'f13': 13
}
idxmap = {
    'f0': 0,
    'f4': 1,
    'f10': 2,
    'f13': 3
}

# Initialize scaler and scale the data
robust_scaler = QuantileTransformer()
scaled_data = robust_scaler.fit_transform(reshaped_data)

# Initialize a dictionary to store the sum of weights for each feature
weight_sums = {name: 0 for name in best_features}
abssum = {name: 0 for name in best_features}
count_sums = {name: 0 for name in best_features}

# Iterate over all possible feature combinations (1 to 4 features)
for r in range(1, len(best_features) + 1):
    for combo in combinations(best_features.keys(), r):
        # Select columns for the current combination of features
        feature_indices = [idxmap[feature] for feature in combo]
        X = scaled_data[:, feature_indices]

        # Train linear model
        model = LinearRegression()
        model.fit(X, goal)
        
        # Update weight sums and counts for each feature in the combination
        for i, feature in enumerate(combo):
            weight_sums[feature] += model.coef_[i]
            abssum[feature] += abs(model.coef_[i])
            count_sums[feature] += 1

# Calculate average weight for each feature
average_weights = {feature: weight_sums[feature] / count_sums[feature] for feature in best_features}
absavg = {feature: abssum[feature] / count_sums[feature] for feature in best_features}

w = []
wabs=[]
features = []  # Store feature names for the x-axis labels
for feature, avg_weight in average_weights.items():
    print(f"{feature}: Average Weight = {avg_weight} {absavg[feature]}")
    w.append(avg_weight)
    wabs.append(absavg[feature])
    features.append(feature)

# Plot the bar chart with feature names as labels
plt.figure(dpi=200)
plt.bar(features, w,label='Avg. Weights')  # Use 'skyblue' for a light blue color
plt.bar(features, wabs, label='Abs. Avg. Weights')  # Use 'skyblue' for a light blue color
# plt.xlabel("Features")
plt.ylabel("Average Weight")
# plt.title("Average Weights of Features from Linear Model")
plt.legend()
plt.show()

#%% TOP X best features
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import QuantileTransformer
from itertools import combinations

# Define best features and index mapping
best_features = {
    'f0': 0,
    'f4': 4,
    'f10': 10,
    'f13': 13
}
idxmap = {
    'f0': 0,
    'f4': 1,
    'f10': 2,
    'f13': 3
}

# Simulate example data (replace with your actual data)
reshaped_data = all_scores.reshape(328, all_scores.shape[-1])
reshaped_data = reshaped_data[:, list(best_features.values())]

goal = np.tile(np.exp(-0.5 * (np.linspace(-1, 1, 41) / 0.3) ** 2), 8)  # Gaussian goal

# Initialize QuantileTransformer and scale the data
scaler = QuantileTransformer()
scaled_data = scaler.fit_transform(reshaped_data)

# Define thresholds as fractions of total combinations
thresholds = [0.25, 0.50, 0.75]

# Initialize dictionary to store feature frequencies per threshold
feature_frequencies = {feature: [] for feature in best_features}

# Iterate over thresholds
for threshold in thresholds:
    error_dict = {}

    # Iterate over all possible feature combinations (1 to 4 features)
    for r in range(1, len(best_features) + 1):
        for combo in combinations(best_features.keys(), r):
            # Select columns for the current combination of features
            feature_indices = [idxmap[feature] for feature in combo]
            X = scaled_data[:, feature_indices]

            # Train linear model
            model = LinearRegression()
            model.fit(X, goal)

            # Calculate error (Mean Absolute Error)
            predictions = model.predict(X)
            error = mean_absolute_error(goal, predictions)

            # Store the error for this combination
            error_dict[combo] = error

    # Sort combinations by error (ascending)
    sorted_combos = sorted(error_dict.items(), key=lambda x: x[1])

    # Select top combinations based on the current threshold
    top_combos = sorted_combos[:int(len(sorted_combos) * threshold)]
    print(len(top_combos))

    # Count how many times each feature appears in the top combinations
    feature_count = {name: 0 for name in best_features}
    for combo, _ in top_combos:
        for feature in combo:
            feature_count[feature] += 1

    # Calculate frequency percentages for the current threshold
    total_top_combos = len(top_combos)
    for feature in best_features:
        feature_frequencies[feature].append((feature_count[feature] / total_top_combos) * 100)

# Plot grouped bar chart for feature frequencies across thresholds
x = np.arange(len(best_features))  # X positions for features
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

# Plot each threshold's data as a group of bars
for i, threshold in enumerate(thresholds):
    threshold_percent = int(threshold * 100)
    ax.bar(x + i * width, [feature_frequencies[feature][i] for feature in best_features],
           width, label=f"Top {threshold_percent}%", alpha=0.8)

# Add labels, legend, and format plot
ax.set_xlabel("Features", fontsize=16)
ax.set_ylabel("Frequency in Top Combinations (%)", fontsize=16)
ax.set_xticks(x + width)
ax.set_xticklabels(best_features.keys(), fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.legend(title="Threshold", fontsize=16, title_fontsize=16)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('(B)', fontsize=16)
plt.tight_layout()
plt.show()


# Plot the errors for each combination
combo_labels = [str(combo) for combo, _ in sorted_combos]  # Labels for the combinations
errors = [error for _, error in sorted_combos]  # Errors for each combination

colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure(figsize=(10, 6), dpi=200)
plt.axhline(y=int(len(sorted_combos)*0.25)-0.5, color=colors[0], linestyle='--')
plt.axhline(y=int(len(sorted_combos)*0.5)-0.5, color=colors[1], linestyle='--')
plt.axhline(y=int(len(sorted_combos)*0.75)-0.5, color=colors[2], linestyle='--')
plt.title('(A)', fontsize=16)

plt.barh(combo_labels, errors, color='steelblue')
plt.xlabel("Mean Absolute Error", fontsize=16)
plt.ylabel("Feature Combinations", fontsize=16)
# plt.title("Error for Each Feature Combination")
plt.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.grid()
plt.show()

#%% INdividual feature box plots w lin mod in all features
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define best features
best_features = {
    'Joint Lines': 0,
    'Conf. Map deriv.': 4,
    'Mean Intensity': 10,
    'Lap. Variance': 13
}

# Gaussian goal array
x = np.linspace(-1, 1, 41)
sigma = 0.3
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
goal = np.tile(gaussian_array, 8)  # Shape (328,)

# List of scalers and corresponding colors
scalers = [
    (MinMaxScaler(), '#1f77b4', 'MinMaxScaler'),
    (QuantileTransformer(), '#ff7f0e', 'QuantileTransformer'),
    (RobustScaler(), '#2ca02c', 'RobustScaler'),
    (StandardScaler(), '#d62728', 'StandardScaler')
]

# Prepare data
reshaped_data = all_scores.reshape(328, all_scores.shape[-1])
train_data = reshaped_data[:, list(best_features.values())]
test_data = flat_t[:, list(best_features.values())]

# Iterate through scalers to plot for each
for scaler, color, scaler_name in scalers:
    # Scale training and test data
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    # Fit the linear model to match the Gaussian goal
    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    model.fit(scaled_train_data, goal)

    # Generate predictions
    fitted_train_output = model.predict(scaled_train_data)
    predicted_test_output = model.predict(scaled_test_data)

    # Determine local min and max for this scaler
    local_min_top = min(fitted_train_output.min(),
                    predicted_test_output.min(),) -0.1
                    # scaled_train_data.min(),
                    # scaled_test_data.min()) -0.25
    local_max_top = max(fitted_train_output.max(),
                    predicted_test_output.max(),) +0.1
                    # scaled_train_data.max(),
                    # scaled_test_data.max()) +0.25
    local_min_btm = min(scaled_train_data.min(),
                    scaled_test_data.min()) -0.25
    local_max_btm = max(scaled_train_data.max(),
                    scaled_test_data.max()) +0.25

    # Prepare subplots for this scaler
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), dpi=300)
    # fig.suptitle(f'Combined Features: {scaler_name}', fontsize=16)

    # Plot fitted training output
    axs[0, 0].plot(goal, linestyle='--', color='black', label='Gaussian Goal')
    axs[0, 0].plot(fitted_train_output, label='Fitted Train Output', color='blue')
    axs[0, 0].set_title('Training Data')
    axs[0, 0].set_xlabel('Data Points')
    axs[0, 0].set_ylabel('Cost')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_ylim(local_min_top, local_max_top)

    # Plot predicted test output
    # axs[0, 1].plot(predicted_test_output, color=color)
    # axs[0, 1].set_title('Test Data')
    # axs[0, 1].set_xlabel('Data Points')
    # axs[0, 1].set_ylabel('Cost')
    # axs[0, 1].grid(True)
    # axs[0, 1].set_ylim(local_min, local_max)

    # Boxplot for combined features
    axs[0, 1].boxplot(predicted_test_output, vert=True, patch_artist=True, boxprops=dict(facecolor=color))
    axs[0, 1].set_title('Test Data Combined Features')
    # axs[0, 1].set_ylabel('Cost')
    axs[0, 1].set_xticks([])
    axs[0, 1].grid(True)
    axs[0, 1].set_ylim(local_min_top, local_max_top)
    
    # Boxplots for individual features
    axs[1, 0].set_title('Train Data Boxplots of Individual Features')
    axs[1, 0].set_ylabel('Scaled Values')
    feature_labels = ['f0', 'f4', 'f10', 'f13']
    # axs[1, 0].set_xticks(range(1, len(feature_labels) + 1))
    axs[1, 0].set_xticklabels(feature_labels)
    axs[1, 0].grid(True)
    axs[1, 0].set_ylim(local_min_btm, local_max_btm)

    for idx, feature in enumerate(best_features):
        feature_data = scaled_train_data[:, idx]
        axs[1, 0].boxplot(
            feature_data, vert=True, positions=[idx + 1], patch_artist=True,
            boxprops=dict(facecolor=color)
        )

    # Boxplots for individual features
    axs[1, 1].set_title('Test Data Boxplots of Individual Features')
    # axs[1, 1].set_ylabel('Scaled Values')
    feature_labels = ['f0', 'f4', 'f10', 'f13']
    # axs[1, 1].set_xticks(range(1, len(feature_labels) + 1))
    axs[1, 1].set_xticklabels(feature_labels)
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim(local_min_btm, local_max_btm)

    for idx, feature in enumerate(best_features):
        feature_data = scaled_test_data[:, idx]
        axs[1, 1].boxplot(
            feature_data, vert=True, positions=[idx + 1], patch_artist=True,
            boxprops=dict(facecolor=color)
        )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print model details
    print(f"Scaler: {scaler_name}")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Training R^2: {model.score(scaled_train_data, goal)}")
    print(f'MAE: {mean_absolute_error(fitted_train_output, goal)}')
    print(f'min max top: {local_min_top+0.1} {local_max_top-0.1}')
    print(f'min max btm: {local_min_btm+0.25} {local_max_btm-0.25}')
    count_in_range = sum(0 <= x <= 1 for x in predicted_test_output)  # Counts items in the range
    percentage = (count_in_range / len(predicted_test_output)) * 100
    print(f'percentage in 0 1 range: {percentage}%\n')

#%% Results of using lin mod on all features
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define best features
best_features = {
    'Joint Lines': 0,
    'Conf. Map deriv.': 4,
    'Mean Intensity': 10,
    'Lap. Variance': 13
}

# Gaussian goal array
x = np.linspace(-1, 1, 41)
sigma = 0.3
gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
goal = np.tile(gaussian_array, 8)  # Shape (328,)

# List of scalers and corresponding colors
scalers = [
    (MinMaxScaler(), '#1f77b4', 'MinMaxScaler'),
    (QuantileTransformer(), '#ff7f0e', 'QuantileTransformer'),
    (RobustScaler(), '#2ca02c', 'RobustScaler'),
    (StandardScaler(), '#d62728', 'StandardScaler')
]

# Prepare data
reshaped_data = all_scores.reshape(328, all_scores.shape[-1])
train_data = reshaped_data[:, list(best_features.values())]
test_data = flat_t[:, list(best_features.values())]

# Iterate through scalers to plot for each
for scaler, color, scaler_name in scalers:
    # Scale training and test data
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    # Fit the linear model to match the Gaussian goal
    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    model.fit(scaled_train_data, goal)

    # Generate predictions
    fitted_train_output = model.predict(scaled_train_data)
    predicted_test_output = model.predict(scaled_test_data)

    # Prepare subplots for this scaler
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), dpi=300)
    fig.suptitle(f'Combined Features: {scaler_name}', fontsize=16)

    # Plot fitted training output
    axs[0, 0].plot(goal, linestyle='--', color='black', label='Gaussian Goal')
    axs[0, 0].plot(fitted_train_output, label='Fitted Train Output', color='blue')
    axs[0, 0].set_title('Training Data')
    axs[0, 0].set_xlabel('Data Points')
    axs[0, 0].set_ylabel('Cost')
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid(True)

    # Plot predicted test output
    axs[0, 1].plot(predicted_test_output, color=color)
    axs[0, 1].set_title('Test Data')
    axs[0, 1].set_xlabel('Data Points')
    axs[0, 1].set_ylabel('Cost')
    axs[0, 1].grid(True)

    # Boxplot for combined features
    axs[1, 0].boxplot(predicted_test_output, vert=True, patch_artist=True, boxprops=dict(facecolor=color))
    axs[1, 0].set_title('Test Data Boxplot (Combined Features)')
    axs[1, 0].set_ylabel('Cost')
    axs[1, 0].set_xticks([])
    axs[1, 0].grid(True)

    # Boxplots for individual features
    axs[1, 1].set_title('Test Data Boxplot (Individual Features)')
    axs[1, 1].set_ylabel('Cost')
    feature_labels = ['f0', 'f4', 'f10', 'f13']
    axs[1, 1].set_xticks(range(1, len(feature_labels) + 1))
    axs[1, 1].set_xticklabels(feature_labels)
    axs[1, 1].grid(True)

    for idx, feature in enumerate(best_features):
        feature_data = scaled_test_data[:, idx]
        axs[1, 1].boxplot(
            feature_data, vert=True, positions=[idx + 1], patch_artist=True,
            boxprops=dict(facecolor=color)
        )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print model details
    print(f"Scaler: {scaler_name}")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Training R^2: {model.score(scaled_train_data, goal)}\n")


#%%

best_features = {
    'f0': 0,
    'f4': 4,
    'f10': 10,
    'f13': 13
}

# Create a new figure for plotting scaled training features alongside the goal
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
# fig2.suptitle('Scaled Train Features vs Goal and Weighted Features', fontsize=16)

# Extract the learned weights from the model
weights = model.coef_

# Define labels for the legend
labels = ['Gaussian Goal', 'Scaled Feature', 'Weighted Feature']
colors = ['black', 'blue', 'orange']

# Iterate over the features to plot
for idx, (feature_name, feature_idx) in enumerate(best_features.items()):
    row, col = divmod(idx, 2)  # Determine subplot position (2x2 grid)
    
    # Extract the scaled feature and compute the weighted feature
    scaled_feature = scaled_train_data[:, idx]
    weighted_feature = scaled_feature * weights[idx]
    
    # Plot the scaled feature alongside the Gaussian goal
    axs2[row, col].plot(goal, linestyle='--', color='black')  # Gaussian Goal
    axs2[row, col].plot(scaled_feature, label=f'Scaled {feature_name}')  # Scaled Feature
    axs2[row, col].plot(weighted_feature, label=f'Weighted {feature_name}')  # Weighted Feature
    
    # Set titles and labels
    axs2[row, col].set_title(f'{feature_name}')
    axs2[row, col].set_xlabel('Data Points')
    axs2[row, col].set_ylabel('Cost')
    axs2[row, col].grid(True)  # Add grid

# Add a single legend to the right of all subplots
lines = [plt.Line2D([0], [0], color=col, linestyle='--' if idx == 0 else '-') for idx, col in enumerate(colors)]
fig2.legend(lines, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=10)

# Adjust the layout to prevent overlap with the legend
plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Reserve space for the legend
plt.show()

#%% STats!
import seaborn as sns
scaler = QuantileTransformer()
scaled_train_data = scaler.fit_transform(train_data) #shape (328,4)

import pandas as pd
corr_matrix = pd.DataFrame(scaled_train_data).corr()
print(corr_matrix)
corr_matrix = pd.DataFrame(scaled_train_data).corr()
feature_labels = ['f0', 'f4', 'f10', 'f13']
# Create a heatmap
# Create an even smaller figure size
plt.figure(figsize=(4, 3),dpi=200)  # Further reduced from (6, 5)

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
            xticklabels=feature_labels,
            yticklabels=feature_labels,
            square=True,
            annot_kws={'size': 10})  # Even smaller font size

plt.title("Correlation Matrix", fontsize=10)
plt.xticks(rotation=45, fontsize=10)  # Smaller font size
plt.yticks(fontsize=10)  # Smaller font size
plt.tight_layout()
plt.show()

from sklearn.feature_selection import f_regression

f_scores, p_values = f_regression(scaled_train_data, goal)
print("F-scores:", f_scores)
print("P-values:", p_values)
from scipy.stats import f

alpha = 0.05 # Significance level
dof_num = 1   # Degrees of freedom for the numerator
dof_den = scaled_train_data.shape[0] - scaled_train_data.shape[1] - 1 # Degrees of freedom for the denominator

# Calculate the critical value
critical_value = f.ppf(1 - alpha, dof_num, dof_den)
print("Critical Value:", critical_value)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = ["f0", "f4", "f10", "f13"]
vif_data["VIF"] = [variance_inflation_factor(scaled_train_data, i) for i in range(scaled_train_data.shape[1])]
print(vif_data)

for i in range(4):
    print(np.corrcoef(scaled_train_data[:,i], goal)[0,1])

#%% sigma compare

# Initialize lists to store learned weights
learned_weights = []

# Define sigma values to experiment with
sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Adjust sigma values as needed

# Loop through each sigma value
for sigma in sigma_values:
    # Create Gaussian target array
    x = np.linspace(-1, 1, 41)
    gaussian_array = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_array = np.tile(gaussian_array, 8)  # Shape (328,)
    
    # Scale the features using the QuantileTransformer
    scaler = QuantileTransformer()
    scaled_features = scaler.fit_transform(train_features)
    
    # Train a new linear regression model for each sigma
    model = LinearRegression()
    model.fit(scaled_features, gaussian_array)  # Fit with Gaussian target
    
    # Store learned weights
    learned_weights.append(model.coef_)

# Convert learned weights to a numpy array
learned_weights = np.array(learned_weights)

# Plot learned weights for different sigma values
plt.figure(figsize=(10, 6))
for i, sigma in enumerate(sigma_values):
    plt.plot(learned_weights[i], label=f'Sigma = {sigma}')
plt.title('Effect of Sigma on Feature Weights')
plt.xlabel('Feature Index')
plt.ylabel('Learned Weights')
plt.xticks(ticks=[0, 1,2,3], labels=['f0','f6','f12','f16']) 
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#%%

import pickle
with open('final2.pkl', 'wb') as f:
    pickle.dump(robust_scaler, f)
    
    
#%% Compare perfect segmentation

linesPath = 'C:/Users/Mateo-drr/Documents/ALU---Autonomous-Lung-Ultrasound/imgProcessing/ml/lines/'

def calcBestFeat(paths,linesf=None,rsize=False,noCrop=False):
    all_scores = []
    for i, path in enumerate(paths):
        scores = []
        for j, w in enumerate(path):
            
            if linesf is not None:
                strt,end=linesf[i,j]
            elif noCrop:
                strt,end=None,None #select everything
            else:
                strt,end=2000,2800
            
            w,cmap = w
            '''
            Joint lines
            '''
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
            '''
            Confidence map
            '''
            #Confidence Map → Crop → Mean Lines → Deriv. → Abs → Prob. → Variance
            if noCrop:
                crop = cmap[-10:]
            else:
                crop = cmap[strt:end]
            lineMean = np.mean(crop,axis=1)
            if rsize:
                lineMean = resize(lineMean, [800], anti_aliasing=True)
            deriv = abs(savgol_filter(lineMean, window_length=len(lineMean)//16,
                                        polyorder=2, deriv=1))
            t=deriv

            probs = t/np.sum(t)
            x = np.arange(len(probs))
            avg = np.sum(x*probs)
            var = np.sum((x - avg)**2 * probs)
            r6 = var
            '''
            Intensity
            '''
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
            r12 = imgc.mean()
            '''
            Laplace
            '''
            #Hilbert → Log → Crop → Laplace → Variance
            himg = byb.envelope(w)
            loghimg = 20*np.log10(himg+1)
            crop = loghimg[strt:end, :]
            if rsize:
                crop = resize(crop, [800,129], anti_aliasing=True)
            # print(crop.shape)
            lap = laplace(crop)
            r16 = lap.var()
            
            scores.append([r0,r6,r12,r16])
        
        all_scores.append(scores)
    return all_scores
    
lines = [[],[],[],[]]
for i in range(4):
    btm=np.load(linesPath + f'btm_lines_{i}.npy')
    top=np.load(linesPath + f'top_lines_{i}.npy')
    lines[i].append([top,btm])
lines=np.array(lines)

linesf=[]
for i in range(4):
    linesf.append(lines[i,0,:,:41])
    linesf.append(lines[i,0,:,41:])        
linesf = np.array(linesf).transpose(0, 2, 1)


justimgs = np.array(alldat)
paths = np.array(justimgs)
paths = np.array(np.split(paths, 8, axis=0))
cstmcrop=calcBestFeat(paths, linesf,rsize=True)
nocrop=calcBestFeat(paths, noCrop=True, rsize=True)
fixedcrop=calcBestFeat(paths, rsize=True)

#these have shape 8,41,4
cstmcrop,nocrop,fixedcrop=np.array(cstmcrop),np.array(nocrop),np.array(fixedcrop)
norm1 = (nocrop - nocrop.min())/(nocrop.max() - nocrop.min())
norm2 = (fixedcrop - fixedcrop.min())/(fixedcrop.max() - fixedcrop.min())
norm3 = (cstmcrop - cstmcrop.min())/(cstmcrop.max() - cstmcrop.min())

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for each category
x_values = list(range(-20, 21))
order = ['(A)', '(B)', '(C)']
ylims=[
       [7000,55000],
       [5000,58000],
       [0.02,0.42],
       [10,150]]
yax=['Variance', 'Variance', 'Mean', 'Variance']

for i in range(4):  # Four features
    # Calculate mean and std of paths
    means = [
        np.mean(nocrop, axis=0),
        np.mean(fixedcrop, axis=0),
        np.mean(cstmcrop, axis=0),
    ]
    stds = [
        np.std(nocrop, axis=0),
        np.std(fixedcrop, axis=0),
        np.std(cstmcrop, axis=0),
    ]
    
    fig, axs = plt.subplots(1, 3, figsize=(25, 6), dpi=200)
    for j, ax in enumerate(axs):
        ax.errorbar(
            x_values, 
            means[j][:, i], 
            yerr=stds[j][:, i], 
            fmt='-o', ecolor='r',capsize=5, capthick=2, color=colors[0]
        )
        ax.set_title(order[j], fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True)
        ax.set_xlabel("Degrees", fontsize=18)
        if j==0:
            ax.set_ylabel(yax[i], fontsize=18)
        ax.set_ylim(ylims[i])
        # ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

    #calc ncv
    mean1,mean2,mean3=norm1.mean(axis=0),norm2.mean(axis=0),norm3.mean(axis=0)
    std1,std2,std3=norm1.std(axis=0),norm2.std(axis=0),norm3.std(axis=0)
    ncv1,ncv2,ncv3=np.mean(100*std1[:,i]/mean1[:,i]),np.mean(100*std2[:,i]/mean2[:,i]),np.mean(100*std3[:,i]/mean3[:,i])

    print(ncv1,ncv2,ncv3)










