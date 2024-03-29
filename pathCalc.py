# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:47:28 2024

@author: mateo-drr
"""

import matplotlib.pyplot as plt
import numpy as np

#Metal phantom
# size: 5x17 [cm]
# it has 9 rough, 8 smooth each 1 cm
# middle point lands in stripe #5 and its position is 2.5x8.5 [cm]

def askConfig(ask):
    if ask:
        rad = input('Radius [cm] from the center point: ')
        maxRot = input('Maximum angle [deg] from the top view: ')
        stops = max(input('Number of stops for imaging: '),3) #minimum of 3 stops -> edges and center
        #if an even number is given just add one stop
        if stops %2 == 0:
            print('Setting number of stops to ',stops, '+ 1')
            stops+=1
    else:
        rad = 5 #[cm]
        maxRot = 45 # TODO probe rot is limited by flange 
        stops = 5
    return rad,maxRot,stops


def drawPath(size, rad, distFromBase=60):    
    #x = middle point of the pantom | y-> = distance from base frame to phantom | z->y = 0
    center = (size/2,distFromBase,0) #[cm] -> here y is behaving as 3d z
    angles = np.linspace(0, np.pi, 101)
    x = center[0] + rad * np.cos(angles)
    x=x[::-1] #flip the array to keep it in ascending order
    y = center[1] - rad * np.sin(angles)
    return x,y,center

def cutPath(maxRot,rad,center,x,y):
    #remove points outside the max rotation
    theta = 90-maxRot
    
    #calculate the maxmin x distance from the angle limit
    xrange = np.cos(np.radians(theta))*rad 
    xmin = center[0]-xrange
    xmax = center[0]+xrange
    
    #cut indexes
    i1,i2 = np.where(x>=xmin)[0][0], np.where(x>=xmax)[0][0]
    xcut = x[i1:i2]
    ycut = y[i1:i2]
    return xcut,ycut

def pitStops(stops,xcut):
    #Select stopping points from xcut
    pits = [0] #short for pit stops, indexes of stops
    if stops >3:
        extras = (stops-3)/2 + 1 
        cuts = len(xcut[:len(xcut)//2])//extras
        #left side pits
        for i in range(int(extras-1)):
            pits.append(int(cuts*(i+1)))
        #middle poitn
        pits.append((len(xcut)-1)//2)
        #right side
        for i in range(int(extras-1)):
            pits.append((len(xcut)-1)//2 + int(cuts*(i+1)))
        
        #last point
        pits.append(len(xcut)-1)
    else:
        #middle poitn
        pits.append((len(xcut)-1)//2)
        #last point
        pits.append(len(xcut)-1)
    return pits


def plotPath(x,y,xcut,ycut,pits):
    #Visualization of the path
    for point in pits:
        plt.plot(xcut[point],ycut[point],marker='x', markersize=10)
    plt.plot(xcut,ycut, linewidth=2)
    plt.plot(x,y)

def projPath3d(xcut,ycut,pits,center,rad,xshift,path='length'):
    #Write the positions in the 3d coordinates from the reference frame
    tcoord,trot=[],[]
    for point in pits:
        
        if path == 'length':
            #add a translation in x and convert to meters
            tcoord.append([xcut[point]*0.01 + xshift-center[0]*0.01,
                           center[-1]*0.01,
                           ycut[point]*0.01])
        elif path == 'width':
            tcoord.append([center[-1]*0.01 + xshift-center[0]*0.01,
                           xcut[point]*0.01,
                           ycut[point]*0.01])
            
        #calculate rotation angle to keep the probe facing the point
        distX = xcut[pits[len(pits)//2]]-xcut[point]
        ang = np.degrees(np.arcsin(distX/rad))
        #-90 x axis points upwards, so we add the angle 
        if path == 'length':
            trot.append([0,-90+ang,0]) 
        elif path =='width':
            trot.append([-ang,-90,0]) 
            
    return tcoord,trot