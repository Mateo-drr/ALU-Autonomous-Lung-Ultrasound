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

def askConfig(ask,angle):
    stops=0
    alpha=0
    if ask:
        rad = float(input('Radius [cm] from the center point: '))
        maxRot = float(input('Maximum angle [deg] from the top view: '))
        if angle:
            while True:
                alpha = float(input('Angle [deg] between stops: '))
                if alpha > maxRot:
                    print("Alpha can't be bigger than the maximum angle")
                else:
                    break
        else:
            stops = max(float(input('Number of stops for imaging: ')),3) #minimum of 3 stops -> edges and center
        #if an even number is given just add one stop
        if stops %2 == 0:
            print('Setting number of stops to ',stops, '+ 1')
            stops+=1
    else:
        rad = 7 #[cm]
        maxRot = 40 # TODO probe rot is limited by flange 
        stops = 5
        alpha = 20
    return rad,maxRot,stops,alpha


def drawPath(size, rad, middlepoint, flip=False):    
    #TODO change from separate arrays into array of tupples - makes code inconvenient
    #x = middle point of the pantom | y-> = distance from base frame to phantom | z->y = 0
    center = (size/2,middlepoint[1],middlepoint[2]) #[cm] -> here y is behaving as 3d z
    angles = np.linspace(0, np.pi, 101)
    x = center[0] + rad * np.cos(angles)
    x=x[::-1] #flip the array to keep it in ascending order
    if flip:
        y = center[1] - rad * np.sin(angles)
    else:
        y = center[1] + rad * np.sin(angles)
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
    #TODO do a safety check if the values are correct
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

def pitStopsAng(alpha_t,maxRot,rad):
    stops_t = maxRot/alpha_t
    stops = round(stops_t)
    alpha = maxRot/stops
    #This would be for half of the path
    #Allowing us to always sample extremes and center point
    stops = stops*2+1
    print(f'Given the angle {alpha_t}, there would be {stops_t*2 +1} stops total',
          f'\nRounding to {alpha} deg., with {stops} stops total')

    theta = 90-maxRot
    pitsA = []
    for i in range(stops):
        xdist = np.cos(np.radians(alpha*i+theta))*rad
        ydist = np.sin(np.radians(alpha*i+theta))*rad
        pitsA.append((xdist,ydist))
    return pitsA,stops    

def calcAlpha(stops_t,maxRot):
    if stops_t % 2 == 0:
        stops_t +=1
        print('Adding an extra stop, total of',stops_t,'stops')
        
    alpha = maxRot/(int(stops_t/2))
    return alpha
    

def plotPath(x,y,xcut,ycut,pits):
    #Visualization of the path
    for point in pits:
        plt.plot(xcut[point],ycut[point],marker='x', markersize=10)
    plt.plot(xcut,ycut, linewidth=2)
    plt.plot(x,y)

def projPath3d(xcut,ycut,pits,shape,rad,middlepoint,path='length',flip=False):
    #Write the positions in the 3d coordinates from the reference frame
    #TODO check quaternions to encode angles - slerp
    tcoord,trot=[],[]
    for point in pits:
        xs,zs,ys = middlepoint
        l,w = shape
        if path == 'length':
            #x -> point + object position - center
            tcoord.append([(xcut[point] + xs - l/2) * 0.01,
                           ys * 0.01,
                           ycut[point] * 0.01])
        elif path == 'width':
            #x -> object position - center
            #y -> point 
            tcoord.append([(xs) * 0.01,
                           (xcut[point] + ys - w/2) * 0.01,
                           ycut[point] * 0.01])
            
        #calculate rotation angle to keep the probe facing the point
        distX = xcut[pits[len(pits)//2]]-xcut[point]
        ang = np.degrees(np.arcsin(distX/rad))
        
        bAng=90
        if flip:
            if path == 'length':
                trot.append([0,-bAng+ang,0]) 
            elif path =='width':
                trot.append([-ang,-bAng,0]) 
        else:
            if path == 'length':
                trot.append([0,bAng-ang,0]) 
            elif path =='width':
                trot.append([ang,bAng,0]) 
        #-90 x axis points upwards, so we add the angle 
        
        # trot.append([0,0,0])
            
    return tcoord,trot

def projPath3dAng(pitsA,middlepoint,shape,rad,path,flip):
    aa,bb=[],[]
    for point in pitsA:
        xs,zs,ys = middlepoint
        l,w = shape

        if path == 'length':
            aa.append([(point[0] + xs) * 0.01,
                        ys * 0.01,
                        (zs - point[1]) * 0.01])
        elif path == 'width':
            aa.append([(xs) * 0.01,
                        (point[0] + ys) * 0.01,
                        (zs - point[1]) * 0.01])
            
        #calculate rotation angle to keep the probe facing the point
        distX = pitsA[len(pitsA)//2][0] - point[0] #center point - stop point
        ang = np.degrees(np.arcsin(distX/rad)) #angle to rotate the end-effector
        
        bAng=90
        if flip:
            if path == 'length':
                bb.append([0,-bAng+ang,0]) 
            elif path =='width':
                bb.append([-ang,-bAng,0]) 
        else:
            if path == 'length':
                bb.append([0,-bAng+ang,0]) 
            elif path =='width':
                bb.append([-ang,-bAng,0]) 
        
    return aa,bb