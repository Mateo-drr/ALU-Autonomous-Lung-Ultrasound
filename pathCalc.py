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
        angle = True if input("Divide path by angle?: ").lower() == 'yes' else False
    else:
        angle=False #if we want to use the angle or the num of stops
    
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
        if stops %2 == 0 and stops !=0 :
            print('Setting number of stops to ',stops, '+ 1')
            stops+=1
    else:
        rad = 7 #[cm]
        maxRot = 40 # TODO probe rot is limited by flange 
        stops = 5
        alpha = 20
    print('')
    return rad,maxRot,stops,alpha

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
    
def plotPathAng(pitsA,rad):
    plt.figure(dpi=200)
    for stop in pitsA:
        plt.plot(stop[0],stop[1],marker='x', markersize=6,linewidth=5)
    plt.xlim(-rad - 0.5, rad + 0.5)  # Set x-axis limits
    plt.ylim(-rad - 0.5, rad + 0.5)  # Set y-axis limits
    plt.show()

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
        if path == 'length':
            bb.append([0,-bAng+ang,0]) 
        elif path =='width':
            bb.append([-ang,-bAng,0]) 
        
    return aa,bb