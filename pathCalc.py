# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:47:28 2024

@author: mateo-drr
"""

import matplotlib.pyplot as plt
import numpy as np
import spatialmath as sm
import copy
from pprint import pprint
from utils import validate,validateLimited

#Metal phantom
# size: 5x17 [cm]
# it has 9 rough, 8 smooth each 1 cm
# middle point lands in stripe #5 and its position is 2.5x8.5 [cm]
table = [64.5,100,89] #[w,l,h][cm]


scenarios = ['curved', 'linear']
configDefault = {'angleDiv':None,
                 'rad': 20,
                 'maxRotL': 40,
                 'maxRotW':40,
                 'alphaL': 20,
                 'alphaW':20,
                 'shape':[17,5],
                 'stopsL': 5,
                 'stopsW': 5,
                 'flange':180,
                 'flangeOffset': (0,2.52+3,20.5), #x,z,y
                 'point-base':(-10,80,10), #x,z,y
                 'point-table':None}

def table2base(pos):
    #TODO
    trans = [0,0,0]
    trans[2] = table[2]-pos[2]
    return (pos[0],pos[1],trans[2])

def base2table(pos):
    #TODO
    trans = [0,0,0]
    return

def curvedScene(config,flip,swift=False):
    if not config['angleDiv']: #ie alpha needs to be calculated
        config['alphaL'] = calcAlpha(config['stopsL'],config['maxRotL'])
        config['alphaW'] = calcAlpha(config['stopsW'],config['maxRotW'])

    #Calculate the positions of the stops along lenght
    config['pitsL'],config['stopsL'] = pitStopsAng(config['alphaL'],
                                                      config['maxRotL'],
                                                      config['rad'])
    #Project the 2d coordinates into 3d
    tcoordL,trotL = projPath3dAng(config['pitsL'],
                                   config,#['point-base'],
                                   #config['rad'],
                                   #config['flange'],
                                   path='length',flip=flip,swift=swift)
    
    #Calculate the position of the stops along width
    config['pitsW'],config['stopsW'] = pitStopsAng(config['alphaW'],
                                                      config['maxRotW'],
                                                      config['rad'])
    #Project the 2d coordinates into 3d
    tcoordW,trotW = projPath3dAng(config['pitsW'],
                                   config,#['point-base'],
                                   #config['rad'],
                                   #config['flange'],
                                   path='width',flip=flip,swift=swift)
    return tcoordL+tcoordW,trotL+trotW

def linearScene(config,flip):
    #Calculate the positions of the stops along lenght
    config['pitsL'] = pitStopsLin(config['shape'],
                          config['stopsL'],
                          config['rad'],
                          path='length')
    #Project the 2d coordinates into 3d
    tcoordL,trotL = projPath3dLin(config['pitsL'],
                              config['point-base'],
                              config['rad'],
                              config['flange'],
                              path='length',flip=False)
    
    #Calculate the positions of the stops along width
    config['pitsW'] = pitStopsLin(config['shape'],
                          config['stopsW'],
                          config['rad'],
                          path='width')
    #Project the 2d coordinates into 3d
    tcoordW,trotW = projPath3dLin(config['pitsW'],
                              config['point-base'],
                              config['rad'],
                              config['flange'],
                              path='width',flip=False)
    return tcoordL+tcoordW,trotL+trotW

def askCurved(config):
    
    #initialize vars    
    angle = config.get('angleDiv')
    rad = config.get('rad')
    maxRotL = config.get('maxRotL')
    maxRotW = config.get('maxRotW')
    stopsL = config.get('stopsL')
    stopsW = config.get('stopsW')
    alphaL = config.get('alphaL')
    alphaW = config.get('alphaW')
    
    angle = True if input("Divide path by angle?: ").lower() == 'yes' else False
    
    #Ask for the radius of the circle
    rad = validate('Radius [cm] from the center point: ', 'Radius')
    
    #Ask for the max angle in lenght
    maxRotL = validateLimited('Maximum angle [deg] from the top view along the length: ',
                       'Maximum angle',
                       90)
            
    #Ask for the max angle in width
    maxRotW = validateLimited('Maximum angle [deg] from the top view along the width: ',
                       'Maximum angle',
                       90)
        
    #If the stops are calculated by the angle    
    if angle:
        
        #Ask for the angle for the stops in lenght
        alphaL = validateLimited('Angle [deg] between stops along the lenght: ',
                           'Alpha',
                           maxRotL)
            
        #Ask for the angle for the stops in width
        alphaW = validateLimited('Angle [deg] between stops along the width: ',
                           'Alpha',
                           maxRotW)
        
    #If the stops are calculated by number
    else:
        
        #Ask for the number of stops in lenght
        stopsL = validate('Number of stops for imaging along the lenght: ', 'Stops')
        stopsL = max(int(stopsL), 3)  # minimum of 3 stops -> edges and center
        # if an even number is given just add one stop
        if stopsL % 2 == 0:
            print('Setting number of stops to ', stopsL, '+ 1')
            stopsL += 1
            
        #Ask for the number of stops in width
        stopsW = validate('Number of stops for imaging along the width: ', 'Stops')
        stopsW = max(int(stopsW), 3)  # minimum of 3 stops -> edges and center
        # if an even number is given just add one stop
        if stopsW % 2 == 0:
            print('Setting number of stops to ', stopsW, '+ 1')
            stopsW += 1

    #save updated vars
    config['angleDiv'] = angle
    config['rad'] = rad
    config['maxRotL'] = maxRotL
    config['maxRotW'] = maxRotW
    config['stopsL'] = stopsL
    config['stopsW'] = stopsW
    config['alphaL'] = alphaL
    config['alphaW'] = alphaW
    return config

def askLinear(config):
    #ask for the shape of the object
    l = validate('What is the length of the path [cm]: ', 'Size')
    w = validate('What is the width of the path [cm]: ', 'Size')
    
    #Ask for the number of stops
    stopsL = int(validate('Number of stops for imaging along the lenght: ', 'Stops'))
    stopsW = int(validate('Number of stops for imaging along the width: ', 'Stops'))
    
    #Ask for the distance to the target
    rad = validate('Distance [cm] from the target: ', 'Distance')
    
    config['rad'] = rad
    config['shape'] = [l,w]
    config['stopsL'] = stopsL
    config['stopsW'] = stopsW
    return config

def askConfig():
    #copy default config
    config = copy.deepcopy(configDefault)
    
    while True:
        #ask path type
        scene = input(f'What path would you like to use: {scenarios} ? ').lower()
        if scene not in scenarios:
            print('Please choose one of the available options')
        else:
            break
            
    while True:
        #ask default or custom config
        use_defaults = input("Would you like to use default values for configuration? (yes/no): ").lower()
        
        if use_defaults == 'yes':
            break
        elif use_defaults == 'no':
            
            #Ask for flange direction
            dir2ang = {'fwd': 180,'bkw': 0,'rgt': 90,'lft': 270}
            while True:
                flange = input("Direction to point the probe?  [fwd,bkw,rgt,lft] ").lower()
                if flange not in dir2ang:
                    print('Choose one of the options!')
                else:
                    break
            config['flange'] = dir2ang[flange]
            
            #Ask for position of the object
            pos=[0,0,0]
            pos[2] = validate('Height of the object [cm]', 'Height') #w r to the table
            pos[0] = float(input('X distance from robot base [cm]'))
            pos[0] = float(input('Y distance from robot base [cm]'))
            #convert to postiion w r to base
            pos = table2base(pos)
            config['point-base'] = pos
            
            
            #Ask parameters for the selected scenario
            if scene == scenarios[0]:
                config = askCurved(config)
            elif scene == scenarios[1]:
                config = askLinear(config)
                
            break
        else:
             print("Invalid input. Please enter 'yes' or 'no'.")
    
    print('Configuration:')
    pprint(config) 
    return config,scene
                

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

def pathOffset(offset):
    xf,zf,yf = offset
    #Offset the targets to match the probe position
    coordinates = sm.SE3.Tx(-xf*0.01) * sm.SE3.Ty(-yf*0.01) * sm.SE3.Tz(-zf*0.01)
    rotation = sm.SE3.Rx(0) * sm.SE3.Ry(0) * sm.SE3.Rz(0)
    targetEndPose = coordinates * rotation
    return targetEndPose

#def projPath3dAng(pitsA,middlepoint,rad,flange,path,flip,swift=False):
def projPath3dAng(pitsA,config,path,flip,swift=False):
    aa,bb=[],[]
    # moved=[]
    
    for point in pitsA:
        xs,zs,ys = config['point-base']
        xf,zf,yf = config['flangeOffset']

        if path == 'length':
            aa.append([(point[0] + xs) * 0.01,
                        ys * 0.01,
                        (zs - point[1]) * 0.01])
            # moved.append([(point[0] + xs) * 0.01,
            #             (ys - yf) * 0.01,
            #             (zs - point[1] - zf) * 0.01])
        elif path == 'width':
            aa.append([(xs) * 0.01,
                        (point[0] + ys) * 0.01,
                        (zs - point[1]) * 0.01])
            # moved.append([(xs) * 0.01,
            #             (point[0] + ys - yf) * 0.01,
            #             (zs - point[1] - zf) * 0.01])
            
        #calculate rotation angle to keep the probe facing the point
        distX = pitsA[len(pitsA)//2][0] - point[0] #center point - stop point
        ang = np.degrees(np.arcsin(distX/config['rad'])) #angle to rotate the end-effector
        
        if swift:
            bAng=90
        else:
            bAng=0
        if path == 'length':
            bb.append([0,-bAng+ang,config['flange']]) 
        elif path =='width':
            bb.append([-ang,-bAng,config['flange']]) 
        #bb.append([0,270,0])
    
    return aa,bb

def encodeStops(tcoord,trot,flangeOffset):
    
    #Calculate offset transformation matrix
    offset = pathOffset(flangeOffset)
    
    targets = []
    probeTargets=[]
    for i in range(len(tcoord)):
        coordinates = sm.SE3.Tx(tcoord[i][0]) * sm.SE3.Ty(tcoord[i][1]) * sm.SE3.Tz(tcoord[i][2])
        rotation = sm.SE3.Rx(trot[i][0], unit='deg') * sm.SE3.Ry(trot[i][1], unit='deg') * sm.SE3.Rz(trot[i][2], unit='deg')
        targetEndPose = coordinates * rotation 
    
        targets.append(targetEndPose * offset)
        probeTargets.append(targetEndPose)
    return targets,probeTargets

def getQuat(target):
    temp = sm.UnitQuaternion(target)
    temp = temp.A
    return temp

def pitStopsLin(shape,stops,rad,path):
    if path == 'length':
        idx=0
    elif path == 'width':
        idx=1
    
    #get the size of the path
    size = shape[idx]
    #get the midpoint
    mid = size/2
    #calculate the stops along the path
    spacing = size/(stops+1)
    
    pits=[]
    for i in range(stops):
        pits.append((spacing*(i+1) - mid,rad))
        
    return pits
    
def projPath3dLin(pits,middlepoint,rad,flange,path,flip,swift=False):
    aa,bb=[],[]
    for point in pits:
        xs,zs,ys = middlepoint

        if path == 'length':
            aa.append([(point[0] + xs) * 0.01,
                        ys * 0.01,
                        (zs - point[1]) * 0.01])
        elif path == 'width':
            aa.append([(xs) * 0.01,
                        (point[0] + ys) * 0.01,
                        (zs - point[1]) * 0.01])
            
        #no rotation needed in this case
        if swift:
            bAng=90
        else:
            bAng=0
        if path == 'length':
            bb.append([0,-bAng,flange]) 
        elif path =='width':
            bb.append([0,-bAng,flange]) 
        
    return aa,bb

#askConfig()
    







