# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:47:28 2024

@author: mateo-drr
"""

import matplotlib.pyplot as plt
import numpy as np
import spatialmath as sm
from roboticstoolbox import quintic
from itertools import chain

#Metal phantom
# size: 5x17 [cm]
# it has 9 rough, 8 smooth each 1 cm
# middle point lands in stripe #5 and its position is 2.5x8.5 [cm]
#table = [64.5,100,89] #[w,l,h][cm]

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
                                   config,
                                   path='length',flip=flip,swift=swift)
    
    #Calculate the position of the stops along width
    config['pitsW'],config['stopsW'] = pitStopsAng(config['alphaW'],
                                                      config['maxRotW'],
                                                      config['rad'])
    #Project the 2d coordinates into 3d
    tcoordW,trotW = projPath3dAng(config['pitsW'],
                                   config,
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
                              path='length',flip=flip)
    
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
                              path='width',flip=flip)
    return tcoordL+tcoordW,trotL+trotW

def rotationScene(config,flip,swift=False):
    #Same as curved scene but simply with a very small radius
    #to avoid being in the water we need to use an offset
    #config['radOffset'] = copy.deepcopy(config['rad'])
    config['rad'] = 1 #this value is used to calculate the rotation and wont
    #affect position
    
    if not config['angleDiv']: #ie alpha needs to be calculated
        config['alphaL'] = calcAlpha(config['stopsL'],config['maxRotL'])
        config['alphaW'] = calcAlpha(config['stopsW'],config['maxRotW'])

    #Calculate the positions of the stops along lenght
    config['pitsL'],config['stopsL'] = pitStopsAng(config['alphaL'],
                                                      config['maxRotL'],
                                                      config['rad'],
                                                      )
    #Project the 2d coordinates into 3d
    tcoordL,trotL = projPath3dRot(config['pitsL'],
                                   config,
                                   path='length',flip=flip,swift=swift)
    
    #Calculate the position of the stops along width
    config['pitsW'],config['stopsW'] = pitStopsAng(config['alphaW'],
                                                      config['maxRotW'],
                                                      config['rad'],
                                                      )
    #Project the 2d coordinates into 3d
    tcoordW,trotW = projPath3dRot(config['pitsW'],
                                   config,
                                   path='width',flip=flip,swift=swift)
    
    return tcoordL+tcoordW,trotL+trotW

def pitStopsAng(alpha_t,maxRot,rad,offset=0):
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
        ydist = np.sin(np.radians(alpha*i+theta))*rad + offset
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
    coordinates = coord2SE3(-xf, -yf, -zf, scaling=0.01)#sm.SE3.Tx(-xf*0.01) * sm.SE3.Ty(-yf*0.01) * sm.SE3.Tz(-zf*0.01)
    rotation = rot2SE3(0, 0, 0)#sm.SE3.Rx(0) * sm.SE3.Ry(0) * sm.SE3.Rz(0) 
    return coordinates * rotation

def projPath3dAng(pitsA,config,path,flip,swift=False):
    aa,bb=[],[]
    
    for point in pitsA:
        xs,zs,ys = config['point-base']
        print(point)

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

def projPath3dRot(pitsA,config,path,flip,swift=False):
    aa,bb=[],[]
    
    for point in pitsA:
        xs,zs,ys = config['point-base']
        print(point)

        aa.append([(xs) * 0.01,
                    ys * 0.01,
                    (zs - config['radOffset']) * 0.01])
            
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

def encodeStops(tcoord,trot,flangeOffset,rotation=True):
    
    #Calculate offset transformation matrix
    offset = pathOffset(flangeOffset)
    
    targets = []
    probeTargets=[]
    for i in range(len(tcoord)):
        coordinates = coord2SE3(*tcoord[i]) #* unpacks the values of the list
        if rotation:
            rotation = rot2SE3(*trot[i])
            targetEndPose = coordinates * rotation 
        else:
            targetEndPose = coordinates
        targets.append(targetEndPose * offset)
        probeTargets.append(targetEndPose)
    return targets,probeTargets

def encodeStop(coord,rot):
    #TODO this is messy, switching z and y
    coordinates = coord2SE3(coord[0],coord[2],coord[1],scaling=0.01) #sm.SE3.Tx(coord[0]*0.01) * sm.SE3.Ty(coord[2]*0.01) * sm.SE3.Tz(coord[1]*0.01)
    rotation = rot2SE3(*rot)#sm.SE3.Rx(rot[0], unit='deg') * sm.SE3.Ry(rot[1], unit='deg') * sm.SE3.Rz(rot[2], unit='deg')
    return coordinates * rotation 
    

def getQuat(target,numpy=True):
    temp = sm.UnitQuaternion(target)
    if numpy:
        return temp.A
    else:
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

def slerp(q0, q1, t):
    """
    Spherical linear interpolation between two quaternions w<xyz>
    
    :type     q0: numpy.array
    :param    q0: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :type     q1: numpy.array
    :param    q1: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :type     t: number
    :param    t: interpolation parameter in the range [0,1]
    :rtype:   numpy.array
    :returns: the 4 x 1 interpolated quaternion
    
    https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py/blob/master/src/general_robotics_toolbox/general_robotics_toolbox.py
    """
    
    assert (t >= 0 and t <= 1), "t must be in the range [0,1]"
    
    q0 = q0/np.linalg.norm(q0)
    q1 = q1/np.linalg.norm(q1)
    
    if (np.dot(q0,q1) < 0):
        q0 = -q0
    
    theta = np.arccos(np.dot(q0,q1))
    
    if (np.abs(theta) < 1e-6):
        return q0
    
    q = (np.sin((1-t)*theta)*q0 + np.sin(t*theta)*q1)/np.sin(theta)
    
    return q/np.linalg.norm(q)

def coord2SE3(xc,yc,zc,scaling=1):
    return sm.SE3.Tx(xc*scaling) * sm.SE3.Ty(yc*scaling) * sm.SE3.Tz(zc*scaling)

def rot2SE3(xr,yr,zr,unit='deg'):
    return sm.SE3.Rx(xr,unit=unit) * sm.SE3.Ry(yr,unit=unit) * sm.SE3.Rz(zr,unit=unit)

def splitCalc(num_splits):
    # Calculate the step size
    step_size = 1 / (num_splits + 1)
    # Generate the split points using a list comprehension
    splits = [step_size * (i + 1) for i in range(num_splits)]
    return splits

def slerpCalc(coord,quatern,split):
    tlist = splitCalc(split)
    interpol = []
    for t in tlist:
        interpol.append(slerp(coord, quatern, t))
    return interpol

def interpolateCoord(tcoord,speed,period):
    checkpoints = []

    for i in range(1,len(tcoord)):
        
        #get the distance
        dist = distCalc(tcoord[i-1], tcoord[i])
        #calculate time needed
        ttot = dist/speed
        #create the time array
        tarr = np.arange(0,ttot+period,period)
        
        #interpolate coordinates
        checkpointsX = quintic(tcoord[i-1][0],tcoord[i][0],tarr)
        checkpointsY = quintic(tcoord[i-1][1],tcoord[i][1],tarr)
        checkpointsZ = quintic(tcoord[i-1][2],tcoord[i][2],tarr)
        
        checkpoints.append([])
        #loop the checkpoints between each target
        for j in range(len(checkpointsX.q)):
            #get just the coordinate/rot
            ckX,ckY,ckZ = checkpointsX.q[j],checkpointsY.q[j],checkpointsZ.q[j]
            #fix starting point
            if j == 0:
                ckX,ckY,ckZ = tcoord[i-1]
            #fix end point and save it only if it's the last target
            if i == len(tcoord)-1 and j == len(checkpointsX.q)-1:
                ckX,ckY,ckZ = tcoord[i]
                checkpoints[i-1].append([ckX,ckY,ckZ])
            
            #save all points except last one
            if j != len(checkpointsX.q)-1:
                checkpoints[i-1].append([ckX,ckY,ckZ])
    
    #get the number of checkpoints between each target
    numint = [len(c) for c in checkpoints]
    
    return checkpoints,numint

def interpolateRot(quaternions,numint):
    
    print(numint)
    print(quaternions)
    
    #calculate slerp
    checkrot = []
    for i in range(1,len(quaternions)):
        #fix num stops because last stop includes the first and last position
        nint = numint[i-1]-1 if i != len(quaternions)-1 else numint[i-1]-2
        #append initial quaternion
        checkrot.append(quaternions[i-1])
        #append interpolated quaternions 
        #if nint > 0 
        checkrot += [q for q in slerpCalc(quaternions[i - 1], quaternions[i], nint)]
        
        #Add the last target only
        if i == len(quaternions)-1:
            checkrot.append(quaternions[i])
            
    return sm.UnitQuaternion(checkrot).A
    
def distCalc(start, end):
    #input has to be [x,y,z]
    xdist = np.abs(start[0]-end[0])
    ydist = np.abs(start[1]-end[1])
    zdist = np.abs(start[2]-end[2])
    
    dist = np.sqrt(xdist**2 + ydist**2 + zdist**2)
    return dist

def interpolateTargets(tcoord,speed,timer_period,targets,offset):
    #Calculate interpolation of position
    tcoordInt,numint = interpolateCoord(tcoord, speed, timer_period)
    #Calculate quaternions
    quat = getQuat(targets)
    #Calculate slerp
    quatInt = interpolateRot(quat, numint)
    
    #flatten the list
    tcoordInt = list(chain.from_iterable(tcoordInt))
    #transform coordinates to se3 without rotation (not necessary)
    _,targetsInt = encodeStops(tcoordInt, None, offset, rotation=False)
    
    return targetsInt, quatInt, numint
    