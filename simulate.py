# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:25:01 2024

@author: Mateo-drr
"""

import pathCalc as pc

import roboticstoolbox as rtb
import swift
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
import qpsolvers as qp
from roboticstoolbox import quintic
import copy
import ask

#Frame
#X is red
#Y is green

#Movement control
#https://github.com/petercorke/robotics-toolbox-python/blob/master/roboticstoolbox/examples/mmc.py
#Sliders
#https://github.com/petercorke/robotics-toolbox-python/blob/master/roboticstoolbox/examples/teach_swift.py

###############################################################################
#PATH PLAN
###############################################################################

#PARAMS
shape = (17,5) #l ,w [cm]
middlepoint = (-50,50,10) #position of the point of interest [cm] x,z,y
flip=False #True if phantom's top points to the base of the robot
numInt = 2

#get path configuration
config,scene = ask.askConfig()
    
# #If we want to get the angle from the number of stops we can call this function
# if not config['angleDiv']: #ie alpha needs to be calculated
#     config['alphaL'] = pc.calcAlpha(config['stopsL'],config['maxRotL'])
#     config['alphaW'] = pc.calcAlpha(config['stopsW'],config['maxRotW'])

# #Calculate the positions of the stops along lenght
# config['pitsL'],config['stopsL'] = pc.pitStopsAng(config['alphaL'],
#                                                   config['maxRotL'],
#                                                   config['rad'])
# #Project the 2d coordinates into 3d
# tcoordL,trotL = pc.projPath3dAng(config['pitsL'],
#                                middlepoint,
#                                config['rad'],
#                                config['flange'],
#                                path='length',flip=flip,swift=True)
# #Plot the path
# pc.plotPathAng(config['pitsL'], config['rad'])

# #Calculate the position of the stops along width
# config['pitsW'],config['stopsW'] = pc.pitStopsAng(config['alphaW'],
#                                                   config['maxRotW'],
#                                                   config['rad'])
# #Project the 2d coordinates into 3d
# tcoordW,trotW = pc.projPath3dAng(config['pitsW'],
#                                middlepoint,
#                                config['rad'],
#                                config['flange'],
#                                path='width',flip=flip,swift=True)
#Plot the path
#pc.plotPathAng(config['pitsW'], config['rad'])

#Join both lists of targets
#tcoord,trot = tcoordL+tcoordW,trotL+trotW
#tcoord,trot = pc.curvedScene(config, flip,swift=True)
tcoord,trot = pc.rotationScene(config, flip,swift=True)
pc.plotPathAng(config['pitsW'], config['rad'])
##### For linear

# pits = pc.pitStopsLin(config['shape'],
#                       config['stopsL'],
#                       config['rad'],
#                       path='length')
# aa,bb = pc.projPath3dLin(pits,
#                          middlepoint,
#                          config['rad'],
#                          config['flange'],
#                          path='length',flip=False,swift=True)

# pc.plotPathAng(pits, config['rad'])

# pits = pc.pitStopsLin(config['shape'],
#                       config['stopsW'],
#                       config['rad'],
#                       path='width')
# aaq,bbq = pc.projPath3dLin(pits,
#                          middlepoint,
#                          config['rad'],
#                          config['flange'],
#                          path='width',flip=False,swift=True)

# pc.plotPathAng(pits, config['rad'])
# tcoord,trot = aa+aaq,bb+bbq




###############################################################################
#SIMULATION
###############################################################################

ur5 = rtb.models.UR5() #define the robot

env = swift.Swift() #define the swift environment
env.launch(realtime=True) #start it

#ur5.q -> joint coordinates
#ur5.qd -> joint velocities
#ur5.qd = [0,-0.1,0,0,0,0] #Add a rotation velocity to a specific joint [rad/s]

ur5.q = ur5.qr #assign a position to the robot

env.add(ur5) #put the robot in swift


###############################################################################
#SLIDERS
###############################################################################
# This is our callback funciton from the sliders in Swift which set
# the joint angles of our robot to the value of the sliders
def set_joint(j, value):
    ur5.q[j] = np.deg2rad(float(value))

# Loop through each link in the Panda and if it is a variable joint,
# add a slider to Swift to control it
j = 0
for link in ur5.links:
    if link.isjoint:

        # We use a lambda as the callback function from Swift
        # j=j is used to set the value of j rather than the variable j
        # We use the HTML unicode format for the degree sign in the unit arg
        env.add(
            swift.Slider(
                lambda x, j=j: set_joint(j, x),
                min=np.round(np.rad2deg(link.qlim[0]), 2),
                max=np.round(np.rad2deg(link.qlim[1]), 2),
                step=1,
                value=np.round(np.rad2deg(ur5.q[j]), 2),
                desc="UR5 Joint " + str(j),
                unit="&#176;",
            )
        )

        j += 1
        
###############################################################################
#TARGETS
###############################################################################

#Calculate offset transformation matrix
offset = pc.pathOffset(config['flangeOffset'])

checkpoints = []

for i in range(1,len(tcoord)):
    #interpolate coordinates
    checkpointsX = quintic(tcoord[i-1][0],tcoord[i][0],numInt+2)
    checkpointsY = quintic(tcoord[i-1][1],tcoord[i][1],numInt+2)
    checkpointsZ = quintic(tcoord[i-1][2],tcoord[i][2],numInt+2)
    
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
            checkpoints.append([ckX,ckY,ckZ])
        
        #save all points except last one
        if j != len(checkpointsX)-1:
            checkpoints.append([ckX,ckY,ckZ])
        
coordbkp = copy.deepcopy(tcoord)
rotbkp = copy.deepcopy(trot)        
#tcoord = checkpoints

targets = []
quaternions=[]
for i in range(len(tcoord)):
    coordinates = pc.coord2SE3(*tcoord[i])#sm.SE3.Tx(tcoord[i][0]) * sm.SE3.Ty(tcoord[i][1]) * sm.SE3.Tz(tcoord[i][2])
    rotation = pc.rot2SE3(*trot[i])#sm.SE3.Rx(trot[i][0], unit='deg') * sm.SE3.Ry(trot[i][1], unit='deg') * sm.SE3.Rz(trot[i][2], unit='deg')
    targetEndPose = coordinates * rotation
    
    #add axes at the desired end-effector pose
    # axes = sg.Axes(length=0.1, base=targetEndPose * offset)
    # env.add(axes) #place them in swift
    targets.append(targetEndPose )#* offset)    
    
    #add axes at the desired end-effector pose
    axes = sg.Axes(length=0.1, base=targetEndPose)
    env.add(axes) #place them in swift

#get the quaternion
quaternions = pc.getQuat(targets)
#rearrange from w,xyz to xyzw
#quaternions = [np.array(q[1:].tolist() + [q[0]]) for q in quaternions]

#calculate slerp
checkrot = []
for i in range(1,len(quaternions)):
    #append initial quaternion
    checkrot.append(quaternions[i-1])
    #append interpolated quaternions 
    checkrot += [q for q in pc.slerpCalc(quaternions[i - 1], quaternions[i], numInt)]
    
    #Add the last target only
    if i == len(quaternions)-1:
        checkrot.append(quaternions[i-1])
    
#rearange from xyzw to wxyz
#checkrot = [np.array(q[-1:] + q[:-1]) for q in checkrot]
#transform the data to sm.unitquaternion format
checkrot = sm.UnitQuaternion(checkrot)
#transform the quaternions to se3
checkrot = [q.SE3() for q in checkrot]

aa = copy.deepcopy(checkpoints)
#transform coordinates to se3 matrix
for i in range(len(checkpoints)):
    checkpoints[i] = pc.coord2SE3(*checkpoints[i])
    #add the rotation to the matrices
    checkpoints[i] *= checkrot[i]
    
    #axes = sg.Axes(length=0.1, base=checkpoints[i])
    #env.add(axes) #place them in swift



###############################################################################
#Code to make the robot reach the goal position
###############################################################################
arrived = False
timeStep = 0.01
n=len(ur5.q)
qdlim = np.full(6, np.pi)

for target in targets:

    while not arrived:
        
        # The pose of the end-effector
        currentPos = ur5.fkine(ur5.q)
        # Transform from the end-effector to desired pose
        eTep = currentPos.inv() * target
        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))
        
        errorVec, arrived = rtb.p_servo(currentPos, target, gain=1, threshold=0.01)
        #returns an error vector. Gain controls how fast the robot movements are. 
        #Threshold is the minimum error needed to consider the robot arrived
        
        # Gain term (lambda) for control minimisation
        gain = 0.01
        # Quadratic component of objective function
        Q = np.eye(n + 6)
        # Joint velocity component of Q
        Q[:n, :n] *= gain
        # Slack component of Q
        Q[n:, n:] = (1 / e) * np.eye(6)
        # The equality contraints
        Aeq = np.c_[ur5.jacobe(ur5.q), np.eye(6)]
        beq = errorVec.reshape((6,))
        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((n + 6, n + 6))
        bin = np.zeros(n + 6)
        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.05
        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9    
        # Form the joint limit velocity damper
        Ain[:n, :n], bin[:n] = ur5.joint_velocity_damper(ps, pi, n)
        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-ur5.jacobm().reshape((n,)), np.zeros(6)]
        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[qdlim[:n], 10 * np.ones(6)]
        ub = np.r_[qdlim[:n], 10 * np.ones(6)]
        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='cvxopt')
        
        # Apply the joint velocities 
        ur5.qd[:n] = qd[:n]
    
        env.step(timeStep)
        
    print('Reached target')
    #_=input('Anything to continue')
    
    
    arrived=False
###############################################################################    

print('Done!')


###############################################################################
#Code to make the robot reach the goal position
###############################################################################
# arrived = False
# timeStep = 0.01
# n=len(ur5.q)

# for target in targets:

#     while not arrived:
        
#         # The pose of the end-effector
#         currentPos = ur5.fkine(ur5.q)
        
#         errorVec, arrived = rtb.p_servo(currentPos, target, gain=1, threshold=0.1)
#         #returns an error vector. Gain controls how fast the robot movements are. 
#         #Threshold is the minimum error needed to consider the robot arrived
        
        
#         jacobian = ur5.jacobe(ur5.q) #get the jacob-e -> end-effector jacobian
    
#         #calculate the joint velocities of the robot using the jac and errorVec
#         newQd = np.linalg.pinv(jacobian) @ errorVec 
        
#         singularity = ur5.manipulability(ur5.q)#ur5.is_singular(newQd)
#         if singularity == 0:
#             print('Reached singularity')
            
#         else:
#             ur5.qd = newQd
    
#         env.step(timeStep)
        
#     print('Reached target')
#     arrived=False
# ###############################################################################    

