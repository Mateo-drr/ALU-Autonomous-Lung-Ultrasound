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

#Frame
#X is red
#Y is green

#Movement control
#https://github.com/petercorke/robotics-toolbox-python/blob/master/roboticstoolbox/examples/mmc.py
#Sliders
#https://github.com/petercorke/robotics-toolbox-python/blob/master/roboticstoolbox/examples/teach_swift.py

#path for metal phantom
#PARAMS
ask=False
shape = (17,5) #l ,w [cm]
middlepoint = (-50,50,10) #position of the point of interest [cm] x,z,y
alpha=15
flip=False #True if phantom's top points to the base of the robot
angle=False #if we want to use the angle or the num of stops
    
#TODO get alpha angle ie the angle between stops
#TODO make option to choose num stops from an angle
#copelia sim, ros package

#points in length dir
if angle:
    rad,maxRot,_,alpha = pc.askConfig(ask,angle=angle)
else:
    rad,maxRot,stops,_ = pc.askConfig(ask,angle=angle)
    
x,y,center = pc.drawPath(shape[0], rad, middlepoint, flip=flip)
xcut,ycut = pc.cutPath(maxRot, rad, center, x, y)
pits = pc.pitStops(stops, xcut)
pc.plotPath(x, y, xcut, ycut, pits)
tcoord,trot = pc.projPath3d(xcut, ycut, pits, shape, rad, middlepoint,path='length', flip=flip)

print(maxRot,rad,stops)
#If we want to get the angle from the number of stops we can call this function
#print(pc.calcAlpha(stops,maxRot))
pitsA,stops = pc.pitStopsAng(alpha,maxRot,rad)
tcoord,trot = pc.projPath3dAng(pitsA,middlepoint,shape,rad,path='length',flip=flip)

aa,bb = pc.projPath3dAng(pitsA,middlepoint,shape,rad,path='width',flip=flip)
tcoord,trot = tcoord+aa,trot+bb

#points in width dir
#rad,maxRot,stops = pc.askConfig(ask) #if different config for w uncomment
# x,y,center = pc.drawPath(shape[1], rad, middlepoint,flip=flip)
# xcut,ycut = pc.cutPath(maxRot, rad, center, x, y)
# pits = pc.pitStops(stops, xcut)
# pc.plotPath(x, y, xcut, ycut, pits)
# aa,bb = pc.projPath3d(xcut, ycut, pits, shape, rad, middlepoint,path='width',flip=flip)

# tcoord, trot = tcoord+aa, trot+bb

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

#target is = to forward kinematic with an x,y,z displacement [m]
#targetEndPose = ur5.fkine(ur5.q) * sm.SE3.Tx(0.0) * sm.SE3.Ty(0.0) * sm.SE3.Tz(0.7)
#tcoord = [[-0.2,0.0,0.5],
#          [-0.5,0.0,0.5],]
#trot = [[0,-90,0],
 #       [0,-90,0]]

targets = []
for i in range(len(tcoord)):
    coordinates = sm.SE3.Tx(tcoord[i][0]) * sm.SE3.Ty(tcoord[i][1]) * sm.SE3.Tz(tcoord[i][2])
    rotation = sm.SE3.Rx(trot[i][0], unit='deg') * sm.SE3.Ry(trot[i][1], unit='deg') * sm.SE3.Rz(trot[i][2], unit='deg')
    targetEndPose = coordinates * rotation
    
    #add axes at the desired end-effector pose
    axes = sg.Axes(length=0.1, base=targetEndPose)
    env.add(axes) #place them in swift
    targets.append(targetEndPose)

# #Make the simulation step
# for _ in range(100):
#     env.step(0.05)
    
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

