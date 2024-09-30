#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:04:25 2024

@author: mateo-drr
"""

from utils import validateFloat,table2base,validateBool,validateInt,validateList
import copy

#FLANGE size [cm]
FHEIGHT = 18.5#8.47 
FOFFSET = 0.0#15.44
#PROBE size [cm]
PHEIGHT = 0 #TODO
#ERROR
err = 0.0
#RADIUS FROM TARGET
RAD = err + 4.8
#TODO CONTAINER HEIGHT
H0 = 20

scenarios = ['curved', 'linear', 'rotation']
dir2ang = {'fwd': 0,'bkw': 180,'rgt': 270,'lft': 90}
ang2dir = {0: 'fwd', 180: 'bkw', 270: 'rgt', 90: 'lft'}
configDefault = {'angleDiv':True,
                 'rad': RAD, #4 max
                 'maxRotL': 20, #30
                 'maxRotW':20, #30
                 'alphaL': 1,
                 'alphaW':1,
                 'shape':[17,5],
                 'stopsL': 5,
                 'stopsW': 5,
                 'flange':0,#Flange direction. See ask config()
                 'flangeOffset': (0,FHEIGHT+PHEIGHT,FOFFSET), #(0,4.724,8.412), #x,z 2.52+3,y [cm]
                 'point-base':(-18-5,88.7-6,30), #x,z,y
                 'point-table':None,
                 'initCoord':(-18-5,65,30), #x,z,y
                 'initRot':(0,0,90), # ? , point up or down, flange to the left or right
                 'numInt':2,
                 'radOffset':RAD,
                 'error':err
                 }

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
    
    angle = validateBool("Divide path by angle?: ")
    #Ask for the radius of the circle
    rad = validateFloat('Radius [cm] from the center point: ', 'Radius')
    #Ask for the max angle in lenght
    maxRotL = validateFloat('Maximum angle [deg] from the top view along the length: ',
                       'Maximum angle',
                       maximum=90)
    #Ask for the max angle in width
    maxRotW = validateFloat('Maximum angle [deg] from the top view along the width: ',
                       'Maximum angle',
                       maximum=90)
        
    #If the stops are calculated by the angle    
    if angle:
        
        #Ask for the angle for the stops in lenght
        alphaL = validateFloat('Angle [deg] between stops along the lenght: ',
                           'Alpha',
                           maximum=maxRotL)
            
        #Ask for the angle for the stops in width
        alphaW = validateFloat('Angle [deg] between stops along the width: ',
                           'Alpha',
                           maximum=maxRotW)
        
    #If the stops are calculated by number
    else:
        
        #Ask for the number of stops in lenght
        stopsL = validateInt('Number of stops for imaging along the lenght: ', 'Stops')
        stopsL = max(stopsL, 3)  # minimum of 3 stops -> edges and center
        # if an even number is given just add one stop
        if stopsL % 2 == 0:
            print('Setting number of stops to ', stopsL, '+ 1')
            stopsL += 1
            
        #Ask for the number of stops in width
        stopsW = validateInt('Number of stops for imaging along the width: ', 'Stops')
        stopsW = max(stopsW, 3)  # minimum of 3 stops -> edges and center
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

def askRotation(config):
    
    #initialize vars    
    angle = config.get('angleDiv')
    rad = config.get('rad') #taking the radius as the distance
    maxRotL = config.get('maxRotL')
    maxRotW = config.get('maxRotW')
    stopsL = config.get('stopsL')
    stopsW = config.get('stopsW')
    alphaL = config.get('alphaL')
    alphaW = config.get('alphaW')
    
    angle = validateBool("Divide path by angle?: ")
    #Ask for the distance from the object
    rad = validateFloat('Probe distance [cm] from the object: ', 'Distance')
    #Ask for the max angle in lenght
    maxRotL = validateFloat('Maximum angle [deg] from the top view along the length: ',
                       'Maximum angle',
                       maximum=90)
    #Ask for the max angle in width
    maxRotW = validateFloat('Maximum angle [deg] from the top view along the width: ',
                       'Maximum angle',
                       maximum=90)
        
    #If the stops are calculated by the angle    
    if angle:
        
        #Ask for the angle for the stops in lenght
        alphaL = validateFloat('Angle [deg] between stops along the lenght: ',
                           'Alpha',
                           maximum=maxRotL)
            
        #Ask for the angle for the stops in width
        alphaW = validateFloat('Angle [deg] between stops along the width: ',
                           'Alpha',
                           maximum=maxRotW)
        
    #If the stops are calculated by number
    else:
        
        #Ask for the number of stops in lenght
        stopsL = validateInt('Number of stops for imaging along the lenght: ', 'Stops')
        stopsL = max(stopsL, 3)  # minimum of 3 stops -> edges and center
        # if an even number is given just add one stop
        if stopsL % 2 == 0:
            print('Setting number of stops to ', stopsL, '+ 1')
            stopsL += 1
            
        #Ask for the number of stops in width
        stopsW = validateInt('Number of stops for imaging along the width: ', 'Stops')
        stopsW = max(stopsW, 3)  # minimum of 3 stops -> edges and center
        # if an even number is given just add one stop
        if stopsW % 2 == 0:
            print('Setting number of stops to ', stopsW, '+ 1')
            stopsW += 1

    #save updated vars
    config['angleDiv'] = angle
    config['rad'] = 0.1#rad
    config['radOffset'] = rad
    config['maxRotL'] = maxRotL
    config['maxRotW'] = maxRotW
    config['stopsL'] = stopsL
    config['stopsW'] = stopsW
    config['alphaL'] = alphaL
    config['alphaW'] = alphaW
    return config

def askLinear(config):
    #ask for the shape of the object
    l = validateFloat('What is the length of the path [cm]: ', 'Size')
    w = validateFloat('What is the width of the path [cm]: ', 'Size')
    
    #Ask for the number of stops
    stopsL = validateInt('Number of stops for imaging along the lenght: ', 'Stops')
    stopsW = validateInt('Number of stops for imaging along the width: ', 'Stops')
    
    #Ask for the distance to the target
    rad = validateFloat('Distance [cm] from the target: ', 'Distance')
    
    config['rad'] = rad
    config['shape'] = [l,w]
    config['stopsL'] = stopsL
    config['stopsW'] = stopsW
    return config

def askConfig():
    #copy default config
    config = copy.deepcopy(configDefault)
    
    scene = validateList('What path would you like to use: ',scenarios)
    
    use_defaults = validateBool("Would you like to use default values for configuration? ")
    
    #Ask for flange direction
    flange = validateList("Direction to point the probe? ", list(dir2ang.keys()))
    config['flange'] = dir2ang[flange]
    
    if not use_defaults:
        
        #Ask for position of the object
        pos=[0,0,0]
        pos[2] = validateFloat('Height of the object [cm] ', 'Height') #w r to the table
        pos[0] = validateFloat('X distance from robot base [cm] ', 'Distance',minimum=-100)
        pos[0] = validateFloat('Y distance from robot base [cm] ', 'Distance',minimum=-100)
        #convert to postiion w r to base
        pos = table2base(pos)
        
        #store settings
        config['point-base'] = pos
        
        #Ask parameters for the selected scenario
        if scene == scenarios[0]:
            config = askCurved(config)
        elif scene == scenarios[1]:
            config = askLinear(config)
        elif scene == scenarios[2]:
            config = askRotation(config)
    
    return config,scene