#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:37:36 2024

@author: mateo-drr
"""
from validator_collection import validators, checkers
from prompt_toolkit import prompt as pt
from prompt_toolkit.completion import WordCompleter

table = [64.5,100,89] #[w,l,h][cm]

def validateFloat(prompt,name,maximum=None,minimum=0):
    while True:
        value = input(prompt)
        if not checkers.is_float(value):
            print('Please enter a number')
        else:
            value = validators.float(value)
            if minimum is not None and value <= minimum:
                print(f'{name} cannot be less than {minimum}')
            elif maximum is not None and value > maximum:
                print(f'{name} cannot be bigger than {maximum} [degrees]')
            else:
                break
    return value

def validateInt(prompt,name):
    while True:
        value = input(prompt)
        if not checkers.is_integer(value):
            print('Please enter a valid number')
        else:
            value = validators.integer(value)
            if value <=0:
                print(f'{name} cannot be negative or 0')
            else:
                break
    return value

def validateBool(prompt):
    while True:
        angle = input(prompt).lower()
        if angle == 'yes' or angle == 'y':
            angle = True
            break
        elif angle == 'no' or angle == 'n':
            angle = False
            break
        else:
            print('Please choose yes or no')
        
    return angle
        
def validateList(prompt,items):
    while True:
        try: #quick fix to be able to run in spyder console
            completer = WordCompleter(items)
            scene = pt(prompt + f'{items} ',completer=completer).lower()
        except:
            scene = input(prompt + f'{items} ').lower()
        if scene not in items:
            print('Please choose one of the available options')
        else:
            break
    return scene

def table2base(pos):
    #TODO
    trans = [0,0,0]
    trans[2] = table[2]-pos[2]
    return (pos[0],pos[1],trans[2])

def base2table(pos):
    #TODO
    trans = [0,0,0]
    return