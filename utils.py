#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:37:36 2024

@author: mateo-drr
"""

def validateLimited(prompt,name,limit):
    while True:
        value = float(input(prompt))
        if value <= 0:
            print(f'{name} cannot be negative or 0')
        elif value > limit:
            print(f'{name} cannot be bigger than {limit} [degrees]')
        else:
            break
    return value

def validate(prompt,name):
    while True:
        value = float(input(prompt))
        if value <=0:
            print(f'{name} cannot be negative or 0')
        else:
            break
    return value