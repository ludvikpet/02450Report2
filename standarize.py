#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:28:26 2020

@author: andreasaspe
"""

from loadingdata import *

def standarize():
  stdX = X.std(ddof=1,axis=0)
  Y = X - np.ones((N,1))*X.mean(axis=0)
  Y=Y/stdX #normalise the data by devision of the
                  #standard deviation
  
  return Y