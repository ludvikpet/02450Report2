#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:28:26 2020

@author: andreasaspe
"""

import numpy as np

def standarize(X):
  mu = np.mean(X, 0)
  sigma = np.std(X, 0)
  X = (X - mu) / sigma
  return X