#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 22:13:04 2020

@author: andreasaspe
"""

from loadingdata import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

K1 = 10
K2 = 10

#Defining the two models
CV1 = model_selection.KFold(K1, shuffle=True) #Outer loop
CV2 = model_selection.KFold(K2, shuffle=True) #Inner loop

#Initalising variables
#Logistic regression
test_error_inner_logistic = np.zeros((K2))
train_error_inner_logistic = np.zeros((K2))
error_outer_fold_logistic = np.zeros(K1)
yhat_log = []

#True valus
y_true = []


mean_test_error_inner_logistic = np.zeros(K2)
mean_train_error_inner_logistic = np.zeros(K2)


opt_lambda = 4
index_outer = 0
for train_index, test_index in CV1.split(X, y):
    print('Crossvalidation outer fold: {0}/{1}'.format(index_outer+1,K1))    
    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    #Normalising data
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    index_inner = 0
    #Inner fold
    for train_index_inner, test_index_inner in CV2.split(X_train, y_train):
        print('   Crossvalidation inner fold: {0}/{1}'.format(index_inner+1,K2))   
            
        X_train_inner = X_train[train_index_inner,:]
        y_train_inner = y_train[train_index_inner]
        X_test_inner = X_train[test_index_inner,:]
        y_test_inner = y_train[test_index_inner]
        
        mu_inner = np.mean(X_train_inner[:, 1:], 0)
        sigma_inner = np.std(X_train_inner[:, 1:], 0)
        
        X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu_inner) / sigma_inner
        X_test_inner[:, 1:] = (X_test_inner[:, 1:] - mu_inner) / sigma_inner
    
        mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1/opt_lambda)
        mdl.fit(X_train_inner,y_train_inner)
        y_test_est_inner = mdl.predict(X_test_inner)
        y_train_est_inner = mdl.predict(X_train_inner)
        test_error_inner_logistic[index_inner] = np.sum(y_test_est_inner!=y_test_inner)/len(y_test_inner)
        train_error_inner_logistic[index_inner] = np.sum(y_train_est_inner!=y_train_inner)/len(y_train_inner)
        
        index_inner += 1
        
        
    #Back to outer fold
    #Training model on original test set
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1/opt_lambda)
    mdl.fit(X_train,y_train)
    y_test_est_log = mdl.predict(X_test)
    error_outer_fold_logistic[index_outer] = np.sum(y_test_est_log!=y_test)/len(y_test)
    
    #Save predictions
    yhat_log.append(y_test_est_log)

    #Save true values
    y_true.append(y_test)

    
    
    index_outer += 1
    
#Weights in last fold:
print(np.round(mdl.coef_,3))
    
