#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:53:41 2020

@author: andreasaspe
"""

from loadingdata import *
from standarize import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

#Optimal complex controlling parameter for logistic regression:
lambda_logistic = np.power(2.,range(-4,8))

#Optimal complex controlling parameter for descision tree:
tc = np.arange(1, 16, 1)

K1 = 5
K2 = 10

#Defining the two models
CV1 = model_selection.KFold(K1, shuffle=True) #Outer loop
CV2 = model_selection.KFold(K2, shuffle=True) #Inner loop

#Initalising variables
#Logistic regression
test_error_inner_logistic = np.zeros((np.size(lambda_logistic),K2))
train_error_inner_logistic = np.zeros((np.size(lambda_logistic),K2))
opt_lambda_logistic = np.zeros(K1)
error_outer_fold_logistic = np.zeros(K1)
yhat_log = []
#Decision tree
test_error_inner_classificationtree = np.zeros((np.size(tc),K2))
train_error_inner_classificationtree = np.zeros((np.size(tc),K2))
opt_tc_classificationtree = np.zeros(K1)
error_outer_fold_classificationtree = np.zeros(K1)
yhat_clas = []
#Baseline
error_outer_fold_baseline = np.zeros(K1)
yhat_baseline = []
#True valus
y_true = []


mean_test_error_inner_logistic = np.zeros(K2)
mean_train_error_inner_logistic = np.zeros(K2)

opt_lambda_logistic = np.zeros(K1)

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
    
        #index = 0
        #Logistic regression
        for i in range(np.size(lambda_logistic)):
            mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1/lambda_logistic[i])
            mdl.fit(X_train_inner,y_train_inner)
            y_test_est_inner = mdl.predict(X_test_inner)
            y_train_est_inner = mdl.predict(X_train_inner)
            test_error_inner_logistic[i,index_inner] = np.sum(y_test_est_inner!=y_test_inner)/len(y_test_inner)
            train_error_inner_logistic[i,index_inner] = np.sum(y_train_est_inner!=y_train_inner)/len(y_train_inner)
        
        #Decision tree
        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train_inner,y_train_inner.ravel())
            y_test_est_inner = dtc.predict(X_test_inner)
            y_train_est_inner = dtc.predict(X_train_inner)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            test_error_inner_classificationtree[i,index_inner] = np.sum(y_test_est_inner != y_test_inner) / float(len(y_test_est_inner))
            train_error_inner_classificationtree[i,index_inner] = np.sum(y_train_est_inner != y_train_inner) / float(len(y_train_est_inner))
        index_inner += 1
        
        
    #Back to outer fold
    #Logistic regression
    mean_test_error_inner_logistic = np.mean(test_error_inner_logistic,1)
    mean_train_error_inner_logistic = np.mean(train_error_inner_logistic,1)
    min_error_logistic = np.min(mean_test_error_inner_logistic.T)
    opt_lambda_idx_logistic = np.argmin(mean_test_error_inner_logistic.T)
    opt_lambda_logistic[index_outer] = lambda_logistic[opt_lambda_idx_logistic]
    #Training model on original test set
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1/opt_lambda_logistic[index_outer])
    mdl.fit(X_train,y_train)
    y_test_est_log = mdl.predict(X_test)
    error_outer_fold_logistic[index_outer] = np.sum(y_test_est_log!=y_test)/len(y_test)
    #Save predictions
    yhat_log.append(y_test_est_log)

    
    #Decision tree
    mean_test_error_inner_classificationtree = np.mean(test_error_inner_classificationtree,1)
    mean_train_error_inner_classificationtree = np.mean(train_error_inner_classificationtree,1)
    min_error_classificationtree = np.min(mean_test_error_inner_classificationtree.T)
    opt_tc_idx_classificationtree = np.argmin(mean_test_error_inner_classificationtree.T)
    opt_tc_classificationtree[index_outer] = tc[opt_tc_idx_classificationtree]
    #Training model on original test set
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_tc_classificationtree[index_outer])
    dtc = dtc.fit(X_train,y_train.ravel())
    y_test_est_clas = dtc.predict(X_test)
    error_outer_fold_classificationtree[index_outer] = np.sum(y_test_est_clas!=y_test)/len(y_test)
    #Save predictions
    yhat_clas.append(y_test_est_clas)
    
    
    #Baseline
    largest_class = np.bincount(y_train).argmax()
    y_test_est_baseline = largest_class*np.ones(len(y_test))
    error_outer_fold_baseline[index_outer] = np.sum(y_test_est_baseline != y_test) / float(len(y_test))
    #Save predictions
    yhat_baseline.append(y_test_est_baseline)


    #Save true values
    y_true.append(y_test)

    
    index_outer += 1

#Getting the right format for statistical testing
yhat_log = np.reshape(yhat_log,-1)
yhat_clas = np.reshape(yhat_clas,-1)
yhat_baseline = np.reshape(yhat_baseline,-1)
y_true = np.reshape(y_true,-1)