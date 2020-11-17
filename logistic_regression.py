#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:05:34 2020

@author: andreasaspe
"""

#Inspired by # exercise 6.3.2, exercise 8.3.3 and exercise 8.1.1

from loadingdata import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn import model_selection
import sklearn.linear_model as lm
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

K = 10
lambdas = np.power(2.,range(-4,8))

CV = model_selection.KFold(K, shuffle=True) #CV = model_selection.LeaveOneOut()

test_error = np.zeros((np.size(lambdas),K))
train_error = np.zeros((np.size(lambdas),K))

index = 0
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(index+1,K))    
    
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
    
    for i in range(np.size(lambdas)):
        mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1/lambdas[i])
        mdl.fit(X_train,y_train)
        y_test_est = mdl.predict(X_test)
        y_train_est = mdl.predict(X_train)
        test_error[i,index] = np.sum(y_test_est!=y_test)/len(y_test)
        train_error[i,index] = np.sum(y_train_est!=y_train)/len(y_train)
    index += 1

test_error_mean = np.mean(test_error,1)
train_error_mean = np.mean(train_error,1)

min_error = np.min(test_error_mean.T)
opt_lambda_idx = np.argmin(test_error_mean.T)
opt_lambda = lambdas[opt_lambda_idx]

figure()
plt.semilogx(lambdas,train_error_mean.T,'b.-',lambdas,test_error_mean.T,'r.-') #plt.semilogx(lambdas,train_error[:,1],'b.-',lambdas,test_error[:,1],'r.-')
plt.semilogx(opt_lambda,min_error,'o') #To get the right index, +1 is needed
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
#plt.ylim([0,0.4])
legend(['Train error','Validation error'])
grid()
show()


# internal_cross_validation = X.shape[0]
                                    
# opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

# figure(0, figsize=(12,8))