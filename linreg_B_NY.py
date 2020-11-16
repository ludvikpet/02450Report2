#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:01:27 2020

@author: andreasaspe
"""
#Linear regression, part b)
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, plot)
import numpy as np, scipy.stats as st
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
from loadingdata import *
import matplotlib.pyplot as plt
from scipy import stats
import torch


# ***** ANN, LINEAR REGRESSION & BASELINE ***** #

#Standardize data - used in linear regression:
meanX = X.mean(axis = 0)
stdX = X.std(ddof=1,axis=0)
y=X[:,0] # puts the y values equal to ozone
X = np.hstack((X[:,1:],y_new))  #cut out the ozone and day
                                #and combine it with 
                                                            
N, M = X.shape

C = 4 #Amount of classes

# Add offset attribute
X_offset = np.concatenate((np.ones((X.shape[0],1)),X),1)

#converts it to a string,
#removes the attribute ozone and season(day)
attributeNames = [u'Offset']+attributeNames[1:9].tolist()+['Spring','Summer','Fall','Winter']

# #Normalize data - used in ANN:
# attributeNames_nn = attributeNames[1:].tolist()+['Spring','Summer','Fall','Winter']
# y_nn=X[:,[0]] # puts the y values equal to ozone
# X_nn = X[:,1:]
# N, M = X.shape
# X_nn = stats.zscore(X) # Normalize data
# X_nn = np.hstack(X_nn,y_new)

#ANN VALUES:
# Parameters for neural network classifier
n_hidden_units = 8      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

errors = [] # make a list for storing generalizaition error in each loop
hidden_unit_used = [] #Chosen hidden unit
learning_curve_list = []

print(attributeNames)
M = M+1

## CROSSVALIDATION VALUES:
# Create crossvalidation partition for evaluation0
K1 = 5
K2 = 10

#Defining the two models
CV1 = model_selection.KFold(K1, shuffle=True) #Outer loop
CV2 = model_selection.KFold(K2, shuffle=True) #Inner loop

# Values of lambda
lambdas = np.power(2.,range(-2,8))

#Set variables for rlr:
Error_train = np.empty((K1,1))
Error_test = np.empty((K1,1))
Error_train_rlr = np.empty((K1,1))
Error_test_rlr = np.empty((K1,1))
Error_train_nofeatures = np.empty((K1,1))
Error_test_nofeatures = np.empty((K1,1))
Error_baseline_nofeatures = np.empty((K1,1))
w_rlr = np.empty((M,K1))
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))
w_noreg = np.empty((M,K1))
opt_lambda_list = []
k=0
k_int = 0

#Added variable for rlr, to use for model comparison:
Error_test_rlr_se = []

index_outer = 0

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
    X_train[1:8] = (X_train[1:8] - mu[1:8]) / sigma[1:8]
    X_test[1:8] = (X_test[1:8] - mu[1:8]) / sigma[1:8]
    
    # Extract training and test set for current CV fold, convert to tensors (USED IN ANN):
    X_train_ANN = torch.Tensor(X[train_index,:])
    y_train_ANN = torch.Tensor(y[train_index])
    X_test_ANN = torch.Tensor(X[test_index,:])
    y_test_ANN = torch.Tensor(y[test_index])
    
    
    # X_train_ANN = (X_train_ANN - mu) / sigma
    # X_test_ANN = (X_test_ANN - mu) / sigma
    
    index_inner = 0
    index_outer = 0
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
        
        


        # #ARTIFICIAL NEURAL NETWORK:
        # errors_h = [] #Errors, that account for number of hidden units
        # for h in range(0, n_hidden_units):
        #     # Define the model
        #     model_inner = lambda: torch.nn.Sequential(
        #                         torch.nn.Linear(M, h+1), #M features to n_hidden_units
        #                         torch.nn.Tanh(),   # 1st transfer function,
        #                         torch.nn.Linear(h+1, 1), # n_hidden_units to 1 output neuron
        #                         # no final tranfer function, i.e. "linear output"
        #                         )
        #     loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
        
        #     #Train the net on training data (inner loop):
        #     net_inner, final_loss_inner, learning_curve_inner = train_neural_net(model_inner,
        #                                                         loss_fn,
        #                                                         X=X_train_inner,
        #                                                         y=y_train_inner,
        #                                                         n_replicates=n_replicates,
        #                                                         max_iter=max_iter)
            # Determine estimated class labels for test set
            # y_test_est_int = net_int(X_test_int)
            
            # # Determine errors and errors
            # se_int = (y_test_est_int[:,0].float()-y_test_int.float())**2 # squared error
            # mse_int = (sum(se_int).type(torch.float)/len(y_test_int)).data.numpy() #mean
            # errors_int[k_int,h] = mse_int # store error rate for current CV fold 
        


        index_inner += 1    
    
    
    
    index_outer += 1

'''
# ***** ARTIFICIAL NEURAL NETWORK ***** #

#Variables:
attributeNames = attributeNames[1:].tolist()+['Spring','Summer','Fall','Winter']
y=X[:,[0]] # puts the y values equal to ozone
X = X[:,:8]
N, M = X.shape
C = 4 #Amount of classes
X = stats.zscore(X) # Normalize data

## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(X,0)
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 2
    X = X @ V[:,:k_pca]


# Parameters for neural network classifier
n_hidden_units = 7      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 2000

# K-fold crossvalidation
K = 2              # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

#print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
hidden_unit_used = [] #Chosen hidden unit
k=0
k_int = 0
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold:
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Extract training and test set for current CV fold, convert to tensors:
    X_train_out = torch.Tensor(X[train_index,:])
    y_train_out = torch.Tensor(y[train_index])
    X_test_out = torch.Tensor(X[test_index,:])
    y_test_out = torch.Tensor(y[test_index])
    
    
    #Original rlr.validate() method:
    #Variables:
    M2 = X_train.shape[1]
    errors_int = [] # make a list for storing generalizaition error in each loop
    #y_train_out = y_train_out.squeeze()
    
    for (k_int, (train_index, test_index)) in enumerate(CV.split(X_train,y_train)):
        #print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        #Internal training and datasets for current CV fold:
        X_train_int = torch.Tensor(X_train[train_index,:])
        y_train_int = torch.Tensor(y_train[train_index])
        X_test_int = torch.Tensor(X_train[test_index,:])
        y_test_int = torch.Tensor(y_train[test_index])
        
        errors_h = []
        for h in range(1, n_hidden_units+1):
            # Define the model
            model_int = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M2, h), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
        
            # Train the net on training data (inner loop):
            net_int, final_loss_int, learning_curve_int = train_neural_net(model_int,
                                                               loss_fn,
                                                               X=X_train_int,
                                                               y=y_train_int,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            # Determine estimated class labels for test set
            y_test_est_int = net_int(X_test_int)
            
            # Determine errors and errors
            se_int = (y_test_est_int.float()-y_test_int.float())**2 # squared error
            mse_int = (sum(se_int).type(torch.float)/len(y_test_int)).data.numpy() #mean
            errors_int.append(mse_int) # store error rate for current CV fold 
        
        errors_h.append(errors_int)
        
    #Optimal hidden units:
    n_hidden_units = np.argmin(np.array(errors_int)[0]) + 1

    for j in range(1,len(errors_h)):
        if np.min(np.array(errors_h)[j-1]) > np.min(np.array(errors_h)[j]):
            n_hidden_units = np.argmin(np.array(errors_h)[j]) + 1
    
    hidden_unit_used.append(n_hidden_units)
    
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    # Train the net on training data (outer loop):
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_out,
                                                       y=y_train_out,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    print('\n\tChosen hidden unit: {}\n'.format(hidden_unit_used[k]))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test_out)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test_out.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test_out)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold 

    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Test mean-squared-error')


print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
#print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

print(errors)


# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy(); y_true = y_test_out.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Ozone concentration: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()

print('Ran Exercise 8.2.5')

# ***** ARTIFICIAL NEURAL NETWORK ***** #
'''