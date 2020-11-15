# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:26:15 2020

@author: ludvi
"""

#Linear regression, part b)
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, plot)
import numpy as np
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
X = X - np.ones((N,1))*meanX
X = X/stdX
X = np.hstack((X[:,1:],y_new))  #cut out the ozone and day
                                #and combine it with 
N, M = X.shape

C = 4 #Amount of classes

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)

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
n_hidden_units = 7      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 2000

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

errors = [] # make a list for storing generalizaition error in each loop
hidden_unit_used = [] #Chosen hidden unit

print(attributeNames)
M = M+1

## CROSSVALIDATION VALUES:
# Create crossvalidation partition for evaluation
K = 2
CV = model_selection.KFold(K, shuffle=True)
# Values of lambda
lambdas = np.power(2.,range(-2,8))

#Set variables for rlr:
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
Error_baseline_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
opt_lambda_list = np.empty(())
k=0
k_int = 0

#Outer fold:
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold:
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Extract training and test set for current CV fold, convert to tensors (USED IN ANN):
    X_train_out = torch.Tensor(X[train_index,:])
    y_train_out = torch.Tensor(y[train_index])
    X_test_out = torch.Tensor(X[test_index,:])
    y_test_out = torch.Tensor(y[test_index])
    
    #opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    #Crossvalidation:
    K2 = 2
    CV2 = model_selection.KFold(K2, shuffle=True)
    M2 = X_train.shape[1]
    w = np.empty((M2,K2,len(lambdas)))
    train_error = np.empty((K2,len(lambdas)))
    test_error = np.empty((K2,len(lambdas)))
    f = 0
    y_train = y_train.squeeze()
    #For ANN:
    errors_int = [] # make a list for storing generalizaition error in each loop
    
    #Internal fold:
    for train_index1, test_index1 in CV2.split(X_train,y_train):
        
        #Define training sets:
        X_train2 = X_train[train_index1]
        y_train2 = y_train[train_index1]
        X_test2 = X_train[test_index1]
        y_test2 = y_train[test_index1]
        
        #Internal training and datasets for current CV fold -> ANN:
        X_train_int = torch.Tensor(X_train[train_index1,:])
        y_train_int = torch.Tensor(y_train[train_index1])
        X_test_int = torch.Tensor(X_train[test_index1,:])
        y_test_int = torch.Tensor(y_train[test_index1])
        
        #REGULARIZED LINEAR REGRESSION:
        
        # Standardize the training and set based on training set moments
        mu_int = np.mean(X_train2[:, 1:], 0)
        sigma_int = np.std(X_train2[:, 1:], 0)
        
        X_train2[:, 1:] = (X_train2[:, 1:] - mu_int) / sigma_int
        X_test2[:, 1:] = (X_test2[:, 1:] - mu_int) / sigma_int
        
        # precompute terms
        Xty_int = X_train2.T @ y_train2
        XtX_int = X_train2.T @ X_train2
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M2)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX_int+lambdaI,Xty_int).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train2-X_train2 @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test2-X_test2 @ w[:,f,l].T,2).mean(axis=0)
    
        f=f+1
        
        #ARTIFICIAL NEURAL NETWORK:
        errors_h = [] #Errors, that account for number of hidden units
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
        
        #Append errors from errors_int into errors_h:
        errors_h.append(errors_int)
        
    #Define cross validation results -> rlr:
    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    #Add optimum lambda value to list:
    #opt_lambda_list.append(opt_lambda)
    
    #ARTIFICIAL NEURAL NETWORK:
    
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
    
    #REGULARIZED LINEAR REGRESSION:
    
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :]
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train


    #Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    Error_baseline_nofeatures[k] =np.square(y_test-y_train.mean()).sum(axis=0)/y_test.shape[0] #Mean squared error for baseline

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
        
        #Plot, that accounts for the baseline error term:
        # subplot(1,2,2)
        # title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        # loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        # plt.axhline(Error_baseline_nofeatures[k],linestyle = '--',
        #  linewidth = 4, color = 'Blue', dashes = (0.5,5.),
        #  dash_capstyle = 'round')
        # xlabel('Regularization factor')
        # ylabel('Squared error (crossvalidation)')
        # legend(['Train error','Validation error'])
        # grid()
        
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print(m)
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

#Display results - ANN:

#First, set y to shape: (x, 1):
#y=X[:,[0]] # puts the y values equal to ozone
    
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

# ***** LINEAR REGRESSION ***** #











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
max_iter = 10000

# K-fold crossvalidation
K = 10                  # only three folds to speed up this example
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