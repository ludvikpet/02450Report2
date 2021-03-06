# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:58:18 2020

@author: ludvi
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

# ***** ARTIFICIAL NEURAL NETWORK ***** #

#Standardize data:
meanX = X.mean(axis = 0)
stdX = X.std(ddof=1,axis=0)
y=X[:,[0]] # puts the y values equal to ozone
X = X - np.ones((N,1))*meanX
X = X/stdX
X = np.hstack((X[:,1:],y_new))  #cut out the ozone and day
                                #and combine it with 

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)

#removes the attribute ozone and season(day)

N, M = X.shape
C = 4 #Amount of classes
#X = stats.zscore(X) # Normalize data
attributeNames = [u'Offset']+attributeNames[1:9].tolist()+['Spring','Summer','Fall','Winter']

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
n_hidden_units = 12      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

# K-fold crossvalidation
K = 5              # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

K2 = 5
CV2 = model_selection.KFold(K2, shuffle=True)

# Values of lambda
lambdas = np.power(2.,range(-2,8))

# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

#Add ANN variables:
errors = [] # make a list for storing generalizaition error in each loop
hidden_unit_used = [] #Chosen hidden unit
learning_curve_list = []
opt_lambda_list = []

#Model comparison
y_est_rlr = []
y_est_ann = []
y_est_baseline = []


k=0

#Add rlr variables:
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
    
    
    #Crossvalidation
    M2 = X_train.shape[1]
    
    w = np.empty((M2,K2,len(lambdas)))
    train_error = np.empty((K2,len(lambdas)))
    test_error = np.empty((K2,len(lambdas)))
    f = 0
    #y_train = y_train.squeeze()
    
    errors_int = np.empty((K2,n_hidden_units)) # make a list for storing generalizaition error in each loop
    
    for (k_int, (train_index1, test_index1)) in enumerate(CV2.split(X_train,y_train)):
        
        #Define training sets:
        X_train2 = X_train[train_index1]
        y_train2 = y_train[train_index1]
        X_test2 = X_train[test_index1]
        y_test2 = y_train[test_index1]
        
        #Internal training and datasets for current CV fold:
        X_train_int = torch.Tensor(X_train[train_index1,:])
        y_train_int = torch.Tensor(y_train[train_index1])
        X_test_int = torch.Tensor(X_train[test_index1,:])
        y_test_int = torch.Tensor(y_train[test_index1])
        
        
        # Standardize the training and set based on training set moments
        mu_int = np.mean(X_train2[:, 1:-4], 0)
        sigma_int = np.std(X_train2[:, 1:-4], 0)
        
        X_train2[:, 1:-4] = (X_train2[:, 1:-4] - mu_int) / sigma_int
        X_test2[:, 1:-4] = (X_test2[:, 1:-4] - mu_int) / sigma_int
        
        X_train_int[:, 1:-4] = (X_train_int[:, 1:-4] - mu_int) / sigma_int
        X_test_int[:, 1:-4] = (X_test_int[:, 1:-4] - mu_int) / sigma_int
        
        #REGULARIZED LINEAR REGRESSION:
        
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
            train_error[f,l] = np.power(y_train2[:,0]-X_train2 @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test2[:,0]-X_test2 @ w[:,f,l].T,2).mean(axis=0)
    
        f=f+1
        
        #ARTIFICIAL NEURAL NETWORK:
        for h in range(0, n_hidden_units):
            # Define the model
            model_int = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M2, h+1), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h+1, 1), # n_hidden_units to 1 output neuron
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
            errors_int[k_int,h] = mse_int # store error rate for current CV fold
    
    #Define cross validation results -> rlr:
    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    #Add optimum lambda value to list:
    opt_lambda_list.append(opt_lambda)
    
    #ARTIFICIAL NEURAL NETWORK:
    
    #Optimal hidden units:
    opt_hidden_units = np.argmin(np.array(errors_int[k])) + 1
    hidden_unit_used.append(opt_hidden_units)
    
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, opt_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(opt_hidden_units, 1), # n_hidden_units to 1 output neuron
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
    #Add learning curve to list:
    learning_curve_list.append(learning_curve)
    
    #REGULARIZED LINEAR REGRESSION:
    
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    #mu[k, :] = np.mean(X_train[:, 1:], 0)
    #sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    #X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    #X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :]
    
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
    Error_train_rlr[k] = np.mean(np.square(y_train[:,0]-X_train @ w_rlr[:,k]))
    Error_test_rlr[k] = np.mean(np.square(y_test[:,0]-X_test @ w_rlr[:,k]))
    
    # Compute squared error with regularization with optimal lambda
    rlr_se = np.abs(y_test[:,0]-X_test @ w_rlr[:,k])**2
    baseline_se = np.abs(y_test-y_train.mean())**2 #Baseline
    
    #Storing loss function of the three models:
    y_est_rlr.append(rlr_se)
    y_est_ann.append(se.data.numpy())
    y_est_baseline.append(baseline_se)
    
    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.mean(np.square(y_train[:,0]-X_train @ w_noreg[:,k]))
    Error_test[k] = np.mean(np.square(y_test[:,0]-X_test @ w_noreg[:,k]))
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train, w_rlr[:,k])
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    #y_est_lr = m.predict(X_test)
    
    
    # Display the results for the last cross-validation fold
    # Also, compute the squared error with regularization and with optimal lambda:
    if k == K-1:
        
        figure(k, figsize=(10,10))
        #subplot(1,2,1)
        #semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        #xlabel('Regularization factor')
        #ylabel('Mean Coefficient Values')
        #grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        #subplot(1,2,2)
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

    k+=1

show()
# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))

# Display the learning curve for the best net in the current fold
for (k,l) in enumerate(learning_curve_list):
    h, = summaries_axes[0].plot(l, color=color_list[k])
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


#Statistics:

#Reshape y_estimates:
y_est_ann = np.reshape(y_est_ann, -1)
y_est_rlr = np.reshape(y_est_rlr, -1)
y_est_baseline = np.reshape(y_est_baseline, -1)

#Concatenate y_estimates:
#y_est_ann = np.concatenate(y_est_ann)
#y_est_rlr = np.concatenate(y_est_rlr)
#y_est_baseline = np.concatenate(y_est_baseline)

# compute z with squared error.
z_nn = y_est_ann            #ANN
z_rlr = y_est_rlr           #rlr
z_b = y_est_baseline        #Baseline

# compute confidence interval of the models:
alpha = 0.05
CIA1 = st.t.interval(1-alpha, df=len(z_nn)-1, loc=np.mean(z_nn), scale=st.sem(z_nn))  # Confidence interval for ANN
CIA2 = st.t.interval(1-alpha, df=len(z_rlr)-1, loc=np.mean(z_rlr), scale=st.sem(z_rlr))  # Confidence interval for rlr
CIA3 = st.t.interval(1-alpha, df=len(z_b)-1, loc=np.mean(z_b), scale=st.sem(z_b))  # Confidence interval for rlr

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis

#Pairwise comparison between ANN and rlr:
z1 = z_nn - z_rlr
CI1 = st.t.interval(1-alpha, len(z1)-1, loc=np.mean(z1), scale=st.sem(z1))  # Confidence interval
p1 = st.t.cdf( -np.abs( np.mean(z1) )/st.sem(z1), df=len(z1)-1)  # p-value

#Pairwise comparison between ANN and baseline:
z2 = z_nn - z_b
CI2 = st.t.interval(1-alpha, len(z2)-1, loc=np.mean(z2), scale=st.sem(z2))  # Confidence interval
p2 = st.t.cdf( -np.abs( np.mean(z2) )/st.sem(z2), df=len(z2)-1)  # p-value

#Pairwise comparison between rlr and baseline:
z3 = z_rlr - z_b
CI3 = st.t.interval(1-alpha, len(z3)-1, loc=np.mean(z3), scale=st.sem(z3))  # Confidence interval
p3 = st.t.cdf( -np.abs( np.mean(z3) )/st.sem(z3), df=len(z3)-1)  # p-value

print('Ran Exercise 8.2.5')

# ***** ARTIFICIAL NEURAL NETWORK ***** #