# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:40:33 2020

@author: Jonathan
"""
###linear regression prob A
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from loadingdata import * 


y=X[:,0] # puts the y values equal to ozone
X=np.hstack((X[:,1:],y_new))  #cut out the ozone and day
                                #and combine it with 


N, M = X.shape
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)

#converts it to a string,
#removes the attribute ozone and season(day)

attributeNames = [u'Offset']+attributeNames[1:].tolist()+['Spring','Summer','Fall','Winter'] 

M = M+1

## Crossvalidation

# Values of lambda
lambdas = np.power(2.,range(-2,8))
internal_cross_validation = 10      
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, internal_cross_validation)  
figure(1,figsize=(12,8))
title('Optimal lambda: {0}'.format(opt_lambda))
semilogx(lambdas,(train_err_vs_lambda/10).T,'b.-',lambdas,(test_err_vs_lambda/10).T,'r.-')
xlabel('Regularization factor')
ylabel('Generalization error ')
legend(['Train error','Test error'])
grid()
show()
print(mean_w_vs_lambda[:,6])# displays the weight for the optimal value of lamdba
