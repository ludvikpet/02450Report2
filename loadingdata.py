#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:48:54 2020

@author: andreasaspe
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:54:20 2020

@author: andreasaspe
"""

#https://web.stanford.edu/~hastie/ElemStatLearn/?fbclid=IwAR05Sn4GMe9yiEURSFyRgpk8cXtTzcbQTx47Yn09DveqSNYaH6Ra0UvsCWs

import numpy as np
import pandas as pd
import os
import csv
import seaborn as sn
import matplotlib.pyplot as plt
#os.chdir('/Users/andreasaspe/Documents/7. semester/Machine Learning (02450)/02450Toolbox_Python/Data')

filename = 'LAozone.data.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
#raw_data = df.get_values()
raw_data = df.to_numpy()

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(0, 9) 
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
classLabels = raw_data[:,-1] # -1 takes the last column


#ASSIGNING CLASSES
#Interval dage: 1:59 er janaur-februar. Dvs. i python domain er dette 0:58
#Interval dage: 60:151 er marts-maj. Dvs. i python domain er dette 59:150
#Interval dage: 152:243 er juni-august. Dvs. i python domain er dette 151:242
#Interval dage: 244:335 er september-november. Dvs. i python domain er dette 243:334
#Intercal dage: 335-365 er december

'''
# ******** MULTI CLASS SETUP ******** #

classLabels_new = ['']*len(classLabels)
for i in range(0,len(classLabels)):
    # select indices belonging to class c:
    if (1 <= classLabels[i] <= 59 or 335 <= classLabels[i] <= 365):
        classLabels_new[i] = "Winter"
    if 60 <= classLabels[i] <= 151:
        classLabels_new[i] = "Spring"
    if 152 <= classLabels[i] <= 243:
        classLabels_new[i] = "Summer"
    if 244 <= classLabels[i] <= 335:
        classLabels_new[i] = "Fall"

# ******** MULTI CLASS SETUP ******** #
'''

# ******** BINARY CLASS SETUP ******** #

classLabels_new = ['']*len(classLabels)
for i in range(0,len(classLabels)):
    # select indices belonging to class c:
    if (80 <= classLabels[i] < 264):
        classLabels_new[i] = "21st of March - 20th of September"
    else:
        classLabels_new[i] = "21st of September - 20th of March"

# ******** BINARY CLASS SETUP ******** #

# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels_new)
# We can assign each type of Iris class with a number by making a
# Python dictionary as so:
classDict = dict(zip(classNames,range(len(classNames))))
# The function zip simply "zips" togetter the classNames with an integer,
# like a zipper on a jacket. 
# For instance, you could zip a list ['A', 'B', 'C'] with ['D', 'E', 'F'] to
# get the pairs ('A','D'), ('B', 'E'), and ('C', 'F'). 
# A Python dictionary is a data object that stores pairs of a key with a value. 
# This means that when you call a dictionary with a given key, you 
# get the stored corresponding value. Try highlighting classDict and press F9.
# You'll see that the first (key, value)-pair is ('Iris-setosa', 0). 
# If you look up in the dictionary classDict with the value 'Iris-setosa', 
# you will get the value 0. Try it with classDict['Iris-setosa']


# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels_new])
# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of 
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the 
# list to a numpy array". 
# Try running this to get a feel for the operation: 
# list = [0,1,2]
# new_list = [element+10 for element in list]

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)

#Remove the comment below to remove outliers:
'''
# ******** OUTLIER REMOVAL ******** #

#We will remove the wind speed with more than 15 mph.
outlier_mask = X[:,2]>15
valid_mask = np.logical_not(outlier_mask)


# Finally we will remove these from the data set
X = X[valid_mask,:]
#Y = Y[valid_mask,:] # Used when standardizing
y = y[valid_mask]
N = len(y)


outlier_mask2 = X[:,5]>4800
valid_mask2 = np.logical_not(outlier_mask2)

X = X[valid_mask2,:]
#Y = Y[valid_mask,:] # Used when standardizing
y = y[valid_mask2]
N = len(y)


# This reveals no further outliers, and we conclude that all outliers have
# been detected and removed.

# ******** OUTLIER REMOVAL ******** #
'''

#Remove the comment below to allow for one-out-of-K encoding:
'''
# ******** ONE-OUT-OF-K ENCODING ******** #

#Transform data, such that we are able to do one-out-of-K
#encoding:
df["Spring"] = 0
df["Summer"] = 0
df["Fall"] = 0
df["Winter"] = 0


#Redefine classlabels_new:
y_new = np.zeros((len(X),4), dtype=int)

#Extract values for the four columns:
for i in range(0,len(y)):
    # select indices belonging to class c:
    if y[i] == 0:   #Fall
        y_new[i,2] = 1
    if y[i] == 1:   #Spring 
        y_new[i,0] = 1
    if y[i] == 2:   #Summer
        y_new[i,1] = 1
    if y[i] == 3:   #Winter
        y_new[i,3] = 1


# ******** ONE-OUT-OF-K ENCODING ******** #
'''