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
raw_data = df.get_values() 

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(0, 10) 
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