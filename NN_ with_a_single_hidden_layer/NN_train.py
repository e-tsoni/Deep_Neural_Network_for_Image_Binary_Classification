# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:01:46 2020

@author: Eftychia Tsoni
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_units import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import operator
from functools import reduce
from nn_model import nn_model
from predict import predict

np.random.seed(1) # set a seed so that the results are consistent

 # Get the dataset
X, Y = load_planar_dataset()
print("X shape is :", X.shape) # 2 features, 400 examples
print("Y shape is :", Y.shape, "\n")
 
# visualize the data
plt.scatter(X[0, :], X[1, :], c=reduce(operator.add, Y), s=40, cmap=plt.cm.Spectral)

# get the number of training examples
m = X.shape[1]
print("m = ", m, "\n")

# number of units in the hidden layer
n_h = 4


# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, learning_rate = 0.01, print_cost=True)

# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')




