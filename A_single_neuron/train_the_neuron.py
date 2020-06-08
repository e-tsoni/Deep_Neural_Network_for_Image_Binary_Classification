# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:23:52 2020

@author: Eftychia Tsoni
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from load_dataset import load_dataset
from model import model

index_train = 150

# Loading the original data (cat / non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print("Shape of original train set: ", train_set_x_orig.shape)
print("Shape of original labels in train set: ", train_set_y.shape)
print("Shape of original test set: ", test_set_x_orig.shape)
print("Shape of original labels in test set: ", test_set_y.shape, "\n")

# Example of a picture in the training set
plt.imshow(train_set_x_orig[index_train])

# Finding the numbers of training examples, test examples, height=width of a picture
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print("number of examples in train set: ", m_train)
print("number of examples in test set: ", m_test)
print("number of pixels, height=width: ", num_px, "\n")

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print("Shape of flatten train set: ", train_set_x_flatten.shape)
print("Shape of flatten test set: ", test_set_x_flatten.shape, "\n")

# Standardize the dataset
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# Train the model
train = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Plot learning curve (with costs)
costs = np.squeeze(train['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(train["learning_rate"]))
plt.show()










