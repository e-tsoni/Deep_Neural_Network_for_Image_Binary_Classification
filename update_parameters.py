# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:07:54 2020

@author: Eftychia Tsoni
"""

import numpy as np
from propagate import propagate

def update_parameters(w, b, X, Y, num_iterations, learning_rate, print_cost = True):
    
    """
    This function optimizes w and b by running a gradient descent algorithm

    Parameters
    ----------
    w : weights, a numpy array of size (num_px * num_px * 3, 1)
    b : bias, a scalar
    X : data of shape (num_px * num_px * 3, number of examples)
    Y : true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations : number of iterations of the optimization loop
    learning_rate : learning rate of the gradient descent update rule
    print_cost : True to print the loss every 100 steps
        
    Returns
    -------
    params : dictionary containing the weights w and bias b
    grads : dictionary containing the gradients of the weights and bias with respect to the cost function
    costs : list of all the costs computed during the optimization, this will be used to plot the learning curve.

    """
    
    # Create a list to keep the cost on every iteration so that we can print it
    costs = []
    
    for i in range(num_iterations):
        # Calculate cost and gradients for every iteration
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads dictionary
        dw = grads["dw"]
        db = grads["db"]
        
        # Update the parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w" : w,
              "b" : b}
    
    grads = {"dw" : dw,
             "db" : db}
    
    return params, grads, costs