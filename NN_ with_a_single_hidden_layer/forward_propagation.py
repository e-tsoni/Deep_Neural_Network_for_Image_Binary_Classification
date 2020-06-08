# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:00:10 2020

@author: Eftychia Tsoni
"""

import numpy as np


def forward_propagation(X, parameters):
    
    """
    Implement Forward Propagation. 
    Compute  Z1,A1,Z2,A[2] (the vector of all predictions on all the examples in the training set).
    Values needed in the backpropagation are stored in "cache". 
    The cache will be given as an input to the backpropagation function.

    Parameters
    ----------
    X : input data of size (n_x, m)
    parameters : python dictionary containing parameters (output of initialization function)

    Returns
    -------
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"

    """
    
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) +b2
    A2 = 1/(1 + np.exp(-Z2))
    
    cache = {"Z1" : Z1,
             "A1" : A1,
             "Z2" : Z2,
             "A2" : A2}
    
    return A2, cache
