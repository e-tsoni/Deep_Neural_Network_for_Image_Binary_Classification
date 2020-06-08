# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:47:06 2020

@author: Eftychia Tsoni
"""

import numpy as np


def initialize_parameters(n_x, n_h, n_y):
    
    """
    initialize the weights matrices with random values.
    initialize the bias vectors as zeros.    

    Parameters
    ----------
    n_x : size of the input layer
    n_h : size of the hidden layer
    n_y : size of the output layer

    Returns
    -------
    params : python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)

    """
    
    np.random.seed(2)
    W1 = np.random.rand(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h)
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}
    
    return parameters
    
    
    