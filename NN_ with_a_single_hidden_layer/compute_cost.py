# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:23:13 2020

@author: Eftychia Tsoni
"""

import numpy as np

def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost

    Parameters
    ----------
    A2 : The sigmoid output of the second activation, of shape (1, number of examples)
    Y :  "true" labels vector of shape (1, number of examples)

    Returns
    -------
    cost : cross-entropy cost

    """
    
    m = Y.shape[1] # number of examples
    
    logprobs = (np.multiply(Y, np.log(A2)) + np.multiply((np.ones(Y.shape) - Y), np.log(np.ones(A2.shape) - A2))) / m
    cost = - np.sum(logprobs)
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
    
    return cost
    
