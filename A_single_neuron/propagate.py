# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:56:03 2020

@author: Eftychia Tsoni
"""

import numpy as np
from sigmoid import sigmoid

def propagate(w, b, X, Y):
    
    """
    Implement the cost function and its gradient for the propagation explained above
    
    Parameters
    ----------
    w : weights, a numpy array of size (num_px * num_px * 3, 1)
        
    b : bias, a scalar
        
    X : data of size (num_px * num_px * 3, number of examples)
        
    Y : true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
        
    Returns
    -------
    cost : negative log-likelihood cost for logistic regression
    dw : gradient of the loss with respect to w, thus same shape as w
    db : gradient of the loss with respect to b, thus same shape as b

    """
    
    # number of training examples
    m = X.shape[1]
    
    
    # FORWARD PROPAGATION (FROM X TO COST)
    
    # Calculate the sigmoid for all the training axamples
    # We have one neuron and one layer, so w is a vector
    A = sigmoid(np.dot(w.T, X) + b)
    # Calculate the cost J for all the training examples
    cost = -(1/m)*np.sum((Y*np.log(A)) + (1-Y)*np.log(1-A))
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dz = A - Y
    dw = (1/m)*np.dot(X, dz.T)
    db = (1/m)*np.sum(dz)
    
    grads = {"dw": dw, 
             "db": db}
    
    return grads, cost
    
