# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:35:49 2020

@author: Eftychia Tsoni
"""
import numpy as np

def backward_propagation(parameters, cache, X, Y):
    
    """
    Implement the backward propagation
    
    Parameters
    ----------
    parameters : python dictionary containing parameters "W1", "W2", "b1", "b2"
    cache : a dictionary containing "Z1", "A1", "Z2" and "A2".
    X : input data of shape (2, number of examples)
    Y : "true" labels vector of shape (1, number of examples)
    
    Returns
    -------
    None.

    """
    
    m = X.shape[1] # number of examples
    # retrieve W1 and W2 from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    # retrieve A1 and A2 from dictionary "cache"
    A1 = cache['A1']
    A2 = cache['A2']
    
    # Backward propagation: calculate dW1, db1, dW2, db2
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

