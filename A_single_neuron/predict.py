# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:35:21 2020

@author: Eftychia Tsoni
"""

import numpy as np
from sigmoid import sigmoid

def predict(w, b, X):
    
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Parameters
    ----------
    w : weights, a numpy array of size (num_px * num_px * 3, 1)
    b : bias, a scalar
    X : data of size (num_px * num_px * 3, number of examples)

    Returns
    -------
    Y_prediction : a numpy array (vector) containing all predictions (0/1) for the examples in X

    """
    
    # number of training examples
    m = X.shape[1]
    
    # initialize
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction[0,i] = np.where(A[0,i]>0.5,1,0)
    
    return Y_prediction
