# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:45:37 2020

@author: Eftychia Tsoni
"""

from forward_propagation import forward_propagation
import numpy as np

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Parameters
    ----------
    parameters : python dictionary containing your parameters 
    X : input data of size (n_x, m)
    
    Returns
    -------
    predictions : vector of predictions of our model (red: 0 / blue: 1)

    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    
    return predictions
    
    