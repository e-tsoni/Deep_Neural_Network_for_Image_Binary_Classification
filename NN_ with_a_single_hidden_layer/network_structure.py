# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:38:50 2020

@author: Eftychia Tsoni
"""


def network_structure(X, Y, neurons_h):
    
    """
    
    Parameters
    ----------
    X : the dataset
    Y : the labels

    Returns
    -------
    n_x : the size of the input layer
    n_h : the size of the hidden layer
    n_y : the size of the output layer

    """
    
    n_x = X.shape[0]
    n_h = neurons_h
    n_y = Y.shape[0]
    
    return (n_x, n_h, n_y)
    
    
    
    
    