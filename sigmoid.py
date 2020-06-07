# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:37:20 2020

@author: Eftychia Tsoni
"""
import numpy as np

def sigmoid(z):
    
    """
    Compute the sigmoid activation of linear z    

    Parameters
    ----------
    z : a scalar or numpy array of any size

    Returns
    -------
    s : the sigmoid function of z

    """
    
    s = 1 / (1 + np.exp(-z))
    
    return s
