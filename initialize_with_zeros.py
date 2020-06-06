# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:46:59 2020

@author: Eftychia Tsoni
"""

import numpy as np

def initialize_with_zeros(dim):
    
    """
    Creates a vector of zeros of shape (dim, 1) for w
    and initialize b to 0.
    
    Parameters
    ----------
    dims : size of w vector

    Returns
    -------
    s : initialized vector of shape (dim, 1)
    b : initialized scalar (corresponds to the bias)
    
    """
    
    w = np.zeros((dim, 1))
    b = 0
    
    return w, b
    
    
    