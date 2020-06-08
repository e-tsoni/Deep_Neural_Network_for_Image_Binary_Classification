# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:26:41 2020

@author: Eftychia Tsoni
"""

def update_parameters(parameters, grads, learning_rate):
    
    """
    Updates parameters using the gradient descent update rule

    Parameters
    ----------
    parameters : python dictionary containing your parameters 
    grads : python dictionary containing your gradients 
    learning_rate : value of the learning rate

    Returns
    -------
    parameters : python dictionary containing updated parameters 

    """
    
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    # Update rule for each parameter
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
