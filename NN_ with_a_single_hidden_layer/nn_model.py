# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:32:27 2020

@author: Eftychia Tsoni
"""
from network_structure import network_structure
import  numpy as np
from initialize_parameters import initialize_parameters
from forward_propagation import forward_propagation
from compute_cost import compute_cost
from backward_propagation import backward_propagation
from update_parameters import update_parameters

def nn_model(X, Y, n_h, num_iterations = 10000, learning_rate = 0.01, print_cost = False):
    """
    

    Parameters
    ----------
    X : dataset of shape (2, number of examples)
    Y : labels of shape (1, number of examples)
    n_h : size of the hidden layer
    num_iterations : Number of iterations in gradient descent loop
    print_cost : if True, print the cost every 1000 iterations

    Returns
    -------
    parameters : parameters learnt by the model. They can then be used to predict.

    """
    
    np.random.seed(3)
    n_x = network_structure(X, Y, n_h)[0]
    n_h = network_structure(X, Y, n_h)[1]
    n_y = network_structure(X, Y, n_h)[2]
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y)
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate = 0.01)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    return parameters
    
    
