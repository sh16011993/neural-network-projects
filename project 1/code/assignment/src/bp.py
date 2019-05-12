#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    activations = [[],[]]
    tempx=x
    for supercount in range(len(biases)):
        for counts in range(len(biases[supercount])):
            activations[supercount].append(sigmoid(np.dot(weightsT[supercount][counts],tempx)+biases[supercount][counts]))
        tempx=activations[supercount]
    delta = (cost).df_wrt_a(activations[1], y)

    for i in range(len(biases)-1, -1, -1):
        activationsD=[]
        for j in range(len(activations[i])):
            activationsD.append(sigmoid_prime(activations[i][j]))
        delta = np.multiply(delta,activationsD)
        nabla_b[i] = delta
        if(i==0):
            nabla_wT[i] = np.dot(nabla_b[i], np.transpose(x))
        else:
            nabla_wT[i] = np.dot(nabla_b[i], np.transpose(activations[i-1]))
        delta = np.dot(np.transpose(weightsT[i]),delta)

    return (nabla_b, nabla_wT)
