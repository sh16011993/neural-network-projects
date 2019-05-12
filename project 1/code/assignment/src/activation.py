#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np
from math import exp

def sigmoid(z):
    """The sigmoid function."""
    return 1/(1+np.exp(-1*z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
