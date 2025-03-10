import numpy as np
from mlp_math import activation_functions, hadamard_product

def forward_propagation(X, weights, biases, activations):
    """
    Enhanced forward propagation with layer-wise activation tracking
    Returns:
    - activations_cache: List of activation matrices for each layer
    - z_cache: List of pre-activation values for each layer
    """
    activations_cache = [X]
    z_cache = []
    
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = np.dot(W, activations_cache[-1]) + b
        z_cache.append(z)
        
        activation_func = activation_functions[activations[i]][0]
        a = activation_func(z)
        activations_cache.append(a)
        
    return activations_cache, z_cache

def backward_propagation(y, x, weights, activations_cache, z_cache, activations, eta):
    """
    Enhanced backward propagation with regularization support
    Returns:
    - gradients: List of weight gradients for each layer
    - delta: Output error signal
    """
    gradients = []
    L = len(weights) - 1  # Last layer index
    
    # Output layer error
    error = activations_cache[-1] - y
    delta = error * activation_functions[activations[-1]][1](z_cache[-1])
    
    # Reverse through hidden layers
    for l in range(L, 0, -1):
        gradients.insert(0, np.dot(delta, activations_cache[l].T) * eta)
        
        # Calculate previous layer's delta
        delta = np.dot(weights[l].T, delta) * activation_functions[activations[l-1]][1](z_cache[l-1])
    
    gradients.insert(0, np.dot(delta, activations_cache[0].T) * eta)
    return gradients

def update_weights(weights, gradients, momentum=0.9, velocity=None):
    """Enhanced weight update with momentum support"""
    if velocity is None:
        velocity = [np.zeros_like(w) for w in weights]
    
    updated_weights = []
    for w, g, v in zip(weights, gradients, velocity):
        v = momentum * v + g
        updated_weights.append(w + v)
    
    return updated_weights, velocity