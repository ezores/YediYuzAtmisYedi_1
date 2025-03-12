import numpy as np
from propagation import forward_propagation,backward_propagation,update_weights
from mlp_math import activation_functions


# Simple test case
X = np.array([[2], [4]])  # Input
weights = [np.array([[1, 2], [3, 4]]), np.array([[0.3, 1.3],[-0.2, -0.4],[0.2,-1.2]])]
biases = [np.array([[2], [1]]), np.array([[-0.1],[0.1],[-0.2]])]
activations = ['sigmoid', 'sigmoid']
y = np.array([[1],[2],[1]])  # Expected output
eta = 0.1

# Forward propagation
activations_cache, z_cache = forward_propagation(X, weights, biases, activations)
print("Activations:", activations_cache)
print("Z values:", z_cache)

# Backward propagation
gradient = backward_propagation(X,y, activations_cache, z_cache, weights, activations, eta)
print("Gradients:", gradient)

# Update weights
#new_weights, velocity = update_weights(weights, gradient)
#print("Updated Weights:", new_weights)

