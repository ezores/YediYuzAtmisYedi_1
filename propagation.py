# propagation.py
import numpy as np
from mlp_math import activation_functions, hadamard_product
from file_utils import print_header, print_matrix

def forward_propagation(X, weights, biases, activations):
    # Ensure X is a column vector (shape: (n, 1))
    X = np.atleast_2d(X)
    if X.shape[0] == 1 and X.shape[1] != 1:
        X = X.T
    a = X
    activations_cache = [a]
    i_cache = []
    for i in range(len(weights)):
        w = weights[i]
        b = biases[i]
        act = activations[i]
        i_val = np.dot(w, a) + b
        i_cache.append(i_val)
        a = activation_functions[act][0](i_val)
        activations_cache.append(a)
        print_header(f"Layer {i+1} pre-activation")
        print_matrix(f"i (Layer {i+1})", i_val)
        print_header(f"Layer {i+1} activation")
        print_matrix(f"a (Layer {i+1})", a)
    return activations_cache, i_cache

def backward_propagation(Y, X, weights, activations_cache, i_cache, activations, learning_rate):
    X = np.atleast_2d(X)
    if X.shape[0] == 1 and X.shape[1] != 1:
        X = X.T
    Y = np.atleast_2d(Y)
    if Y.shape[0] == 1 and Y.shape[1] != 1:
        Y = Y.T
    print_header("Backward Propagation: Error Signal")
    deltas = [None] * len(weights)
    gradients_W = [None] * len(weights)
    L = len(weights) - 1
    delta_L = Y - activations_cache[-1]
    if activations[L] not in ['softmax', 'sigmoide']:
        deriv_out = activation_functions[activations[L]][1](i_cache[-1])
        delta_L = hadamard_product(delta_L, deriv_out)
    elif activations[L] == 'sigmoide':
        deriv_sig = activation_functions['sigmoide'][1](activations_cache[-1])
        delta_L = hadamard_product(delta_L, deriv_sig)
    deltas[L] = delta_L
    print_matrix(f"Delta Layer {L+1}", delta_L)
    for l in range(L-1, -1, -1):
        temp = np.dot(weights[l+1].T, deltas[l+1])
        deriv = activation_functions[activations[l]][1](i_cache[l])
        deltas[l] = hadamard_product(temp, deriv)
        print_matrix(f"Delta Layer {l+1}", deltas[l])
    print_header("Backward Propagation: Weight Gradients")
    gradients_W[0] = np.dot(deltas[0], X.T) * learning_rate
    print_matrix("Gradient W Layer 1", gradients_W[0])
    for l in range(1, len(weights)):
        gradients_W[l] = np.dot(deltas[l], activations_cache[l].T) * learning_rate
        print_matrix(f"Gradient W Layer {l+1}", gradients_W[l])
    return gradients_W

def update_weights(weights, biases, gradients_W, learning_rate):
    print_header("Update Weights")
    for l in range(len(weights)):
        weights[l] = weights[l] + gradients_W[l]
        print_matrix(f"Updated Weights Layer {l+1}", weights[l])
    return weights
