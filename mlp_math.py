import numpy as np

# Enhanced activation functions with numerical stability
def sigmoid(z, derive=False):
    z = np.clip(z, -500, 500)  # Prevent overflow
    s = 1 / (1 + np.exp(-z))
    return z* (1 - z) if derive else s


def relu(z, derive=False):
    return np.where(z > 0, 1, 0) if derive else np.maximum(0, z)

def leaky_relu(z, alpha=0.01, derive=False):
    if derive:
        return np.where(z > 0, 1, alpha)
    return np.where(z > 0, z, alpha * z)

def tanh(z, derive=False):
    t = np.tanh(z)
    return 1 - t**2 if derive else t

# def softmax(z, derive=False):
#     e_z = np.exp(z - np.max(z))  # Numerical stability
#     sm = e_z / e_z.sum(axis=0)
#     if derive:
#         return sm * (1 - sm)  # Simplified derivative for cross-entropy
#     return sm
def softmax(z, derive=False):
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    sm = e_z / np.sum(e_z, axis=0, keepdims=True)
    return sm * (1 - sm) if derive else sm  # Dérivée corrigée

def cross_entropy_loss(Y_pred, Y_true, epsilon=1e-12):
    Y_pred = np.clip(Y_pred, epsilon, 1. - epsilon)
    m = Y_true.shape[1]
    return -np.sum(Y_true * np.log(Y_pred)) / m

activation_functions = {
    'sigmoid': (sigmoid, sigmoid),
    'relu': (relu, relu),
    'leakyrelu': (leaky_relu, leaky_relu),
    'tanh': (tanh, tanh),
    'softmax': (softmax, softmax)
}

def hadamard_product(a, b):
    """Element-wise product with broadcasting support"""
    return a * b

def l2_regularization(weights, lambda_=0.01):
    """Calculate L2 regularization term"""
    return 0.5 * lambda_ * sum(np.sum(w**2) for w in weights)