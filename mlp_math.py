# mlp_math.py
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tan(x):
    return np.tan(x)

def tan_derivative(x):
    return 1.0 / np.cos(x)**2

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    shiftx = x - np.max(x)
    ex = np.exp(shiftx)
    return ex / np.sum(ex)

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

def custom_activation(x):
    return x

def custom_activation_derivative(x):
    return np.ones_like(x)

def sinus(x):
    return np.sin(x)

def sinus_derivative(x):
    return np.cos(x)

activation_functions = {
    'sigmoide': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'tan': (tan, tan_derivative),
    'softmax': (softmax, softmax_derivative),
    'personnalisée': (custom_activation, custom_activation_derivative),
    'sinus': (sinus, sinus_derivative)
}

def hadamard_product(A, B):
    return A * B
