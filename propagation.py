import numpy as np
from mlp_math import activation_functions

def forward_propagation(X, weights, biases, activations, training=True):
    activations_cache = [X]
    z_cache = []
    bn_cache = []

    for i, (W, b) in enumerate(zip(weights, biases)):
        z = W @ activations_cache[-1] + b

        # Batch Normalization
        if i < len(weights) - 1 and training:
            mean = np.mean(z, axis=1, keepdims=True)
            var = np.var(z, axis=1, keepdims=True)
            z_hat = (z - mean) / np.sqrt(var + 1e-5)
            gamma, beta = 1.0, 0.0  # À remplacer par des paramètres appris
            z = gamma * z_hat + beta
            bn_cache.append((mean, var, gamma, beta))

        z_cache.append(z)
        a = activation_functions[activations[i]][0](z)
        activations_cache.append(a)

    return activations_cache, z_cache, bn_cache

def backward_propagation(y, activations_cache, z_cache, weights, activations, eta):
    gradients = []
    delta = activations_cache[-1] - y  # Softmax + entropie croisée

    for l in reversed(range(len(weights))):
        grad = (delta @ activations_cache[l].T) * eta
        gradients.insert(0, grad)

        if l > 0:
            delta = (weights[l].T @ delta) * activation_functions[activations[l-1]][1](z_cache[l-1])

    return gradients

def update_weights(weights, gradients, momentum=0.9, velocity=None, clip_value=1.0):  # ← Seuil réduit
    gradients = [np.clip(g, -clip_value, clip_value) for g in gradients]
    velocity = [momentum * v + g for v, g in zip(velocity, gradients)] if velocity else gradients
    return [w + v for w, v in zip(weights, velocity)], velocity