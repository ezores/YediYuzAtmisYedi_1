import numpy as np
from mlp_math import activation_functions, hadamard_product


def forward_propagation(X, weights, biases, activations, bn_params=None,
                        training=True, dropout_rate=0.0):
    """
    Enhanced forward propagation with batch normalization and dropout
    Returns:
    - activations_cache: List of activation matrices
    - z_cache: List of pre-activation matrices
    - cache: Tuple of (dropout_masks, bn_cache)
    """
    activations_cache = [X]
    z_cache = []
    dropout_masks = []
    bn_cache = []

    for i, (W, b) in enumerate(zip(weights, biases)):
        # Linear transformation
        z = W @ activations_cache[-1] + b

        # Batch normalization
        if bn_params and bn_params[i] and training:
            gamma = bn_params[i]['gamma']
            beta = bn_params[i]['beta']

            # Batch statistics
            mu = np.mean(z, axis=1, keepdims=True)
            sigma2 = np.var(z, axis=1, keepdims=True)

            # Normalize
            z_norm = (z - mu) / np.sqrt(sigma2 + 1e-8)
            z = gamma * z_norm + beta

            # Update running statistics
            if training:
                bn_params[i]['running_mean'] = 0.9 * bn_params[i]['running_mean'] + 0.1 * mu
                bn_params[i]['running_var'] = 0.9 * bn_params[i]['running_var'] + 0.1 * sigma2

            # bn_cache.append((z_norm, mu, sigma2, gamma))

        else:
            bn_cache.append(None)

        z_cache.append(z)

        # Activation function
        a = activation_functions[activations[i]][0](z)

        # Dropout
        if training and dropout_rate > 0:
            mask = (np.random.rand(*a.shape) < (1 - dropout_rate)) / (1 - dropout_rate)
            a *= mask
            dropout_masks.append(mask)
        else:
            dropout_masks.append(None)

        activations_cache.append(a)

    return activations_cache, z_cache, (dropout_masks, bn_cache)


def backward_propagation(X, Y, activations_cache, z_cache, weights, activations, eta, cache):
    deltas = [None] * len(weights)
    gradients = [None] * len(weights)
    L = len(weights) - 1
    dropout_masks, bn_cache = cache

    # Output layer delta
    delta_L = Y - activations_cache[-1]

    if activations[L] != 'sigmoid':
        deriv = activation_functions[activations[L]][1](z_cache[-1])
        delta_L = hadamard_product(delta_L, deriv)

    deltas[L] = delta_L

    # Hidden layers
    for l in range(L - 1, -1, -1):
        delta_next = deltas[l + 1]

        # Compute delta for current layer
        deriv = activation_functions[activations[l]][1](z_cache[l])
        delta = hadamard_product(weights[l + 1].T @ delta_next, deriv)

        # Apply dropout mask
        if dropout_masks[l] is not None:
            delta = hadamard_product(delta, dropout_masks[l])

        # Batch norm backprop
        if bn_cache and bn_cache[l]:
            z_norm, mu, sigma2, gamma = bn_cache[l]
            m = z_norm.shape[1]

            # Gradient through batch norm
            dz_norm = delta * gamma
            dsigma2 = np.sum(dz_norm * (z_cache[l] - mu) * (-0.5) * (sigma2 + 1e-8) ** (-1.5), axis=1, keepdims=True)
            dmu = np.sum(dz_norm * (-1 / np.sqrt(sigma2 + 1e-8)), axis=1, keepdims=True) + \
                  dsigma2 * np.mean(-2 * (z_cache[l] - mu), axis=1, keepdims=True)

            delta = (dz_norm / np.sqrt(sigma2 + 1e-8)) + \
                    dsigma2 * 2 * (z_cache[l] - mu) / m + \
                    dmu / m

            # Store scaled gamma/beta gradients (FIX 1: Add eta scaling)
            dgamma = np.sum(delta * z_norm, axis=1, keepdims=True) * eta
            dbeta = np.sum(delta, axis=1, keepdims=True) * eta
            bn_cache[l] = (dgamma, dbeta)

        deltas[l] = delta

    # Compute weight gradients (without eta scaling)
    gradients[0] = (deltas[0] @ X.T)
    for l in range(1, len(weights)):
        gradients[l] = (deltas[l] @ activations_cache[l].T)

    return gradients, bn_cache


def update_weights(weights, gradients, bn_params, bn_grads, momentum=0.9, velocity=None, clip_value=1.0):
    """Updated with correct batch norm handling"""
    # Clip weight gradients
    gradients = [np.clip(g, -clip_value, clip_value) for g in gradients]

    # Update batch norm parameters (FIX 2: Use bn_grads instead of weight gradients)
    if bn_params and bn_grads:
        for l in range(len(bn_params)):
            if bn_params[l] and bn_grads[l]:
                dgamma, dbeta = bn_grads[l]
                bn_params[l]['gamma'] += dgamma
                bn_params[l]['beta'] += dbeta

    # Momentum update for weights
    if velocity is None:
        velocity = [np.zeros_like(w) for w in weights]

    for i in range(len(velocity)):
        velocity[i] = momentum * velocity[i] + gradients[i]

    new_weights = [w + v for w, v in zip(weights, velocity)]
    return new_weights, velocity