import numpy as np
from mlp_math import activation_functions,hadamard_product

def forward_propagation(X, weights, biases, activations, training=True):
    activations_cache = [X]
    z_cache = []

    for i, (W, b) in enumerate(zip(weights, biases)):
        z = W @ activations_cache[-1] + b

        z_cache.append(z)
        a = activation_functions[activations[i]][0](z)
        activations_cache.append(a)

    return activations_cache, z_cache

def backward_propagation(x,y, activations_cache, z_cache, weights, activations, eta):
    deltas = [None] * len(weights)
    gradients = [None] * len(weights)
    L = len(weights)-1
    
    delta_L = y-activations_cache[-1]  

    if activations[L] !=  'sigmoid':
        delta_L = hadamard_product(delta_L, activation_functions[activations[L]][1](z_cache[-1]))
        deltas[L] = delta_L
         
            
        for l in range(L - 1, -1, -1):
            temp = weights[l + 1].T @ deltas[l + 1]
            deriv = activation_functions[activations[l]][1](z_cache[l])
            deltas[l] = hadamard_product(temp, deriv)


    elif activations[L] == 'sigmoid':
        der_sig = activation_functions[activations[-1]][1](activations_cache[-1])
        delta_L = hadamard_product(delta_L, der_sig)
        deltas[L] = delta_L

        for l in range(L - 1, -1, -1):
            temp = weights[l + 1].T @ deltas[l + 1]
            deriv = activation_functions[activations[l]][1](activations_cache[l+1])
            deltas[l] = hadamard_product(temp, deriv)
    

  
    gradients[0]= eta*x*deltas[0].T
  

    for l in range(1,len(weights)):
        gradients[l] = eta*activations_cache[l]*deltas[l].T
    

    return gradients



def update_weights(weights, gradients, momentum=0.9, velocity=None, clip_value=1.0):
    import numpy as np  # Ensure NumPy is imported

    # Étape 1: Clip les gradients pour éviter les explosions
    gradients = [np.clip(g, -clip_value, clip_value) for g in gradients]
    
    # Étape 2: Transposer les gradients si nécessaire pour correspondre aux poids

    adjusted_gradients = []
    for w, g in zip(weights, gradients):
        if g.shape != w.shape:
            adjusted_grad = g.T  # Transpose si les shapes ne matchent pas
        else:
            adjusted_grad = g
        adjusted_gradients.append(adjusted_grad)
    
    # Étape 3: Initialiser velocity si elle est None
    if velocity is None:
        velocity = [np.zeros_like(w) for w in weights]
    
    # Étape 4: Mettre à jour velocity avec momentum
    for i in range(len(velocity)):
        velocity[i] = momentum * velocity[i] + adjusted_gradients[i]
    
    # Étape 5: Mettre à jour les poids (sans transposition)
    new_weights = [w + v for w, v in zip(weights, velocity)]
    
    # **Transposer chaque poids avant de retourner**
    transposed_weights = [w.T for w in new_weights]

    return transposed_weights, velocity  # Retourne les poids transposés
