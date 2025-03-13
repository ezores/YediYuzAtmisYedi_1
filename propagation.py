import numpy as np
from mlp_math import activation_functions,hadamard_product

def forward_propagation(X, weights, biases, activations, training=True):
    """
    Propagation avant du réseau de neurones

    Args:
        X (np.ndarray): Entrée du réseau
        weights (List[np.ndarray]): Poids du réseau
        biases (List[np.ndarray]): Biais du réseau
        activations (List[str]): Fonctions d'activation pour chaque couche
        training (bool): Mode d'entraînement ou d'évaluation

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
            Activations, z et dropout_masks pour chaque couche
    """
    activations_cache = [X]
    z_cache = []

    for i, (W, b) in enumerate(zip(weights, biases)):
        z = W @ activations_cache[-1] + b

        z_cache.append(z)
        a = activation_functions[activations[i]][0](z)
        activations_cache.append(a)

    return activations_cache, z_cache, []

def backward_propagation(x, y, activations_cache, z_cache, weights, activations, eta):
    """
    Propagation arrière du réseau de neurones

    Args:
        x (np.ndarray): Entrée du réseau
        y (np.ndarray): Sortie attendue
        activations_cache (List[np.ndarray]): Cache des activations
        z_cache (List[np.ndarray]): Cache des z
        weights (List[np.ndarray]): Poids du réseau
        activations (List[str]): Fonctions d'activation pour chaque couche
        eta (float): Taux d'apprentissage

    Returns:
        List[np.ndarray]: Gradients pour chaque couche
    """
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
    

    gradients[0] = eta * np.dot(deltas[0], x.T)  # Shape: (hidden_size, input_size)
    for l in range(1, len(weights)):
        gradients[l] = eta * np.dot(deltas[l], activations_cache[l].T)  # Shape: (current_layer_size, prev_layer_size)

    return gradients



def update_weights(weights, gradients, momentum=0.9, velocity=None, clip_value=1.0):
    """
    Met à jour les poids du réseau avec descente de gradient et momentum

    Args:
        weights (List[np.ndarray]): Poids du réseau
        gradients (List[np.ndarray]): Gradients calculés
        momentum (float): Facteur de momentum
        velocity (List[np.ndarray]): Vitesse précédente
        clip_value (float): Valeur de clipping pour les gradients

    Returns:
        List[np.ndarray]: Nouveaux poids
    """
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
    return new_weights, velocity  # Remove transposition