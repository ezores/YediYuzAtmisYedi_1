import time
import numpy as np
from propagation import forward_propagation, backward_propagation, update_weights
from mlp_math import activation_functions, cross_entropy_loss

class TrainingProgress:
    def __init__(self, max_epochs):
        """
        Initialisation de la classe TrainingProgress

        Args:
            max_epochs (int): Nombre maximal d'époques

        Returns:
            None
        """
        self.max_epochs = max_epochs
        self.epoch_times = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []

    def add_epoch(self, epoch_time, train_loss, val_loss, train_acc, val_acc, lr):
        """
        Ajoute une nouvelle époque aux métriques de progression

        Args:
            epoch_time (float): Durée de l'époque en secondes
            train_loss (float): Perte sur l'ensemble d'entraînement
            val_loss (float): Perte sur l'ensemble de validation
            train_acc (float): Précision sur l'ensemble d'entraînement
            val_acc (float): Précision sur l'ensemble de validation
            lr (float): Taux d'apprentissage

        Returns:
            None
        """
        self.epoch_times.append(epoch_time)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)

def compute_accuracy(Y_pred, Y_true):
    """
    Calcule la précision des prédictions

    Args:
        Y_pred (np.ndarray): Prédictions du modèle
        Y_true (np.ndarray): Labels de vérité terrain

    Returns:
        float: Précision des prédictions
    """
    return np.mean(np.argmax(Y_pred, axis=0) == np.argmax(Y_true, axis=0))

def add_noise(X, sigma):
    """
    Ajoute du bruit gaussien aux données d'entrée

    Args:
        X (np.ndarray): Données d'entrée
        sigma (float): Écart-type du bruit gaussien

    Returns:
        np.ndarray: Données bruitées
    """
    return X + np.random.normal(0, sigma, X.shape) if sigma > 0 else X

def train_model(
    X_train, Y_train, X_val, Y_val, weights, biases, activations,
    learning_rate=0.001, max_epochs=100, patience=5, noise_sigma=0.1,
    momentum=0.9, adaptive_eta=True, lambda_l2=0.001, verbose=True
):
    """
    Entraîne un modèle de réseau de neurones avec descente de gradient stochastique

    Args:
        X_train (np.ndarray): Données d'entraînement (features)
        Y_train (np.ndarray): Labels d'entraînement (one-hot encoded)
        X_val (np.ndarray): Données de validation (features)
        Y_val (np.ndarray): Labels de validation (one-hot encoded)
        weights (List[np.ndarray]): Poids du réseau
        biases (List[np.ndarray]): Biais du réseau
        activations (List[str]): Fonctions d'activation pour chaque couche
        learning_rate (float): Taux d'apprentissage initial
        max_epochs (int): Nombre maximal d'époques
        patience (int): Nombre d'époques sans amélioration avant arrêt
        noise_sigma (float): Écart-type du bruit gaussien à ajouter aux données
        momentum (float): Facteur de momentum pour la descente de gradient
        adaptive_eta (bool): Activer le taux d'apprentissage adaptatif
        lambda_l2 (float): Coefficient de régularisation L2
        verbose (bool): Afficher les métriques d'entraînement

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], TrainingProgress]: Poids, biais et progression de l'entraînement
    """
    progress = TrainingProgress(max_epochs)
    best_weights = [w.copy() for w in weights]
    best_biases = [b.copy() for b in biases]
    best_loss = float('inf')
    current_lr = learning_rate
    velocities = [np.zeros_like(w) for w in weights]

    # Formatage des données
    X_train, Y_train = X_train.T, Y_train.T
    # X_train, Y_train = X_train, Y_train  # No transpose
    X_val = X_val.T if X_val.size > 0 else X_val

    for epoch in range(max_epochs):
        epoch_start = time.time()
        X_noisy = add_noise(X_train, noise_sigma)

        a_cache, z_cache, bn_cache = forward_propagation(
            X_noisy,  # ✅ Keep data as (input_size, num_samples)
            weights,
            biases,
            activations,
            training=True
        )

        # Calcul de la perte avec L2
        train_ce = cross_entropy_loss(a_cache[-1], Y_train)
        l2_term = 0.5 * lambda_l2 * sum(np.sum(w**2) for w in weights)
        train_loss = train_ce + l2_term

        # Backward propagation
        grads = backward_propagation(
            # Y_train, a_cache, z_cache, weights, activations, current_lr
            X_noisy,  # Add X input (was missing)
            Y_train,
            a_cache,
            z_cache,
            weights,
            activations,
            current_lr  # This is the eta parameter
        )

        # Mise à jour des poids avec gradient clipping
        weights, velocities = update_weights(
            weights, grads, momentum, velocities, clip_value=1.0
        )

        # Validation
        val_loss, val_acc = float('inf'), 0.0
        if X_val.size > 0:
            val_a, _, _ = forward_propagation(X_val, weights, biases, activations, training=False)
            val_ce = cross_entropy_loss(val_a[-1], Y_val.T)
            val_loss = val_ce + 0.5 * lambda_l2 * sum(np.sum(w**2) for w in weights)
            val_acc = compute_accuracy(val_a[-1], Y_val.T)

        # Adaptation du taux d'apprentissage
        if adaptive_eta:
            if epoch < 10:
                current_lr = learning_rate * (epoch + 1) / 10  # Warmup
            elif val_loss < best_loss * 0.99:
                current_lr *= 1.05
                best_loss = val_loss
                best_weights = [w.copy() for w in weights]
                best_biases = [b.copy() for b in biases]
            else:
                current_lr *= 0.7

        # Stockage des métriques
        progress.add_epoch(
            time.time() - epoch_start,
            train_loss,
            val_loss,
            compute_accuracy(a_cache[-1], Y_train),
            val_acc,
            current_lr
        )

        # Affichage
        if verbose and epoch % 1 == 0:
            print(f"Epoch {epoch+1:03d}/{max_epochs} | LR: {current_lr:.5f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return best_weights, best_biases, progress