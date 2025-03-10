import time
import numpy as np
from propagation import forward_propagation, backward_propagation, update_weights
from mlp_math import activation_functions

class TrainingProgress:
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self.epoch_times = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []

    def add_epoch(self, epoch_time, train_loss, val_loss, train_acc, val_acc, lr):
        self.epoch_times.append(epoch_time)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)

    def get_last_metrics(self):
        return {
            'time': self.epoch_times[-1] if self.epoch_times else 0,
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'val_loss': self.val_losses[-1] if self.val_losses else 0,
            'train_acc': self.train_accs[-1] if self.train_accs else 0,
            'val_acc': self.val_accs[-1] if self.val_accs else 0,
            'lr': self.learning_rates[-1] if self.learning_rates else 0
        }

def compute_loss(Y_pred, Y_true):
    m = Y_true.shape[1]
    return np.sum((Y_pred - Y_true)**2) / (2*m)

def compute_accuracy(Y_pred, Y_true):
    return np.mean(np.argmax(Y_pred, axis=0) == np.argmax(Y_true, axis=0))

def add_noise(X, sigma):
    if sigma > 0:
        noise = np.random.normal(0, sigma, X.shape)
        return X + noise
    return X

def train_model(X_train, Y_train, X_val, Y_val, weights, biases, activations,
                learning_rate=0.1, max_epochs=100, patience=5, noise_sigma=0.0,
                momentum=0.0, adaptive_eta=True, verbose=True):
    """
    Enhanced training loop with adaptive learning rate, noise injection and momentum
    """
    # Initialize training state
    progress = TrainingProgress(max_epochs)
    best_weights = [w.copy() for w in weights]
    best_biases = [b.copy() for b in biases]
    best_loss = float('inf')
    patience_count = 0
    current_lr = learning_rate
    
    # Initialize momentum buffers
    velocities = [np.zeros_like(w) for w in weights]
    
    # Convert data to numpy arrays
    X_train = X_train.T
    Y_train = Y_train.T
    X_val = X_val.T if X_val.size > 0 else X_val

    for epoch in range(max_epochs):
        epoch_start = time.time()
        
        # Add noise to training data
        X_noisy = add_noise(X_train, noise_sigma)

        # Forward propagation
        a_cache, i_cache = forward_propagation(X_noisy, weights, biases, activations)
        
        # Backward propagation
        grads_W = backward_propagation(Y_train, X_noisy, weights, a_cache, i_cache, 
                                      activations, current_lr)
        
        # Update weights with momentum
        for i in range(len(weights)):
            velocities[i] = momentum * velocities[i] + grads_W[i]
            weights[i] += velocities[i]

        # Calculate training metrics
        train_loss = compute_loss(a_cache[-1], Y_train)
        train_acc = compute_accuracy(a_cache[-1], Y_train)
        
        # Validation metrics
        val_loss = float('inf')
        val_acc = 0.0
        if X_val.size > 0:
            val_a, _ = forward_propagation(X_val, weights, biases, activations)
            val_loss = compute_loss(val_a[-1], Y_val.T)
            val_acc = compute_accuracy(val_a[-1], Y_val.T)

        # Adaptive learning rate
        if adaptive_eta and X_val.size > 0:
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = [w.copy() for w in weights]
                best_biases = [b.copy() for b in biases]
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 2:  # Reduce LR after 2 consecutive non-improvements
                    current_lr *= 0.5
                    patience_count = 0
                    # Revert to best weights
                    weights = [w.copy() for w in best_weights]
                    biases = [b.copy() for b in best_biases]
                    print(f"Reducing learning rate to {current_lr:.4f}")

        # Early stopping
        if patience_count >= patience:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

        # Store progress
        progress.add_epoch(
            epoch_time=time.time() - epoch_start,
            train_loss=train_loss,
            val_loss=val_loss if X_val.size > 0 else 0,
            train_acc=train_acc,
            val_acc=val_acc if X_val.size > 0 else 0,
            lr=current_lr
        )

        # Print progress
        if verbose:
            last_metrics = progress.get_last_metrics()
            print(f"Epoche {epoch+1:03d}/{max_epochs} - "
                  f"eta = {current_lr:.4f} ({last_metrics['time']:.3f}sec) "
                  f"Train Loss: {last_metrics['train_loss']:.4f} Acc: {last_metrics['train_acc']:.2%} "
                  f"Val Loss: {last_metrics['val_loss']:.4f} Acc: {last_metrics['val_acc']:.2%}")

    # Restore best weights if using adaptive learning
    if adaptive_eta and X_val.size > 0:
        weights = best_weights
        biases = best_biases

    return weights, biases, progress