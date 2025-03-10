# training.py
import random
import time
import numpy as np
from file_utils import print_header, print_matrix, save_hidden_units
from propagation import forward_propagation, backward_propagation, update_weights

def split_data(samples, labels, cv_split=0.2):
    combined = list(zip(samples, labels))
    print(f"Combined data length: {len(combined)}")  # Debug print
    random.shuffle(combined)
    idx = int(len(combined) * (1 - cv_split))
    print(f"Index for split: {idx}")  # Debug print
    train_data = combined[:idx]
    val_data = combined[idx:]
    if not train_data:
        print("Warning: No training data after split.")  # Debug print
    if not val_data:
        print("Warning: No validation data after split.")  # Debug print
    X_train, Y_train = zip(*train_data) if train_data else ([], [])
    X_val, Y_val = zip(*val_data) if val_data else ([], [])
    return list(X_train), list(Y_train), list(X_val), list(Y_val)

def compute_loss(sample, label, weights, biases, activations):
    acts, _ = forward_propagation(sample, weights, biases, activations)
    diff = acts[-1] - label
    return np.sum(diff**2)

def train_model(X_samples, Y_samples, weights, biases, activations, learning_rate,
                max_time=60, patience=5, cv_split=0.2):
    start_time = time.time()
    X_train, Y_train, X_val, Y_val = split_data(X_samples, Y_samples, cv_split)
    best_val_loss = float('inf')
    best_hidden = None
    patience_count = 0
    epoch = 0
    while time.time() - start_time < max_time and patience_count < patience:
        epoch += 1
        combined = list(zip(X_train, Y_train))
        random.shuffle(combined)
        for sample, label in combined:
            acts_cache, i_cache = forward_propagation(sample, weights, biases, activations)
            grads_W = backward_propagation(label, sample, weights, acts_cache, i_cache, activations, learning_rate)
            weights = update_weights(weights, biases, grads_W, learning_rate)
        val_losses = []
        for sample, label in zip(X_val, Y_val):
            val_losses.append(float(compute_loss(sample, label, weights, biases, activations)))
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_count = 0
            hidden_acts = []
            for samp in X_train:
                a_cache, _ = forward_propagation(samp, weights, biases, activations)
                if len(a_cache) > 1:
                    hidden_acts.append(a_cache[1])
            best_hidden = hidden_acts
        else:
            patience_count += 1
    if best_hidden is not None:
        save_hidden_units(best_hidden, "hidden_units.txt")
    print(f"Training stopped after {epoch} epochs. Best validation loss: {best_val_loss:.6f}")
    return weights, biases

def get_test_input_case1():
    # Simple test case:
    X = np.array([[0.5], [-0.5]])
    Y = np.array([[1]])
    layer_sizes = [2, 2, 1]
    activations = ['sigmoide', 'sigmoide']
    w1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    w2 = np.array([[0.5, -0.5]])
    b1 = np.array([[0.1], [0.1]])
    b2 = np.array([[0.0]])
    return X, Y, layer_sizes, activations, [w1, w2], [b1, b2]
