# training.py
import random
import time
import numpy as np
from file_utils import print_header, save_hidden_units, split_data
from propagation import forward_propagation, backward_propagation, update_weights

def compute_loss(sample, label, weights, biases, activations):
    acts, _ = forward_propagation(sample, weights, biases, activations)
    diff = acts[-1] - label
    return np.sum(diff**2)

def compute_accuracy(samples, labels, weights, biases, activations):
    correct = 0
    total = len(samples)
    for sample, label in zip(samples, labels):
        acts, _ = forward_propagation(sample, weights, biases, activations)
        pred = np.argmax(acts[-1])
        expected = np.argmax(label)
        if pred == expected:
            correct += 1
    return correct / total if total > 0 else 0

def train_model(X_samples, Y_samples, weights, biases, activations, learning_rate,
                max_time=60, patience=5, cv_split=0.2):
    start_time = time.time()
    X_train, Y_train, X_val, Y_val = split_data(X_samples, Y_samples, cv_split)
    best_val_loss = float('inf')
    best_hidden = None
    patience_count = 0
    epoch = 0
    while time.time() - start_time < max_time and patience_count < patience:
        epoch_start = time.time()
        epoch += 1
        print_header(f"Epoch {epoch}")
        combined = list(zip(X_train, Y_train))
        random.shuffle(combined)
        for sample, label in combined:
            acts_cache, i_cache = forward_propagation(sample, weights, biases, activations)
            grads_W = backward_propagation(label, sample, weights, acts_cache, i_cache, activations, learning_rate)
            weights = update_weights(weights, biases, grads_W, learning_rate)
        # Compute validation loss and accuracies.
        val_losses = [compute_loss(s, l, weights, biases, activations) for s, l in zip(X_val, Y_val)]
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        
        train_accuracy = compute_accuracy(X_train, Y_train, weights, biases, activations)
        val_accuracy = compute_accuracy(X_val, Y_val, weights, biases, activations)
        
        elapsed = time.time() - epoch_start
        print(f"Epoche eta = {learning_rate} : ({elapsed:.3f} sec) Performance avec données de test : {train_accuracy} Performance avec données de VC : {val_accuracy}")
        
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
            print(f"No improvement. Patience count: {patience_count}")
    if best_hidden is not None:
        save_hidden_units(best_hidden, "hidden_units.txt")
    print(f"Training stopped after {epoch} epochs. Best validation loss: {best_val_loss:.6f}")
    return weights, biases

def get_test_input_case1():
    # A simple test case
    X = np.array([[0.5], [-0.5]])
    Y = np.array([[1]])
    layer_sizes = [2, 2, 1]
    activations = ['sigmoide', 'sigmoide']
    w1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    w2 = np.array([[0.5, -0.5]])
    b1 = np.array([[0.1], [0.1]])
    b2 = np.array([[0.0]])
    return X, Y, layer_sizes, activations, [w1, w2], [b1, b2]
