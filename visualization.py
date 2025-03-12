# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_training_history(history, filename="training_curves.png"):
    """
    Plot training/validation loss and accuracy curves.

    Args:
        history: TrainingProgress object from training.py
        filename: Output file path
    """
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.train_losses, label='Train')
    plt.plot(history.val_losses, label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.train_accs, label='Train')
    plt.plot(history.val_accs, label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(cm, class_names=None, filename="confusion_matrix.png"):
    """
    Plot confusion matrix without sklearn dependency.

    Args:
        cm: Numpy array confusion matrix from MLP.confusion_matrix()
        class_names: List of class labels
        filename: Output file path
    """
    plt.figure(figsize=(10, 8))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_weight_distribution(weights, filename="weight_distribution.png"):
    """
    Visualize weight distribution across layers.

    Args:
        weights: List of weight matrices from MLP.weights
        filename: Output file path
    """
    plt.figure(figsize=(10, 6))

    for i, w in enumerate(weights):
        plt.hist(w.flatten(), bins=100, alpha=0.5,
                 label=f'Layer {i + 1} ({w.shape[0]}x{w.shape[1]})')

    plt.title("Weight Distribution Across Layers")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_activation_distribution(activations_cache, filename="activation_distribution.png"):
    """
    Visualize activation distribution across layers.

    Args:
        activations_cache: From forward_propagation()
        filename: Output file path
    """
    plt.figure(figsize=(10, 6))

    for i, a in enumerate(activations_cache[1:-1]):  # Skip input and output
        plt.hist(a.flatten(), bins=100, alpha=0.5,
                 label=f'Layer {i + 1} ({a.shape[0]} units)')

    plt.title("Activation Distribution Across Hidden Layers")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def generate_network_visualization(mlp, filename="network_structure.html"):
    """
    Generate interactive network visualization (D3.js based).
    Simplified version for educational purposes.
    """
    # Basic text-based visualization
    with open(filename, 'w') as f:
        f.write("<html><body>")
        f.write("<h1>Network Architecture</h1>")
        f.write(f"<p>Input Size: {mlp.input_size}</p>")

        for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases)):
            f.write(f"<h3>Layer {i + 1}</h3>")
            f.write(f"<p>Weights: {w.shape[0]}x{w.shape[1]}</p>")
            f.write(f"<p>Biases: {b.shape[0]}x1</p>")

        f.write("</body></html>")