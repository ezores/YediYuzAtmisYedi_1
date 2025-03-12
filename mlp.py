# mlp.py
import numpy as np
from propagation import forward_propagation
from training import train_model
from file_utils import save_hidden_units
from mlp_math import activation_functions

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation='sigmoid', learning_rate=0.1, max_epochs=100,
                 patience=5, adaptive_eta=False, noise_sigma=0.0, momentum=0.0,
                 use_batch_norm=False, dropout_rate=0.0):
        """
        Enhanced MLP with batch normalization and dropout support.

        Parameters:
        use_batch_norm (bool): Enable batch normalization
        dropout_rate (float): Dropout probability (0.0-1.0)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.adaptive_eta = adaptive_eta
        self.noise_sigma = noise_sigma
        self.momentum = momentum
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Initialize parameters with batch norm weights
        self.weights, self.biases, self.bn_params = self._initialize_parameters()
        self.best_weights = None
        self.best_biases = None
        self.training_history = None

    def _initialize_parameters(self):
        """Initialize weights, biases, and batch norm parameters"""
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        weights = []
        biases = []
        bn_params = []

        for i in range(len(layer_sizes) - 1):
            # Weight initialization
            if self.activation.lower() in ['relu', 'leakyrelu']:
                std = np.sqrt(2.0 / layer_sizes[i])
            else:
                std = np.sqrt(1.0 / layer_sizes[i])

            weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * std)
            biases.append(np.zeros((layer_sizes[i + 1], 1)))

            # Batch norm parameters
            if self.use_batch_norm and i < len(layer_sizes) - 2:
                bn_params.append({
                    'gamma': np.ones((layer_sizes[i + 1], 1)),
                    'beta': np.zeros((layer_sizes[i + 1], 1)),
                    'running_mean': np.zeros((layer_sizes[i + 1], 1)),
                    'running_var': np.ones((layer_sizes[i + 1], 1))
                })
            else:
                bn_params.append(None)

        return weights, biases, bn_params

    def train(self, X_train, Y_train, X_val=None, Y_val=None):
        """Enhanced training with batch norm and dropout support"""
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        activations = [self.activation] * len(self.hidden_sizes) + ['sigmoid']

        # Train with modified forward/backward passes
        final_weights, final_biases, history = train_model(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            weights=self.weights,
            biases=self.biases,
            bn_params=self.bn_params,
            activations=activations,
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            patience=self.patience,
            noise_sigma=self.noise_sigma,
            momentum=self.momentum,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            adaptive_eta=self.adaptive_eta
        )

        self.weights = final_weights
        self.biases = final_biases
        self.training_history = history
        return history

    def predict(self, X):
        """Prediction with disabled dropout and batch norm in eval mode"""
        X = np.array(X).T
        activations, _, _ = forward_propagation(
            X,
            self.weights,
            self.biases,
            [self.activation] * len(self.hidden_sizes) + ['sigmoid'],
            training=False,  # Disable dropout/batch norm during inference
            bn_params=self.bn_params,
            dropout_rate=0.0
        )
        return activations[-1].T

    def evaluate(self, X, Y):
        """Enhanced evaluation with confusion matrix support"""
        predictions = self.predict(X)
        y_true = np.argmax(Y, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        return np.mean(y_pred == y_true)

    def confusion_matrix(self, X, Y):
        """Generate confusion matrix without external dependencies"""
        y_pred = np.argmax(self.predict(X), axis=1)
        y_true = np.argmax(Y, axis=1)

        # Get all possible classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        # Initialize matrix
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        # Populate matrix
        for t, p in zip(y_true, y_pred):
            matrix[t, p] += 1

    def save_hidden_units(self, X):
        """Save hidden activations with dropout disabled"""
        X = np.array(X).T
        activations, _, _ = forward_propagation(
            X,
            self.weights,
            self.biases,
            [self.activation] * len(self.hidden_sizes) + ['sigmoid'],
            training=False  # Disable dropout for feature extraction
        )

        hidden_activations = activations[1:-1]
        formatted_activations = [layer_acts.T for layer_acts in hidden_activations]
        save_hidden_units(formatted_activations)

    def get_architecture(self):
        """Return enhanced architecture description"""
        return {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_sizes,
            'output_size': self.output_size,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'adaptive_learning': self.adaptive_eta,
            'noise_level': self.noise_sigma,
            'momentum': self.momentum,
            'batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate
        }

    def reset_parameters(self):
        """Re-initialize all parameters including batch norm"""
        self.weights, self.biases, self.bn_params = self._initialize_parameters()
        self.training_history = None