import numpy as np
from propagation import forward_propagation
from training import train_model
from file_utils import save_hidden_units
from mlp_math import activation_functions

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, 
                 activation='sigmoid', learning_rate=0.1, max_epochs=100,
                 patience=5, adaptive_eta=False, noise_sigma=0.0, momentum=0.0):
        """
        Multilayer Perceptron with enhanced training capabilities
        
        Parameters:
        input_size (int): Number of input features
        hidden_sizes (list): List of integers specifying hidden layer sizes
        output_size (int): Number of output neurons
        activation (str): Activation function for hidden layers
        learning_rate (float): Initial learning rate
        max_epochs (int): Maximum number of training epochs
        patience (int): Early stopping patience
        adaptive_eta (bool): Enable adaptive learning rate
        noise_sigma (float): Standard deviation for input noise
        momentum (float): Momentum factor (0-1)
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
        
        # Initialize network parameters
        self.weights, self.biases = self._initialize_parameters()
        self.best_weights = None
        self.best_biases = None
        self.training_history = None

    def _initialize_parameters(self):
        """Initialize weights with He initialization and biases to zero"""
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        weights = []
        biases = []
        
        for i in range(len(layer_sizes)-1):
            # He initialization for ReLU variants, sqrt(2/n) for others
            if self.activation in ['relu', 'leakyrelu']:
                std = np.sqrt(2.0 / layer_sizes[i])
            else:
                std = np.sqrt(1.0 / layer_sizes[i])
                
            weights.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]) * std)
            biases.append(np.zeros((layer_sizes[i+1], 1)))
            
        return weights, biases

    def train(self, X_train, Y_train, X_val=None, Y_val=None):
        """
        Train the network with optional validation set
        X_train: Training samples (n_samples, n_features)
        Y_train: Training labels (n_samples, n_outputs)
        X_val: Validation samples (optional)
        Y_val: Validation labels (optional)
        """
        # Convert to numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_val = np.array(X_val) if X_val is not None else np.array([])
        Y_val = np.array(Y_val) if Y_val is not None else np.array([])
        
        # Build activation list (hidden layers + output)
        activations = [self.activation] * len(self.hidden_sizes) + ['sigmoid']
        
        # Train the network
        final_weights, final_biases, history = train_model(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            weights=self.weights,
            biases=self.biases,
            activations=activations,
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            patience=self.patience,
            noise_sigma=self.noise_sigma,
            momentum=self.momentum,
            adaptive_eta=self.adaptive_eta
        )
        
        # Store trained parameters and history
        self.weights = final_weights
        self.biases = final_biases
        self.training_history = history
        
        return history

    def predict(self, X):
        """Make predictions for input samples"""
        X = np.array(X).T  # Convert to column vectors
        activations, _ = forward_propagation(X, self.weights, self.biases, 
                                           [self.activation]*len(self.hidden_sizes) + ['sigmoid'])
        return activations[-1].T  # Return as row vectors

    def evaluate(self, X, Y):
        """Calculate accuracy for given samples and labels"""
        predictions = self.predict(X)
        y_true = np.argmax(Y, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        return np.mean(y_pred == y_true)

    def save_hidden_units(self, X):
        """Save activations of all hidden layers for given input"""
        X = np.array(X).T  # Convert to column vectors
        activations, _ = forward_propagation(X, self.weights, self.biases,
                                           [self.activation]*len(self.hidden_sizes) + ['sigmoid'])
        
        # Extract hidden layer activations (exclude input and output)
        hidden_activations = activations[1:-1]
        
        # Format and save activations
        formatted_activations = []
        for layer_idx, layer_acts in enumerate(hidden_activations):
            # Transpose to get samples as rows, neurons as columns
            formatted_activations.append(layer_acts.T)
            
        save_hidden_units(formatted_activations)
        
    def get_architecture(self):
        """Return network architecture description"""
        return {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_sizes,
            'output_size': self.output_size,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'adaptive_learning': self.adaptive_eta,
            'noise_level': self.noise_sigma,
            'momentum': self.momentum
        }

    def reset_parameters(self):
        """Re-initialize network parameters"""
        self.weights, self.biases = self._initialize_parameters()
        self.training_history = None