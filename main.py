# main.py
import os
import sys
import argparse
import configparser
import logging
import numpy as np
from file_utils import process_file, getES, split_data, kfold_split, DEFAULT_OUTPUT_ENCODING
from mlp import MLP
from visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_weight_distribution,
    plot_activation_distribution,
    generate_network_visualization
)
from mlp_math import activation_functions

# Configuration defaults
DEFAULT_CONFIG = {
    'eta': '0.1',
    'nb_epoches': '100',
    'neurones_par_couche_cachee': '50',
    'fct': 'sigmoid',
    'base_donnees': '40',
    'adaptive_eta': 'False',
    'noise_sigma': '0.0',
    'momentum': '0.0',
    'batch_norm': 'False',
    'dropout_rate': '0.0',
    'k_folds': '5'
}


def setup_logging():
    """Configure logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('mlp_training.log'),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """Enhanced argument parser with new features"""
    parser = argparse.ArgumentParser(description="MLP Speech Recognition System")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train the model')
    mode_group.add_argument('--vc', action='store_true', help='Run k-fold cross-validation')
    mode_group.add_argument('--test', action='store_true', help='Test trained model')

    # Model parameters
    parser.add_argument('--eta', type=float, help='Initial learning rate (0.001-0.5)')
    parser.add_argument('--neurons', nargs='+', type=int, help='Hidden layer sizes (e.g., 128 64)')
    parser.add_argument('--activations', nargs='+',
                        help=f"Activation functions ({', '.join(activation_functions.keys())})")
    parser.add_argument('--base', type=int, choices=[40, 50, 60], help='Database size: 40|50|60')
    parser.add_argument('--epochs', type=int, help='Max training epochs (50-500)')
    parser.add_argument('--folds', type=int, default=5, help='Number of k-folds for cross-validation')
    parser.add_argument('--adaptive', action='store_true', help='Enable adaptive learning rate')
    parser.add_argument('--noise', type=float, help='Input noise sigma (0.0-0.5)')
    parser.add_argument('--batch_norm', action='store_true', help='Enable batch normalization')
    parser.add_argument('--dropout', type=float, help='Dropout rate (0.0-0.9)')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')

    return parser.parse_args()


def interactive_config(args):
    """Enhanced interactive configuration with new parameters"""
    print("\n=== Network Configuration ===")

    # Existing parameters
    if args.eta is None:
        args.eta = float(input(f"Initial learning rate [{DEFAULT_CONFIG['eta']}]: ") or DEFAULT_CONFIG['eta'])

    if not args.neurons:
        default_neurons = DEFAULT_CONFIG['neurones_par_couche_cachee']
        args.neurons = list(map(int, input(
            f"Hidden layer sizes (e.g., '64' or '128 64') [{default_neurons}]: ").split())) or default_neurons.split(
            ',')

    if not args.activations:
        print(f"Available activations: {', '.join(activation_functions.keys())}")
        args.activations = [input(f"Activation function [{DEFAULT_CONFIG['fct']}]: ") or DEFAULT_CONFIG['fct']]

    if args.base is None:
        args.base = int(
            input(f"Database size (40/50/60) [{DEFAULT_CONFIG['base_donnees']}]: ") or DEFAULT_CONFIG['base_donnees'])

    if args.epochs is None:
        args.epochs = int(input(f"Max epochs [{DEFAULT_CONFIG['nb_epoches']}]: ") or DEFAULT_CONFIG['nb_epoches'])

    # New parameters
    if not args.adaptive:
        args.adaptive = input("Use adaptive learning rate? (y/n) [n]: ").lower() in ['y', 'yes']

    if args.noise is None:
        args.noise = float(
            input(f"Input noise sigma [{DEFAULT_CONFIG['noise_sigma']}]: ") or DEFAULT_CONFIG['noise_sigma'])

    if not args.batch_norm:
        args.batch_norm = input("Enable batch normalization? (y/n) [n]: ").lower() in ['y', 'yes']

    if args.dropout is None:
        args.dropout = float(
            input(f"Dropout rate (0-0.9) [{DEFAULT_CONFIG['dropout_rate']}]: ") or DEFAULT_CONFIG['dropout_rate'])

    if args.augment is None:
        args.augment = input("Enable data augmentation? (y/n) [n]: ").lower() in ['y', 'yes']

    return args


def prepare_datasets(base_sizes: list, input_dir: str = "InputFiles",
                     output_dir: str = "OutputFiles", augment: bool = False,
                     noise_sigma: float = 0.0) -> dict:
    """Generate multiple datasets with optional augmentation"""
    datasets = {}
    for size in base_sizes:
        output_path = os.path.join(output_dir, f"data_train_{size}_ligne.txt")
        if not os.path.exists(output_path):
            process_file(
                os.path.join(input_dir, "data_train.txt"),
                output_path,
                elements_per_segment=26,
                selected_elements=12,
                total_segments=size,
                augment=augment,
                noise_sigma=noise_sigma
            )
        datasets[size] = output_path
    return datasets


def main():
    setup_logging()
    args = parse_arguments()
    args = interactive_config(args)

    try:
        # Prepare all three datasets
        logging.info("Preparing datasets...")
        datasets = prepare_datasets(
            base_sizes=[40, 50, 60],
            augment=args.augment,
            noise_sigma=args.noise
        )

        # Load main dataset
        data_file = datasets[args.base]
        logging.info(f"Using dataset: {data_file}")
        X, Y = getES(data_file, DEFAULT_OUTPUT_ENCODING)

        # Initialize MLP with new parameters
        mlp = MLP(
            input_size=X.shape[1],
            hidden_sizes=args.neurons,
            output_size=10,
            activation=args.activations[0],
            learning_rate=args.eta,
            max_epochs=args.epochs,
            adaptive_eta=args.adaptive,
            noise_sigma=args.noise,
            momentum=0.0,
            use_batch_norm=args.batch_norm,
            dropout_rate=args.dropout
        )

        # Training workflow
        if args.train or args.vc:
            logging.info("\n=== Starting Training ===")

            if args.vc:
                logging.info(f"Running {args.folds}-fold cross-validation")
                fold_accuracies = []

                for fold, (X_train, Y_train, X_val, Y_val) in enumerate(kfold_split(X, Y, k=args.folds)):
                    logging.info(f"\n--- Fold {fold + 1}/{args.folds} ---")
                    mlp.train(X_train, Y_train, X_val, Y_val)
                    val_acc = mlp.evaluate(X_val, Y_val)
                    fold_accuracies.append(val_acc)
                    logging.info(f"Fold {fold + 1} Accuracy: {val_acc:.2%}")

                logging.info("\n=== Cross-Validation Results ===")
                logging.info(f"Mean Accuracy: {np.mean(fold_accuracies):.2%}")
                logging.info(f"Std Deviation: {np.std(fold_accuracies):.2%}")

            else:  # Standard training
                X_train, Y_train, X_val, Y_val = split_data(X, Y, cv_split=0.2)
                mlp.train(X_train, Y_train, X_val, Y_val)

                # Visualizations
                plot_training_history(mlp.training_history)
                plot_weight_distribution(mlp.weights)

                if X_val.size > 0:
                    cm = mlp.confusion_matrix(X_val, Y_val)
                    plot_confusion_matrix(cm, class_names=list(map(str, range(10))))

        # Testing workflow
        if args.test:
            logging.info("\n=== Testing ===")
            test_file = os.path.join("InputFiles", "data_test.txt")
            X_test, Y_test = getES(test_file, DEFAULT_OUTPUT_ENCODING)
            test_acc = mlp.evaluate(X_test, Y_test)
            logging.info(f"Test Accuracy: {test_acc:.2%}")

            # Generate final visualizations
            generate_network_visualization(mlp)
            plot_activation_distribution(mlp.save_hidden_units(X_test))

    except Exception as e:
        logging.error(f"Critical error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()