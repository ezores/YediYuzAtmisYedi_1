import os
import sys
import argparse
import configparser
import numpy as np
from file_utils import process_file, getES, split_data, DEFAULT_OUTPUT_ENCODING
from mlp import MLP
from visualization import generate_network_visualization
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
    'momentum': '0.0'
}


def parse_arguments():
    """
    Parse command-line arguments for the MLP speech recognition system
    """
    parser = argparse.ArgumentParser(description="MLP Speech Recognition System")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train the model')
    mode_group.add_argument('--vc', action='store_true', help='Run k-fold cross-validation')
    mode_group.add_argument('--test', action='store_true', help='Test trained model')
    parser.add_argument('--eta', type=float, help='Initial learning rate (suggested: 0.001-0.5)')
    parser.add_argument('--neurons', nargs='+', type=int, help='Hidden layer sizes (e.g., 128 64 for two layers)')
    parser.add_argument('--activations', nargs='+',
                        help=f"Activation functions ({', '.join(activation_functions.keys())})")
    parser.add_argument('--base', type=int, choices=[40, 50, 60], help='Database size: 40|50|60 segments')
    parser.add_argument('--epochs', type=int, help='Max training epochs (typically 50-500)')
    parser.add_argument('--adaptive', action='store_true', help='Enable adaptive learning rate')
    parser.add_argument('--noise', type=float, help='Input noise sigma (suggested: 0.0-0.5)')
    return parser.parse_args()


def interactive_config(args):
    """
    Interactive configuration for the MLP speech recognition system
    """
    print("\n=== Network Configuration ===")
    activations_list = list(activation_functions.keys())
    if args.eta is None:
        args.eta = float(
            input(f"Initial learning rate (e.g., 0.01-0.5) [{DEFAULT_CONFIG['eta']}]: ") or DEFAULT_CONFIG['eta'])
    if not args.neurons:
        default_neurons = DEFAULT_CONFIG['neurones_par_couche_cachee']
        args.neurons = list(map(int, input(
            f"Hidden layer sizes (e.g., '64' or '128 64') [{default_neurons}]: ").split())) or default_neurons.split(
            ',')
    if not args.activations:
        print(f"Available activations: {', '.join(activations_list)}")
        args.activations = [
            input(f"Activation function ({'/'.join(activations_list)}) [{DEFAULT_CONFIG['fct']}]: ") or DEFAULT_CONFIG[
                'fct']]
    if args.base is None:
        args.base = int(input("Database size (40=small/50=medium/60=large) [40]: ") or DEFAULT_CONFIG['base_donnees'])
    if args.epochs is None:
        args.epochs = int(input("Max training epochs (recommended: 100-200) [100]: ") or DEFAULT_CONFIG['nb_epoches'])
    if not args.adaptive:
        args.adaptive = input("Use adaptive learning? (y/n) [n]: ").lower() in ['y', 'yes']
    if args.noise is None:
        args.noise = float(input("Input noise sigma (0.0 to 0.5, 0=disable) [0]: ") or 0)
    return args


def load_or_create_config(args):
    """'
    Load or create configuration file for the MLP speech recognition system
    """
    config = configparser.ConfigParser()
    if os.path.exists('config.ini'):
        config.read('config.ini')
    else:
        config['DEFAULT'] = DEFAULT_CONFIG
    config['DEFAULT']['eta'] = str(args.eta)
    config['DEFAULT']['neurones_par_couche_cachee'] = ','.join(map(str, args.neurons))
    config['DEFAULT']['fct'] = args.activations[0]
    config['DEFAULT']['base_donnees'] = str(args.base)
    config['DEFAULT']['nb_epoches'] = str(args.epochs)
    config['DEFAULT']['adaptive_eta'] = str(args.adaptive)
    config['DEFAULT']['noise_sigma'] = str(args.noise)
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    return config


def prepare_datasets(base_size: int, input_dir: str = "InputFiles", output_dir: str = "OutputFiles") -> str:
    input_path = os.path.join(input_dir, "data_train.txt")
    output_path = os.path.join(output_dir, f"data_train_{base_size}_ligne.txt")
    """
    Process the training data file for the MLP speech recognition system
    This function will process the input file and generate a new file with the selected elements
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_path):
        print("\n=== Starting Data Processing ===")
        process_file(
            input_path,
            output_path,
            elements_per_segment=26,
            selected_elements=12,
            total_segments=base_size
        )
    return output_path


def main():
    """
    Main function for the MLP speech recognition system
    """
    args = parse_arguments()
    args = interactive_config(args)
    config = load_or_create_config(args)

    try:
        # Prepare dataset
        data_file = prepare_datasets(args.base)
        print(f"Loading processed file: {data_file}")

        # Validate data file
        print("\n=== Input File Validation ===")
        with open(data_file, 'r') as f:
            for _ in range(3):
                line = f.readline().strip()
                if not line: continue
                if ':' not in line:
                    print(f"Validation Error: Missing colon in sample line: '{line[:50]}...'")
                else:
                    identifier, values = line.split(':', 1)
                    print(f"Sample Line OK - ID: {identifier}, Values: {len(values.split())} features")

        X, Y = getES(data_file, DEFAULT_OUTPUT_ENCODING)
        if X.size == 0 or Y.size == 0:
            raise ValueError("Empty dataset loaded")

    except Exception as e:
        print(f"\nCritical error: {e}")
        print("Possible solutions:")
        print("1. Verify input files exist in InputFiles/ directory")
        print("2. Check data files use format: 'ID: val1 val2 ...'")
        print("3. Ensure numeric values after colon separator")
        print("4. Confirm identifiers are digits 0-9")
        sys.exit(1)

    # Split data
    if args.train:
        X_train, Y_train, X_val, Y_val = split_data(X, Y, cv_split=0.2)
    elif args.vc:
        X_train, Y_train, X_val, Y_val = split_data(X, Y, cv_split=0.5)
    else:
        X_train, Y_train, X_val, Y_val = X, Y, np.array([]), np.array([])

    # Initialize MLP
    mlp = MLP(
        input_size=X.shape[1],
        hidden_sizes=args.neurons,
        output_size=10,
        activation=args.activations[0],
        learning_rate=args.eta,
        max_epochs=args.epochs,
        adaptive_eta=args.adaptive,
        noise_sigma=args.noise
    )

    # Training process
    if args.train or args.vc:
        print("\n=== Starting Training ===")
        try:
            mlp.train(X_train, Y_train, X_val, Y_val)
            mlp.save_hidden_units(X_train)
            if args.train:
                print("\n=== Training Performance ===")
                train_acc = mlp.evaluate(X_train, Y_train)
                print(f"Training Accuracy: {train_acc:.2%}")
                print("\n=== Validation Performance ===")
                val_acc = mlp.evaluate(X_val, Y_val)
                print(f"Validation Accuracy: {val_acc:.2%}")
            if args.vc:
                print("\n=== Cross-Validation Results ===")
                vc_acc = mlp.evaluate(X_val, Y_val)
                print(f"Cross-Validation Accuracy: {vc_acc:.2%}")

        except Exception as e:
            print(f"\nTraining failed: {str(e)}")
            sys.exit(1)

    if args.test:
        print("\n=== Testing Mode ===")
        try:
            # Process test data with SAME parameters as training
            test_input_file = os.path.join("InputFiles", "data_test.txt")
            test_processed_file = os.path.join("OutputFiles", f"data_test_{args.base}_ligne.txt")

            process_file(
                test_input_file,
                test_processed_file,
                elements_per_segment=26,
                selected_elements=12,
                total_segments=args.base  # Critical fix
            )

            X_test, Y_test = getES(test_processed_file, DEFAULT_OUTPUT_ENCODING)
            test_acc = mlp.evaluate(X_test, Y_test)
            print(f"Test Accuracy: {test_acc:.2%}")
        except Exception as e:
            print(f"\nTesting failed: {str(e)}")
            sys.exit(1)

    generate_network_visualization(mlp, "Visualisation_MLP.html")


if __name__ == "__main__":
    main()