# main.py
import sys
import os
import numpy as np
from file_utils import process_file, load_matrix, clear_pictures_folder, print_header
from training import train_model, get_test_input_case1
from visualization import visualize_network, generate_interactive_html
from propagation import forward_propagation, backward_propagation, update_weights
import configparser

def read_config(config_filename="config.ini"):
    config = configparser.ConfigParser()
    if os.path.exists(config_filename):
        config.read(config_filename)
        cfg = config['DEFAULT']
        try:
            num_layers = int(cfg.get('num_layers'))
            # Example: "2,2,1" (input, one hidden, output)
            layer_sizes = [int(x.strip()) for x in cfg.get('layer_sizes').split(',')]
            activation = cfg.get('activation')
            learning_rate = float(cfg.get('learning_rate'))
            max_time = int(cfg.get('max_time'))
            patience = int(cfg.get('patience'))
            cv_split = float(cfg.get('cv_split'))
            return {
                'num_layers': num_layers,
                'layer_sizes': layer_sizes,
                'activation': activation,
                'learning_rate': learning_rate,
                'max_time': max_time,
                'patience': patience,
                'cv_split': cv_split
            }
        except Exception as e:
            print("Error reading configuration:", e)
    else:
        print("No configuration file found.")
    return None

def get_user_input():
    config = read_config()
    if config:
        print("Loaded configuration:")
        print(config)
        layer_sizes = config['layer_sizes']
        num_layers = config['num_layers']
        activations = [config['activation']] * (num_layers - 1)
        learning_rate = config['learning_rate']
        max_time = config['max_time']
        patience = config['patience']
        cv_split = config['cv_split']
    else:
        num_layers = int(input("Entrez le nombre total de couches (entrée, cachées, sortie) : "))
        layer_sizes = []
        act = input("Entrez la fonction d'activation (sigmoide/tan/tanh/softmax/personnalisée/sinus) : ").strip()
        for i in range(num_layers):
            layer_sizes.append(int(input(f"Entrez le nombre de neurones dans la couche {i+1} : ")))
        activations = [act] * (num_layers - 1)
        learning_rate = float(input("Entrez le taux d'apprentissage: "))
        max_time = int(input("Entrez le temps maximum d'apprentissage (en secondes): "))
        patience = int(input("Entrez le nombre de patience: "))
        cv_split = float(input("Entrez le pourcentage de validation croisée (ex: 0.2): "))
    # Initialize weights and biases with small random numbers
    weights = []
    biases = []
    for i in range(len(layer_sizes)-1):
        prev_size = layer_sizes[i]
        cur_size = layer_sizes[i+1]
        w = np.random.uniform(-0.1, 0.1, (cur_size, prev_size))
        b = np.zeros((cur_size, 1))
        weights.append(w)
        biases.append(b)
    Y_dummy = np.zeros((layer_sizes[-1], 1))
    return Y_dummy, layer_sizes, activations, weights, biases, learning_rate, max_time, patience, cv_split

def main():
    if "--test" in sys.argv:
        # Process three datasets: 40, 50, 60 segments
        input_file = os.path.join("InputFiles", "data_train.txt")
        out_40 = os.path.join("OutputFiles", "data_train_40_ligne.txt")
        out_50 = os.path.join("OutputFiles", "data_train_50_ligne.txt")
        out_60 = os.path.join("OutputFiles", "data_train_60_ligne.txt")
        process_file(input_file, out_40, total_segments=40)
        process_file(input_file, out_50, total_segments=50)
        process_file(input_file, out_60, total_segments=60)
        for f in [out_40, out_50, out_60]:
            if os.path.exists(f):
                print(f"SUCCESS: Processed file {f} exists.")
            else:
                print(f"ERROR: Processed file {f} does NOT exist.")
        # Load one set (e.g., 40 segments) and create samples
        X_mat = load_matrix(out_40)
        X_samples = [np.array(row).reshape(-1, 1) for row in X_mat]
        Y_samples = [np.zeros((X_mat.shape[1], 1)) for _ in X_samples]
        # Visualization test using test input case 1:
        test_X, test_Y, layer_sizes_test, acts_test, w_test, b_test = get_test_input_case1()
        visualize_network(layer_sizes_test, test_X, b_test, filename="test_network_vis.png")
        generate_interactive_html(layer_sizes_test, test_X, w_test, b_test, filename="test_network.html")
        # Training test:
        print("Running training test on test case 1...")
        s_test = [test_X] * 10
        y_test = [test_Y] * 10
        train_model(s_test, y_test, w_test, b_test, acts_test, 0.1, max_time=10, patience=3, cv_split=0.2)
    else:
        out_40 = os.path.join("OutputFiles", "data_train_40_ligne.txt")
        if os.path.exists(out_40):
            X_mat = load_matrix(out_40)
            X_samples = [np.array(row).reshape(-1, 1) for row in X_mat]
            Y_samples = [np.zeros((X_mat.shape[1], 1)) for _ in X_samples]
            big_input_sample = X_samples[0]
        else:
            print("No processed file found, using test case 1 as fallback.")
            test_X, test_Y, layer_sizes_test, acts_test, w_test, b_test = get_test_input_case1()
            X_samples = [test_X]
            Y_samples = [test_Y]
            big_input_sample = test_X
        Y_dummy, layer_sizes, activations, weights, biases, learning_rate, max_time, patience, cv_split = get_user_input()
        clear_pictures_folder()
        print("Layer sizes:", layer_sizes)
        print("Activations:", activations)
        print_header("Visualisation du Réseau (Manual)")
        visualize_network(layer_sizes, big_input_sample, biases, filename="network_vis_manual.png")
        generate_interactive_html(layer_sizes, big_input_sample, weights, biases, filename="network_manual.html")
        trained_weights, trained_biases = train_model(X_samples, Y_samples, weights, biases, activations,
                                                      learning_rate, max_time, patience, cv_split)
if __name__ == "__main__":
    main()
