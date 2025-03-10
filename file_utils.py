# file_utils.py
import os
import shutil
import numpy as np

def print_header(title):
    border = "=" * len(title)
    print(f"\n{border}\n{title}\n{border}\n")

def print_matrix(label, matrix):
    # Use numpy’s array2string for formatted printing.
    print(f"{label}:\n{np.array2string(matrix, precision=4)}\n")

def clear_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder)

def clear_pictures_folder():
    clear_folder("pictures")

def clear_html_folder():
    clear_folder("html")

def process_file(input_file, output_file, elements_per_segment=26, selected_elements=12, total_segments=40):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    processed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 1561:
            continue
        data = parts[1:]
        extracted = []
        for i in range(total_segments):
            start = i * elements_per_segment
            end = start + selected_elements
            extracted.extend(data[start:end])
        processed_lines.append(" ".join(extracted))
    with open(output_file, 'w') as outfile:
        outfile.write("\n".join(processed_lines))
    print(f"Processed file saved to: {os.path.abspath(output_file)}")

def load_matrix(filename):
    try:
        mat = np.loadtxt(filename)
    except Exception as e:
        print(f"Error loading matrix from {filename}: {e}")
        mat = np.array([])
    print(f"Loaded matrix from {filename} with {mat.shape[0]} rows and {mat.shape[1] if mat.ndim>1 else 0} columns.")
    return mat

def save_hidden_units(hidden_activations, filename="hidden_units.txt"):
    with open(filename, 'w') as f:
        for i, act in enumerate(hidden_activations):
            f.write(f"Hidden Layer {i+1} activations:\n")
            f.write(np.array2string(act, precision=4))
            f.write("\n\n")
    print(f"Hidden unit activations saved to: {os.path.abspath(filename)}")
