import os
import numpy as np
from typing import List, Tuple, Dict, Union

DEFAULT_OUTPUT_ENCODING = {str(i): [1 if j == i else 0 for j in range(10)] for i in range(10)}


def process_file(input_file, output_file, elements_per_segment=26, selected_elements=12, total_segments=40):
    """
    Process raw data file into formatted output file with selected elements
    This function is used to preprocess the raw data file into a format that can be used for training

    Args:
        input_file: Path to raw input data
        output_file: Path to save processed output
        elements_per_segment: Number of elements per segment
        selected_elements: Number of selected elements per segment
        total_segments: Total number of segments

    Returns:
        None
    """
    all_values = []
    VALID_IDS = set(DEFAULT_OUTPUT_ENCODING.keys())

    # Première passe : collecter toutes les valeurs
    with open(input_file, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            values_str = line.split(':', 1)[1].strip()
            try:
                values = list(map(float, values_str.split()))
                all_values.extend(values)
            except:
                continue

    # Calcul des statistiques globales
    global_mean = np.mean(all_values)
    global_std = np.std(all_values) + 1e-8
    #values = (values - global_mean) / (global_std + 1e-8)

    # Deuxième passe : traitement avec normalisation globale
    processed_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(':', 1)
            if len(parts) != 2:
                continue
            identifier, values_str = parts
            identifier = identifier.strip().lower().replace('o', '0')

            if identifier not in VALID_IDS:
                continue

            try:
                values = (np.array(list(map(float, values_str.split()))) - global_mean) / global_std
                extracted = []
                for i in range(total_segments):
                    start = i * elements_per_segment
                    end = start + selected_elements
                    #extracted.extend(values[start:end])
                    extracted.extend(values[start:end].tolist())  # Conversion explicite pour éviter les erreurs de type
                #processed_lines.append(f"{identifier}: {' '.join(map('{:.6f}'.format, extracted))}")
                    # Formatage corrigé avec des flottants Python
                formatted_values = [f"{v:.6f}" for v in extracted]
                processed_line = f"{identifier}: {' '.join(formatted_values)}"
                processed_lines.append(processed_line)
            except:
                continue

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("\n".join(processed_lines))


def getES(filename, output_encoding):
    """
    Load and return input samples and labels from processed file
    Enhanced data loading with:
    - Additional sanity checks
    - Automatic value casting

    Args:
        filename: Path to processed input file
        output_encoding: Dictionary mapping identifiers to labels

    Returns:
        Tuple of (samples, labels)
    """
    samples = []
    labels = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue

            parts = line.split(':', 1)
            identifier = parts[0].strip().lower().replace('o', '0')
            values_str = parts[1].strip()

            if identifier not in output_encoding:
                continue

            try:
                # Load and verify values
                values = np.array([float(x) for x in values_str.split()],
                                  dtype=np.float32)
                if len(values) == 0:
                    continue

                samples.append(values)
                labels.append(output_encoding[identifier])
            except ValueError:
                continue

    # Final dataset validation
    if len(samples) == 0:
        raise ValueError("No valid samples loaded")

    return np.array(samples), np.array(labels)

def split_data(samples: np.ndarray, labels: np.ndarray,
              cv_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset into training and validation sets with shuffling
    
    Args:
        samples: Input features matrix
        labels: Corresponding labels
        cv_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (X_train, Y_train, X_val, Y_val)
    """
    if len(samples) != len(labels):
        raise ValueError("Samples and labels must have same length")
        
    indices = np.random.permutation(len(samples))
    split_idx = int(len(indices) * (1 - cv_split))
    
    return (samples[indices[:split_idx]], labels[indices[:split_idx]],
            samples[indices[split_idx:]], labels[indices[split_idx:]])

def debug_data_processing(input_file: str, output_file: str):
    """
    Debug tool for data processing pipeline
    
    Args:
        input_file: Path to raw input data
        output_file: Path to save processed output

    Returns:
        None
    """
    print("\n=== Data Processing Debug ===")
    
    # Show input samples
    print("\nInput File Samples:")
    try:
        with open(input_file, 'r') as f:
            for i in range(3):
                line = next(f).strip()
                print(f"  Line {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
    except StopIteration:
        print("  Input file has less than 3 lines")
    except FileNotFoundError:
        print("  Input file not found")
        return

    # Process file
    try:
        process_file(input_file, output_file)
    except Exception as e:
        print(f"\nProcessing failed: {str(e)}")
        return

    # Show output samples
    print("\nOutput File Samples:")
    try:
        with open(output_file, 'r') as f:
            for i in range(3):
                line = next(f).strip()
                print(f"  Line {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
    except StopIteration:
        print("  Output file has less than 3 lines")

def auto_detect_parameters(input_file: str) -> Tuple[int, int, int]:
    """
    Automatically detects data format parameters
    
    Args:
        input_file: Path to raw input data file
        
    Returns:
        Tuple of (elements_per_segment, selected_elements, total_segments)
    """
    try:
        with open(input_file, 'r') as f:
            line = next(f).strip()
    except StopIteration:
        raise ValueError("Input file is empty")
    
    if ':' not in line:
        raise ValueError("No colon separator in first line")
    
    _, values_part = line.split(':', 1)
    total_values = len(values_part.split())
    
    # Try common configurations
    common_configs = [
        (40, 26, 40),  # 40 segments × 26 elements
        (26, 12, 40),  # 40 segments × 26 elements with 12 selected
        (40, 40, 26)   # 26 segments × 40 elements
    ]
    
    for elements, selected, segments in common_configs:
        if total_values == segments * elements:
            return elements, selected, segments
    
    # Fallback to divisible configuration
    for elements in [26, 40, 12]:
        if total_values % elements == 0:
            segments = total_values // elements
            return elements, elements, segments
    
    raise ValueError(f"Could not detect parameters from {total_values} values")

def save_hidden_units(hidden_activations: List[np.ndarray], 
                     filename: str = "hidden_units.txt") -> None:
    """
    Saves hidden layer activations with formatting

    Args:
        hidden_activations: List of hidden layer activations
        filename: Output filename for saved activations

    Returns:
        None
    """
    try:
        with open(filename, 'w') as f:
            for layer_idx, activations in enumerate(hidden_activations, 1):
                f.write(f"=== Hidden Layer {layer_idx} Activations ===\n")
                f.write(f"Shape: {activations.shape}\n")
                np.savetxt(f, activations, fmt='%.4f', delimiter='\t')
                f.write("\n\n")
        print(f"Saved hidden units to {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Error saving hidden units: {str(e)}")
        raise