import os
import shutil
import numpy as np
from typing import List, Tuple, Dict, Union

# Constants
DEFAULT_OUTPUT_ENCODING = {
    '0': [1,0,0,0,0,0,0,0,0,0],
    '1': [0,1,0,0,0,0,0,0,0,0],
    '2': [0,0,1,0,0,0,0,0,0,0],
    '3': [0,0,0,1,0,0,0,0,0,0],
    '4': [0,0,0,0,1,0,0,0,0,0],
    '5': [0,0,0,0,0,1,0,0,0,0],
    '6': [0,0,0,0,0,0,1,0,0,0],
    '7': [0,0,0,0,0,0,0,1,0,0],
    '8': [0,0,0,0,0,0,0,0,1,0],
    '9': [0,0,0,0,0,0,0,0,0,1]
}

VALID_IDS = set(DEFAULT_OUTPUT_ENCODING.keys())

def process_file(input_file: str, output_file: str, 
                elements_per_segment: int = 26, 
                selected_elements: int = 12,
                total_segments: int = 40) -> None:
    """
    Processes colon-separated input files with ID normalization
    """
    VALID_IDS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    
    print("\n=== Raw Input Validation ===")
    with open(input_file, 'r') as f:
        sample_lines = [next(f).strip() for _ in range(5)]
    
    for idx, line in enumerate(sample_lines, 1):
        if ':' not in line:
            print(f"INVALID LINE {idx}: '{line[:50]}...'")
        else:
            print(f"VALID LINE {idx}: '{line[:50]}...'")

    error_report = {
        'total_lines': 0,
        'empty_lines': 0,
        'missing_colon': 0,
        'invalid_id': 0,
        'value_error': 0,
        'insufficient_values': 0,
        'segment_errors': 0,
        'successful': 0
    }

    processed_lines = []
    
    with open(input_file, 'r') as infile:
        for line_num, line in enumerate(infile, 1):
            error_report['total_lines'] += 1
            line = line.strip()
            
            if not line:
                error_report['empty_lines'] += 1
                continue

            # Split on colon
            if ':' not in line:
                error_report['missing_colon'] += 1
                print(f"[Line {line_num}] Missing colon: '{line[:50]}...'")
                continue

            parts = line.split(':', 1)
            identifier = parts[0].strip().lower().replace('o', '0').replace(':', '')
            values_str = parts[1].strip()

            # Validate identifier
            if not identifier.isdigit() or identifier not in VALID_IDS:
                error_report['invalid_id'] += 1
                print(f"[Line {line_num}] Invalid ID '{identifier}': '{line[:50]}...'")
                continue

            # Convert values to floats
            try:
                values = [float(x) for x in values_str.split()]
            except ValueError as e:
                error_report['value_error'] += 1
                print(f"[Line {line_num}] Value error: {e} in '{values_str[:50]}...'")
                continue

            # Check total values
            required_values = total_segments * elements_per_segment
            if len(values) < required_values:
                error_report['insufficient_values'] += 1
                print(f"[Line {line_num}] Insufficient values ({len(values)} < {required_values})")
                continue

            # Extract features
            extracted = []
            try:
                for i in range(total_segments):
                    start = i * elements_per_segment
                    end = start + selected_elements
                    segment = values[start:end]
                    
                    if len(segment) != selected_elements:
                        raise ValueError(f"Segment {i} length {len(segment)} != {selected_elements}")
                        
                    extracted.extend(segment)
            except (IndexError, ValueError) as e:
                error_report['segment_errors'] += 1
                print(f"[Line {line_num}] Segment error: {str(e)}")
                continue

            # Format output line
            formatted_values = [f"{v:.4f}" for v in extracted]
            processed_line = f"{identifier}: {' '.join(formatted_values)}"
            processed_lines.append(processed_line)
            error_report['successful'] += 1

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as outfile:
        outfile.write("\n".join(processed_lines))

    print("\n=== Data Processing Report ===")
    print(f"Total lines processed: {error_report['total_lines']}")
    print(f"Successfully converted: {error_report['successful']}")
    print(f"Empty lines: {error_report['empty_lines']}")
    print(f"Missing colons: {error_report['missing_colon']}")
    print(f"Invalid IDs: {error_report['invalid_id']}")
    print(f"Value errors: {error_report['value_error']}")
    print(f"Insufficient values: {error_report['insufficient_values']}")
    print(f"Segment errors: {error_report['segment_errors']}")

    if error_report['successful'] == 0:
        raise ValueError("No valid data processed - check input format")


def getES(filename: str, output_encoding: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads processed files with ID normalization
    """
    VALID_IDS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    samples = []
    labels = []
    error_count = 0
    max_errors_to_show = 10

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Split using colon
            if ':' not in line:
                if error_count < max_errors_to_show:
                    print(f"[Line {line_num}] Missing colon: '{line[:50]}...'")
                error_count += 1
                continue

            parts = line.split(':', 1)
            identifier = parts[0].strip().lower().replace('o', '0').replace(':', '')
            values_str = parts[1].strip()

            if not identifier.isdigit() or identifier not in VALID_IDS:
                if error_count < max_errors_to_show:
                    print(f"[Line {line_num}] Invalid ID '{identifier}': '{line[:50]}...'")
                error_count += 1
                continue

            try:
                values = [float(x) for x in values_str.split()]
            except ValueError as e:
                if error_count < max_errors_to_show:
                    print(f"[Line {line_num}] Value error: {e} in '{values_str[:50]}...'")
                error_count += 1
                continue

            samples.append(values)
            labels.append(output_encoding[identifier])

    if error_count > 0:
        print(f"\nEncountered {error_count} errors during loading")
        if error_count > max_errors_to_show:
            print(f"Showing first {max_errors_to_show} errors...")

    if not samples:
        raise ValueError("No valid samples loaded - check file format")

    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.float32)

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
    """Saves hidden layer activations with formatting"""
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