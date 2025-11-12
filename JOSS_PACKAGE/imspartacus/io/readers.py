"""
File reading utilities for IMS data.

This module handles loading data from various file formats used in ion mobility
mass spectrometry, including:
- Text files (.txt) from MassLynx
- CSV files from TWIMExtract
"""

import re
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


def is_valid_calibrant_file(file_path: Path) -> bool:
    """
    Check if a file is a valid calibrant data file.
    
    This function determines if a file should be processed as calibrant data
    by checking:
    1. File extension (.txt or .csv)
    2. Filename pattern indicating charge state
    
    For .txt files: The filename should start with a number (the charge state)
    Example: "24.txt" for charge state 24
    
    For .csv files: The filename should contain a charge state pattern like:
    - "range_24.txt" 
    - "range_24_raw"
    - "DT_sample_range_24.txt_raw"
    
    Args:
        file_path: Path object pointing to the file to check
        
    Returns:
        True if the file is valid for processing, False otherwise
        
    Examples:
        >>> is_valid_calibrant_file(Path("24.txt"))
        True
        >>> is_valid_calibrant_file(Path("range_24.csv"))
        True
        >>> is_valid_calibrant_file(Path("notes.txt"))
        False
    """
    # Check if it's a CSV file
    if file_path.suffix.lower() == '.csv':
        # Get filename without the .csv extension
        filename_without_ext = file_path.stem
        
        # These patterns match different ways charge states appear in filenames
        patterns = [
            r'range_(\d+)\.txt',   # Matches "range_24.txt"
            r'range_(\d+)_',        # Matches "range_24_"
            r'_(\d+)\.txt_raw',     # Matches "_24.txt_raw"
            r'_(\d+)_raw$',         # Matches "_24_raw" at the end
            r'_(\d+)$'              # Matches "_24" at the end
        ]
        
        # Try each pattern to see if we can find a charge state
        for pattern in patterns:
            if re.search(pattern, filename_without_ext):
                return True
        return False
    
    # Check if it's a .txt file
    elif file_path.suffix == '.txt':
        # For .txt files, the filename should start with a digit
        # Example: "24.txt" for charge state 24
        return file_path.name[0].isdigit()
    
    # If it's neither .txt nor .csv, it's not valid
    else:
        return False


def extract_charge_state_from_filename(filename: str) -> Optional[int]:
    """
    Extract the charge state from a calibrant filename.
    
    This function looks for numeric patterns in filenames that indicate
    the charge state of the ion.
    
    For .txt files: "24.txt" -> 24
    For .csv files: "DT_sample_range_24.txt_raw.csv" -> 24
    
    Args:
        filename: The filename (with or without extension) to parse
        
    Returns:
        The charge state as an integer, or None if not found
        
    Examples:
        >>> extract_charge_state_from_filename("24.txt")
        24
        >>> extract_charge_state_from_filename("range_18_raw.csv")
        18
        >>> extract_charge_state_from_filename("unknown.txt")
        None
    """
    # Remove file extension to work with just the name
    file_path = Path(filename)
    filename_without_ext = file_path.stem
    
    # First, try the simple case: filename is just a number
    # Example: "24.txt" -> stem is "24"
    try:
        return int(filename_without_ext)
    except ValueError:
        pass  # Not just a number, try regex patterns
    
    # Patterns to find charge state in more complex filenames
    patterns = [
        r'range_(\d+)\.txt',   # Matches "range_24.txt"
        r'range_(\d+)_',        # Matches "range_24_"
        r'_(\d+)\.txt_raw',     # Matches "_24.txt_raw"
        r'_(\d+)_raw$',         # Matches "_24_raw" at end
        r'_(\d+)$'              # Matches "_24" at end
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, filename_without_ext)
        if match:
            # match.group(1) gets the number inside the parentheses in the pattern
            return int(match.group(1))
    
    # If no pattern matched, return None
    return None


def load_atd_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load drift time and intensity data from an ATD (arrival time distribution) file.
    
    This function supports two formats:
    1. .txt files: Two-column space-separated data from MassLynx
    2. .csv files: Comma-separated data from TWIMExtract (can include # comments)
    
    Args:
        file_path: Path to the ATD data file
        
    Returns:
        A tuple of two numpy arrays: (drift_time, intensity)
        - drift_time: Array of drift time values (usually in ms)
        - intensity: Array of intensity values corresponding to each drift time
        
    Raises:
        ValueError: If no valid data is found in the file
        FileNotFoundError: If the file doesn't exist
        
    Examples:
        >>> drift_time, intensity = load_atd_data(Path("24.txt"))
        >>> print(f"Loaded {len(drift_time)} data points")
        Loaded 500 data points
    """
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Handle CSV files (from TWIMExtract)
    if file_path.suffix.lower() == '.csv':
        data_rows = []
        
        # Read file line by line
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip comment lines (start with #)
                if line.startswith('#'):
                    continue
                
                # Try to parse the data
                try:
                    # Split by comma
                    values = line.split(',')
                    
                    # We need at least 2 values (drift time and intensity)
                    if len(values) >= 2:
                        drift_time = float(values[0])
                        intensity = float(values[1])
                        data_rows.append([drift_time, intensity])
                        
                except (ValueError, IndexError):
                    # If line can't be parsed, skip it
                    continue
        
        # Check if we got any data
        if not data_rows:
            raise ValueError(f"No valid data found in CSV file: {file_path}")
        
        # Convert list to numpy array
        data = np.array(data_rows)
        
        # Return as two separate arrays
        # data[:, 0] means "all rows, first column" (drift time)
        # data[:, 1] means "all rows, second column" (intensity)
        return data[:, 0], data[:, 1]
    
    # Handle .txt files (from MassLynx)
    else:
        # np.loadtxt() automatically reads space/tab separated files
        # It expects two columns: drift time and intensity
        data = np.loadtxt(file_path)
        
        return data[:, 0], data[:, 1]


def load_multiple_atd_files(folder_path: Path) -> dict:
    """
    Load all valid ATD files from a folder.
    
    This is a convenience function that processes an entire folder
    of calibrant files at once.
    
    Args:
        folder_path: Path to folder containing ATD files
        
    Returns:
        Dictionary mapping charge states to (drift_time, intensity) tuples
        Example: {24: (drift_array, intensity_array), 25: (...), ...}
        
    Examples:
        >>> data = load_multiple_atd_files(Path("myoglobin"))
        >>> print(f"Found charge states: {list(data.keys())}")
        Found charge states: [24, 25, 26, 27]
    """
    results = {}
    
    # Iterate through all files in the folder
    for file_path in folder_path.iterdir():
        # Skip if not a valid calibrant file
        if not is_valid_calibrant_file(file_path):
            continue
        
        # Extract charge state from filename
        charge_state = extract_charge_state_from_filename(file_path.name)
        if charge_state is None:
            continue
        
        # Try to load the data
        try:
            drift_time, intensity = load_atd_data(file_path)
            results[charge_state] = (drift_time, intensity)
        except Exception:
            # If loading fails, skip this file
            continue
    
    return results
