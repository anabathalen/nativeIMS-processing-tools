"""
File writing utilities for IMS data.

This module handles writing data to various output formats used in
ion mobility mass spectrometry workflows.
"""

import io
import zipfile
from pathlib import Path
from typing import Optional, Dict
import pandas as pd


def write_imscal_dat(
    data: pd.DataFrame,
    output_path: Path,
    velocity: Optional[float] = None,
    voltage: Optional[float] = None,
    pressure: Optional[float] = None,
    length: Optional[float] = None
) -> str:
    """
    Write data in IMSCal .dat format.
    
    IMSCal is a calibration tool that can accept two different formats:
    1. Calibration format: Header with instrument parameters + calibrant data
    2. Input format: Simple data table without header
    
    For calibration (with header), provide all instrument parameters and a DataFrame with:
    - 'protein': protein name
    - 'charge state': integer charge state
    - 'mass': protein mass in Da
    - 'calibrant_value': CCS value in nm² (will be converted to Ų)
    - 'drift time': measured drift time in ms
    
    For input files (no header), just provide a DataFrame with:
    - 'index': row index
    - 'mass': protein mass in Da
    - 'charge': integer charge state
    - 'intensity': signal intensity
    - 'drift_time': measured drift time in ms
    
    Args:
        data: DataFrame with data to write
        output_path: Where to save the .dat file
        velocity: Wave velocity in m/s (optional, for calibration format)
        voltage: Wave height in V (optional, for calibration format)
        pressure: IMS pressure in mbar (optional, for calibration format)
        length: Drift cell length in m (optional, for calibration format)
        
    Returns:
        The .dat file content as a string
        
    Example (calibration format):
        >>> df = pd.DataFrame({
        ...     'protein': ['myoglobin', 'myoglobin'],
        ...     'charge state': [24, 25],
        ...     'mass': [16952.3, 16952.3],
        ...     'calibrant_value': [31.2, 29.8],
        ...     'drift time': [5.23, 4.87]
        ... })
        >>> content = write_imscal_dat(df, Path("out.dat"), 281.0, 20.0, 1.63, 0.98)
        
    Example (input format):
        >>> df = pd.DataFrame({
        ...     'index': [0, 1, 2],
        ...     'mass': [16952.3, 16952.3, 16952.3],
        ...     'charge': [24, 24, 24],
        ...     'intensity': [100, 200, 150],
        ...     'drift_time': [5.1, 5.2, 5.3]
        ... })
        >>> content = write_imscal_dat(df, Path("input.dat"))
    """
    # Detect format based on columns and parameters
    has_instrument_params = all(p is not None for p in [velocity, voltage, pressure, length])
    is_calibration_format = 'protein' in data.columns and 'calibrant_value' in data.columns
    
    if has_instrument_params and is_calibration_format:
        # Calibration format with header
        header = (
            f"# length {length}\n"
            f"# velocity {velocity}\n"
            f"# voltage {voltage}\n"
            f"# pressure {pressure}\n"
        )
        
        content_lines = []
        for _, row in data.iterrows():
            protein = row['protein']
            charge_state = int(row['charge state'])
            mass = row['mass']
            calibrant_value = row['calibrant_value'] * 100  # nm² to Ų
            drift_time = row['drift time']
            
            line = f"{protein}_{charge_state} {mass} {charge_state} {calibrant_value} {drift_time}"
            content_lines.append(line)
        
        full_content = header + "\n".join(content_lines)
    else:
        # Input format without header - simple space-delimited table
        content_lines = []
        for _, row in data.iterrows():
            # Format: index mass charge intensity drift_time
            line = f"{int(row['index'])} {row['mass']} {int(row['charge'])} {row['intensity']} {row['drift_time']}"
            content_lines.append(line)
        
        full_content = "\n".join(content_lines)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(full_content)
    
    return full_content


def dataframe_to_csv_string(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a CSV-formatted string.
    
    This is useful for creating downloadable CSV content in web apps
    without actually writing to disk.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        CSV-formatted string
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> csv_str = dataframe_to_csv_string(df)
        >>> print(csv_str)
        A,B
        1,3
        2,4
    """
    # Use StringIO to write CSV to a string instead of a file
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def write_calibration_results_csv(
    calibrant_data: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Write calibration results to a CSV file.
    
    This creates a human-readable CSV with all the fitting results.
    
    Args:
        calibrant_data: DataFrame with calibration results
        output_path: Where to save the CSV file
        
    Example:
        >>> write_calibration_results_csv(results_df, Path("results.csv"))
    """
    calibrant_data.to_csv(output_path, index=False)


def generate_zip_archive(sample_paths: Dict[str, Path]) -> io.BytesIO:
    """
    Create a ZIP archive containing .dat files from multiple sample folders.
    
    This function is used to bundle processed data files for download. It searches
    each sample folder for .dat files and packages them into a single ZIP file
    with a folder structure that preserves the sample organization.
    
    Args:
        sample_paths: Dictionary mapping sample names to their folder paths
        
    Returns:
        BytesIO buffer containing the ZIP file data (ready for download)
        
    Example:
        >>> sample_paths = {
        ...     'myoglobin': Path('temp/myoglobin'),
        ...     'ubiquitin': Path('temp/ubiquitin')
        ... }
        >>> zip_buffer = generate_zip_archive(sample_paths)
        >>> # Can now use zip_buffer.getvalue() for downloads
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for sample_name, sample_path in sample_paths.items():
            # Find all .dat files in this sample folder
            dat_files = list(Path(sample_path).glob('*.dat'))
            
            for dat_file in dat_files:
                # Archive path: sample_name/filename.dat
                archive_path = f"{sample_name}/{dat_file.name}"
                zipf.write(dat_file, arcname=archive_path)
    
    # Reset buffer position to beginning for reading
    zip_buffer.seek(0)
    return zip_buffer


def dataframe_to_csv_buffer(df: pd.DataFrame) -> io.BytesIO:
    """
    Convert a DataFrame to a BytesIO buffer containing CSV data.
    
    This is useful for creating downloadable CSV files in web apps
    without writing to disk.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        BytesIO buffer containing the CSV data (ready for download)
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> buffer = dataframe_to_csv_buffer(df)
        >>> # Can use buffer for download buttons or file operations
    """
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer
