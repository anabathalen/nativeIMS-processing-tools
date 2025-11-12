"""Range file generation for TWIMExtract.

This module provides functionality to generate range files for TWIMExtract
analysis based on protein mass and charge states.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import io
import zipfile


@dataclass
class RangeParameters:
    """Parameters for range file generation.
    
    Attributes:
        mass: Protein molecular mass in Daltons
        mz_range_size: Total m/z window size
        charge_range: Tuple of (min_charge, max_charge)
        rt_start: Retention time start in minutes
        rt_end: Retention time end in minutes
        dt_start: Drift time start in bins
        dt_end: Drift time end in bins
        folder_name: Name for output folder
    """
    mass: float
    mz_range_size: float
    charge_range: Tuple[int, int]
    rt_start: float
    rt_end: float
    dt_start: int
    dt_end: int
    folder_name: str


@dataclass
class RangeFileResult:
    """Results from range file generation.
    
    Attributes:
        generated_files: List of generated filenames
        charge_states: List of charge states
        mz_values: Dictionary mapping charge state to m/z value
    """
    generated_files: List[str]
    charge_states: List[int]
    mz_values: Dict[int, float]


class RangeFileGenerator:
    """Generate TWIMExtract range files from protein parameters."""
    
    def __init__(self, params: RangeParameters):
        """Initialize generator with parameters.
        
        Args:
            params: RangeParameters object
        """
        self.params = params
    
    def calculate_mz(self, charge: int) -> float:
        """Calculate m/z for given charge state.
        
        Uses the formula: m/z = (mass + charge) / charge
        
        Args:
            charge: Charge state
            
        Returns:
            Calculated m/z value
        """
        return (self.params.mass + charge) / charge
    
    def generate_range_content(self, charge: int) -> str:
        """Generate content for a single range file.
        
        Args:
            charge: Charge state for this range file
            
        Returns:
            String content for the range file
        """
        mz = self.calculate_mz(charge)
        half_range = self.params.mz_range_size / 2.0
        
        mz_start = mz - half_range
        mz_end = mz + half_range
        
        content = f"""MZ_start: {mz_start:.1f}
MZ_end: {mz_end:.1f}
RT_start_(minutes): {self.params.rt_start:.1f}
RT_end_(minutes): {self.params.rt_end:.1f}
DT_start_(bins): {self.params.dt_start}
DT_end_(bins): {self.params.dt_end}"""
        
        return content
    
    def generate_all_files(self, output_dir: str) -> RangeFileResult:
        """Generate all range files for the specified charge range.
        
        Args:
            output_dir: Directory to write range files to
            
        Returns:
            RangeFileResult containing generated file information
        """
        generated_files = []
        charge_states = []
        mz_values = {}
        
        min_charge, max_charge = self.params.charge_range
        
        for charge in range(min_charge, max_charge + 1):
            filename = f"range_{charge}.txt"
            filepath = os.path.join(output_dir, filename)
            
            content = self.generate_range_content(charge)
            mz = self.calculate_mz(charge)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            generated_files.append(filename)
            charge_states.append(charge)
            mz_values[charge] = mz
        
        return RangeFileResult(generated_files, charge_states, mz_values)
    
    def generate_preview_data(self, max_preview: int = 5) -> List[Dict[str, str]]:
        """Generate preview data for charge states and m/z values.
        
        Args:
            max_preview: Maximum number of charge states to preview
            
        Returns:
            List of dictionaries with preview information
        """
        preview_data = []
        min_charge, max_charge = self.params.charge_range
        
        for charge in range(min_charge, min(max_charge + 1, min_charge + max_preview)):
            mz = self.calculate_mz(charge)
            half_range = self.params.mz_range_size / 2.0
            preview_data.append({
                "Charge": f"{charge}+",
                "m/z": f"{mz:.2f}",
                "Range": f"{mz - half_range:.1f} - {mz + half_range:.1f}"
            })
        
        if max_charge - min_charge >= max_preview:
            preview_data.append({
                "Charge": "...",
                "m/z": "...",
                "Range": f"(+{max_charge - min_charge + 1 - max_preview} more)"
            })
        
        return preview_data


class RangeFilePackager:
    """Package range files into downloadable formats."""
    
    @staticmethod
    def create_zip(output_dir: str, result: RangeFileResult, folder_name: str) -> io.BytesIO:
        """Create ZIP file containing all range files.
        
        Args:
            output_dir: Directory containing the range files
            result: RangeFileResult with file information
            folder_name: Name for folder inside ZIP
            
        Returns:
            BytesIO buffer containing ZIP file
        """
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for filename in result.generated_files:
                filepath = os.path.join(output_dir, filename)
                # Add files to a folder inside the ZIP
                zipf.write(filepath, os.path.join(folder_name, filename))
        
        zip_buffer.seek(0)
        return zip_buffer
    
    @staticmethod
    def get_zip_filename(folder_name: str) -> str:
        """Generate appropriate ZIP filename.
        
        Args:
            folder_name: Base folder name
            
        Returns:
            ZIP filename
        """
        return f"{folder_name}_range_files.zip"
