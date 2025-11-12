"""
IMSpartacus - Core library for processing native ion mobility mass spectrometry data.

This library provides tools for:
- Calibrant data processing and Gaussian fitting
- CCS (collision cross section) calculations
- Data visualization and analysis
- File I/O for various IMS data formats
- Input file generation for IMSCal software
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Make key classes/functions available at package level
from imspartacus.calibration.processor import CalibrantProcessor
from imspartacus.calibration.database import CalibrantDatabase, load_bush_database
from imspartacus.io.readers import load_atd_data, is_valid_calibrant_file
from imspartacus.io.writers import generate_zip_archive
from imspartacus.extraction.input_generator import InputProcessor, InputParams, InputProcessingResult
from imspartacus.extraction.output_processor import OutputFileProcessor, ProteinOutput, OutputProcessingResult
from imspartacus.visualization import CCSDData, GaussianFitData, PlotSettings, CCSDPlotter

__all__ = [
    "CalibrantProcessor",
    "CalibrantDatabase", 
    "load_bush_database",
    "load_atd_data",
    "is_valid_calibrant_file",
    "generate_zip_archive",
    "InputProcessor",
    "InputParams",
    "InputProcessingResult",
    "OutputFileProcessor",
    "ProteinOutput",
    "OutputProcessingResult",
    "CCSDData",
    "GaussianFitData",
    "PlotSettings",
    "CCSDPlotter",
]
