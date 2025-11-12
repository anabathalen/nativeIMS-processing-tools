# Core Library Extraction Plan for 1_calibrate.py

## Overview
This document outlines which components from `1_calibrate.py` should be extracted to the `nativeims` core library.

---

## 1. nativeims/io/readers.py

### Functions to Extract:
```python
def is_valid_calibrant_file(file_path: Path, file_format: str = 'auto') -> bool:
    """
    Check if file is a valid calibrant data file.
    
    Args:
        file_path: Path to the file
        file_format: 'csv', 'txt', or 'auto' for automatic detection
    
    Returns:
        True if file is valid, False otherwise
    """
    # Extract from CalibrantProcessor._is_valid_data_file()

def load_atd_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load drift time and intensity data from ATD file.
    
    Supports both .txt and .csv formats (including TWIMExtract CSV).
    
    Args:
        file_path: Path to ATD data file
    
    Returns:
        Tuple of (drift_time, intensity) arrays
    
    Raises:
        ValueError: If no valid data found
    """
    # Extract from CalibrantProcessor._load_data_from_file()

def extract_charge_state_from_filename(filename: str) -> Optional[int]:
    """
    Extract charge state from calibrant filename.
    
    Supports patterns like:
    - "24.txt" (direct charge state)
    - "range_24.txt"
    - "DT_..._range_24.txt_raw"
    
    Args:
        filename: Filename to parse
    
    Returns:
        Charge state as integer, or None if not found
    """
    # Extract regex logic from _process_single_file()
```

---

## 2. nativeims/calibration/calibrant_db.py

### Classes/Functions to Extract:
```python
# Calibrant naming conventions
CALIBRANT_FOLDER_MAPPING = {
    'Denatured Myoglobin': 'myoglobin',
    'Denatured Cytochrome C': 'cytochromec',
    'Polyalanine Peptide of Length X': 'polyalanineX',
    'Denatured Ubiquitin': 'ubiquitin',
    'Native BSA': 'BSA',
    'GRGDS Peptide': 'GRGDS',
    'SDGRG Peptide': 'SDGRG'
}

class CalibrantDatabase:
    """Interface to Bush calibrant database."""
    
    def __init__(self, bush_df: pd.DataFrame):
        """Initialize with Bush database DataFrame."""
    
    def get_calibrant_column(self, gas_type: str) -> str:
        """Get column name for CCS values based on drift gas."""
    
    def lookup_calibrant(
        self, 
        protein: str, 
        charge_state: int, 
        gas_type: str
    ) -> Optional[Dict[str, float]]:
        """
        Look up calibrant CCS and mass values.
        
        Returns:
            Dict with 'ccs', 'mass' keys, or None if not found
        """

def load_bush_database(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the Bush calibrant database from CSV."""
```

---

## 3. nativeims/calibration/gaussian_fitting.py

### Classes/Functions to Extract:
```python
@dataclass
class GaussianFitResult:
    """Results from Gaussian fitting of ATD."""
    amplitude: float
    apex: float
    std_dev: float
    r_squared: float
    fitted_values: np.ndarray
    drift_time: np.ndarray
    intensity: np.ndarray

@dataclass
class CalibrantMeasurement:
    """Single calibrant measurement result."""
    protein: str
    mass: float
    charge_state: int
    drift_time: float
    r_squared: float
    ccs_literature: float
    fit_result: GaussianFitResult

class CalibrantProcessor:
    """Process calibrant ATD files and extract drift times."""
    
    def __init__(
        self, 
        calibrant_db: CalibrantDatabase,
        min_r2: float = 0.9
    ):
        """
        Initialize processor.
        
        Args:
            calibrant_db: CalibrantDatabase instance
            min_r2: Minimum R² threshold for accepting fits
        """
    
    def process_file(
        self,
        file_path: Path,
        protein_name: str,
        gas_type: str = 'helium'
    ) -> Optional[CalibrantMeasurement]:
        """
        Process a single calibrant ATD file.
        
        Args:
            file_path: Path to ATD file
            protein_name: Name of calibrant protein
            gas_type: Drift gas ('helium' or 'nitrogen')
        
        Returns:
            CalibrantMeasurement or None if processing failed
        """
    
    def process_folder(
        self,
        folder_path: Path,
        protein_name: str,
        gas_type: str = 'helium'
    ) -> List[CalibrantMeasurement]:
        """
        Process all ATD files in a folder.
        
        Args:
            folder_path: Path to folder containing ATD files
            protein_name: Name of calibrant protein
            gas_type: Drift gas type
        
        Returns:
            List of CalibrantMeasurement objects
        """
    
    def process_calibrant_set(
        self,
        base_path: Path,
        gas_type: str = 'helium'
    ) -> pd.DataFrame:
        """
        Process complete calibrant dataset.
        
        Expects folder structure:
        base_path/
            myoglobin/
                24.txt
                25.txt
            cytochromec/
                ...
        
        Returns:
            DataFrame with all calibrant measurements
        """
```

---

## 4. nativeims/calibration/drift_to_ccs.py

### Functions to Extract:
```python
@dataclass
class InstrumentParams:
    """IMS instrument parameters."""
    wave_velocity: float  # m/s
    wave_height: float    # V
    pressure: float       # mbar
    drift_length: float   # m
    instrument_type: str  # 'cyclic' or 'synapt'
    inject_time: float = 0.0  # ms (for cyclic only)

def adjust_drift_time_for_injection(
    drift_time: float,
    inject_time: float,
    instrument_type: str
) -> float:
    """
    Adjust drift time for injection time (Cyclic IMS only).
    
    Args:
        drift_time: Measured drift time (ms)
        inject_time: Injection time (ms)
        instrument_type: 'cyclic' or 'synapt'
    
    Returns:
        Adjusted drift time
    """

def create_imscal_reference_file(
    calibrant_data: pd.DataFrame,
    instrument_params: InstrumentParams,
    output_path: Path
) -> None:
    """
    Create .dat reference file for IMSCal.
    
    Args:
        calibrant_data: DataFrame with calibrant measurements
        instrument_params: Instrument parameters
        output_path: Where to save .dat file
    """
```

---

## 5. nativeims/io/writers.py

### Functions to Extract:
```python
def write_imscal_dat(
    calibrant_data: pd.DataFrame,
    instrument_params: InstrumentParams,
    output_path: Path
) -> None:
    """Write calibration data in IMSCal .dat format."""

def dataframe_to_csv_string(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string for download."""
```

---

## 6. Keep in Streamlit (1_calibrate.py)

### UI-Specific Components:
- `CalibrationParams` dataclass (UI-specific configuration)
- `ProcessingResult` dataclass (UI-specific result container)
- `ResultsDisplayer` class (all visualization methods)
- `FileGenerator.create_download_buttons()` (Streamlit-specific)
- `UIComponents` class (all UI input methods)
- `main()` function (Streamlit app logic)
- Custom CSS and styling

---

## Refactored Streamlit App Structure

After extraction, `1_calibrate.py` becomes a thin wrapper:

```python
import streamlit as st
from pathlib import Path
from nativeims.calibration.calibrant_db import CalibrantDatabase, load_bush_database
from nativeims.calibration.gaussian_fitting import CalibrantProcessor
from nativeims.calibration.drift_to_ccs import InstrumentParams, create_imscal_reference_file
from nativeims.io.writers import write_imscal_dat

def main():
    # Load database
    bush_df = load_bush_database()
    calibrant_db = CalibrantDatabase(bush_df)
    
    # Get user inputs (UI logic)
    params = get_user_parameters()  # Streamlit-specific
    uploaded_files = get_uploaded_files()  # Streamlit-specific
    
    # Core processing (library calls)
    processor = CalibrantProcessor(calibrant_db, min_r2=params.min_r2)
    results = processor.process_calibrant_set(
        uploaded_files, 
        gas_type=params.gas_type
    )
    
    # Display results (UI logic)
    display_results(results)  # Streamlit-specific
    create_download_buttons(results, params)  # Streamlit-specific
```

---

## Migration Checklist

### Phase 1: Core IO
- [ ] Create `nativeims/io/readers.py`
- [ ] Extract file validation logic
- [ ] Extract data loading logic
- [ ] Extract charge state parsing
- [ ] Write unit tests for IO functions

### Phase 2: Database
- [ ] Create `nativeims/calibration/calibrant_db.py`
- [ ] Extract Bush database loading
- [ ] Extract lookup logic
- [ ] Add database validation
- [ ] Write unit tests

### Phase 3: Fitting
- [ ] Create `nativeims/calibration/gaussian_fitting.py`
- [ ] Extract CalibrantProcessor (refactored)
- [ ] Create dataclasses for results
- [ ] Separate fitting from UI logic
- [ ] Write unit tests with fixtures

### Phase 4: Calibration
- [ ] Create `nativeims/calibration/drift_to_ccs.py`
- [ ] Extract drift time adjustment
- [ ] Extract .dat file generation
- [ ] Write unit tests

### Phase 5: Writers
- [ ] Create `nativeims/io/writers.py`
- [ ] Extract file writing logic
- [ ] Write unit tests

### Phase 6: Integration
- [ ] Refactor `1_calibrate.py` to use new library
- [ ] Update imports
- [ ] Test end-to-end workflow
- [ ] Update documentation

---

## Benefits of This Structure

1. **Reusability**: Core functions can be used in CLI, notebooks, or other apps
2. **Testability**: Each module can be tested independently
3. **Maintainability**: Clear separation of concerns
4. **Documentation**: API reference can be auto-generated
5. **Performance**: Core library can be optimized without touching UI
6. **Distribution**: Can be pip-installed separately from Streamlit app

---

## Example Usage (Post-Refactor)

### In Python Script:
```python
from nativeims.calibration import CalibrantProcessor, load_bush_database
from nativeims.calibration import InstrumentParams

# Load database
db = load_bush_database()

# Process calibrants
processor = CalibrantProcessor(db, min_r2=0.9)
results = processor.process_folder('data/myoglobin', 'myoglobin')

# Create reference file
params = InstrumentParams(
    wave_velocity=281.0,
    wave_height=20.0,
    pressure=1.63,
    drift_length=0.98,
    instrument_type='cyclic'
)
create_imscal_reference_file(results, params, 'output.dat')
```

### In Jupyter Notebook:
```python
import pandas as pd
from nativeims.calibration import CalibrantProcessor

# Interactive analysis
processor = CalibrantProcessor(db)
results = processor.process_file('myoglobin_24.txt', 'myoglobin')

# Visualize fit
import matplotlib.pyplot as plt
plt.plot(results.fit_result.drift_time, results.fit_result.intensity)
plt.plot(results.fit_result.drift_time, results.fit_result.fitted_values)
```

---

## Notes

- Keep all Streamlit-specific UI code in the `streamlit_app/pages/` files
- Core library should have NO Streamlit dependencies
- Use dataclasses for structured data passing
- Add comprehensive docstrings following NumPy style
- Include type hints for all public APIs
