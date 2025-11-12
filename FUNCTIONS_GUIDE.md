# Core Library Functions - Complete Guide for Beginners

This document explains **every function** you need to understand in the new `nativeims` library.

---

## 📁 File Structure

```
nativeims/
├── __init__.py                    # Package entry point
├── io/                            # File reading/writing
│   ├── __init__.py
│   ├── readers.py                 # Load data from files
│   └── writers.py                 # Save data to files
├── calibration/                   # Main calibration logic
│   ├── __init__.py
│   ├── database.py                # Bush database interface
│   ├── processor.py               # Gaussian fitting & processing
│   └── utils.py                   # Helper functions
└── utils/                         # General utilities
    └── (future modules)
```

---

## 📖 Module 1: `nativeims/io/readers.py`

**Purpose**: Load ATD data from text or CSV files

### Function 1: `is_valid_calibrant_file(file_path)`

**What it does**: Checks if a file is valid calibrant data

**Inputs**:
- `file_path`: A Path object pointing to a file

**Outputs**:
- `True` if file is valid, `False` otherwise

**How it works**:
1. Checks file extension (.txt or .csv)
2. For .txt: filename must start with a number
3. For .csv: filename must contain a charge state pattern

**Example**:
```python
from pathlib import Path
from nativeims.io import is_valid_calibrant_file

file = Path("24.txt")
if is_valid_calibrant_file(file):
    print("This is a valid file!")
```

---

### Function 2: `extract_charge_state_from_filename(filename)`

**What it does**: Finds the charge state number in a filename

**Inputs**:
- `filename`: A string with the filename (like "24.txt" or "range_24.csv")

**Outputs**:
- An integer (the charge state), or `None` if not found

**How it works**:
1. Tries to convert the whole filename to a number (for simple cases like "24.txt")
2. If that fails, uses regex patterns to find numbers like "range_24"

**Example**:
```python
from nativeims.io import extract_charge_state_from_filename

charge = extract_charge_state_from_filename("24.txt")
print(charge)  # Output: 24

charge = extract_charge_state_from_filename("DT_sample_range_18.csv")
print(charge)  # Output: 18
```

---

### Function 3: `load_atd_data(file_path)`

**What it does**: Loads drift time and intensity data from a file

**Inputs**:
- `file_path`: Path to the ATD file

**Outputs**:
- A tuple: `(drift_time_array, intensity_array)`
- Both are numpy arrays with the same length

**How it works**:
1. Checks file extension
2. For .csv: reads line by line, skips comments (#), parses comma-separated values
3. For .txt: uses `np.loadtxt()` to read space-separated data
4. Returns two arrays

**Example**:
```python
from pathlib import Path
from nativeims.io import load_atd_data

drift_time, intensity = load_atd_data(Path("24.txt"))
print(f"Loaded {len(drift_time)} data points")
print(f"First drift time: {drift_time[0]} ms")
print(f"First intensity: {intensity[0]}")
```

---

### Function 4: `load_multiple_atd_files(folder_path)`

**What it does**: Loads ALL valid ATD files in a folder

**Inputs**:
- `folder_path`: Path to a folder containing ATD files

**Outputs**:
- A dictionary: `{charge_state: (drift_time, intensity), ...}`

**How it works**:
1. Loops through all files in the folder
2. Checks if each file is valid
3. Extracts charge state
4. Loads the data
5. Stores in dictionary with charge state as key

**Example**:
```python
from pathlib import Path
from nativeims.io import load_multiple_atd_files

data = load_multiple_atd_files(Path("myoglobin"))
print(f"Found charge states: {list(data.keys())}")

# Access data for charge state 24
drift_24, intensity_24 = data[24]
```

---

## 📖 Module 2: `nativeims/io/writers.py`

**Purpose**: Save processed data to files

### Function 1: `write_imscal_dat(calibrant_data, velocity, voltage, pressure, length, output_path)`

**What it does**: Creates a .dat file for IMSCal software

**Inputs**:
- `calibrant_data`: A DataFrame with your results
- `velocity`: Wave velocity (m/s)
- `voltage`: Wave height (V)
- `pressure`: IMS pressure (mbar)
- `length`: Drift length (m)
- `output_path`: Where to save the file (or None to just return string)

**Outputs**:
- A string containing the file contents
- Also writes to file if `output_path` is provided

**How it works**:
1. Creates header with instrument parameters
2. Loops through DataFrame rows
3. Formats each row as: `protein_charge mass charge CCS drift_time`
4. Converts CCS from nm² to Ų (multiply by 100)

**Example**:
```python
from pathlib import Path
import pandas as pd
from nativeims.io.writers import write_imscal_dat

# Your results
results = pd.DataFrame({
    'protein': ['myoglobin'],
    'charge state': [24],
    'mass': [16952.3],
    'calibrant_value': [31.2],  # CCS in nm²
    'drift time': [5.23]
})

# Write .dat file
write_imscal_dat(
    results,
    velocity=281.0,
    voltage=20.0,
    pressure=1.63,
    length=0.98,
    output_path=Path("output.dat")
)
```

---

### Function 2: `dataframe_to_csv_string(df)`

**What it does**: Converts a DataFrame to CSV text

**Inputs**:
- `df`: A pandas DataFrame

**Outputs**:
- A string in CSV format

**How it works**:
- Uses `StringIO` to create CSV in memory (not on disk)

**Example**:
```python
import pandas as pd
from nativeims.io.writers import dataframe_to_csv_string

df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
csv_text = dataframe_to_csv_string(df)
print(csv_text)
# Output:
# A,B
# 1,3
# 2,4
```

---

### Function 3: `write_calibration_results_csv(calibrant_data, output_path)`

**What it does**: Saves results to a CSV file

**Inputs**:
- `calibrant_data`: DataFrame with results
- `output_path`: Where to save

**Outputs**:
- None (writes file to disk)

**How it works**:
- Simple wrapper around `DataFrame.to_csv()`

**Example**:
```python
from pathlib import Path
from nativeims.io.writers import write_calibration_results_csv

write_calibration_results_csv(results_df, Path("results.csv"))
```

---

## 📖 Module 3: `nativeims/calibration/database.py`

**Purpose**: Interface to the Bush CCS database

### Variable: `CALIBRANT_FOLDER_MAPPING`

**What it is**: A dictionary mapping protein names to folder names

**Example**:
```python
from nativeims.calibration import CALIBRANT_FOLDER_MAPPING

print(CALIBRANT_FOLDER_MAPPING['Denatured Myoglobin'])
# Output: 'myoglobin'
```

---

### Function: `load_bush_database(file_path)`

**What it does**: Loads the Bush database CSV file

**Inputs**:
- `file_path`: Path to bush.csv (optional - will search for it if None)

**Outputs**:
- A pandas DataFrame with columns:
  - `protein`: protein name
  - `charge`: charge state
  - `mass`: mass in Da
  - `CCS_he`: CCS in helium (nm²)
  - `CCS_n2`: CCS in nitrogen (nm²)

**How it works**:
1. If no path provided, searches common locations
2. Reads CSV file
3. Returns DataFrame

**Example**:
```python
from pathlib import Path
from nativeims.calibration import load_bush_database

bush_df = load_bush_database(Path("data/bush.csv"))
print(bush_df.head())
```

---

### Class: `CalibrantDatabase`

**What it is**: A class that helps you query the Bush database

**How to create**:
```python
bush_df = load_bush_database()
db = CalibrantDatabase(bush_df)
```

#### Method 1: `get_calibrant_column(gas_type)`

**What it does**: Returns the column name for a gas type

**Inputs**:
- `gas_type`: "helium" or "nitrogen"

**Outputs**:
- "CCS_he" or "CCS_n2"

**Example**:
```python
column = db.get_calibrant_column('helium')
print(column)  # Output: CCS_he
```

---

#### Method 2: `lookup_calibrant(protein, charge_state, gas_type)`

**What it does**: Looks up CCS and mass for a specific protein and charge

**Inputs**:
- `protein`: Protein name (e.g., "myoglobin")
- `charge_state`: Integer (e.g., 24)
- `gas_type`: "helium" or "nitrogen"

**Outputs**:
- A dictionary: `{'ccs': 31.2, 'mass': 16952.3}`
- Or `None` if not found

**How it works**:
1. Filters database for matching protein AND charge
2. Gets the CCS value for the specified gas
3. Returns CCS and mass

**Example**:
```python
result = db.lookup_calibrant('myoglobin', 24, 'helium')
if result:
    print(f"CCS: {result['ccs']} nm²")
    print(f"Mass: {result['mass']} Da")
```

---

#### Method 3: `get_available_charge_states(protein)`

**What it does**: Gets all charge states in database for a protein

**Inputs**:
- `protein`: Protein name

**Outputs**:
- List of integers (charge states)

**Example**:
```python
charges = db.get_available_charge_states('myoglobin')
print(charges)  # Output: [18, 19, 20, 21, 22, 23, 24, 25]
```

---

#### Method 4: `get_available_proteins()`

**What it does**: Gets all proteins in the database

**Outputs**:
- List of protein names

**Example**:
```python
proteins = db.get_available_proteins()
print(proteins)  # Output: ['BSA', 'GRGDS', 'myoglobin', ...]
```

---

## 📖 Module 4: `nativeims/calibration/processor.py`

**Purpose**: Main calibrant processing with Gaussian fitting

### Dataclass 1: `GaussianFitResult`

**What it is**: Stores results from fitting a Gaussian

**Attributes**:
- `amplitude`: Peak height
- `apex`: Peak center (the drift time we want!)
- `std_dev`: Peak width
- `r_squared`: Quality of fit (0-1, higher is better)
- `fitted_values`: The fitted curve (numpy array)
- `drift_time`: Original x-axis data
- `intensity`: Original y-axis data

**Example**:
```python
# After fitting
print(f"Drift time at peak: {fit_result.apex} ms")
print(f"Fit quality: {fit_result.r_squared}")
```

---

### Dataclass 2: `CalibrantMeasurement`

**What it is**: Complete result for one calibrant

**Attributes**:
- `protein`: Protein name
- `mass`: Mass in Da
- `charge_state`: Charge state
- `drift_time`: Measured drift time (from Gaussian apex)
- `r_squared`: Quality of fit
- `ccs_literature`: Literature CCS value from database
- `fit_result`: GaussianFitResult object (optional)
- `filename`: Original filename (optional)

**Example**:
```python
measurement = CalibrantMeasurement(
    protein='myoglobin',
    mass=16952.3,
    charge_state=24,
    drift_time=5.23,
    r_squared=0.95,
    ccs_literature=31.2
)
```

---

### Class: `CalibrantProcessor`

**What it is**: The main class that processes calibrant files

**How to create**:
```python
from nativeims.calibration import CalibrantDatabase, CalibrantProcessor, load_bush_database

bush_df = load_bush_database()
db = CalibrantDatabase(bush_df)
processor = CalibrantProcessor(db, min_r2=0.9)
```

**Parameters**:
- `calibrant_db`: A CalibrantDatabase object
- `min_r2`: Minimum R² to accept a fit (default 0.9)
- `fitting_function`: Custom fitting function (optional)

---

#### Method 1: `process_file(file_path, protein_name, gas_type)`

**What it does**: Process ONE calibrant file

**Inputs**:
- `file_path`: Path to the file
- `protein_name`: Name like "myoglobin"
- `gas_type`: "helium" or "nitrogen"

**Outputs**:
- A `CalibrantMeasurement` object
- Or `None` if processing failed

**How it works**:
1. Extracts charge state from filename
2. Loads the data
3. Fits a Gaussian
4. Looks up literature CCS
5. Creates CalibrantMeasurement object

**Example**:
```python
result = processor.process_file(
    Path("myoglobin/24.txt"),
    "myoglobin",
    "helium"
)

if result:
    print(f"Drift time: {result.drift_time} ms")
    print(f"R²: {result.r_squared}")
```

---

#### Method 2: `process_folder(folder_path, protein_name, gas_type)`

**What it does**: Process ALL files in a folder

**Inputs**:
- `folder_path`: Path to folder
- `protein_name`: Protein name
- `gas_type`: Gas type

**Outputs**:
- A tuple: `(measurements, skipped)`
  - `measurements`: List of CalibrantMeasurement objects
  - `skipped`: List of strings (filenames that failed)

**How it works**:
1. Loops through all files
2. Processes each valid file
3. Separates successful vs. failed

**Example**:
```python
measurements, skipped = processor.process_folder(
    Path("myoglobin"),
    "myoglobin",
    "helium"
)

print(f"Success: {len(measurements)}")
print(f"Failed: {len(skipped)}")
```

---

#### Method 3: `process_calibrant_set(base_path, gas_type)`

**What it does**: Process MULTIPLE protein folders

**Inputs**:
- `base_path`: Path to folder containing protein subfolders
- `gas_type`: Gas type

**Outputs**:
- A pandas DataFrame with all results

**Folder structure expected**:
```
base_path/
  myoglobin/
    24.txt
    25.txt
  cytochromec/
    18.txt
    19.txt
```

**How it works**:
1. Loops through subfolders
2. Uses folder name as protein name
3. Processes all files in each folder
4. Combines into one DataFrame

**Example**:
```python
results_df = processor.process_calibrant_set(
    Path("calibrants"),
    "helium"
)

print(results_df)
```

---

### Function: `measurements_to_dataframe(measurements)`

**What it does**: Converts a list of CalibrantMeasurement to DataFrame

**Inputs**:
- `measurements`: List of CalibrantMeasurement objects

**Outputs**:
- pandas DataFrame

**Example**:
```python
from nativeims.calibration import measurements_to_dataframe

df = measurements_to_dataframe(measurements)
df.to_csv("results.csv")
```

---

## 📖 Module 5: `nativeims/calibration/utils.py`

**Purpose**: Helper functions for calibration

### Dataclass: `InstrumentParams`

**What it is**: Stores instrument parameters

**Attributes**:
- `wave_velocity`: m/s
- `wave_height`: V
- `pressure`: mbar
- `drift_length`: m
- `instrument_type`: "cyclic" or "synapt"
- `inject_time`: ms (for cyclic only)

**Example**:
```python
from nativeims.calibration import InstrumentParams

params = InstrumentParams(
    wave_velocity=281.0,
    wave_height=20.0,
    pressure=1.63,
    drift_length=0.98,
    instrument_type='cyclic',
    inject_time=0.3
)
```

---

### Function 1: `adjust_drift_time_for_injection(drift_time, inject_time, instrument_type)`

**What it does**: Subtracts injection time for Cyclic IMS

**Inputs**:
- `drift_time`: Measured time (ms)
- `inject_time`: Injection time (ms)
- `instrument_type`: "cyclic" or "synapt"

**Outputs**:
- Adjusted drift time (ms)

**How it works**:
- If cyclic: returns `drift_time - inject_time`
- If synapt: returns `drift_time` (no change)

**Example**:
```python
from nativeims.calibration import adjust_drift_time_for_injection

adjusted = adjust_drift_time_for_injection(5.5, 0.3, 'cyclic')
print(adjusted)  # Output: 5.2
```

---

### Function 2: `adjust_dataframe_drift_times(df, instrument_params)`

**What it does**: Adjusts ALL drift times in a DataFrame

**Inputs**:
- `df`: DataFrame with 'drift time' column
- `instrument_params`: InstrumentParams object

**Outputs**:
- New DataFrame with adjusted drift times

**How it works**:
1. Makes a copy of the DataFrame
2. Applies `adjust_drift_time_for_injection` to each row
3. Returns the copy

**Example**:
```python
from nativeims.calibration import adjust_dataframe_drift_times, InstrumentParams

params = InstrumentParams(
    wave_velocity=281.0,
    wave_height=20.0,
    pressure=1.63,
    drift_length=0.98,
    instrument_type='cyclic',
    inject_time=0.3
)

adjusted_df = adjust_dataframe_drift_times(results_df, params)
```

---

## 🎯 Complete Workflow Example

Here's how all the functions work together:

```python
from pathlib import Path
from nativeims.calibration import (
    load_bush_database,
    CalibrantDatabase,
    CalibrantProcessor,
    InstrumentParams,
    adjust_dataframe_drift_times
)
from nativeims.io.writers import write_imscal_dat, write_calibration_results_csv

# Step 1: Load Bush database
bush_df = load_bush_database(Path("data/bush.csv"))
db = CalibrantDatabase(bush_df)

# Step 2: Create processor
processor = CalibrantProcessor(db, min_r2=0.9)

# Step 3: Process all calibrants
results_df = processor.process_calibrant_set(
    Path("calibrants"),
    gas_type="helium"
)

# Step 4: Define instrument parameters
params = InstrumentParams(
    wave_velocity=281.0,
    wave_height=20.0,
    pressure=1.63,
    drift_length=0.98,
    instrument_type='cyclic',
    inject_time=0.3
)

# Step 5: Adjust drift times
adjusted_df = adjust_dataframe_drift_times(results_df, params)

# Step 6: Save results
write_calibration_results_csv(adjusted_df, Path("results.csv"))
write_imscal_dat(
    adjusted_df,
    velocity=params.wave_velocity,
    voltage=params.wave_height,
    pressure=params.pressure,
    length=params.drift_length,
    output_path=Path("calibration.dat")
)

print("Done!")
```

---

## 🔄 How This Relates to Your Streamlit App

Your Streamlit app (`1_calibrate.py`) will become **much simpler**:

### Old way (everything in Streamlit):
- File validation logic
- Data loading logic
- Gaussian fitting logic
- Database queries
- File writing logic
- UI display logic

### New way (thin wrapper):
```python
import streamlit as st
from nativeims.calibration import (
    load_bush_database,
    CalibrantDatabase,
    CalibrantProcessor
)

def main():
    st.title("Calibration Tool")
    
    # Load database
    bush_df = load_bush_database()
    db = CalibrantDatabase(bush_df)
    
    # Get user inputs
    uploaded_file = st.file_uploader("Upload ZIP")
    min_r2 = st.number_input("Min R²", value=0.9)
    
    # Process using library
    processor = CalibrantProcessor(db, min_r2=min_r2)
    results = processor.process_calibrant_set(uploaded_file)
    
    # Display results
    st.dataframe(results)
```

**Much cleaner!** The core logic is in the library, Streamlit just handles UI.

---

## ✅ Summary

You now have **23 functions** organized into **5 modules**:

1. **readers.py** (4 functions): Load data from files
2. **writers.py** (3 functions): Save data to files
3. **database.py** (1 function + 1 class with 4 methods): Bush database
4. **processor.py** (2 dataclasses + 1 class with 3 methods + 1 function): Main processing
5. **utils.py** (1 dataclass + 2 functions): Helper utilities

Each function has:
- ✅ Clear purpose
- ✅ Type hints
- ✅ Docstrings
- ✅ Examples
- ✅ Error handling
