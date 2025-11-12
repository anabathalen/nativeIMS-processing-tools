# 📚 BEGINNER'S COMPLETE GUIDE

## What Just Happened?

I've created a **complete core library** by extracting all the scientific logic from your `1_calibrate.py` file. Let me explain everything step by step.

---

## 🗂️ Files Created

Here's what I built for you:

### 1. Core Library Files (The Science)

```
nativeims/
├── __init__.py                           ← Package setup
├── README.md                             ← Library documentation
│
├── io/                                   ← File reading/writing
│   ├── __init__.py
│   ├── readers.py                        ← 4 functions to load data
│   └── writers.py                        ← 3 functions to save data
│
├── calibration/                          ← Main calibration logic
│   ├── __init__.py
│   ├── database.py                       ← Bush database (5 functions)
│   ├── processor.py                      ← Gaussian fitting (6 functions)
│   └── utils.py                          ← Helper functions (3 functions)
│
└── utils/                                ← General utilities (future)
```

**Total: 23 functions** organized into **5 modules**

---

### 2. Documentation Files (Learning Resources)

```
EXTRACTION_PLAN.md          ← What to extract and why
FUNCTIONS_GUIDE.md          ← DETAILED guide to EVERY function (for beginners!)
VISUAL_SUMMARY.md           ← Visual diagrams and flowcharts
QUICKSTART.md               ← Get started quickly
USAGE_EXAMPLES.py           ← Working code examples
simple_tests.py             ← Test everything works
```

---

## 📖 What Each File Does

### Core Library Files

#### `nativeims/io/readers.py`
**Purpose**: Load data from files

**What's inside**:
1. `is_valid_calibrant_file()` - Checks if a file is valid
2. `extract_charge_state_from_filename()` - Gets charge state from filename
3. `load_atd_data()` - Loads drift time and intensity
4. `load_multiple_atd_files()` - Loads entire folder

**Example**:
```python
from nativeims.io import load_atd_data

drift_time, intensity = load_atd_data(Path("24.txt"))
```

---

#### `nativeims/io/writers.py`
**Purpose**: Save results to files

**What's inside**:
1. `write_imscal_dat()` - Creates .dat file for IMSCal
2. `dataframe_to_csv_string()` - Converts DataFrame to CSV text
3. `write_calibration_results_csv()` - Saves CSV file

**Example**:
```python
from nativeims.io.writers import write_imscal_dat

write_imscal_dat(results, velocity=281.0, voltage=20.0, 
                 pressure=1.63, length=0.98, 
                 output_path=Path("output.dat"))
```

---

#### `nativeims/calibration/database.py`
**Purpose**: Interface to Bush calibrant database

**What's inside**:
1. `CALIBRANT_FOLDER_MAPPING` - Dictionary of protein names
2. `load_bush_database()` - Loads bush.csv file
3. `CalibrantDatabase` class:
   - `get_calibrant_column()` - Gets column name for gas type
   - `lookup_calibrant()` - Finds CCS value for protein/charge
   - `get_available_charge_states()` - Lists available charges
   - `get_available_proteins()` - Lists all proteins

**Example**:
```python
from nativeims.calibration import load_bush_database, CalibrantDatabase

bush_df = load_bush_database(Path("data/bush.csv"))
db = CalibrantDatabase(bush_df)

result = db.lookup_calibrant('myoglobin', 24, 'helium')
print(f"CCS: {result['ccs']} nm²")
```

---

#### `nativeims/calibration/processor.py`
**Purpose**: Main processing with Gaussian fitting

**What's inside**:
1. `GaussianFitResult` - Dataclass for fit results
2. `CalibrantMeasurement` - Dataclass for measurements
3. `CalibrantProcessor` class:
   - `process_file()` - Process one file
   - `process_folder()` - Process all files in folder
   - `process_calibrant_set()` - Process multiple proteins
4. `measurements_to_dataframe()` - Convert to DataFrame

**Example**:
```python
from nativeims.calibration import CalibrantProcessor

processor = CalibrantProcessor(db, min_r2=0.9)
result = processor.process_file(Path("24.txt"), "myoglobin", "helium")

print(f"Drift time: {result.drift_time} ms")
print(f"R²: {result.r_squared}")
```

---

#### `nativeims/calibration/utils.py`
**Purpose**: Helper functions

**What's inside**:
1. `InstrumentParams` - Dataclass for instrument settings
2. `adjust_drift_time_for_injection()` - Adjusts one drift time
3. `adjust_dataframe_drift_times()` - Adjusts all drift times

**Example**:
```python
from nativeims.calibration import InstrumentParams, adjust_dataframe_drift_times

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

## 🎓 Understanding the Code (Beginner Concepts)

### What's a "module"?
A module is just a Python file. When you write `from nativeims.io import readers`, you're importing the `readers.py` file.

### What's a "package"?
A package is a folder containing modules. The `nativeims` folder is a package because it has `__init__.py` inside.

### What's a "class"?
A class is like a template for creating objects. Think of it as a blueprint:

```python
# Define the class (blueprint)
class CalibrantDatabase:
    def __init__(self, bush_df):
        self.df = bush_df
    
    def lookup_calibrant(self, protein, charge):
        # ... code ...

# Create an object from the class (build from blueprint)
db = CalibrantDatabase(bush_df)

# Use the object's methods
result = db.lookup_calibrant('myoglobin', 24)
```

### What's a "dataclass"?
A dataclass is a simple way to create classes that just store data:

```python
@dataclass
class CalibrantMeasurement:
    protein: str
    charge_state: int
    drift_time: float

# Create one
m = CalibrantMeasurement('myoglobin', 24, 5.23)

# Access attributes
print(m.protein)       # 'myoglobin'
print(m.drift_time)    # 5.23
```

### What are "type hints"?
Type hints tell you what type each variable should be:

```python
def load_atd_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    #                       ^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                       input type  output type
```

This helps:
- Your editor autocomplete
- Find bugs before running code
- Understand what the function expects

### What's `Optional`?
`Optional[X]` means "either X or None":

```python
def process_file(...) -> Optional[CalibrantMeasurement]:
    if successful:
        return CalibrantMeasurement(...)  # Returns the object
    else:
        return None  # Or returns None
```

---

## 🔍 How to Read the Code

### Step 1: Start with imports
```python
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
```
These bring in tools you need (like `Path` for file paths).

### Step 2: Look at function signature
```python
def load_atd_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    #                       ^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                       input       output
```
This tells you: "Give me a Path, I'll return two numpy arrays"

### Step 3: Read the docstring
```python
    """
    Load drift time and intensity data from an ATD file.
    
    Args:
        file_path: Path to the ATD data file
        
    Returns:
        A tuple of two numpy arrays: (drift_time, intensity)
    """
```
This explains what it does in plain English.

### Step 4: Look at the code
```python
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1]
```
This is the actual work being done.

---

## 🚀 How to Use It

### Workflow 1: Process One File

```python
from pathlib import Path
from nativeims.calibration import (
    load_bush_database,
    CalibrantDatabase,
    CalibrantProcessor
)

# Step 1: Load database
bush_df = load_bush_database(Path("data/bush.csv"))
db = CalibrantDatabase(bush_df)

# Step 2: Create processor
processor = CalibrantProcessor(db, min_r2=0.9)

# Step 3: Process file
result = processor.process_file(
    Path("myoglobin/24.txt"),
    "myoglobin",
    "helium"
)

# Step 4: Check result
if result:
    print(f"Drift time: {result.drift_time} ms")
    print(f"CCS: {result.ccs_literature} nm²")
```

---

### Workflow 2: Process Entire Dataset

```python
from pathlib import Path
from nativeims.calibration import (
    load_bush_database,
    CalibrantDatabase,
    CalibrantProcessor,
    InstrumentParams,
    adjust_dataframe_drift_times
)
from nativeims.io.writers import write_imscal_dat

# 1. Setup
bush_df = load_bush_database(Path("data/bush.csv"))
db = CalibrantDatabase(bush_df)
processor = CalibrantProcessor(db, min_r2=0.9)

# 2. Process all calibrants
results_df = processor.process_calibrant_set(
    Path("calibrants"),  # Folder with myoglobin/, cytochromec/, etc.
    gas_type="helium"
)

# 3. Define instrument
params = InstrumentParams(
    wave_velocity=281.0,
    wave_height=20.0,
    pressure=1.63,
    drift_length=0.98,
    instrument_type='cyclic',
    inject_time=0.3
)

# 4. Adjust drift times (for Cyclic IMS)
adjusted_df = adjust_dataframe_drift_times(results_df, params)

# 5. Save results
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

## ✅ Testing Your Library

Run the tests to make sure everything works:

```bash
cd c:\Users\h87023ab\Documents\GITHUB\PROCESSING_TOOLS\nativeIMS-processing-tools
python simple_tests.py
```

You should see:
```
Test 1: Checking imports...
  ✓ IO module imported successfully
  ✓ Calibration module imported successfully
✅ All imports successful!

Test 2: Testing filename parsing...
  ✓ '24.txt' -> 24
  ✓ 'range_18.csv' -> 18
✅ Filename parsing works!

...

Results: 7/7 tests passed
🎉 All tests passed! Library is working correctly.
```

---

## 📚 Learning Path

### Level 1: Understanding
1. Read **QUICKSTART.md** - Get overview
2. Read **VISUAL_SUMMARY.md** - See diagrams
3. Read this file - Understand concepts

### Level 2: Usage
4. Read **FUNCTIONS_GUIDE.md** - Learn each function
5. Run **simple_tests.py** - See it work
6. Run **USAGE_EXAMPLES.py** - See examples

### Level 3: Application
7. Try using the library yourself
8. Integrate into your Streamlit app
9. Create your own examples

---

## 🎯 What You've Gained

### Before:
- ❌ Everything in one 600-line file
- ❌ Hard to test without Streamlit
- ❌ Can't reuse in other projects
- ❌ Mixed UI and science logic

### After:
- ✅ Clean separation (23 functions in 5 modules)
- ✅ Easy to test each function
- ✅ Reusable in notebooks, scripts, CLI tools
- ✅ Science logic separate from UI

---

## 🔄 Next Steps

### 1. Test the Library
```bash
python simple_tests.py
```

### 2. Try an Example
```bash
python USAGE_EXAMPLES.py
```

### 3. Update Your Streamlit App
Refactor `1_calibrate.py` to use the library instead of having all the code inline.

### 4. Extract Other Pages
Apply the same pattern to:
- `2_get_input_files.py` → `nativeims/extraction/`
- `3_process_output_files.py` → `nativeims/processing/`
- etc.

---

## 💡 Tips for Using the Library

### Tip 1: Import What You Need
```python
# Good - specific imports
from nativeims.calibration import CalibrantProcessor

# Also good - import module
from nativeims import calibration
processor = calibration.CalibrantProcessor(...)
```

### Tip 2: Use Type Hints
Your editor (VS Code) will autocomplete if you use type hints:
```python
from nativeims.calibration import CalibrantMeasurement

def my_function(measurement: CalibrantMeasurement):
    # VS Code will autocomplete measurement. ← press dot and see magic!
    print(measurement.drift_time)
```

### Tip 3: Check Return Values
Many functions return `Optional[X]`, so always check:
```python
result = processor.process_file(...)

if result is not None:  # Always check!
    print(result.drift_time)
else:
    print("Processing failed")
```

### Tip 4: Read Error Messages
If you get an error, it will tell you what's wrong:
```
TypeError: load_atd_data() takes 1 positional argument but 2 were given
```
This means you gave it too many inputs. Check the function signature!

---

## 📞 Common Questions

### Q: "I get 'ModuleNotFoundError: No module named nativeims'"
**A**: Make sure you're running from the project root directory:
```bash
cd c:\Users\h87023ab\Documents\GITHUB\PROCESSING_TOOLS\nativeIMS-processing-tools
python your_script.py
```

### Q: "How do I know what inputs a function needs?"
**A**: Look at the type hints or docstring:
```python
def load_atd_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    #                       ^^^^
    # It needs a Path object
```

### Q: "What's the difference between a class and a dataclass?"
**A**: 
- **Class**: Has methods (functions) that do things
- **Dataclass**: Just stores data, minimal methods

### Q: "Do I need to understand everything to use it?"
**A**: No! Start with the high-level functions like `process_calibrant_set()`. You can learn the details later.

---

## 🎉 Summary

You now have:
- ✅ **23 functions** in a reusable library
- ✅ **6 documentation files** to help you learn
- ✅ **Clean separation** of science vs. UI code
- ✅ **Type hints** for autocomplete
- ✅ **Tests** to verify it works
- ✅ **Examples** to learn from

**You're ready to go!** 🚀

Start with `python simple_tests.py`, then explore the documentation files to learn more.
