# Quick Start Guide - nativeIMS Core Library

Welcome! This guide will help you understand what you've built and how to use it.

---

## 🎯 What Did We Just Build?

We extracted the **core scientific logic** from your Streamlit app and put it into a **reusable library**.

### Before (Everything in Streamlit):
```
1_calibrate.py (600+ lines)
├── File loading code
├── Gaussian fitting code
├── Database queries
├── File writing code
└── UI display code
```

### After (Separated):
```
nativeims/ (Core library - no Streamlit)
├── io/
│   ├── readers.py      ← Load files
│   └── writers.py      ← Save files
└── calibration/
    ├── database.py     ← Bush database
    ├── processor.py    ← Gaussian fitting
    └── utils.py        ← Helper functions

1_calibrate.py (now ~100 lines)
└── Just UI code, calls the library
```

---

## 📦 What's in Each File?

### `nativeims/io/readers.py` - Load Data
- ✅ Load .txt files from MassLynx
- ✅ Load .csv files from TWIMExtract
- ✅ Extract charge states from filenames
- ✅ Validate file formats

### `nativeims/io/writers.py` - Save Data
- ✅ Write .dat files for IMSCal
- ✅ Write CSV results
- ✅ Convert DataFrames to strings

### `nativeims/calibration/database.py` - Bush Database
- ✅ Load bush.csv file
- ✅ Look up CCS values
- ✅ Query available proteins/charges

### `nativeims/calibration/processor.py` - Main Processing
- ✅ Fit Gaussians to ATDs
- ✅ Extract drift times
- ✅ Match with database values
- ✅ Quality control (R² thresholds)

### `nativeims/calibration/utils.py` - Helpers
- ✅ Store instrument parameters
- ✅ Adjust drift times (Cyclic IMS)

---

## 🚀 How to Use It

### Example 1: Process a single file

```python
from pathlib import Path
from nativeims.calibration import (
    load_bush_database,
    CalibrantDatabase,
    CalibrantProcessor
)

# Setup
bush_df = load_bush_database(Path("data/bush.csv"))
db = CalibrantDatabase(bush_df)
processor = CalibrantProcessor(db, min_r2=0.9)

# Process one file
result = processor.process_file(
    Path("myoglobin/24.txt"),
    "myoglobin",
    "helium"
)

# Check result
if result:
    print(f"Drift time: {result.drift_time:.3f} ms")
    print(f"R²: {result.r_squared:.3f}")
    print(f"CCS: {result.ccs_literature:.2f} nm²")
```

---

### Example 2: Process a folder

```python
# Process all files in a folder
measurements, skipped = processor.process_folder(
    Path("myoglobin"),
    "myoglobin",
    "helium"
)

print(f"Processed: {len(measurements)} files")
print(f"Skipped: {len(skipped)} files")

for m in measurements:
    print(f"Charge {m.charge_state}: {m.drift_time:.3f} ms")
```

---

### Example 3: Complete workflow

```python
from nativeims.calibration import (
    load_bush_database,
    CalibrantDatabase,
    CalibrantProcessor,
    InstrumentParams,
    adjust_dataframe_drift_times
)
from nativeims.io.writers import write_imscal_dat

# 1. Load database
bush_df = load_bush_database(Path("data/bush.csv"))
db = CalibrantDatabase(bush_df)

# 2. Process calibrants
processor = CalibrantProcessor(db, min_r2=0.9)
results_df = processor.process_calibrant_set(
    Path("calibrants"),
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

# 4. Adjust drift times
adjusted_df = adjust_dataframe_drift_times(results_df, params)

# 5. Save results
write_imscal_dat(
    adjusted_df,
    velocity=params.wave_velocity,
    voltage=params.wave_height,
    pressure=params.pressure,
    length=params.drift_length,
    output_path=Path("output.dat")
)

print("Done!")
```

---

## 🧪 Testing Your Library

Run the simple tests:

```bash
python simple_tests.py
```

You should see:
```
Test 1: Checking imports...
  ✓ IO module imported successfully
  ✓ Calibration module imported successfully
  ✓ Writers module imported successfully
✅ All imports successful!

Test 2: Testing filename parsing...
  ✓ '24.txt' -> 24
  ✓ 'range_18.csv' -> 18
  ...
✅ All tests passed!
```

---

## 📚 Learning Resources

### For Beginners:

1. **FUNCTIONS_GUIDE.md** - Detailed explanation of EVERY function
2. **USAGE_EXAMPLES.py** - Working code examples
3. **simple_tests.py** - See how each part works

### Code Documentation:

Every function has:
- **Type hints**: Shows what inputs/outputs are expected
- **Docstrings**: Explains what it does
- **Examples**: Shows how to use it

Example:
```python
def load_atd_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load drift time and intensity data from an ATD file.
    
    Args:
        file_path: Path to the ATD data file
        
    Returns:
        A tuple of two numpy arrays: (drift_time, intensity)
        
    Example:
        >>> drift_time, intensity = load_atd_data(Path("24.txt"))
    """
```

---

## 🔄 Next Steps: Update Your Streamlit App

Now you can simplify `1_calibrate.py`:

### Before:
```python
class CalibrantProcessor:
    def __init__(self, bush_df):
        # ... 50 lines of code
    
    def _load_data_from_file(self, file_path):
        # ... 30 lines of code
    
    def _process_single_file(self, file_path):
        # ... 100 lines of code
    
    # etc... 600 total lines
```

### After:
```python
from nativeims.calibration import CalibrantProcessor, CalibrantDatabase

# Setup (just 3 lines!)
bush_df = load_bush_database()
db = CalibrantDatabase(bush_df)
processor = CalibrantProcessor(db)

# Process (just 1 line!)
results = processor.process_calibrant_set(uploaded_path)

# Display results in Streamlit
st.dataframe(results)
```

**Much simpler!** 🎉

---

## ✅ Benefits of This Structure

### 1. **Reusability**
Use the same code in:
- Streamlit apps
- Jupyter notebooks
- Command-line scripts
- Other Python programs

### 2. **Testability**
- Each function can be tested independently
- No need for Streamlit to test the logic
- Easy to find bugs

### 3. **Maintainability**
- Core logic separate from UI
- Change the science without touching the UI
- Change the UI without touching the science

### 4. **Documentation**
- Auto-generate API docs
- Clear function purposes
- Easy for others to understand

### 5. **Distribution**
- Can be pip-installed
- Share with collaborators
- Use in publications

---

## 🎓 Understanding the Code (For Beginners)

### What's a "dataclass"?

A dataclass is like a container for related data:

```python
@dataclass
class CalibrantMeasurement:
    protein: str
    charge_state: int
    drift_time: float
```

Instead of:
```python
measurement = {
    'protein': 'myoglobin',
    'charge_state': 24,
    'drift_time': 5.23
}
```

You can do:
```python
measurement = CalibrantMeasurement(
    protein='myoglobin',
    charge_state=24,
    drift_time=5.23
)

print(measurement.protein)  # 'myoglobin'
```

**Why?** Type checking, autocomplete, clearer code!

---

### What's a "class method"?

A method is a function that belongs to a class:

```python
class CalibrantDatabase:
    def lookup_calibrant(self, protein, charge):
        # This is a method
        pass

# Use it like this:
db = CalibrantDatabase(bush_df)
result = db.lookup_calibrant('myoglobin', 24)
```

The `self` parameter is automatic - it refers to the object itself (`db` in this case).

---

### What are "type hints"?

Type hints tell you (and your editor) what types are expected:

```python
def load_atd_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    #                   ^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                   Input type   Output type
```

This means:
- **Input**: `file_path` should be a `Path` object
- **Output**: Returns a tuple of two numpy arrays

**Why?** Your editor can warn you about mistakes before you run the code!

---

### What's `Optional[...]`?

`Optional[X]` means "either X or None":

```python
def process_file(...) -> Optional[CalibrantMeasurement]:
    if success:
        return CalibrantMeasurement(...)
    else:
        return None  # This is okay because of Optional
```

---

## 🐛 Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'nativeims'"

**Solution**: Make sure you're running from the project root:
```bash
cd c:\Users\h87023ab\Documents\GITHUB\PROCESSING_TOOLS\nativeIMS-processing-tools
python simple_tests.py
```

---

### Issue 2: "Cannot import fit_gaussian_with_retries"

**Solution**: The processor needs your existing `myutils.data_tools` module. Make sure it's available, or provide a custom fitting function:

```python
from myutils import data_tools

processor = CalibrantProcessor(
    db,
    fitting_function=data_tools.fit_gaussian_with_retries
)
```

---

### Issue 3: "Bush database not found"

**Solution**: Provide the full path:
```python
bush_df = load_bush_database(Path("data/bush.csv"))
```

---

## 📖 File Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `nativeims/io/readers.py` | Load files | `load_atd_data()`, `is_valid_calibrant_file()` |
| `nativeims/io/writers.py` | Save files | `write_imscal_dat()`, `write_calibration_results_csv()` |
| `nativeims/calibration/database.py` | Bush DB | `load_bush_database()`, `CalibrantDatabase` |
| `nativeims/calibration/processor.py` | Main logic | `CalibrantProcessor` |
| `nativeims/calibration/utils.py` | Helpers | `adjust_drift_time_for_injection()` |

---

## 🎯 Summary

You now have:
- ✅ **23 well-documented functions**
- ✅ **Clean separation** (science vs. UI)
- ✅ **Reusable library** (use anywhere)
- ✅ **Type hints** (autocomplete in VS Code)
- ✅ **Examples** (see how to use it)
- ✅ **Tests** (verify it works)

**Next**: Apply the same pattern to your other Streamlit pages! 🚀
