# What We Built - Visual Summary

## 📊 The Big Picture

```
OLD STRUCTURE (Everything mixed together):
┌─────────────────────────────────────────┐
│  1_calibrate.py (600+ lines)            │
│  ┌───────────────────────────────────┐  │
│  │ UI Code (Streamlit)               │  │
│  │ File Loading                      │  │
│  │ Gaussian Fitting                  │  │
│  │ Database Queries                  │  │
│  │ File Writing                      │  │
│  │ All mixed together! 😵            │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

Problem: Hard to test, reuse, or maintain!


NEW STRUCTURE (Separated):
┌──────────────────────────────────────────────────────────┐
│  nativeims/ (Core Library - Pure Python, no UI)         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  io/                                               │  │
│  │  ├── readers.py    ← Load .txt/.csv files         │  │
│  │  └── writers.py    ← Save .dat/.csv files         │  │
│  │                                                     │  │
│  │  calibration/                                      │  │
│  │  ├── database.py   ← Bush database queries        │  │
│  │  ├── processor.py  ← Gaussian fitting logic       │  │
│  │  └── utils.py      ← Helper functions             │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                          ▲
                          │ uses
                          │
┌──────────────────────────────────────────────────────────┐
│  1_calibrate.py (now ~100 lines)                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Just UI code:                                     │  │
│  │  • Get user inputs                                 │  │
│  │  • Call library functions                          │  │
│  │  • Display results                                 │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

Benefits: Testable, reusable, maintainable! 🎉
```

---

## 📦 What's in Each Module?

### Module 1: `nativeims/io/readers.py`
```
┌─────────────────────────────────────────┐
│  File Reading Functions                 │
├─────────────────────────────────────────┤
│  1. is_valid_calibrant_file()           │
│     ↳ Check if file should be processed │
│                                          │
│  2. extract_charge_state_from_filename() │
│     ↳ Get charge state from filename    │
│                                          │
│  3. load_atd_data()                     │
│     ↳ Load drift time & intensity       │
│                                          │
│  4. load_multiple_atd_files()           │
│     ↳ Load entire folder                │
└─────────────────────────────────────────┘
```

### Module 2: `nativeims/io/writers.py`
```
┌─────────────────────────────────────────┐
│  File Writing Functions                 │
├─────────────────────────────────────────┤
│  1. write_imscal_dat()                  │
│     ↳ Create .dat file for IMSCal       │
│                                          │
│  2. dataframe_to_csv_string()           │
│     ↳ Convert DataFrame to CSV text     │
│                                          │
│  3. write_calibration_results_csv()     │
│     ↳ Save results as CSV               │
└─────────────────────────────────────────┘
```

### Module 3: `nativeims/calibration/database.py`
```
┌─────────────────────────────────────────┐
│  Bush Database Interface                │
├─────────────────────────────────────────┤
│  Variable: CALIBRANT_FOLDER_MAPPING     │
│     ↳ Protein name ↔ folder name        │
│                                          │
│  Function: load_bush_database()         │
│     ↳ Load bush.csv file                │
│                                          │
│  Class: CalibrantDatabase               │
│  ├── get_calibrant_column()             │
│  ├── lookup_calibrant()                 │
│  ├── get_available_charge_states()      │
│  └── get_available_proteins()           │
└─────────────────────────────────────────┘
```

### Module 4: `nativeims/calibration/processor.py`
```
┌─────────────────────────────────────────┐
│  Main Processing (Gaussian Fitting)     │
├─────────────────────────────────────────┤
│  Dataclass: GaussianFitResult           │
│     ↳ Stores fit results                │
│                                          │
│  Dataclass: CalibrantMeasurement        │
│     ↳ Stores one measurement            │
│                                          │
│  Class: CalibrantProcessor              │
│  ├── process_file()                     │
│  │   ↳ Process one file                 │
│  ├── process_folder()                   │
│  │   ↳ Process all files in folder      │
│  └── process_calibrant_set()            │
│      ↳ Process multiple proteins        │
│                                          │
│  Function: measurements_to_dataframe()  │
│     ↳ Convert results to DataFrame      │
└─────────────────────────────────────────┘
```

### Module 5: `nativeims/calibration/utils.py`
```
┌─────────────────────────────────────────┐
│  Helper Functions                       │
├─────────────────────────────────────────┤
│  Dataclass: InstrumentParams            │
│     ↳ Store instrument settings         │
│                                          │
│  Function: adjust_drift_time_for_injection() │
│     ↳ Subtract inject time (Cyclic IMS) │
│                                          │
│  Function: adjust_dataframe_drift_times() │
│     ↳ Adjust all drift times in DataFrame │
└─────────────────────────────────────────┘
```

---

## 🔄 How Data Flows Through the System

```
1. USER UPLOADS FILE
   │
   ├──> is_valid_calibrant_file()
   │    (Check if file is valid)
   │
   └──> If valid...
        │
        ├──> extract_charge_state_from_filename()
        │    (Get charge state: 24)
        │
        ├──> load_atd_data()
        │    (Load drift time & intensity arrays)
        │
        ├──> CalibrantProcessor._fit_gaussian()
        │    (Fit Gaussian, get apex & R²)
        │
        ├──> CalibrantDatabase.lookup_calibrant()
        │    (Get literature CCS value)
        │
        └──> Create CalibrantMeasurement object
             ├── protein: 'myoglobin'
             ├── charge_state: 24
             ├── drift_time: 5.23
             ├── r_squared: 0.95
             └── ccs_literature: 31.2

2. PROCESS MULTIPLE FILES
   │
   └──> CalibrantProcessor.process_folder()
        │
        ├──> Calls process_file() for each file
        ├──> Collects all CalibrantMeasurement objects
        └──> Returns list of measurements

3. CREATE OUTPUT FILES
   │
   ├──> measurements_to_dataframe()
   │    (Convert to DataFrame)
   │
   ├──> adjust_dataframe_drift_times()
   │    (Subtract inject time if Cyclic)
   │
   ├──> write_calibration_results_csv()
   │    (Save as CSV)
   │
   └──> write_imscal_dat()
        (Save as .dat for IMSCal)
```

---

## 🎯 Function Call Examples

### Example 1: Single File Processing
```python
# Input: One file
Path("myoglobin/24.txt")

# Processing chain
result = processor.process_file(
    file_path,      # ← Load & validate
    "myoglobin",    # ← Look up in database
    "helium"        # ← Get CCS value
)

# Output: CalibrantMeasurement object
result.protein          # 'myoglobin'
result.charge_state     # 24
result.drift_time       # 5.23
result.r_squared        # 0.95
result.ccs_literature   # 31.2
```

### Example 2: Folder Processing
```python
# Input: Folder with multiple files
Path("myoglobin/")
├── 24.txt
├── 25.txt
└── 26.txt

# Processing
measurements, skipped = processor.process_folder(
    Path("myoglobin"),
    "myoglobin",
    "helium"
)

# Output: List of measurements
measurements[0]  # CalibrantMeasurement for charge 24
measurements[1]  # CalibrantMeasurement for charge 25
measurements[2]  # CalibrantMeasurement for charge 26
```

### Example 3: Multiple Proteins
```python
# Input: Folder structure
Path("calibrants/")
├── myoglobin/
│   ├── 24.txt
│   └── 25.txt
└── cytochromec/
    ├── 18.txt
    └── 19.txt

# Processing
df = processor.process_calibrant_set(
    Path("calibrants"),
    "helium"
)

# Output: DataFrame
   protein       charge_state  drift_time  r2    ccs_literature
0  myoglobin     24           5.23        0.95  31.2
1  myoglobin     25           4.87        0.93  29.8
2  cytochromec   18           4.12        0.96  25.3
3  cytochromec   19           3.98        0.94  24.1
```

---

## 📊 Data Types Reference

### Path
```python
from pathlib import Path

file_path = Path("myoglobin/24.txt")
folder_path = Path("myoglobin")
```

### Tuple
```python
# Two values returned together
drift_time, intensity = load_atd_data(file_path)
#          ↑ numpy array
#                    ↑ numpy array
```

### Optional
```python
# Can be the type OR None
result: Optional[CalibrantMeasurement]

if result is not None:
    print(result.drift_time)
```

### DataFrame
```python
import pandas as pd

df = pd.DataFrame({
    'protein': ['myoglobin'],
    'charge_state': [24],
    'drift_time': [5.23]
})

print(df['drift_time'][0])  # Access: 5.23
```

### Dataclass
```python
@dataclass
class CalibrantMeasurement:
    protein: str
    drift_time: float

m = CalibrantMeasurement('myoglobin', 5.23)
print(m.protein)      # 'myoglobin'
print(m.drift_time)   # 5.23
```

---

## ✅ Complete Function Count

| Module | Functions | Classes | Total |
|--------|-----------|---------|-------|
| `io/readers.py` | 4 | 0 | 4 |
| `io/writers.py` | 3 | 0 | 3 |
| `calibration/database.py` | 1 | 1 (4 methods) | 5 |
| `calibration/processor.py` | 1 | 1 (3 methods) + 2 dataclasses | 6 |
| `calibration/utils.py` | 2 | 0 + 1 dataclass | 3 |
| **TOTAL** | **11** | **2 + 3 dataclasses** | **23** |

---

## 🎓 For Your Reference

### Quick Import Cheat Sheet
```python
# Load files
from nativeims.io import load_atd_data, is_valid_calibrant_file

# Save files
from nativeims.io.writers import write_imscal_dat

# Database
from nativeims.calibration import load_bush_database, CalibrantDatabase

# Processing
from nativeims.calibration import CalibrantProcessor

# Utils
from nativeims.calibration import InstrumentParams, adjust_drift_time_for_injection
```

### Common Workflow
```python
# 1. Setup
bush_df = load_bush_database(Path("data/bush.csv"))
db = CalibrantDatabase(bush_df)
processor = CalibrantProcessor(db, min_r2=0.9)

# 2. Process
results_df = processor.process_calibrant_set(Path("calibrants"), "helium")

# 3. Adjust (if Cyclic)
params = InstrumentParams(...)
adjusted_df = adjust_dataframe_drift_times(results_df, params)

# 4. Save
write_imscal_dat(adjusted_df, ...)
```

---

## 🎉 You're Ready!

You now have a **complete, documented, reusable library** with:
- ✅ 23 functions organized into 5 modules
- ✅ Type hints for autocomplete
- ✅ Docstrings with examples
- ✅ Clean separation of concerns
- ✅ Easy to test and maintain

**Next step**: Run `python simple_tests.py` to verify everything works!
