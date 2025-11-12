# Page 2 Refactoring: Get Input Files

## Summary
Successfully refactored `pages/2_get_input_files.py` to use the nativeims core library, following the same pattern as page 1.

## What Was Extracted

### New Module: `nativeims/extraction/input_generator.py`

**Purpose**: Generate IMSCal input files from sample arrival time distribution data

**Extracted Functions/Classes**:

1. **InputParams** (dataclass)
   - Stores: drift_mode, inject_time, sample_mass_map
   - Purpose: Configuration for input processing

2. **InputProcessingResult** (dataclass)
   - Stores: processed_files, failed_files, sample_paths
   - Purpose: Tracking processing outcomes

3. **InputProcessor** (class)
   - Methods:
     - `process_sample_folder()` - Process single sample folder → creates .dat files
     - `process_all()` - Process all sample folders in batch
   - Purpose: Core business logic for input file generation

### Updated Module: `nativeims/io/writers.py`

**Added Function**: `generate_zip_archive()`
- Purpose: Bundle processed .dat files into downloadable ZIP
- Input: Dictionary of {sample_name: folder_path}
- Output: BytesIO buffer with ZIP file

**Updated Function**: `write_imscal_dat()`
- Now supports TWO formats:
  1. **Calibration format** (with header) - for calibrant data
  2. **Input format** (no header) - for sample data
- Auto-detects format based on DataFrame columns and parameters

## Code Reuse

The refactoring **reused existing library functions** instead of duplicating code:

### From `nativeims/io/readers.py`:
- ✅ `is_valid_calibrant_file()` - File validation
- ✅ `extract_charge_state_from_filename()` - Parse charge from filename
- ✅ `load_atd_data()` - Load drift time & intensity arrays

### From `nativeims/calibration/utils.py`:
- ✅ `adjust_drift_time_for_injection()` - Cyclic IMS correction

### Original page had duplicate versions of all these! 

## Refactored Streamlit Page

**File**: `pages/2_get_input_files_refactored.py`

**Before**: ~343 lines with embedded business logic  
**After**: ~247 lines - pure UI code

**Structure**:
```python
class UI:
    # Pure Streamlit components
    show_main_header()
    show_info_card()
    get_uploaded_zip()
    get_instrument_settings()
    get_sample_masses()
    show_processing_results()
    show_download_button()
    show_references()

def main():
    # Thin orchestration layer
    # All processing delegated to nativeims library
```

## Key Improvements

1. **Eliminated Code Duplication**
   - Original: 3 duplicate file handling methods
   - Refactored: Reuses existing library functions
   
2. **Separation of Concerns**
   - Core logic: `nativeims/extraction/input_generator.py`
   - UI logic: `pages/2_get_input_files_refactored.py`
   - No Streamlit dependencies in core library ✅

3. **JOSS Compliance**
   - Core library can be used independently
   - Easy to test without Streamlit
   - Clear API boundaries

## Testing the Refactored Page

To test the new page:

```python
# In Python terminal
from nativeims.extraction import InputProcessor, InputParams
from pathlib import Path

# Create parameters
params = InputParams(
    drift_mode='Cyclic',
    inject_time=0.5,
    sample_mass_map={'myoglobin': 16952.3}
)

# Process data
processor = InputProcessor(base_path=Path('temp'), params=params)
result = processor.process_all(['myoglobin'])

# Check results
print(result.processed_files)
print(result.failed_files)
```

## Files Created/Modified

### Created:
- ✅ `nativeims/extraction/__init__.py`
- ✅ `nativeims/extraction/input_generator.py`
- ✅ `pages/2_get_input_files_refactored.py`

### Modified:
- ✅ `nativeims/io/writers.py` - Added `generate_zip_archive()`, updated `write_imscal_dat()`
- ✅ `nativeims/__init__.py` - Exported extraction classes

## Next Steps

Continue refactoring remaining pages:
- `pages/3_process_output_files.py`
- `pages/4_get_calibrated_scaled_data.py`
- `pages/5_plot_CCSDs.py`
- `pages/6_fit_data.py`
- `pages/7_plot_aIMS(CIU).py`
- `pages/8_plot_pretty_MS.py`
