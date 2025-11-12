# nativeIMS - Core Library

A Python library for processing native ion mobility mass spectrometry data.

## What is this?

This library provides reusable functions for:
- Loading ATD (arrival time distribution) data from MassLynx and TWIMExtract
- Fitting Gaussians to extract drift times
- Looking up literature CCS values from the Bush database
- Generating calibration files for IMSCal
- Processing complete calibrant datasets

## Installation

Currently, this is a local package. To use it:

```python
# From the project root directory
import sys
sys.path.append('/path/to/nativeIMS-processing-tools')

from nativeims.calibration import CalibrantProcessor
```

## Quick Example

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

# Process calibrants
results = processor.process_calibrant_set(
    Path("calibrants"),
    gas_type="helium"
)

print(results)
```

## Modules

### `nativeims.io`
File input/output operations:
- `load_atd_data()` - Load drift time and intensity data
- `write_imscal_dat()` - Write IMSCal calibration files
- `is_valid_calibrant_file()` - Validate file formats

### `nativeims.calibration`
Calibration workflow:
- `CalibrantDatabase` - Interface to Bush database
- `CalibrantProcessor` - Gaussian fitting and processing
- `InstrumentParams` - Store instrument parameters

## Documentation

See these files for detailed documentation:
- **QUICKSTART.md** - Get started quickly
- **FUNCTIONS_GUIDE.md** - Detailed explanation of every function
- **VISUAL_SUMMARY.md** - Visual overview of the library
- **USAGE_EXAMPLES.py** - Working code examples

## Testing

Run the simple test suite:

```bash
python simple_tests.py
```

## Requirements

- Python 3.7+
- pandas
- numpy
- pathlib (built-in)

## License

See LICENSE file in the project root.

## References

Bush, M. F., et al. "Collision Cross Sections of Protein Ions."
Journal of the American Society for Mass Spectrometry, 2010, 21, 1003-1010.
