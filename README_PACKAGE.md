# nativeIMS

A Python library for processing native ion mobility mass spectrometry (IMS) data with an optional web interface.

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

nativeIMS provides tools for:
- **Calibrant data processing** with automated Gaussian fitting
- **CCS (collision cross section) calculations** using the Bush database
- **Drift time extraction** from arrival time distributions (ATDs)
- **IMSCal file generation** for instrument calibration
- Support for both **MassLynx** (.txt) and **TWIMExtract** (.csv) file formats

**Key Feature**: This package provides a standalone Python library that can be used independently OR through an optional web interface built with Streamlit.

## Installation

### Core Library (Required)

```bash
# Clone the repository
git clone https://github.com/yourusername/nativeIMS-processing-tools.git
cd nativeIMS-processing-tools

# Install the core library
pip install .

# Or install in development mode (recommended for development)
pip install -e .
```

### With Web Interface (Optional)

```bash
# Install with Streamlit web interface
pip install .[web]
```

### For Development

```bash
# Install with testing and development tools
pip install .[dev]
```

## Quick Start

### Using the Core Library (No Web Interface)

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

# 1. Load the Bush calibrant database
bush_df = load_bush_database(Path("data/bush.csv"))
db = CalibrantDatabase(bush_df)

# 2. Create a calibrant processor
processor = CalibrantProcessor(db, min_r2=0.9)

# 3. Process your calibrant data
results_df = processor.process_calibrant_set(
    Path("path/to/your/calibrant/folders"),
    gas_type="helium"
)

# 4. Define instrument parameters
params = InstrumentParams(
    wave_velocity=281.0,  # m/s
    wave_height=20.0,     # V
    pressure=1.63,        # mbar
    drift_length=0.98,    # m (0.98 for Cyclic, 0.25 for Synapt)
    instrument_type='cyclic',
    inject_time=0.3       # ms (for Cyclic IMS only)
)

# 5. Adjust drift times (for Cyclic IMS)
adjusted_df = adjust_dataframe_drift_times(results_df, params)

# 6. Save results
write_imscal_dat(
    adjusted_df,
    velocity=params.wave_velocity,
    voltage=params.wave_height,
    pressure=params.pressure,
    length=params.drift_length,
    output_path=Path("calibration.dat")
)
```

### Using the Web Interface (Optional)

```bash
# Run the Streamlit app
streamlit run pages/1_calibrate_refactored.py

# Or with the full Python path
python -m streamlit run pages/1_calibrate_refactored.py
```

Then open your browser to http://localhost:8501

## Core Library Structure

```
nativeims/
├── __init__.py                 # Package initialization
├── io/                         # File I/O operations
│   ├── readers.py             # Load ATD data from .txt/.csv files
│   └── writers.py             # Write results to .dat/.csv files
├── calibration/               # Calibration workflows
│   ├── database.py           # Bush database interface
│   ├── processor.py          # Gaussian fitting & processing
│   └── utils.py              # Helper functions
└── utils/                     # General utilities (future expansion)
```

## Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get up and running quickly
- **[Functions Guide](FUNCTIONS_GUIDE.md)** - Detailed explanation of every function
- **[Visual Summary](VISUAL_SUMMARY.md)** - Diagrams and flowcharts
- **[Beginner's Guide](BEGINNERS_GUIDE.md)** - Complete guide for beginners
- **[Usage Examples](USAGE_EXAMPLES.py)** - Working code examples

## Testing

Run the test suite:

```bash
# Simple tests (no installation required)
python simple_tests.py

# Or if you installed with [dev]
pytest tests/
```

## Data Format

### Input Files

**Folder structure:**
```
calibrants/
├── myoglobin/
│   ├── 24.txt      # Charge state 24
│   ├── 25.txt      # Charge state 25
│   └── 26.txt
└── cytochromec/
    ├── 18.txt
    └── 19.txt
```

**Supported formats:**
- `.txt` files: Two-column space-separated data from MassLynx (drift time, intensity)
- `.csv` files: Comma-separated data from TWIMExtract (supports comments with #)

### Output Files

- **CSV**: Human-readable results with Gaussian fit parameters
- **DAT**: IMSCal-compatible calibration file with instrument parameters

## Requirements

- Python 3.7+
- pandas
- numpy
- scipy
- matplotlib
- streamlit (optional, for web interface)

## References

1. Bush, M. F., et al. "Collision Cross Sections of Protein Ions." *Journal of the American Society for Mass Spectrometry*, 2010, 21, 1003-1010.
2. Haynes, S. E., et al. "Variable-Velocity Traveling-Wave Ion Mobility Separation Enhancing Peak Capacity for Data-Independent Acquisition Proteomics." *Analytical Chemistry*, 2017, 89, 5669–5672.
3. Sergent, I., et al. "IMSCal: A software for generic processing of TWIMS data using calibration approaches." *International Journal of Mass Spectrometry*, 2023, 492, 117112.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{nativeims,
  author = {Your Name},
  title = {nativeIMS: A Python library for native ion mobility mass spectrometry data processing},
  year = {2025},
  url = {https://github.com/yourusername/nativeIMS-processing-tools}
}
```

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/nativeIMS-processing-tools/issues)
- **Email**: your.email@example.com

---

**Note**: This is a research tool. Always validate results with your experimental data and domain expertise.
