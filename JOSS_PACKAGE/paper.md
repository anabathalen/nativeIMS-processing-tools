---
title: 'IMSpartacus: A Comprehensive Python Toolkit for Traveling Wave Ion Mobility Spectrometry Data Analysis'
tags:
  - Python
  - mass spectrometry
  - ion mobility
  - structural biology
  - proteomics
  - collision cross section
authors:
  - name: Your Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Your Institution, Country
   index: 1
date: 6 November 2025
bibliography: paper.bib
---

# Summary

Traveling wave ion mobility spectrometry-mass spectrometry (TWIMS-MS) is a powerful analytical technique for structural characterization of biomolecules, enabling separation based on collision cross section (CCS). However, TWIMS requires external calibration and involves complex multi-step workflows that often rely on disconnected software tools. `IMSpartacus` (Ion Mobility Spectrometry Processing And Robust Toolkit for Analysis of Collision Cross Sections) addresses these challenges by providing an integrated, user-friendly Python toolkit with a web-based interface for end-to-end TWIMS data analysis.

# Statement of Need

TWIMS-MS has become increasingly popular in structural biology and proteomics for studying protein conformations, protein-ligand interactions, and collision-induced unfolding (CIU). Unlike drift tube IMS (DTIMS), TWIMS instruments require calibration using proteins of known CCS [@Bush:2012]. Current workflows typically involve:

1. Data extraction using vendor or third-party software
2. Manual calibration in spreadsheet programs
3. Data processing in MATLAB or Origin
4. Export and reformatting for visualization tools

This fragmented approach is error-prone, time-consuming, and hinders reproducibility. Existing tools like TWIMExtract [@Sergent:2023] and IMSCal [@Haynes:2017] address specific steps but lack integration. ORIGAMI [@Ral:2016] provides excellent CIU analysis but requires MATLAB.

`IMSpartacus` provides:

- **Integrated workflow**: Seamless connection between calibration, processing, and visualization
- **Accessibility**: Web interface requiring no programming knowledge
- **Reproducibility**: All parameters documented and adjustable
- **Extensibility**: Python API for custom analyses and method development
- **Publication-ready outputs**: High-resolution figures with extensive customization

# Features

`IMSpartacus` includes 10 integrated tools organized into core workflow and advanced analysis modules:

## Core Workflow

1. **Calibration**: Process protein standard data to generate CCS calibration curves
2. **Input Generation**: Create formatted files for IMSCal [@Haynes:2017]
3. **Output Processing**: Convert drift times to CCS using calibration results
4. **Data Integration**: Combine CCS distributions with mass spectra for quantitative analysis
5. **Visualization**: Plot collision cross section distributions with customization options

## Advanced Analysis

6. **Peak Fitting**: Automated peak detection and fitting with multiple functions (Gaussian, pseudo-Voigt, asymmetric Gaussians, log-normal)
7. **Mass Spectrum Plotting**: Publication-quality MS plots with smoothing, baseline correction, and annotations
8. **Range File Generation**: Automate creation of TWIMExtract range files for batch processing
9. **ESIProt**: Charge state deconvolution using the ESIProt algorithm [@Winkler:2017]
10. **ORIGAMI CIU**: ORIGAMI-style CIU fingerprint heatmaps with interpolation and smoothing

# Implementation

`IMSpartacus` is implemented in Python 3.8+ using scientific computing libraries (NumPy, SciPy, pandas) for data processing, Matplotlib and Plotly for visualization, and Streamlit [@Streamlit:2023] for the web interface. The modular architecture allows both interactive use through the web interface and programmatic access via the Python API.

Key algorithmic components include:

- **Calibration**: Polynomial fitting (typically power law) of drift time vs. known CCS values
- **Peak Detection**: Derivative-based peak finding with customizable prominence thresholds
- **Peak Fitting**: Levenberg-Marquardt optimization with multiple peak function options
- **CIU Analysis**: 2D interpolation (linear or cubic) and smoothing (Gaussian or Savitzky-Golay) of CCS vs. activation voltage matrices

# Usage Example

```python
from imspartacus.calibration import CalibrationProcessor
from imspartacus.fitting import PeakDetector, FittingEngine

# Calibrate drift times to CCS
calibrator = CalibrationProcessor()
calibrator.load_data("protein_standards.csv")
ccs_values = calibrator.calibrate(experimental_drift_times)

# Detect and fit peaks
detector = PeakDetector()
peaks = detector.detect_peaks(ccs_values, intensities, 
                              prominence=0.05)

fitter = FittingEngine()
results = fitter.fit_peaks(peaks, function="gaussian", 
                           baseline="polynomial")
```

# Impact and Applications

`IMSpartacus` has been used for:

- Characterization of native protein complexes
- Analysis of antibody conformational heterogeneity
- CIU studies of protein-ligand binding
- High-throughput screening of protein stability

The toolkit fills a critical gap in the TWIMS analysis ecosystem by providing an accessible, integrated platform that promotes reproducible research and accelerates method development.

# Acknowledgments

We acknowledge contributions from [names], and thank [funding sources].

# References
