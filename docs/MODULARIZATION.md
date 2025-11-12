# Modularization plan for nativeIMS-processing-tools

This document identifies logic currently embedded in Streamlit pages that can be extracted into a reusable, testable core library to better meet JOSS’ guidance for web-based tools.

Goals
- Separate UI (Streamlit) from core algorithms, IO, and plotting primitives
- Make the core importable and testable (pytest), runnable headlessly (optional CLI)
- Keep changes incremental and low-risk by moving well-defined functions/classes

Recommended package layout (using existing `myutils`)
- myutils/
  - dtims.py — Mason–Schamp CCS, DTIMS parsing, regression helpers
  - origami.py — TWIM extract parsing, CCS conversion via calibration, fingerprint matrix ops (interpolate/normalize/smooth)
  - esiprot.py — ESI deconvolution and m/z calculators
  - range_files.py — Range file generation, zip packaging
  - ms_processing.py — Baseline, integration, general MS transforms
  - ms_plot.py — Plot primitives for MS/CCS (return matplotlib Figure or Plotly Figure)
  - fitting.py — Peak shapes, baseline models, fitting engines and analyzers
  - io.py — Common CSV parsing utilities, safe conversions, duplicate handling
  - settings.py — Save/load settings dict <-> JSON (optional)

Per-file extraction map and candidates

pages/12_origami.py (CCS conversion + fingerprint)
- Move to myutils/origami.py
  - def safe_float_conversion(value) -> float  (or myutils.io)
  - def remove_duplicate_values(values: Sequence[float], tolerance=1e-6) -> tuple[np.ndarray, np.ndarray]
  - def parse_twim_extract(file) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]
    - returns: trap_cv_values (V), drift_times_ms, intensity dataframe/array
  - def build_fingerprint_matrix(twim_df, trap_cv_values, drift_times_ms, cal_drift_s, cal_ccs, charge) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    - steps: convert drift->CCS via interpolation, sort by CCS, de-duplicate CCS rows
    - returns: ccs_values, trap_cv_values, intensity_matrix
  - def normalize_columns(matrix) -> np.ndarray
  - def interpolate_matrix(ccs, trapcv, matrix, method="linear", multiplier=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]
  - def smooth_matrix_gaussian(matrix, sigma, truncate) -> np.ndarray
  - def smooth_matrix_savgol(matrix, window_length, polyorder, mode) -> np.ndarray
- Move to myutils/ms_plot.py
  - def plot_fingerprint_plotly(ccs, trapcv, matrix, style: dict) -> plotly.graph_objects.Figure
  - def render_fingerprint_png(ccs, trapcv, matrix, style: dict) -> bytes (PNG buffer)
- Streamlit page keeps only: UI widgets, calling the above, and download buttons

pages/11_DTIMS_calibration.py (DTIMS physics + regression)
- Move to myutils/dtims.py
  - @dataclass class PhysicalConstants
  - def parse_dtims_csv(file) -> (pd.DataFrame, list[str], list[str])
  - def find_max_drift_time(df, column) -> (float|None, float|None)
  - def calculate_reduced_mass(m_analyte_da, m_buffer_da) -> float (kg)
  - def calculate_ccs_mason_schamp(drift_time_ms, voltage_v, temperature_k, pressure_mbar, mass_analyte_da, mass_buffer_da, charge=1, length_cm=25.05, return_mobility=False)
  - def calculate_true_voltage(helium_cell_dc, bias, transfer_dc_entrance, helium_exit_dc) -> float
  - def perform_linear_regression(x, y) -> dict (slope, intercept, r2, etc.)
- Move plotting helpers (calibration, CCS-voltage) to myutils/ms_plot.py (return Figures)

pages/10_ESIProt.py (ESI deconvolution)
- Move to myutils/esiprot.py
  - def calc_mw(mz_values) -> dict (charges, mw, errors, average, stdev)
  - def calc_mz_values(mass, charge_min, charge_max) -> list[dict]
  - def create_download_data(results, mz_inputs) -> pd.DataFrame
  - def create_calculation_download_data(calculations) -> pd.DataFrame
- UI tab functions call the above and render results

pages/9_generate_range_files.py (TWIMExtract ranges)
- Move to myutils/range_files.py
  - @dataclass class RangeParams
  - @dataclass class RangeFileResult
  - class RangeFileGenerator with calculate_mz, generate_range_content, generate_all_files
  - class OutputGenerator.generate_zip
- UI class stays in page and calls myutils

pages/8_plot_pretty_MS.py (plotting/processing)
- Move to myutils/ms_processing.py
  - SafeSpectrumData
  - read_spectrum_file_safe(file_or_path)
  - apply_processing(spectrum_data, options)
  - calculate_safe_y_limits(...)
- Move to myutils/ms_plot.py
  - add_annotations_safe, create_safe_plot, add_custom_spectrum_label
  - get_color_palette, apply_plot_styling, add_vertical_lines_safe
- Ensure functions return Figure objects and accept style dicts; Streamlit handles display

pages/6_fit_data.py (peak models + fitting engines)
- Move to myutils/fitting.py
  - Peak shape functions: gaussian_peak, lorentzian_peak, voigt_peak, asymmetric_gaussian, EMG, bigaussian
  - Baselines: linear_baseline, polynomial_baseline, exponential_baseline
  - Engines/classes: OriginPeakDetector, OriginParameterEstimator, OriginParameterManager, OriginFittingEngine, OriginDataProcessor, OriginResultAnalyzer, CCSDDataProcessor, OriginPeakManager
- Keep only UI orchestration/plots in the page

pages/5_plot_CCSDs.py
- Move core computations (smoothing, local extrema, transformations) to myutils/ms_processing.py
- Keep plotting in myutils/ms_plot.py

pages/4_get_calibrated_scaled_data.py
- Move to myutils/ms_processing.py
  - fit_baseline_and_integrate, get_automatic_range, core logic in CalibratedDriftProcessor
- Plot primitives to myutils/ms_plot.py

pages/1_calibrate.py
- Move the calibration math and file parsing to myutils/calibration.py
- Retain Streamlit-only UI components in page

pages/2_get_input_files.py and 3_process_output_files.py
- Move data classes and processors to myutils/io.py (or split input_processing.py and output_processing.py)
- Keep `UI` classes in the pages

Existing myutils modules
- data_tools.py: consider merging gaussian fit utilities with fitting.py and centralizing r_squared
- import_tools.py: stays as-is (zip handling and cached reads)
- styling.py: UI-only; keep in place

Small, testable contracts (examples)
- origami.interpolate_matrix:
  - inputs: ccs: 1D array (strictly increasing), trapcv: 1D array (strictly increasing), matrix: 2D array, method in {"linear","cubic"}, multiplier:int>=1
  - outputs: (ccs_new, trapcv_new, matrix_new)
  - errors: ValueError if inputs non-monotonic or shapes mismatch
- dtims.calculate_ccs_mason_schamp:
  - inputs: physical params; returns CCS in Å² (and optionally mobility)
  - errors: NaN or ValueError on invalid parameters

Minimal unit test plan (pytest)
- tests/test_dtims.py: sanity/monotonic tests for Mason–Schamp; regression helper
- tests/test_origami.py: parsing sample TWIM extract, duplicate removal, interpolation shape
- tests/test_esiprot.py: calc_mw consistency on synthetic m/z; calc_mz_values ranges
- tests/test_ms_processing.py: baseline/integration and smoothing boundaries
- tests/test_range_files.py: content formatting and zip packaging
- tests/test_ms_plot.py: Agg backend render smoke tests (no UI)

Optional CLI entry points (headless)
- nativeims-origami: convert TWIM+calibration -> fingerprint CSV/PNG
- nativeims-dtims: compute CCS for CSV of drift times and conditions
- nativeims-esiprot: deconvolute m/z list

Implementation order (low risk → high impact)
1) Extract origami core (parse, build matrix, interpolate/smooth) and dtims physics
2) Add pytest with small fixtures in tests/data/
3) Extract ESIProt, range files
4) Move plotting primitives (ms_plot)
5) Extract fitting engines

Reviewer local run (towards JOSS)
- Core import: `from myutils.origami import build_fingerprint_matrix` (no Streamlit)
- Tests: `pytest -q`
- UI: `streamlit run app.py` remains unchanged

This plan keeps your Streamlit pages as thin UI shells while exposing a clean, reusable core suitable for testing and reuse.