import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import interpolate, ndimage
from scipy.signal import savgol_filter
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

st.title("CCS Conversion and Fingerprint Analysis")
st.write("Convert drift times to CCS values and create fingerprint heatmaps")

# Settings save/load functionality
def save_settings_to_dict():
    """Save current settings to a dictionary"""
    settings = {}
    
    # Data settings
    if 'is_cyclic' in st.session_state:
        settings['is_cyclic'] = st.session_state.is_cyclic
    if 'inject_time' in st.session_state:
        settings['inject_time'] = st.session_state.inject_time
    if 'interp_multiplier' in st.session_state:
        settings['interp_multiplier'] = st.session_state.interp_multiplier
    if 'interp_method' in st.session_state:
        settings['interp_method'] = st.session_state.interp_method
    if 'normalize_cv' in st.session_state:
        settings['normalize_cv'] = st.session_state.normalize_cv
    
    # Smoothing settings
    if 'apply_smoothing' in st.session_state:
        settings['apply_smoothing'] = st.session_state.apply_smoothing
    if 'smoothing_method' in st.session_state:
        settings['smoothing_method'] = st.session_state.smoothing_method
    if 'gaussian_sigma' in st.session_state:
        settings['gaussian_sigma'] = st.session_state.gaussian_sigma
    if 'gaussian_truncate' in st.session_state:
        settings['gaussian_truncate'] = st.session_state.gaussian_truncate
    if 'sg_window_length' in st.session_state:
        settings['sg_window_length'] = st.session_state.sg_window_length
    if 'sg_polyorder' in st.session_state:
        settings['sg_polyorder'] = st.session_state.sg_polyorder
    if 'sg_mode' in st.session_state:
        settings['sg_mode'] = st.session_state.sg_mode
    
    # Figure customization
    if 'use_custom_color' in st.session_state:
        settings['use_custom_color'] = st.session_state.use_custom_color
    if 'hex_color' in st.session_state:
        settings['hex_color'] = st.session_state.hex_color
    if 'color_scheme' in st.session_state:
        settings['color_scheme'] = st.session_state.color_scheme
    if 'reverse_colors' in st.session_state:
        settings['reverse_colors'] = st.session_state.reverse_colors
    if 'font_family' in st.session_state:
        settings['font_family'] = st.session_state.font_family
    if 'font_size' in st.session_state:
        settings['font_size'] = st.session_state.font_size
    if 'figure_width_inches' in st.session_state:
        settings['figure_width_inches'] = st.session_state.figure_width_inches
    if 'figure_height_inches' in st.session_state:
        settings['figure_height_inches'] = st.session_state.figure_height_inches
    if 'figure_dpi' in st.session_state:
        settings['figure_dpi'] = st.session_state.figure_dpi
    if 'show_colorbar' in st.session_state:
        settings['show_colorbar'] = st.session_state.show_colorbar
    
    # Axis limits
    if 'auto_x_limits' in st.session_state:
        settings['auto_x_limits'] = st.session_state.auto_x_limits
    if 'x_min' in st.session_state:
        settings['x_min'] = st.session_state.x_min
    if 'x_max' in st.session_state:
        settings['x_max'] = st.session_state.x_max
    if 'auto_y_limits' in st.session_state:
        settings['auto_y_limits'] = st.session_state.auto_y_limits
    if 'y_min' in st.session_state:
        settings['y_min'] = st.session_state.y_min
    if 'y_max' in st.session_state:
        settings['y_max'] = st.session_state.y_max
    
    # Other settings
    if 'colorbar_title' in st.session_state:
        settings['colorbar_title'] = st.session_state.colorbar_title
    if 'custom_title' in st.session_state:
        settings['custom_title'] = st.session_state.custom_title
    
    return settings

def load_settings_from_dict(settings):
    """Load settings from dictionary into session state"""
    for key, value in settings.items():
        st.session_state[key] = value

def safe_float_conversion(value):
    """Safely convert a value to float, handling sequences and invalid data"""
    try:
        # If it's already a number, return it
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        
        # If it's a string, try to convert
        if isinstance(value, str):
            value = value.strip()
            if value == '' or value.lower() in ['nan', 'null', 'none']:
                return 0.0
            return float(value)
        
        # If it's a sequence (list, array, etc.), take the first valid element
        if hasattr(value, '__iter__') and not isinstance(value, str):
            for item in value:
                try:
                    return safe_float_conversion(item)
                except:
                    continue
            return 0.0
        
        # If all else fails, return 0
        return 0.0
        
    except (ValueError, TypeError, AttributeError):
        return 0.0

def remove_duplicate_values(values, tolerance=1e-6):
    """Remove duplicate values from an array, keeping the first occurrence"""
    if len(values) <= 1:
        return values, []
    
    values = np.array(values)
    unique_mask = np.ones(len(values), dtype=bool)
    
    for i in range(1, len(values)):
        # Check if current value is too close to any previous value
        for j in range(i):
            if unique_mask[j] and abs(values[i] - values[j]) < tolerance:
                unique_mask[i] = False
                break
    
    removed_indices = np.where(~unique_mask)[0]
    return values[unique_mask], removed_indices

# Initialize settings loaded flag if not exists
if 'settings_loaded' not in st.session_state:
    st.session_state.settings_loaded = False

# Settings management section
with st.expander("Settings Management"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Save Settings")
        if st.button("Save Current Settings"):
            settings = save_settings_to_dict()
            settings_json = json.dumps(settings, indent=2)
            
            st.download_button(
                label="Download Settings File",
                data=settings_json,
                file_name="origami_settings.json",
                mime="application/json",
                help="Download your current settings to reuse later"
            )
            st.success("Settings ready for download!")
    
    with col2:
        st.subheader("Load Settings")
        
        # Clear file uploader button
        if st.button("Clear Settings File"):
            # Clear the file uploader by resetting the key
            if 'settings_file_key' not in st.session_state:
                st.session_state.settings_file_key = 0
            st.session_state.settings_file_key += 1
            st.session_state.settings_loaded = False
            st.rerun()
        
        # File uploader with dynamic key to allow clearing
        if 'settings_file_key' not in st.session_state:
            st.session_state.settings_file_key = 0
            
        settings_file = st.file_uploader(
            "Upload Settings File",
            type=['json'],
            help="Upload a previously saved settings file",
            key=f"settings_uploader_{st.session_state.settings_file_key}"
        )
        
        # Only process the file if it's uploaded and we haven't already loaded these settings
        if settings_file is not None and not st.session_state.settings_loaded:
            try:
                settings = json.loads(settings_file.read().decode('utf-8'))
                load_settings_from_dict(settings)
                st.session_state.settings_loaded = True
                st.success("Settings loaded successfully! The page will refresh with your saved settings.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading settings: {str(e)}")
        
        # Reset the loaded flag when no file is uploaded
        if settings_file is None:
            st.session_state.settings_loaded = False

# File uploaders
st.header("File Upload")
col1, col2 = st.columns(2)

with col1:
    calibration_file = st.file_uploader(
        "Upload Calibration File",
        type=['csv', 'txt'],
        help="CSV file with columns: Z, Drift, CCS, CCS Std.Dev."
    )

with col2:
    twim_file = st.file_uploader(
        "Upload TWIM Extract File",
        type=['csv', 'txt'],
        help="CSV file from TWIMExtract with drift times and intensities"
    )

# Early charge state selection and data loading
if calibration_file and twim_file:
    try:
        # Read calibration file
        cal_df = pd.read_csv(calibration_file)
        
        # Remove rows where error is larger than CCS value
        initial_rows = len(cal_df)
        cal_df = cal_df[cal_df['CCS Std.Dev.'] <= cal_df['CCS']]
        removed_rows = initial_rows - len(cal_df)
        
        st.success(f"Calibration file loaded: {len(cal_df)} calibration points")
        if removed_rows > 0:
            st.warning(f"Removed {removed_rows} calibration points where error ≥ CCS value")
        
        # Read TWIM extract file with improved parsing
        twim_content = twim_file.read().decode('utf-8').split('\n')
        
        # Find the TrapCV row
        trap_cv_values = None
        data_start_idx = None
        
        for i, line in enumerate(twim_content):
            if line.startswith('$TrapCV:'):
                trap_cv_str = line.split(',')[1:]
                trap_cv_values = []
                for x in trap_cv_str:
                    cleaned = x.strip()
                    if cleaned != '':
                        try:
                            trap_cv_values.append(float(cleaned))
                        except ValueError:
                            st.warning(f"Could not parse TrapCV value: {cleaned}")
                            continue
                data_start_idx = i + 1
                break
        
        if trap_cv_values is None or len(trap_cv_values) == 0:
            st.error("Could not find valid $TrapCV: row in the TWIM extract file")
            st.stop()
        
        # Remove duplicate TrapCV values BEFORE processing data
        original_trap_cv_count = len(trap_cv_values)
        trap_cv_values_clean, removed_trapcv_indices = remove_duplicate_values(trap_cv_values)
        
        if len(removed_trapcv_indices) > 0:
            st.warning(f"Removed {len(removed_trapcv_indices)} duplicate TrapCV values: {[trap_cv_values[i] for i in removed_trapcv_indices]}")
            st.info(f"Original TrapCV count: {original_trap_cv_count} → Clean TrapCV count: {len(trap_cv_values_clean)}")
        
        # Update trap_cv_values with cleaned version
        trap_cv_values = trap_cv_values_clean
        
        # Parse the data rows with improved error handling
        data_rows = []
        invalid_rows = 0
        
        for line_num, line in enumerate(twim_content[data_start_idx:], start=data_start_idx+1):
            if line.strip() == '':
                continue
            
            values = line.split(',')
            if len(values) <= 1:
                continue
            
            try:
                # Parse drift time
                drift_time = safe_float_conversion(values[0])
                
                # Parse intensities with safe conversion, but only for non-duplicate TrapCV columns
                intensities = []
                for i in range(1, len(values)):
                    original_trapcv_idx = i - 1
                    
                    # Skip if this TrapCV index was marked as duplicate
                    if original_trapcv_idx in removed_trapcv_indices:
                        continue
                    
                    # Only process up to the number of clean TrapCV values
                    if len(intensities) < len(trap_cv_values):
                        intensity_val = safe_float_conversion(values[i])
                        intensities.append(intensity_val)
                
                # Pad with zeros if needed
                while len(intensities) < len(trap_cv_values):
                    intensities.append(0.0)
                
                # Only take the first len(trap_cv_values) intensities
                intensities = intensities[:len(trap_cv_values)]
                
                data_rows.append([drift_time] + intensities)
                
            except Exception as e:
                invalid_rows += 1
                if invalid_rows <= 5:  # Only show first 5 errors
                    st.warning(f"Error parsing line {line_num}: {str(e)}")
                continue
        
        if invalid_rows > 5:
            st.warning(f"... and {invalid_rows - 5} more parsing errors")
        
        if len(data_rows) == 0:
            st.error("No valid data rows found in TWIM extract file")
            st.stop()
        
        # Create DataFrame with validated data
        columns = ['Drift_Time'] + [f'TrapCV_{cv}' for cv in trap_cv_values]
        twim_df = pd.DataFrame(data_rows, columns=columns)
        
        # Additional validation - ensure all columns are numeric
        for col in twim_df.columns:
            if col != 'Drift_Time':
                twim_df[col] = twim_df[col].apply(safe_float_conversion)
        
        st.success(f"TWIM extract file loaded: {len(twim_df)} drift time points, {len(trap_cv_values)} TrapCV values")
        if invalid_rows > 0:
            st.info(f"Skipped {invalid_rows} invalid rows during parsing")
        
        # Charge state selection - do this FIRST
        charge_states = cal_df['Z'].unique()
        
        if len(charge_states) > 1:
            selected_charge = st.selectbox(
                "Select charge state for CCS conversion:",
                charge_states,
                help="Multiple charge states found in calibration file"
            )
        else:
            selected_charge = charge_states[0]
            st.info(f"Using charge state: {selected_charge}")
        
        # Data settings
        st.header("Data Settings")
        col1, col2, col3 = st.columns(3)

        with col1:
            is_cyclic = st.checkbox(
                "Is this cyclic data?",
                key="is_cyclic"
            )
            inject_time = None
            if is_cyclic:
                inject_time = st.number_input(
                    "Inject time (ms)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    help="Inject time in milliseconds to subtract from drift times",
                    key="inject_time"
                )

        with col2:
            interp_multiplier = st.number_input(
                "Interpolation multiplier",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                help="Multiply the number of data points in both dimensions (e.g., 4x = 16x total points)",
                key="interp_multiplier"
            )
            
            interp_method = st.selectbox(
                "Interpolation method",
                ["linear", "cubic"],
                help="Choose interpolation method for adding data points",
                key="interp_method"
            )

        with col3:
            normalize_cv = st.checkbox(
                "Normalize each CV slice",
                help="Normalize intensity in each TrapCV column to its maximum value",
                key="normalize_cv"
            )

        # Smoothing settings
        st.header("Smoothing Settings")
        apply_smoothing = st.checkbox(
            "Apply smoothing",
            key="apply_smoothing"
        )

        if apply_smoothing:
            smoothing_method = st.selectbox(
                "Smoothing method",
                ["Gaussian", "Savitzky-Golay"],
                help="Choose smoothing method",
                key="smoothing_method"
            )
            
            col1, col2, col3 = st.columns(3)
            
            if smoothing_method == "Gaussian":
                with col1:
                    gaussian_sigma = st.number_input(
                        "Gaussian sigma",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        help="Standard deviation for Gaussian kernel (higher = more smoothing)",
                        key="gaussian_sigma"
                    )
                with col2:
                    gaussian_truncate = st.number_input(
                        "Truncate",
                        min_value=1.0,
                        max_value=8.0,
                        value=4.0,
                        step=0.5,
                        help="Truncate the filter at this many standard deviations",
                        key="gaussian_truncate"
                    )
            
            elif smoothing_method == "Savitzky-Golay":
                with col1:
                    sg_window_length = st.number_input(
                        "Window length",
                        min_value=5,
                        max_value=51,
                        value=11,
                        step=2,
                        help="Length of the filter window (must be odd and >= polyorder + 2)",
                        key="sg_window_length"
                    )
                    if sg_window_length % 2 == 0:
                        sg_window_length += 1
                        st.warning(f"Window length adjusted to {sg_window_length} (must be odd)")
                
                with col2:
                    sg_polyorder = st.number_input(
                        "Polynomial order",
                        min_value=1,
                        max_value=min(6, sg_window_length - 1),
                        value=3,
                        step=1,
                        help="Order of polynomial used to fit samples",
                        key="sg_polyorder"
                    )
                    
                with col3:
                    sg_mode = st.selectbox(
                        "Edge mode",
                        ["mirror", "constant", "nearest", "wrap"],
                        index=0,
                        help="How to handle edges of the data",
                        key="sg_mode"
                    )

        # Figure customization settings
        st.header("Figure Customization")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Custom color scheme with hex code
            use_custom_color = st.checkbox(
                "Use custom color scheme",
                key="use_custom_color"
            )
            
            if use_custom_color:
                hex_color = st.text_input(
                    "Hex color (100% intensity)",
                    value="#FF0000",
                    help="Hex color code for maximum intensity (e.g., #FF0000 for red)",
                    key="hex_color"
                )
                
                # Validate hex color
                try:
                    # Remove # if present and validate
                    hex_clean = hex_color.lstrip('#')
                    if len(hex_clean) != 6:
                        st.error("Please enter a valid 6-digit hex color code")
                        hex_color = "#FF0000"
                    else:
                        int(hex_clean, 16)  # Test if valid hex
                except ValueError:
                    st.error("Please enter a valid hex color code")
                    hex_color = "#FF0000"
            else:
                color_scheme = st.selectbox(
                    "Color scheme",
                    ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Reds", "Greens", 
                     "YlOrRd", "YlGnBu", "RdYlBu", "Spectral", "Coolwarm", "Jet", "Hot", "Cool"],
                    help="Choose color scheme for the heatmap",
                    key="color_scheme"
                )
            
            reverse_colors = st.checkbox(
                "Reverse colors",
                help="Reverse the color scale",
                key="reverse_colors"
            )

        with col2:
            font_family = st.selectbox(
                "Font family",
                ["Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana", 
                 "Georgia", "Palatino", "Garamond", "Comic Sans MS", "Trebuchet MS"],
                help="Choose font family for all text",
                key="font_family"
            )
            
            font_size = st.number_input(
                "Font size",
                min_value=8,
                max_value=24,
                value=12,
                step=1,
                help="Font size for all text elements",
                key="font_size"
            )

        with col3:
            figure_width_inches = st.number_input(
                "Figure width (inches)",
                min_value=3.0,
                max_value=20.0,
                value=8.0,
                step=0.5,
                help="Width of the figure in inches",
                key="figure_width_inches"
            )
            
            figure_height_inches = st.number_input(
                "Figure height (inches)",
                min_value=3.0,
                max_value=15.0,
                value=6.0,
                step=0.5,
                help="Height of the figure in inches",
                key="figure_height_inches"
            )

        with col4:
            figure_dpi = st.number_input(
                "Figure DPI",
                min_value=72,
                max_value=600,
                value=300,
                step=50,
                help="Resolution in dots per inch for static figures",
                key="figure_dpi"
            )
            
            show_colorbar = st.checkbox(
                "Show colorbar",
                value=True,
                help="Display the color scale bar",
                key="show_colorbar"
            )

        # Axis limits settings
        st.subheader("Axis Limits")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**X-axis (TrapCV) Limits**")
            auto_x_limits = st.checkbox(
                "Auto X-axis limits", 
                value=True,
                key="auto_x_limits"
            )
            if not auto_x_limits:
                x_min = st.number_input(
                    "X-axis minimum", 
                    value=0.0, 
                    step=0.1, 
                    key="x_min"
                )
                x_max = st.number_input(
                    "X-axis maximum", 
                    value=100.0, 
                    step=0.1, 
                    key="x_max"
                )

        with col2:
            st.write("**Y-axis (CCS) Limits**")
            auto_y_limits = st.checkbox(
                "Auto Y-axis limits", 
                value=True,
                key="auto_y_limits"
            )
            if not auto_y_limits:
                y_min = st.number_input(
                    "Y-axis minimum", 
                    value=0.0, 
                    step=10.0, 
                    key="y_min"
                )
                y_max = st.number_input(
                    "Y-axis maximum", 
                    value=1000.0, 
                    step=10.0, 
                    key="y_max"
                )

        # Convert inches to pixels for Plotly (96 DPI for screen display)
        figure_width = int(figure_width_inches * 96)
        figure_height = int(figure_height_inches * 96)

        colorbar_title = st.text_input(
            "Colorbar title",
            value="Intensity",
            help="Title for the colorbar",
            key="colorbar_title"
        )

        # Simplified advanced settings
        with st.expander("Advanced Figure Settings"):
            custom_title = st.text_input(
                "Custom title",
                value="",
                help="Override the default title (leave blank for auto-generated)",
                key="custom_title"
            )

        # Process the data
        if st.button("Generate Fingerprint"):
            with st.spinner("Processing data..."):
                
                # Get drift times (in ms from TWIM extract)
                drift_times_ms = twim_df['Drift_Time'].values
                
                # Apply cyclic correction if specified
                if is_cyclic and inject_time is not None:
                    drift_times_ms = drift_times_ms - inject_time
                
                # Convert drift times from ms to seconds for calibration lookup
                drift_times_s = drift_times_ms / 1000.0
                
                # Filter calibration data for selected charge state
                cal_filtered = cal_df[cal_df['Z'] == selected_charge]
                
                # Create interpolation function for CCS lookup
                if len(cal_filtered) >= 2:
                    # Sort by drift time for interpolation
                    cal_drift_s = cal_filtered['Drift'].values
                    cal_ccs = cal_filtered['CCS'].values
                    
                    sort_idx = np.argsort(cal_drift_s)
                    cal_drift_sorted = cal_drift_s[sort_idx]
                    cal_ccs_sorted = cal_ccs[sort_idx]
                    
                    # Create interpolation function
                    f_interp = interpolate.interp1d(
                        cal_drift_sorted, 
                        cal_ccs_sorted, 
                        kind='linear', 
                        fill_value='extrapolate'
                    )
                    
                    # Convert ALL drift times to CCS first
                    ccs_values_raw = f_interp(drift_times_s)
                    
                    # Create initial fingerprint matrix from original data
                    original_ccs_points = len(ccs_values_raw)
                    original_trapcv_points = len(trap_cv_values)
                    
                    # Create intensity matrix from original data with safe conversion
                    intensity_matrix_original = np.zeros((original_ccs_points, original_trapcv_points))
                    
                    for i in range(len(ccs_values_raw)):
                        for j, trap_cv in enumerate(trap_cv_values):
                            intensity_col = f'TrapCV_{trap_cv}'
                            if intensity_col in twim_df.columns:
                                # Use safe conversion to handle any remaining sequence issues
                                raw_value = twim_df.iloc[i][intensity_col]
                                intensity_matrix_original[i, j] = safe_float_conversion(raw_value)
                            else:
                                intensity_matrix_original[i, j] = 0.0
                    
                    # SORT BY CCS VALUES and remove duplicates/handle non-monotonic issues
                    ccs_sort_idx = np.argsort(ccs_values_raw)
                    ccs_values_sorted = ccs_values_raw[ccs_sort_idx]
                    intensity_matrix_sorted = intensity_matrix_original[ccs_sort_idx, :]
                    
                    # Remove duplicate CCS values that cause interpolation issues
                    # Keep only unique CCS values (within a small tolerance)
                    unique_mask = np.ones(len(ccs_values_sorted), dtype=bool)
                    tolerance = 1e-6  # Small tolerance for floating point comparison
                    
                    for i in range(1, len(ccs_values_sorted)):
                        if abs(ccs_values_sorted[i] - ccs_values_sorted[i-1]) < tolerance:
                            unique_mask[i] = False
                    
                    # Apply the unique mask
                    ccs_values_unique = ccs_values_sorted[unique_mask]
                    intensity_matrix_unique = intensity_matrix_sorted[unique_mask, :]
                    
                    # Update variables to use cleaned data
                    ccs_values = ccs_values_unique
                    intensity_matrix_original = intensity_matrix_unique
                    
                    # Update original points count after cleaning
                    original_ccs_points = len(ccs_values)
                    
                    removed_duplicates = len(ccs_values_sorted) - len(ccs_values_unique)
                    if removed_duplicates > 0:
                        st.info(f"Removed {removed_duplicates} duplicate/near-duplicate CCS values for interpolation stability")
                    
                    st.info(f"Sorted data by CCS values (range: {ccs_values.min():.1f} - {ccs_values.max():.1f} Ų)")
                    st.info(f"Final data dimensions: {len(ccs_values)} CCS values × {len(trap_cv_values)} TrapCV values")
                    
                    # Ensure TrapCV values are monotonic (they should be clean now)
                    trap_cv_values = np.array(trap_cv_values)
                    
                    # Sort TrapCV if not already sorted
                    if not np.all(np.diff(trap_cv_values) > 0):
                        # Sort TrapCV values and corresponding intensity columns
                        trap_cv_sort_idx = np.argsort(trap_cv_values)
                        trap_cv_values = trap_cv_values[trap_cv_sort_idx]
                        intensity_matrix_original = intensity_matrix_original[:, trap_cv_sort_idx]
                        st.info("Sorted TrapCV values for interpolation compatibility")
                    
                    # Final verification of TrapCV monotonicity
                    trapcv_diffs = np.diff(trap_cv_values)
                    if np.any(trapcv_diffs <= 0):
                        st.error(f"TrapCV values still not strictly increasing after cleaning. Min diff: {np.min(trapcv_diffs)}")
                        st.error(f"TrapCV values: {trap_cv_values}")
                        st.stop()
                    
                    # Apply CV slice normalization if requested
                    if normalize_cv:
                        for j in range(intensity_matrix_original.shape[1]):
                            col_max = np.max(intensity_matrix_original[:, j])
                            if col_max > 0:
                                intensity_matrix_original[:, j] = intensity_matrix_original[:, j] / col_max
                        st.info("Applied CV slice normalization (each TrapCV column normalized to its maximum)")
                    
                    # Apply 2D interpolation if multiplier > 1 (now on cleaned CCS data)
                    if interp_multiplier > 1:
                        new_ccs_points = original_ccs_points * interp_multiplier
                        new_trapcv_points = original_trapcv_points * interp_multiplier
                        
                        # Create new interpolated grids in CCS space
                        new_ccs_values = np.linspace(ccs_values.min(), ccs_values.max(), new_ccs_points)
                        new_trap_cv_values = np.linspace(trap_cv_values.min(), trap_cv_values.max(), new_trapcv_points)
                        
                        # Verify that original coordinates are strictly monotonic
                        ccs_diffs = np.diff(ccs_values)
                        trapcv_diffs = np.diff(trap_cv_values)
                        
                        if np.any(ccs_diffs <= 0):
                            st.error(f"CCS values are not strictly increasing. Min diff: {np.min(ccs_diffs)}")
                            st.stop()
                        
                        if np.any(trapcv_diffs <= 0):
                            st.error(f"TrapCV values are not strictly increasing. Min diff: {np.min(trapcv_diffs)}")
                            st.stop()
                        
                        # Perform 2D interpolation on the CCS matrix
                        try:
                            if interp_method == 'linear':
                                # Use RegularGridInterpolator for linear interpolation
                                interp_func = interpolate.RegularGridInterpolator(
                                    (ccs_values, trap_cv_values),
                                    intensity_matrix_original,
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=0.0
                                )
                                
                                # Create meshgrids for interpolation
                                new_trapcv_grid, new_ccs_grid = np.meshgrid(new_trap_cv_values, new_ccs_values)
                                new_points = np.array([new_ccs_grid.ravel(), new_trapcv_grid.ravel()]).T
                                new_intensities = interp_func(new_points).reshape(new_ccs_points, new_trapcv_points)
                                
                            else:  # cubic
                                # Use RectBivariateSpline for cubic interpolation
                                interp_func = interpolate.RectBivariateSpline(
                                    ccs_values,
                                    trap_cv_values,
                                    intensity_matrix_original,
                                    kx=min(3, len(ccs_values)-1),
                                    ky=min(3, len(trap_cv_values)-1)
                                )
                                
                                new_intensities = interp_func(new_ccs_values, new_trap_cv_values)
                            
                            # Update variables with interpolated data
                            ccs_values = new_ccs_values
                            trap_cv_values = new_trap_cv_values
                            intensity_matrix_final = new_intensities
                            
                            st.info(f"2D interpolation: {original_ccs_points}×{original_trapcv_points} → {new_ccs_points}×{new_trapcv_points} points using {interp_method} method")
                            st.info(f"Total points increased from {original_ccs_points * original_trapcv_points} to {new_ccs_points * new_trapcv_points}")
                            
                        except Exception as interp_error:
                            st.error(f"Interpolation failed: {str(interp_error)}")
                            st.info("Using original data without interpolation")
                            intensity_matrix_final = intensity_matrix_original
                            
                    else:
                        intensity_matrix_final = intensity_matrix_original
                    
                    # Apply smoothing if requested
                    if apply_smoothing:
                        if smoothing_method == "Gaussian":
                            intensity_matrix_final = ndimage.gaussian_filter(
                                intensity_matrix_final, 
                                sigma=gaussian_sigma,
                                truncate=gaussian_truncate
                            )
                            st.info(f"Applied Gaussian smoothing (σ={gaussian_sigma}, truncate={gaussian_truncate})")
                        
                        elif smoothing_method == "Savitzky-Golay":
                            # Apply Savitzky-Golay filter to each row and column
                            # First apply to rows (TrapCV direction)
                            for i in range(intensity_matrix_final.shape[0]):
                                if intensity_matrix_final.shape[1] >= sg_window_length:
                                    intensity_matrix_final[i, :] = savgol_filter(
                                        intensity_matrix_final[i, :], 
                                        sg_window_length, 
                                        sg_polyorder,
                                        mode=sg_mode
                                    )
                            
                            # Then apply to columns (CCS direction)
                            for j in range(intensity_matrix_final.shape[1]):
                                if intensity_matrix_final.shape[0] >= sg_window_length:
                                    intensity_matrix_final[:, j] = savgol_filter(
                                        intensity_matrix_final[:, j], 
                                        sg_window_length, 
                                        sg_polyorder,
                                        mode=sg_mode
                                    )
                            
                            st.info(f"Applied Savitzky-Golay smoothing (window={sg_window_length}, poly_order={sg_polyorder}, mode={sg_mode})")
                    
                    # Prepare color scheme
                    if use_custom_color:
                        # Create custom colorscale from white to hex color
                        hex_clean = hex_color.lstrip('#')
                        rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
                        
                        if reverse_colors:
                            # Hex color at 0, white at 1
                            colorscale = [
                                [0, f'rgb{rgb}'],
                                [1, 'rgb(255, 255, 255)']
                            ]
                        else:
                            # White at 0, hex color at 1
                            colorscale = [
                                [0, 'rgb(255, 255, 255)'],
                                [1, f'rgb{rgb}']
                            ]
                    else:
                        colorscale = color_scheme.lower()
                        if reverse_colors:
                            colorscale += "_r"
                    
                    # Create fingerprint data for download
                    fingerprint_data = []
                    for i, ccs in enumerate(ccs_values):
                        for j, trap_cv in enumerate(trap_cv_values):
                            intensity = intensity_matrix_final[i, j]
                            fingerprint_data.append({
                                'TrapCV': trap_cv,
                                'CCS': ccs,
                                'Intensity': intensity
                            })
                    
                    fingerprint_df = pd.DataFrame(fingerprint_data)
                    
                    if len(fingerprint_df) > 0:
                        # Create heatmap with custom styling
                        colorbar_config = dict(
                            title=colorbar_title,
                            titlefont=dict(size=font_size, family=font_family),
                            tickfont=dict(size=font_size, family=font_family)
                        ) if show_colorbar else None
                        
                        fig_matrix = go.Figure(data=go.Heatmap(
                            z=intensity_matrix_final,
                            x=trap_cv_values,
                            y=ccs_values,
                            colorscale=colorscale,
                            colorbar=colorbar_config,
                            showscale=show_colorbar
                        ))
                        
                        # Build title with processing info
                        if custom_title:
                            title = custom_title
                        else:
                            title_parts = [f'CCS Fingerprint Matrix (Charge State {selected_charge})']
                            if interp_multiplier > 1:
                                title_parts.append(f'{interp_method.capitalize()} {interp_multiplier}x Interpolation')
                            if normalize_cv:
                                title_parts.append('CV Normalized')
                            if apply_smoothing:
                                title_parts.append(f'{smoothing_method} Smoothing')
                            title = ' - '.join(title_parts)
                        
                        fig_matrix.update_layout(
                            title=dict(
                                text=title,
                                font=dict(size=font_size, family=font_family),
                                x=0.5
                            ),
                            xaxis=dict(
                                title=dict(
                                    text='Trap CV (V)',
                                    font=dict(size=font_size, family=font_family)
                                ),
                                tickfont=dict(size=font_size, family=font_family),
                                showline=True,
                                linewidth=2,
                                linecolor='black',
                                mirror=True
                            ),
                            yaxis=dict(
                                title=dict(
                                    text='Collision Cross Section (Å²)',
                                    font=dict(size=font_size, family=font_family)
                                ),
                                tickfont=dict(size=font_size, family=font_family),
                                showline=True,
                                linewidth=2,
                                linecolor='black',
                                mirror=True
                            ),
                            width=figure_width,
                            height=figure_height,
                            font=dict(size=font_size, family=font_family),
                            plot_bgcolor='white'
                        )
                        
                        # Apply axis limits if specified
                        if not auto_x_limits:
                            fig_matrix.update_xaxes(range=[x_min, x_max])
                        if not auto_y_limits:
                            fig_matrix.update_yaxes(range=[y_min, y_max])
                        
                        st.plotly_chart(fig_matrix, use_container_width=True)
                        
                        # Display processing info
                        st.info(f"Matrix dimensions: {len(ccs_values)} CCS values × {len(trap_cv_values)} TrapCV values")
                        
                        processing_steps = []
                        if normalize_cv:
                            processing_steps.append("CV slice normalization")
                        if interp_multiplier > 1:
                            processing_steps.append(f"{interp_multiplier}x {interp_method} interpolation")
                        if apply_smoothing:
                            processing_steps.append(f"{smoothing_method} smoothing")
                        
                        if processing_steps:
                            st.info(f"Applied: {', '.join(processing_steps)}")
                        
                        # Download options
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Download data
                            download_df = fingerprint_df[['TrapCV', 'CCS', 'Intensity']].copy()
                            download_df = download_df.sort_values(['TrapCV', 'CCS'])
                            
                            csv_buffer = io.StringIO()
                            download_df.to_csv(csv_buffer, index=False)
                            
                            # Create filename with processing info
                            filename_parts = [f"ccs_fingerprint_z{selected_charge}"]
                            if interp_multiplier > 1:
                                filename_parts.append(f"{interp_method}_{interp_multiplier}x")
                            if normalize_cv:
                                filename_parts.append("normalized")
                            if apply_smoothing:
                                filename_parts.append(f"{smoothing_method.lower()}_smooth")
                            
                            filename = "_".join(filename_parts) + ".csv"
                            
                            st.download_button(
                                label="Download CCS Fingerprint Data",
                                data=csv_buffer.getvalue(),
                                file_name=filename,
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Download interactive figure as HTML
                            html_buffer = io.StringIO()
                            fig_matrix.write_html(html_buffer)
                            
                            html_filename = filename.replace('.csv', '_interactive.html')
                            
                            st.download_button(
                                label="Download Interactive Figure (HTML)",
                                data=html_buffer.getvalue(),
                                file_name=html_filename,
                                mime="text/html"
                            )
                        
                        with col3:
                            # Download static figure as PNG using matplotlib
                            # Create matplotlib figure
                            fig_static, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches), dpi=figure_dpi)
                            
                            # Convert colorscale for matplotlib
                            if use_custom_color:
                                # Create custom matplotlib colormap
                                from matplotlib.colors import LinearSegmentedColormap
                                hex_clean = hex_color.lstrip('#')
                                rgb_norm = tuple(int(hex_clean[i:i+2], 16)/255.0 for i in (0, 2, 4))
                                
                                if reverse_colors:
                                    # Hex color at 0, white at 1
                                    colors = [rgb_norm, (1.0, 1.0, 1.0)]
                                else:
                                    # White at 0, hex color at 1
                                    colors = [(1.0, 1.0, 1.0), rgb_norm]
                                
                                cmap = LinearSegmentedColormap.from_list("custom", colors)
                            else:
                                # Use standard matplotlib colormaps
                                colormap_name = color_scheme.lower()
                                if colormap_name == 'viridis':
                                    cmap = plt.cm.viridis
                                elif colormap_name == 'plasma':
                                    cmap = plt.cm.plasma
                                elif colormap_name == 'inferno':
                                    cmap = plt.cm.inferno
                                elif colormap_name == 'magma':
                                    cmap = plt.cm.magma
                                elif colormap_name == 'cividis':
                                    cmap = plt.cm.cividis
                                elif colormap_name == 'blues':
                                    cmap = plt.cm.Blues
                                elif colormap_name == 'reds':
                                    cmap = plt.cm.Reds
                                elif colormap_name == 'greens':
                                    cmap = plt.cm.Greens
                                elif colormap_name == 'ylord':
                                    cmap = plt.cm.YlOrRd
                                elif colormap_name == 'ylgnbu':
                                    cmap = plt.cm.YlGnBu
                                elif colormap_name == 'rdylbu':
                                    cmap = plt.cm.RdYlBu
                                elif colormap_name == 'spectral':
                                    cmap = plt.cm.Spectral
                                elif colormap_name == 'coolwarm':
                                    cmap = plt.cm.coolwarm
                                elif colormap_name == 'jet':
                                    cmap = plt.cm.jet
                                elif colormap_name == 'hot':
                                    cmap = plt.cm.hot
                                elif colormap_name == 'cool':
                                    cmap = plt.cm.cool
                                else:
                                    cmap = plt.cm.viridis
                                
                                if reverse_colors and not use_custom_color:
                                    cmap = cmap.reversed()
                            
                            # Create coordinate grids that match the actual data coordinates
                            # Need to create edges for pcolormesh (one more point than data)
                            if len(trap_cv_values) > 1:
                                trap_cv_spacing = (trap_cv_values[-1] - trap_cv_values[0]) / (len(trap_cv_values) - 1)
                                trap_cv_edges = np.linspace(
                                    trap_cv_values[0] - trap_cv_spacing/2,
                                    trap_cv_values[-1] + trap_cv_spacing/2,
                                    len(trap_cv_values) + 1
                                )
                            else:
                                trap_cv_edges = np.array([trap_cv_values[0] - 0.5, trap_cv_values[0] + 0.5])
                            
                            if len(ccs_values) > 1:
                                ccs_spacing = (ccs_values[-1] - ccs_values[0]) / (len(ccs_values) - 1)
                                ccs_edges = np.linspace(
                                    ccs_values[0] - ccs_spacing/2,
                                    ccs_values[-1] + ccs_spacing/2,
                                    len(ccs_values) + 1
                                )
                            else:
                                ccs_edges = np.array([ccs_values[0] - 0.5, ccs_values[0] + 0.5])
                            
                            # Create meshgrids for pcolormesh
                            X_edges, Y_edges = np.meshgrid(trap_cv_edges, ccs_edges)
                            
                            # Use pcolormesh instead of imshow for proper coordinate handling
                            im = ax.pcolormesh(
                                X_edges,
                                Y_edges,
                                intensity_matrix_final,
                                cmap=cmap,
                                shading='flat'
                            )
                            
                            # Set axis limits if specified
                            if not auto_x_limits:
                                ax.set_xlim(x_min, x_max)
                            if not auto_y_limits:
                                ax.set_ylim(y_min, y_max)
                            
                            # Set labels and title
                            ax.set_xlabel('Trap CV (V)', fontsize=font_size, fontfamily=font_family.lower())
                            ax.set_ylabel('Collision Cross Section (Å²)', fontsize=font_size, fontfamily=font_family.lower())
                            ax.set_title(title, fontsize=font_size, fontfamily=font_family.lower(), pad=20)
                            
                            # Add black box around the plot
                            for spine in ax.spines.values():
                                spine.set_edgecolor('black')
                                spine.set_linewidth(2)
                            
                            # Add colorbar if requested
                            if show_colorbar:
                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label(colorbar_title, fontsize=font_size, fontfamily=font_family.lower())
                                cbar.ax.tick_params(labelsize=font_size)
                                # Add black border to colorbar
                                cbar.outline.set_edgecolor('black')
                                cbar.outline.set_linewidth(2)
                            
                            # Set tick font sizes
                            ax.tick_params(axis='both', which='major', labelsize=font_size, colors='black')
                            
                            # Adjust layout automatically
                            plt.tight_layout()
                            
                            # Save to buffer
                            png_buffer = io.BytesIO()
                            plt.savefig(png_buffer, format='png', dpi=figure_dpi, bbox_inches='tight')
                            plt.close(fig_static)
                            
                            png_filename = filename.replace('.csv', '_static.png')
                            
                            st.download_button(
                                label="Download Static Figure (PNG)",
                                data=png_buffer.getvalue(),
                                file_name=png_filename,
                                mime="image/png"
                            )
                        
                    else:
                        st.error("No data available for heatmap generation")
                        
                else:
                    st.error("Insufficient calibration points for interpolation. Need at least 2 points.")
                
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        import traceback
        st.error("Full error details:")
        st.code(traceback.format_exc())

else:
    st.info("Please upload both calibration and TWIM extract files to proceed.")

# Help section
with st.expander("Help"):
    st.markdown("""
    ### Settings Management:
    
    **Save Settings:**
    - Click "Save Current Settings" to prepare your settings for download
    - Download the JSON file to save all your current parameter choices
    - Settings include: data processing options, smoothing parameters, color schemes, figure dimensions, axis limits, etc.
    
    **Load Settings:**
    - Upload a previously saved settings JSON file
    - All parameters will be restored to their saved values
    - The page will refresh automatically with your loaded settings
    - Use "Clear Settings File" button to remove the uploaded file and stop refreshing
    
    ### File Formats Expected:
    
    **Calibration File:**
    - CSV format with columns: Z, Drift, CCS, CCS Std.Dev.
    - Drift times should be in **seconds**
    - Multiple charge states supported
    - Rows with error ≥ CCS value are automatically removed
    
    **TWIM Extract File:**
    - First two rows: Range file names and Raw file names (ignored)
    - Third row: $TrapCV: followed by TrapCV values
    - Subsequent rows: drift_time,intensity1,intensity2,intensity3...
    - Drift times are in **milliseconds**
    - Should have ~200 drift time points per TrapCV value
    
    ### Processing Workflow:
    
    1. **Charge State Selection**: Choose charge state first
    2. **CCS Conversion**: All drift times converted to CCS using calibration
    3. **Data Sorting**: CCS values sorted in ascending order (required for interpolation)
    4. **Duplicate Removal**: Remove duplicate CCS values that cause interpolation issues
    5. **CV Slice Normalization**: Applied to CCS matrix (if enabled)
    6. **2D Interpolation**: Applied to CCS matrix (if multiplier > 1)
    7. **Smoothing**: Applied to final CCS matrix (if enabled)
    
    ### Processing Options:
    
    **2D Interpolation:**
    - **Multiplier**: Increases data points by the specified factor in BOTH dimensions
    - Example: 4x multiplier on 200×50 data = 800×200 = 160,000 total points
    - **Linear**: RegularGridInterpolator for linear 2D interpolation
    - **Cubic**: RectBivariateSpline for smooth cubic 2D interpolation
    - Applied to the CCS-space matrix for accurate scaling
    - Automatically handles coordinate validation and monotonicity
    
    **CV Slice Normalization:**
    - Normalizes each TrapCV column to its maximum intensity value
    - Useful for comparing relative peak shapes across different voltages
    - Applied before interpolation and smoothing
    
    **Smoothing Methods:**
    - **Gaussian**: Standard Gaussian blur with configurable sigma and truncation
    - **Savitzky-Golay**: Polynomial smoothing that preserves peak shapes better than Gaussian
    
    ### Figure Customization:
    
    **Custom Color Scheme:**
    - Use hex color codes for custom color gradients
    - White represents 0% intensity, hex color represents 100%
    - Reverse option swaps this relationship
    
    **Size & Resolution:**
    - Set figure dimensions in inches (width × height)
    - DPI setting controls resolution for static PNG output
    - Interactive figure uses 96 DPI equivalent for screen display
    
    **Axis Limits:**
    - Auto limits: Use full data range
    - Manual limits: Set custom X (TrapCV) and Y (CCS) ranges
    - Applied to both interactive and static figures
    
    **Typography:**
    - Select font family for all text elements
    - Single font size control for consistent appearance
    
    ### Download Options:
    
    **Data (CSV):** Raw fingerprint data with TrapCV, CCS, and Intensity columns
    
    **Interactive Figure (HTML):** 
    - Plotly figure with zoom, pan, hover tooltips
    - Preserves all styling and axis limits
    - Best for data exploration
    
    **Static Figure (PNG):**
    - High-resolution matplotlib figure with black borders
    - Publication-ready quality at specified DPI
    - Exact size control in inches
    - Best for papers and presentations
    
    **Settings (JSON):**
    - Save all current parameter settings
    - Load previously saved settings to reproduce exact results
    - Useful for batch processing multiple datasets with identical parameters
    
    ### Data Validation & Error Handling:
    
    **Automatic Data Cleaning:**
    - Invalid values converted to 0.0
    - Sequences/arrays automatically converted to scalar values
    - Empty cells handled gracefully
    - Parsing errors reported with line numbers
    - Duplicate TrapCV and CCS values automatically removed
    
    **File Format Tolerance:**
    - Handles malformed CSV files
    - Skips invalid rows with detailed reporting
    - Validates all numeric conversions
    - Ensures data integrity before processing
    
    ### Troubleshooting:
    
    **Endless Refresh Issue:**
    - If the page keeps refreshing after loading settings, click "Clear Settings File"
    - This removes the uploaded settings file and stops the refresh loop
    - Your loaded settings will remain active until you change them manually
    """)