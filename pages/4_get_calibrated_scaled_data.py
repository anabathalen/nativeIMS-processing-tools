from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st
import zipfile
import tempfile
import os
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import integrate
from sklearn.linear_model import LinearRegression
import re

from myutils import styling

PROTON_MASS = 1.007276

def fit_baseline_and_integrate(x: np.ndarray, y: np.ndarray, integration_range: Tuple[float, float]) -> Tuple[float, np.ndarray]:
    """
    Fit a linear baseline and integrate the peak above the baseline.
    
    Parameters:
        x: m/z values
        y: intensity values
        integration_range: (min_mz, max_mz) for integration
        
    Returns:
        area: integrated area above baseline
        baseline: fitted baseline values
    """
    # Get data within integration range
    mask = (x >= integration_range[0]) & (x <= integration_range[1])
    if np.sum(mask) < 3:
        return 0.0, np.zeros_like(y)
    
    x_region = x[mask]
    y_region = y[mask]
    
    # Use the first and last 10% of points to fit baseline
    n_points = len(x_region)
    baseline_fraction = max(0.1, 3.0 / n_points)  # At least 10% or 3 points
    n_baseline = max(3, int(n_points * baseline_fraction))
    
    # Get baseline points from edges
    baseline_x = np.concatenate([x_region[:n_baseline], x_region[-n_baseline:]])
    baseline_y = np.concatenate([y_region[:n_baseline], y_region[-n_baseline:]])
    
    # Fit linear baseline
    try:
        baseline_model = LinearRegression()
        baseline_model.fit(baseline_x.reshape(-1, 1), baseline_y)
        
        # Calculate baseline for entire range
        baseline_region = baseline_model.predict(x_region.reshape(-1, 1))
        
        # Ensure baseline doesn't go above the signal
        baseline_region = np.minimum(baseline_region, y_region)
        
        # Calculate area above baseline using trapezoidal rule
        corrected_y = y_region - baseline_region
        corrected_y = np.maximum(corrected_y, 0)  # Remove negative values
        
        if len(corrected_y) > 1:
            # Use numpy's trapz instead of scipy's deprecated trapz
            area = np.trapz(corrected_y, x_region)
        else:
            area = 0.0
            
        # Create full baseline array for plotting
        full_baseline = np.zeros_like(y)
        full_baseline[mask] = baseline_region
        
        return max(0.0, area), full_baseline
        
    except Exception as e:
        st.warning(f"Baseline fitting failed: {str(e)}. Using simple integration.")
        # Fallback to simple integration
        area = np.trapz(y_region, x_region)
        return max(0.0, area), np.zeros_like(y)

def get_automatic_range(mz_center: float, percent: float) -> Tuple[float, float]:
    """Get automatic integration range based on percentage of m/z."""
    delta = mz_center * (percent / 100.0)
    return mz_center - delta, mz_center + delta

def plot_and_integrate_with_baseline(ms_df: pd.DataFrame, mz: float, selected_range: Tuple[float, float], 
                                   smoothing_window: int, show_zoomed: bool = True) -> Tuple[Optional[float], bool]:
    """Plot spectrum with baseline fitting and integration."""
    try:
        if show_zoomed:
            mz_window_min = mz * 0.90
            mz_window_max = mz * 1.10
            ms_df_window = ms_df[(ms_df["m/z"] >= mz_window_min) & (ms_df["m/z"] <= mz_window_max)].copy()
            title_suffix = " (Zoomed)"
        else:
            ms_df_window = ms_df.copy()
            title_suffix = " (Full Spectrum)"
        
        if len(ms_df_window) == 0:
            st.warning("No data in selected range.")
            return None, False
        
        ms_df_window["Smoothed"] = ms_df_window["Intensity"].rolling(
            window=max(1, smoothing_window), center=True, min_periods=1
        ).mean()
        
        # Get data for integration
        integration_mask = (ms_df_window["m/z"] >= selected_range[0]) & (ms_df_window["m/z"] <= selected_range[1])
        integration_df = ms_df_window[integration_mask]
        
        # Check if range extends beyond current view
        range_outside_view = (selected_range[0] < ms_df_window["m/z"].min() or 
                             selected_range[1] > ms_df_window["m/z"].max())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the spectrum
        ax.plot(ms_df_window["m/z"], ms_df_window["Smoothed"], color="blue", linewidth=1.5, label="Smoothed spectrum")
        ax.axvline(mz, color="red", linestyle="--", alpha=0.7, label=f"Theoretical m/z: {mz:.3f}")
        
        area = None
        if len(integration_df) >= 3:
            x = integration_df["m/z"].values
            y = integration_df["Smoothed"].values
            
            # Fit baseline and integrate
            area, baseline = fit_baseline_and_integrate(
                ms_df_window["m/z"].values, 
                ms_df_window["Smoothed"].values, 
                selected_range
            )
            
            # Plot baseline in integration region
            baseline_mask = (ms_df_window["m/z"] >= selected_range[0]) & (ms_df_window["m/z"] <= selected_range[1])
            if np.any(baseline_mask):
                baseline_x = ms_df_window["m/z"].values[baseline_mask]
                baseline_y = baseline[baseline_mask]
                ax.plot(baseline_x, baseline_y, 'g--', linewidth=2, label="Fitted baseline")
                
                # Fill area above baseline
                spectrum_y = ms_df_window["Smoothed"].values[baseline_mask]
                ax.fill_between(baseline_x, baseline_y, spectrum_y, 
                               where=(spectrum_y >= baseline_y), 
                               color="orange", alpha=0.4, label="Integrated area")
            
            # Add integration bounds
            ax.axvline(selected_range[0], color="green", linestyle="-", alpha=0.8, linewidth=2)
            ax.axvline(selected_range[1], color="green", linestyle="-", alpha=0.8, linewidth=2)
            
            if area <= 0:
                st.warning("No positive area detected above baseline. Please adjust the integration range.")
                area = None
        else:
            if range_outside_view:
                st.warning("Integration range extends beyond current view. Click 'Toggle Zoom' to see complete range.")
            else:
                st.warning("Integration range too small (less than 3 points). Please expand the range.")
        
        ax.set_xlabel("m/z")
        ax.set_ylabel("Smoothed Intensity")
        ax.set_title(f"Integration region: {selected_range[0]:.3f} - {selected_range[1]:.3f} m/z{title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add area annotation if calculated
        if area is not None:
            ax.text(0.02, 0.98, f"Area: {area:.2e}", transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)  # Explicitly close to prevent memory leaks
        
        return area, range_outside_view
        
    except Exception as e:
        st.error(f"Error in plotting/integration: {str(e)}")
        return None, False

def plot_full_mass_spectrum_with_ranges(ms_df: pd.DataFrame, protein_name: str, protein_masses: Dict[str, float], 
                                       charge_range: Tuple[int, int], selected_charge: int,
                                       scale_ranges: Dict = None) -> None:
    """Plot the full mass spectrum with vertical lines for all charge states and their integration ranges."""
    mass = protein_masses[protein_name]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ms_df["m/z"], ms_df["Intensity"], color="gray", linewidth=1, alpha=0.7, label="Mass spectrum")
    
    # Add vertical lines for each charge state within the range
    min_charge, max_charge = charge_range
    for charge in range(min_charge, max_charge + 1):
        mz = (mass + PROTON_MASS * charge) / charge
        color = "red" if charge == selected_charge else "blue"
        alpha = 0.9 if charge == selected_charge else 0.5
        linestyle = "-" if charge == selected_charge else "--"
        linewidth = 2 if charge == selected_charge else 1
        
        ax.axvline(mz, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)
        
        # Add charge state label
        label_height = ax.get_ylim()[1] * 0.9
        ax.text(mz, label_height, f"{charge}+", 
               color=color, ha="center", va="top", 
               fontsize=10, fontweight='bold' if charge == selected_charge else 'normal',
               bbox=dict(facecolor='white', alpha=0.8, pad=2))
        
        # Show integration range if defined
        if scale_ranges and (protein_name, charge) in scale_ranges:
            range_min, range_max = scale_ranges[(protein_name, charge)]
            ax.axvspan(range_min, range_max, alpha=0.2, 
                      color=color, label=f"Integration range {charge}+" if charge == selected_charge else "")
    
    # Add legend
    ax.plot([], [], color="red", linestyle="-", linewidth=2, label=f"Selected: {selected_charge}+")
    ax.plot([], [], color="blue", linestyle="--", alpha=0.5, label="Other charge states")
    
    # Set labels and title
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Mass Spectrum for {protein_name} - All Charge States and Integration Ranges")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

# --- Data Classes ---
@dataclass
class CalibratedDriftResult:
    output_buffers: Dict[str, List[pd.DataFrame]]
    processed_files: int
    matched_points: int

# --- Processing Logic ---
class CalibratedDriftProcessor:
    @staticmethod
    def load_and_normalize_atd(file_path: str, instrument_type: str, inject_time: float) -> Optional[pd.DataFrame]:
        """Load ATD file and normalize intensities to max value of 1"""
        try:
            raw_df = pd.read_csv(file_path, sep="\t", header=None, names=["Drift", "Intensity"])
            
            # Apply instrument-specific drift time correction
            if instrument_type == "Cyclic" and inject_time is not None:
                raw_df["Drift"] = raw_df["Drift"] - inject_time
            raw_df["Drift"] = raw_df["Drift"] / 1000.0
            
            # Normalize ATD intensities so maximum is 1
            max_intensity = raw_df["Intensity"].max()
            if max_intensity > 0:
                raw_df["Intensity"] = raw_df["Intensity"] / max_intensity
            
            return raw_df
        except Exception:
            return None

    @staticmethod
    def load_mass_spectrum(ms_path: str) -> Optional[pd.DataFrame]:
        """Load mass spectrum file with error handling"""
        try:
            if os.path.exists(ms_path):
                ms_df = pd.read_csv(ms_path, sep="\t", header=None, names=["m/z", "Intensity"])
                ms_df.dropna(inplace=True)
                return ms_df
        except Exception:
            pass
        return None

    @staticmethod
    def calculate_scale_factor(ms_df: pd.DataFrame, protein_name: str, charge_state: int, 
                              protein_mass: float, scale_ranges: Dict, use_max_intensity: bool = False) -> Tuple[Optional[float], float]:
        """Calculate scaling factor from mass spectrum integration with baseline fitting or max intensity"""
        if ms_df is None or protein_mass is None:
            return None, None
            
        mz = (protein_mass + PROTON_MASS * charge_state) / charge_state
        range_key = (protein_name, charge_state)
        
        if range_key not in scale_ranges:
            return None, mz
            
        mz_min, mz_max = scale_ranges[range_key]
        
        try:
            # Get data in range
            mask = (ms_df["m/z"] >= mz_min) & (ms_df["m/z"] <= mz_max)
            if np.sum(mask) < 3:
                return None, mz
            
            # Apply smoothing
            smoothed_intensity = ms_df["Intensity"].rolling(window=51, center=True, min_periods=1).mean()
            
            if use_max_intensity:
                # Use maximum intensity in the range
                scale_factor = smoothed_intensity[mask].max()
            else:
                # Use baseline fitting for more accurate integration
                scale_factor, _ = fit_baseline_and_integrate(
                    ms_df["m/z"].values, 
                    smoothed_intensity.values, 
                    (mz_min, mz_max)
                )
            
            return scale_factor if scale_factor > 0 else None, mz
            
        except Exception as e:
            st.warning(f"Scale factor calculation failed for {protein_name} charge {charge_state}: {str(e)}")
            return None, mz

    @staticmethod
    def match_and_calibrate(
        drift_zip: BytesIO,
        cal_csvs: List,
        instrument_type: str,
        inject_time: float,
        charge_ranges: Dict[str, Tuple[int, int]],
        scale_ranges: Dict[Tuple[str, int], Tuple[float, float]],
        protein_masses: Dict[str, float],
        use_max_intensity: bool = False  # <-- Add this parameter
    ) -> CalibratedDriftResult:
        output_buffers = {}
        processed_files = 0
        matched_points = 0
        skipped_files = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract ZIP file
            drift_zip_path = os.path.join(tmpdir, "drift.zip")
            with open(drift_zip_path, "wb") as f:
                f.write(drift_zip.getvalue())
            with zipfile.ZipFile(drift_zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # Load calibration data once
            calibration_lookup = {}
            for file in cal_csvs:
                protein_name = file.name.replace(".csv", "")
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    key = (protein_name, int(row["Z"]))
                    calibration_lookup.setdefault(key, []).append({
                        "Drift": row["Drift"],
                        "CCS": row["CCS"],
                        "CCS Std.Dev.": row["CCS Std.Dev."]
                    })

            # Process each protein directory
            for root, _, files in os.walk(tmpdir):
                protein_name = os.path.basename(root)
                if protein_name == os.path.basename(tmpdir):  # Skip root directory
                    continue

                # Get charge range for this protein
                charge_range = charge_ranges.get(protein_name, (2, 4))  # Default if missing

                # Load mass spectrum once per protein
                mass_spectrum_path = os.path.join(root, "mass_spectrum.txt")
                ms_df = CalibratedDriftProcessor.load_mass_spectrum(mass_spectrum_path)
                protein_mass = protein_masses.get(protein_name, None)

                # Process ATD files for each charge state
                atd_files = [f for f in files if f.endswith(".txt") and f.split(".")[0].isdigit()]

                for file in atd_files:
                    charge_state = int(file.split(".")[0])

                    # Skip if outside charge range
                    if charge_state < charge_range[0] or charge_state > charge_range[1]:
                        continue
                    
                    # Skip if no calibration data available
                    key = (protein_name, charge_state)
                    cal_data = calibration_lookup.get(key)
                    if not cal_data:
                        skipped_files.append(f"{protein_name} charge {charge_state}: No calibration data")
                        continue
                    
                    # Skip if no scale factor defined for this protein/charge combination
                    if key not in scale_ranges:
                        skipped_files.append(f"{protein_name} charge {charge_state}: No scale factor defined")
                        continue
                    
                    # Load and normalize ATD
                    file_path = os.path.join(root, file)
                    normalized_df = CalibratedDriftProcessor.load_and_normalize_atd(
                        file_path, instrument_type, inject_time
                    )
                    
                    if normalized_df is None:
                        skipped_files.append(f"{protein_name} charge {charge_state}: Could not load ATD")
                        continue
                    
                    processed_files += 1
                    
                    # Calculate scaling factor from mass spectrum
                    scale_factor, mz = CalibratedDriftProcessor.calculate_scale_factor(
                        ms_df, protein_name, charge_state, protein_mass, scale_ranges, use_max_intensity
                    )
                    
                    # Skip if scale factor calculation failed
                    if scale_factor is None:
                        skipped_files.append(f"{protein_name} charge {charge_state}: Scale factor calculation failed")
                        continue
                    
                    # Match calibration points to normalized ATD data
                    out_rows = []
                    for entry in cal_data:
                        drift_val = entry["Drift"]
                        
                        # Find closest drift time in normalized data
                        closest_idx = (normalized_df["Drift"] - drift_val).abs().idxmin()
                        normalized_intensity = normalized_df.loc[closest_idx, "Intensity"]
                        
                        # Apply scaling to normalized intensity
                        scaled_intensity = normalized_intensity * scale_factor
                        
                        out_rows.append({
                            "Charge": charge_state,
                            "Drift": drift_val,
                            "CCS": entry["CCS"],
                            "CCS Std.Dev.": entry["CCS Std.Dev."],
                            "Normalized_Intensity": normalized_intensity,
                            "Scaled_Intensity": scaled_intensity,
                            "m/z": mz
                        })
                        matched_points += 1
                    
                    if out_rows:
                        out_df = pd.DataFrame(out_rows)
                        out_key = f"{protein_name}.csv"
                        output_buffers.setdefault(out_key, []).append(out_df)

        # Show skipped files if any
        if skipped_files and len(skipped_files) > 0:
            st.warning(f"Skipped {len(skipped_files)} files due to missing data or scale factors:")
            for msg in skipped_files[:10]:  # Show first 10 to avoid clutter
                st.write(f"• {msg}")
            if len(skipped_files) > 10:
                st.write(f"...and {len(skipped_files) - 10} more")

        return CalibratedDriftResult(output_buffers, processed_files, matched_points)

    @staticmethod
    def prepare_zip(output_buffers: Dict[str, List[pd.DataFrame]]) -> BytesIO:
        """Prepare ZIP file with all results"""
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:
            for filename, dfs in output_buffers.items():
                combined = pd.concat(dfs, ignore_index=True)
                csv_bytes = combined.to_csv(index=False).encode("utf-8")
                zip_out.writestr(filename, csv_bytes)
        zip_buffer.seek(0)
        return zip_buffer

    @staticmethod
    def get_optimal_charge_ranges(ms_df: pd.DataFrame, protein_mass: float, min_charge: int = None, max_charge: int = 20) -> Dict[int, Tuple[float, float]]:
        """Define optimal m/z ranges for each charge state with better error handling."""
        try:
            # Get global m/z bounds from the data
            global_min_mz = ms_df["m/z"].min()
            global_max_mz = ms_df["m/z"].max()
            
            # Determine minimum charge state if not provided
            if min_charge is None:
                # Find charge state where m/z falls within spectrum range
                for c in range(max_charge, 0, -1):
                    mz = (protein_mass + PROTON_MASS * c) / c
                    if mz <= global_max_mz:
                        min_charge = c
                        break
                
                # If still None, default to charge state 1
                if min_charge is None:
                    min_charge = 1
            
            # Calculate theoretical m/z for each charge state
            charge_mz = {}
            for c in range(min_charge, max_charge + 1):
                mz = (protein_mass + PROTON_MASS * c) / c
                if global_min_mz <= mz <= global_max_mz:  # Only include if within spectrum
                    charge_mz[c] = mz
            
            if not charge_mz:
                return {}  # No valid charge states in range
            
            # Sort charge states by m/z (descending)
            sorted_charges = sorted(charge_mz.keys(), key=lambda c: charge_mz[c], reverse=True)
            
            # Define boundaries between charge states
            charge_ranges = {}
            
            for i, c in enumerate(sorted_charges):
                current_mz = charge_mz[c]
                
                # For first (highest m/z) charge state
                if i == 0:
                    upper_bound = global_max_mz
                else:
                    prev_c = sorted_charges[i-1]
                    prev_mz = charge_mz[prev_c]
                    # Boundary between this charge state and the previous one
                    upper_bound = (current_mz + prev_mz) / 2
                
                # For last (lowest m/z) charge state
                if i == len(sorted_charges) - 1:
                    lower_bound = global_min_mz
                else:
                    next_c = sorted_charges[i+1]
                    next_mz = charge_mz[next_c]
                    # Boundary between this charge state and the next one
                    lower_bound = (current_mz + next_mz) / 2
                
                # Store the range for this charge state
                charge_ranges[c] = (lower_bound, upper_bound)
            
            return charge_ranges
            
        except Exception as e:
            st.error(f"Error calculating optimal charge ranges: {str(e)}")
            return {}

# --- UI Components ---
class UI:
    @staticmethod
    def show_main_header():
        st.markdown("""
        <div class="main-header">
            <h1>Calibrate Drift Files & Scale Normalized ATDs</h1>
            <p>Normalize ATD intensities to max=1, then match calibration data and scale using mass spectrum integration</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_info_card():
        st.markdown("""
        <div class="info-card">
            <p>Use this page to generate calibrated and scaled CCSDs for each protein using the processed IMSCal<sup>1</sup> output files. This step completes the calibration process by normalizing ATD intensities to a maximum of 1, then matching calibration data and scaling using mass spectrum integration with baseline fitting.</p>
            <p>This process is particularly useful when you have performed multiple experiments on the same protein (e.g., activated ion mobility experiments at different collision voltages). In such cases, you only need to calibrate once using IMSCal, then use this tool to process all your experimental conditions.</p>
            <p><strong>What you'll need:</strong></p>
            <ul>
                <li><strong>ZIP file containing raw drift files:</strong> Each protein folder should contain X.txt files (where X is the charge state) and a mass_spectrum.txt file</li>
                <li><strong>CSV files from the 'Process Output Files' step:</strong> These contain the calibration data generated in the previous step</li>
                <li><strong>Protein masses:</strong> Molecular mass (Da) for each protein</li>
                <li><strong>Charge state range:</strong> Specify which charge states to include in the analysis</li>
                <li><strong>Integration ranges:</strong> Define m/z ranges for mass spectrum integration to calculate scaling factors</li>
            </ul>
            <p><strong>Note:</strong> This step performs baseline fitting and integration for accurate scaling. The output includes both normalized intensities (max=1) and scaled intensities (normalized × mass spectrum scale factor).</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_upload_section():
        st.markdown("""
        <div class="section-card">
            <div class="section-header">📁 Step 1: Upload Raw Drift Files</div>
        </div>
        """, unsafe_allow_html=True)
        return st.file_uploader(
            "Upload zipped folder of raw drift files", 
            type="zip",
            help="ZIP file should contain folders with X.txt files and mass_spectrum.txt"
        )

    @staticmethod
    def show_calibration_upload():
        st.markdown("""
        <div class="section-card">
            <div class="section-header">📊 Step 2: Upload Calibration Data</div>
        </div>
        """, unsafe_allow_html=True)
        return st.file_uploader(
            "Upload the CSV files from the 'Process Output Files' page", 
            type="csv", 
            accept_multiple_files=True,
            help="Select all CSV files generated in the previous step"
        )

    @staticmethod
    def get_protein_masses(protein_names: List[str]) -> Dict[str, float]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">⚖️ Step 3: Enter Protein Masses</div>
        </div>
        """, unsafe_allow_html=True)
        masses = {}
        cols = st.columns(min(3, len(protein_names)))
        for i, name in enumerate(protein_names):
            with cols[i % len(cols)]:
                mass = st.number_input(f"Mass (Da) for {name}", min_value=0.0, key=f"mass_{name}")
                masses[name] = mass
        return masses

    @staticmethod
    def get_charge_range() -> Tuple[int, int]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">🔋 Step 4: Select Charge State Range</div>
        </div>
        """, unsafe_allow_html=True)
        min_charge = st.number_input("Minimum charge state", min_value=1, value=2)
        max_charge = st.number_input("Maximum charge state", min_value=min_charge, value=min_charge+2)
        return (min_charge, max_charge)

    @staticmethod
    def show_instrument_settings() -> Tuple[str, Optional[float], bool]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">⚙️ Step 5: Instrument Configuration</div>
        </div>
        """, unsafe_allow_html=True)
        
        instrument_type = st.radio(
            "Select your instrument type:",
            ["Synapt", "Cyclic"],
            help="This affects how drift times are processed"
        )
        
        inject_time = None
        if instrument_type == "Cyclic":
            inject_time = st.number_input(
                "Enter the injection time (ms)",
                min_value=0.0,
                value=0.0,
                step=0.1,
                help="This value will be subtracted from all drift times"
            )
        
        # Add scale factor method selection
        use_max_intensity = st.checkbox(
            "Use maximum intensity as scale factor",
            value=False,
            help="If checked, uses the maximum intensity in the integration range instead of baseline-corrected integration"
        )
        
        return instrument_type, inject_time, use_max_intensity

    @staticmethod
    def show_processing_status(processed_files: int, matched_points: int, output_files: int):
        """Show processing status summary"""
        st.markdown("""
        <div class="section-card">
            <div class="section-header">✅ Processing Complete</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Processed", processed_files)
        with col2:
            st.metric("Data Points Matched", matched_points)
        with col3:
            st.metric("Output Files Generated", output_files)

    @staticmethod
    def show_protein_card(protein_name: str, total_points: int, charge_states: int):
        """Show summary card for each protein"""
        with st.expander(f"📊 {protein_name} Summary", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Data Points:** {total_points}")
            with col2:
                st.write(f"**Charge States:** {charge_states}")

    @staticmethod
    def show_next_steps():
        """Show next steps information"""
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Next Steps</h4>
            <p>Your normalized and calibrated drift data is ready! The output includes:</p>
            <ul>
                <li><strong>Normalized_Intensity:</strong> Original intensity normalized to max=1 per charge state</li>
                <li><strong>Scaled_Intensity:</strong> Normalized intensity × mass spectrum integration scale factor</li>
                <li><strong>CCS values:</strong> Matched calibration data</li>
                <li><strong>m/z values:</strong> Theoretical m/z for each charge state</li>
            </ul>
            <p>Use this data for further analysis or visualization.</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_integration_range_selector(tmpdir_name: str, protein_names: List[str], protein_masses: Dict[str, float], use_max_intensity: bool = False) -> Tuple[Dict, Dict]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">🎚️ Step 6: Select Integration Ranges & Charge States</div>
        </div>
        """, unsafe_allow_html=True)

        # Session state setup
        charge_ranges = st.session_state.setdefault("charge_ranges", {name: (2, 4) for name in protein_names})
        scale_ranges = st.session_state.setdefault("scale_ranges", {})
        scale_factors = st.session_state.setdefault("scale_factors", {})

        # --- Synchronize session state with current protein list ---
        # Add missing proteins with defaults
        for name in protein_names:
            if name not in charge_ranges:
                charge_ranges[name] = (2, 4)
        # Remove proteins that are no longer present
        for name in list(charge_ranges.keys()):
            if name not in protein_names:
                del charge_ranges[name]
        # Prune stale entries in scale_* dicts
        for k in list(scale_ranges.keys()):
            prot = k[0] if isinstance(k, tuple) and len(k) >= 1 else None
            if prot not in protein_names:
                del scale_ranges[k]
        for k in list(scale_factors.keys()):
            prot = k[0] if isinstance(k, tuple) and len(k) >= 1 else None
            if prot not in protein_names:
                del scale_factors[k]

        st.session_state["charge_ranges"] = charge_ranges
        st.session_state["scale_ranges"] = scale_ranges
        st.session_state["scale_factors"] = scale_factors

        # Controls section
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            smoothing_window = st.slider(
                "Smoothing window (number of points)", min_value=0, max_value=1000, value=51, step=1
            )
        
        with col2:
            auto_percent = st.number_input(
                "Auto range (%)", min_value=0.01, max_value=5.0, value=0.1, step=0.01,
                help="Percentage of m/z to use for automatic range (±%)"
            )
        
        with col3:
            if st.button("🔄 Auto-set all ranges"):
                for prot in protein_names:
                    mass = protein_masses[prot]
                    if mass > 0:
                        min_c, max_c = charge_ranges[prot]
                        for c in range(int(min_c), int(max_c)+1):
                            mz = (mass + PROTON_MASS * c) / c
                            auto_min, auto_max = get_automatic_range(mz, auto_percent)
                            scale_ranges[(prot, c)] = (auto_min, auto_max)
                
                st.session_state["scale_ranges"] = scale_ranges
                st.success(f"Auto-set all ranges to ±{auto_percent}% of theoretical m/z")

        # Protein and charge selection
        selected_protein = st.selectbox("Select protein for integration setup", protein_names)

        # Use safe keys for widgets (avoid collisions with special chars)
        safe_key = re.sub(r"[^0-9A-Za-z_]+", "_", selected_protein)
        current_min, current_max = charge_ranges.get(selected_protein, (2, 4))

        col1, col2 = st.columns(2)
        with col1:
            min_charge = st.number_input(
                f"Minimum charge for {selected_protein}", min_value=1,
                value=current_min, key=f"min_charge_{safe_key}"
            )
        with col2:
            max_charge = st.number_input(
                f"Maximum charge for {selected_protein}", min_value=min_charge,
                value=max(current_max, min_charge), key=f"max_charge_{safe_key}"
            )

        charge_ranges[selected_protein] = (int(min_charge), int(max_charge))
        st.session_state["charge_ranges"] = charge_ranges

        selected_charge = st.number_input(
            f"Select charge state for {selected_protein}", min_value=int(min_charge),
            max_value=int(max_charge), value=int(min_charge), key=f"charge_{safe_key}"
        )

        # Load mass spectrum and show overview
        protein_dir = os.path.join(tmpdir_name, selected_protein)
        ms_path = os.path.join(protein_dir, "mass_spectrum.txt")
        ms_df = CalibratedDriftProcessor.load_mass_spectrum(ms_path)
        
        if ms_df is not None:
            plot_full_mass_spectrum_with_ranges(
                ms_df, selected_protein, protein_masses, 
                charge_ranges[selected_protein], selected_charge, scale_ranges
            )

        # Manual range selection for current protein/charge
        if os.path.exists(ms_path) and ms_df is not None and len(ms_df) > 0:
            mass = protein_masses[selected_protein]
            if mass > 0:
                mz = (mass + PROTON_MASS * selected_charge) / selected_charge
                
                # Zoom control
                show_zoomed = st.session_state.get(f"show_zoomed_{safe_key}_{selected_charge}", True)
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col2:
                    if st.button("🔍 Toggle Zoom", key=f"zoom_{safe_key}_{selected_charge}"):
                        st.session_state[f"show_zoomed_{safe_key}_{selected_charge}"] = not show_zoomed
                        st.rerun()
                
                with col3:
                    if st.button("🎯 Auto Range", key=f"auto_{safe_key}_{selected_charge}"):
                        auto_min, auto_max = get_automatic_range(mz, auto_percent)
                        scale_ranges[(selected_protein, selected_charge)] = (auto_min, auto_max)
                        st.session_state["scale_ranges"] = scale_ranges
                        st.rerun()
                
                if show_zoomed:
                    mz_window_min = mz * 0.90
                    mz_window_max = mz * 1.10
                    ms_df_window = ms_df[(ms_df["m/z"] >= mz_window_min) & (ms_df["m/z"] <= mz_window_max)].copy()
                else:
                    ms_df_window = ms_df.copy()
                
                if len(ms_df_window) > 0:
                    ms_df_window["Smoothed"] = ms_df_window["Intensity"].rolling(
                        window=max(1, smoothing_window), center=True, min_periods=1
                    ).mean()
                    
                    mz_min = float(ms_df_window["m/z"].min())
                    mz_max = float(ms_df_window["m/z"].max())
                    
                    # Get current range or use automatic default
                    current_range = scale_ranges.get(
                        (selected_protein, selected_charge),
                        get_automatic_range(mz, auto_percent)
                    )
                    
                    # Ensure slider range covers the detected range
                    slider_min = min(mz_min, current_range[0] - 0.01)
                    slider_max = max(mz_max, current_range[1] + 0.01)
                    
                    selected_range = st.slider(
                        f"Integration range for {selected_protein} charge {selected_charge}",
                        min_value=slider_min,
                        max_value=slider_max,
                        value=current_range,
                        step=0.001,
                        format="%.3f",
                        key=f"slider_{safe_key}_{selected_charge}"
                    )
                    
                    st.write(f"Selected integration range: {selected_range[0]:.3f} - {selected_range[1]:.3f} m/z")
                    
                    # Calculate scale factor based on method
                    if use_max_intensity:
                        # Calculate max intensity in range
                        mask = (ms_df_window["m/z"] >= selected_range[0]) & (ms_df_window["m/z"] <= selected_range[1])
                        if np.sum(mask) >= 1:
                            area = ms_df_window.loc[mask, "Smoothed"].max()
                            
                            # Plot for visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(ms_df_window["m/z"], ms_df_window["Smoothed"], color="blue", linewidth=1.5, label="Smoothed spectrum")
                            ax.axvline(mz, color="red", linestyle="--", alpha=0.7, label=f"Theoretical m/z: {mz:.3f}")
                            ax.axvline(selected_range[0], color="green", linestyle="-", alpha=0.8, linewidth=2)
                            ax.axvline(selected_range[1], color="green", linestyle="-", alpha=0.8, linewidth=2)
                            
                            # Highlight the maximum point
                            max_idx = ms_df_window.loc[mask, "Smoothed"].idxmax()
                            max_mz = ms_df_window.loc[max_idx, "m/z"]
                            ax.plot(max_mz, area, 'ro', markersize=8, label=f"Maximum intensity: {area:.2e}")
                            

                            ax.set_xlabel("m/z")
                            ax.set_ylabel("Smoothed Intensity")
                            ax.set_title(f"Maximum Intensity in Range: {selected_range[0]:.3f} - {selected_range[1]:.3f} m/z")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                            
                            range_outside_view = False
                        else:
                            area = None
                            range_outside_view = True
                    else:
                        # Use baseline fitting
                        area, range_outside_view = plot_and_integrate_with_baseline(
                            ms_df, mz, selected_range, smoothing_window, show_zoomed
                        )
                    
                    if range_outside_view and show_zoomed:
                        st.info("💡 Integration range extends beyond zoomed view. Use 'Toggle Zoom' to see full spectrum.")
                    
                    if area is not None and area > 0:
                        scale_ranges[(selected_protein, selected_charge)] = selected_range
                        st.session_state["scale_ranges"] = scale_ranges
                        scale_factors[(selected_protein, selected_charge)] = area
                        st.session_state["scale_factors"] = scale_factors
                        method_text = "Maximum intensity" if use_max_intensity else "Integration area"
                        st.success(f"✅ {method_text} (scale factor): {area:.2e}")
                    else:
                        st.warning("⚠️ No valid peak detected for this charge state. Try adjusting the range.")

        # Show summary table
        if scale_factors or scale_ranges:
            method_text = "Maximum Intensity" if use_max_intensity else "Integration Area"
            st.markdown(f"#### Integration Ranges and Scale Factors ({method_text} Method)")
            rows = []
            for prot in protein_names:
                min_c, max_c = charge_ranges[prot]
                for c in range(int(min_c), int(max_c)+1):
                    factor = scale_factors.get((prot, c), None)
                    rng = scale_ranges.get((prot, c), None)
                    
                    rows.append({
                        "Protein": prot,
                        "Charge": c,
                        "Integration Range (m/z)": f"{rng[0]:.3f} - {rng[1]:.3f}" if rng else "Not set",
                        "Scale Factor": f"{factor:.2e}" if factor is not None else "Not calculated"
                    })
            
            df_display = pd.DataFrame(rows)
            st.dataframe(df_display, use_container_width=True)

        return scale_ranges, charge_ranges

# --- Main App ---
def main():
    styling.load_custom_css()
    UI.show_main_header()
    UI.show_info_card()
    
    # Clear cache button inside info card for consistent styling
    if st.button("🧹 Clear Cache & Restart App"):
        # import_tools.clear_cache()
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.session_state.clear()
        st.rerun()

    # Step 1: Upload files
    drift_zip = UI.show_upload_section()
    cal_csvs = UI.show_calibration_upload()
    if not drift_zip or not cal_csvs:
        # Add references section when no files are uploaded
        st.markdown("""
        <div class="info-card">
            <h3>📚 References</h3>
            <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, <em> International Journal of Mass Spectrometry,</em>,2023, 492, 117112.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Upload both drift ZIP and calibration CSVs to continue.")
        return

    # Step 2: Get protein info
    protein_names = [file.name.replace(".csv", "") for file in cal_csvs]
    protein_masses = UI.get_protein_masses(protein_names)

    # Validate masses
    if any(mass == 0.0 for mass in protein_masses.values()):
        st.warning("Please enter a mass for every protein.")
        return

    # Step 3: Instrument settings (moved up to get use_max_intensity early)
    instrument_type, inject_time, use_max_intensity = UI.show_instrument_settings()

    # Step 4: Integration range and charge selection
    if "tmpdir" not in st.session_state:
        tmpdir = tempfile.TemporaryDirectory()
        drift_zip_path = os.path.join(tmpdir.name, "drift.zip")
        with open(drift_zip_path, "wb") as f:
            f.write(drift_zip.getvalue())
        with zipfile.ZipFile(drift_zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir.name)
        st.session_state["tmpdir"] = tmpdir
    else:
        tmpdir = st.session_state["tmpdir"]

    scale_ranges, charge_ranges = UI.show_integration_range_selector(
        tmpdir.name, protein_names, protein_masses, use_max_intensity
    )

    # Step 5: Process data
    if st.button("🚀 Process Normalized ATDs", type="primary"):
        with st.spinner("Processing normalized ATD data..."):
            result = CalibratedDriftProcessor.match_and_calibrate(
                drift_zip, cal_csvs, instrument_type, inject_time,
                charge_ranges, scale_ranges, protein_masses, use_max_intensity
            )

        if result.output_buffers:
            UI.show_processing_status(result.processed_files, result.matched_points, len(result.output_buffers))
            for filename, dfs in result.output_buffers.items():
                protein_name = filename.replace('.csv', '')
                total_points = sum(len(df) for df in dfs)
                UI.show_protein_card(protein_name, total_points, len(dfs))

            # Add custom filename input
            st.markdown("### 📁 Download Options")
            custom_filename = st.text_input(
                "Custom filename (without .zip extension)",
                value="normalized_calibrated_drift_data",
                help="Enter a custom name for your download file"
            )
            
            # Ensure filename ends with .zip
            if not custom_filename.endswith('.zip'):
                download_filename = f"{custom_filename}.zip"
            else:
                download_filename = custom_filename

            zip_buffer = CalibratedDriftProcessor.prepare_zip(result.output_buffers)
            st.download_button(
                label="📦 Download Normalized & Calibrated Data (ZIP)",
                data=zip_buffer,
                file_name=download_filename,
                mime="application/zip"
            )
            UI.show_next_steps()
        else:
            st.error("No matching data found. Please check your file formats and naming.")
    
    # Add references section at the end
    st.markdown("""
    <div class="info-card">
        <h3>📚 References</h3>
        <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, <em> International Journal of Mass Spectrometry,</em>,2023, 492, 117112.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()