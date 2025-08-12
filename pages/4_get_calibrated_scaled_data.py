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

from myutils import styling

PROTON_MASS = 1.007276

def autodetect_peak_and_range(ms_df_window: pd.DataFrame, full_ms_df: pd.DataFrame = None, mz_center: float = None) -> Tuple[float, float]:
    """Detect peak bounds centered on the charge state m/z rather than highest intensity.
    If the detected range is wider than the window, use the full spectrum."""
    y = ms_df_window["Smoothed"].values
    x = ms_df_window["m/z"].values
    
    # If no center provided, use middle of window
    if mz_center is None:
        mz_center = (x[0] + x[-1]) / 2
    
    # Find the closest point to the charge state m/z
    center_idx = np.argmin(np.abs(x - mz_center))
    
    # Find local minima to the left and right of the center
    left_min = np.argmin(y[:center_idx]) if center_idx > 0 else 0
    right_min = np.argmin(y[center_idx:]) + center_idx if center_idx < len(y)-1 else len(y)-1
    
    detected_min = x[left_min]
    detected_max = x[right_min]
    
    # If detection goes to window edges and we have full spectrum, expand search
    if full_ms_df is not None and (left_min == 0 or right_min == len(y)-1):
        # Use full spectrum for wider peak detection
        full_smoothed = full_ms_df["Intensity"].rolling(window=51, center=True, min_periods=1).mean()
        
        # Find the closest point to the charge state m/z in full spectrum
        full_center_idx = np.argmin(np.abs(full_ms_df["m/z"].values - mz_center))
        
        # Expand search in full spectrum
        full_left_min = np.argmin(full_smoothed.iloc[:full_center_idx]) if full_center_idx > 0 else 0
        full_right_min = np.argmin(full_smoothed.iloc[full_center_idx:]) + full_center_idx if full_center_idx < len(full_smoothed)-1 else len(full_smoothed)-1
        
        detected_min = full_ms_df["m/z"].iloc[full_left_min]
        detected_max = full_ms_df["m/z"].iloc[full_right_min]
    
    return detected_min, detected_max

def plot_and_integrate_with_unzoom(ms_df: pd.DataFrame, mz: float, selected_range: Tuple[float, float], 
                                  smoothing_window: int, show_zoomed: bool = True) -> Tuple[Optional[float], bool]:
    """Plot spectrum with option to unzoom if peak is wider than window."""
    if show_zoomed:
        mz_window_min = mz * 0.90
        mz_window_max = mz * 1.10
        ms_df_window = ms_df[(ms_df["m/z"] >= mz_window_min) & (ms_df["m/z"] <= mz_window_max)].copy()
        title_suffix = " (Zoomed)"
    else:
        ms_df_window = ms_df.copy()
        title_suffix = " (Full Spectrum)"
    
    ms_df_window["Smoothed"] = ms_df_window["Intensity"].rolling(
        window=max(1, smoothing_window), center=True, min_periods=1
    ).mean()
    
    shade_df = ms_df_window[(ms_df_window["m/z"] >= selected_range[0]) & (ms_df_window["m/z"] <= selected_range[1])]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ms_df_window["m/z"], ms_df_window["Smoothed"], color="blue")
    ax.axvline(mz, color="red", linestyle="--", alpha=0.7, label="Selected m/z")
    
    # Check if range extends beyond current view
    range_outside_view = (selected_range[0] < ms_df_window["m/z"].min() or 
                         selected_range[1] > ms_df_window["m/z"].max())
    
    area = None
    if len(shade_df) >= 2:
        x = shade_df["m/z"].values
        y = shade_df["Smoothed"].values
        baseline = np.linspace(y[0], y[-1], len(y))
        ax.fill_between(x, y, baseline, color="orange", alpha=0.4)
        integral = np.trapz(y - baseline, x)
        if integral > 0:
            area = integral
        else:
            st.warning("No peak detected or negative area. Please adjust the integration range.")
    else:
        if range_outside_view:
            st.warning("Integration range extends beyond current view. Click 'Show Full Spectrum' to see complete range.")
        else:
            ax.fill_between(shade_df["m/z"], shade_df["Smoothed"], color="orange", alpha=0.4)
            st.warning("No peak detected in selected range.")
    
    ax.set_xlabel("m/z")
    ax.set_ylabel("Smoothed Intensity")
    ax.set_title(f"Integration region: {selected_range[0]:.3f} - {selected_range[1]:.3f} m/z{title_suffix}")
    ax.legend()
    st.pyplot(fig, clear_figure=True)
    
    return area, range_outside_view

def plot_full_mass_spectrum(ms_df: pd.DataFrame, protein_name: str, protein_masses: Dict[str, float], 
                          charge_range: Tuple[int, int], selected_charge: int) -> None:
    """Plot the full mass spectrum with vertical lines for all charge states and label them."""
    mass = protein_masses[protein_name]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ms_df["m/z"], ms_df["Intensity"], color="gray", linewidth=1)
    
    # Add vertical lines for each charge state within the range
    min_charge, max_charge = charge_range
    for charge in range(min_charge, max_charge + 1):
        mz = (mass + PROTON_MASS * charge) / charge
        color = "red" if charge == selected_charge else "blue"
        alpha = 0.9 if charge == selected_charge else 0.5
        linestyle = "-" if charge == selected_charge else "--"
        ax.axvline(mz, color=color, linestyle=linestyle, alpha=alpha)
        
        # Add charge state label
        label_height = ax.get_ylim()[1] * 0.9
        ax.text(mz, label_height, f"{charge}+", 
               color=color, ha="center", va="top", 
               fontsize=9, rotation=90, 
               bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    # Add selected charge state in legend
    ax.plot([], [], color="red", linestyle="-", label=f"Selected: {selected_charge}+")
    ax.plot([], [], color="blue", linestyle="--", alpha=0.5, label="Other charge states")
    
    # Set labels and title
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Full Mass Spectrum for {protein_name}")
    ax.legend(loc="upper right")
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
                              protein_mass: float, scale_ranges: Dict) -> Tuple[Optional[float], float]:
        """Calculate scaling factor from mass spectrum integration"""
        if ms_df is None or protein_mass is None:
            return None, None
            
        mz = (protein_mass + PROTON_MASS * charge_state) / charge_state
        mz_min, mz_max = scale_ranges.get((protein_name, charge_state), (mz * 0.995, mz * 1.005))
        
        try:
            scale_factor = ms_df[(ms_df["m/z"] >= mz_min) & (ms_df["m/z"] <= mz_max)]["Intensity"].sum()
            return scale_factor, mz
        except Exception:
            return None, mz

    @staticmethod
    def match_and_calibrate(
        drift_zip: BytesIO,
        cal_csvs: List,
        instrument_type: str,
        inject_time: float,
        charge_ranges: Dict[str, Tuple[int, int]],  # <-- Change here
        scale_ranges: Dict[Tuple[str, int], Tuple[float, float]],
        protein_masses: Dict[str, float]
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
                        ms_df, protein_name, charge_state, protein_mass, scale_ranges
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
        """
        Define optimal m/z ranges for each charge state, scaling with 1/n.
        
        Parameters:
            ms_df: DataFrame containing the mass spectrum data
            protein_mass: Mass of the protein in Da
            min_charge: Minimum charge state (will be calculated if None)
            max_charge: Maximum charge state to consider
            
        Returns:
            Dictionary mapping charge states to their (min_mz, max_mz) ranges
        """
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
            <p><strong>NEW:</strong> ATD intensities are now normalized so each charge state has a maximum intensity of 1 before scaling is applied.</p>
            <p><strong>What you'll need:</strong></p>
            <ul>
                <li>ZIP file containing raw drift files (X.txt format and mass_spectrum.txt per protein)</li>
                <li>CSV files from the 'Process Output Files' step</li>
                <li>Protein masses for each protein</li>
                <li>Charge state range to include</li>
            </ul>
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
    def show_instrument_settings() -> Tuple[str, Optional[float]]:
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
        return instrument_type, inject_time

    @staticmethod
    def show_integration_range_selector(tmpdir_name: str, protein_names: List[str], protein_masses: Dict[str, float]) -> Tuple[Dict, Dict]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">🎚️ Step 6: Select Integration Ranges & Charge States</div>
        </div>
        """, unsafe_allow_html=True)

        # Session state setup
        charge_ranges = st.session_state.setdefault("charge_ranges", {name: (2, 4) for name in protein_names})
        scale_ranges = st.session_state.setdefault("scale_ranges", {})
        scale_factors = st.session_state.setdefault("scale_factors", {})

        # Smoothing slider (now 0 to 1000 points)
        smoothing_window = st.slider(
            "Smoothing window (number of points)", min_value=0, max_value=1000, value=51, step=1
        )

        # Autogenerate all scale factors button
        autogen = st.button("Autodetect & Autogenerate scale factors for all proteins/charges")

        selected_protein = st.selectbox("Select protein for integration setup", protein_names)
        min_charge = st.number_input(
            f"Minimum charge for {selected_protein}", min_value=1, value=charge_ranges[selected_protein][0], key=f"min_charge_{selected_protein}"
        )
        max_charge = st.number_input(
            f"Maximum charge for {selected_protein}", min_value=min_charge, value=charge_ranges[selected_protein][1], key=f"max_charge_{selected_protein}"
        )
        charge_ranges[selected_protein] = (min_charge, max_charge)
        st.session_state["charge_ranges"] = charge_ranges

        selected_charge = st.number_input(
            f"Select charge state for {selected_protein}", min_value=min_charge, max_value=max_charge, value=min_charge, key=f"charge_{selected_protein}"
        )

        protein_dir = os.path.join(tmpdir_name, selected_protein)
        ms_path = os.path.join(protein_dir, "mass_spectrum.txt")
        ms_df = CalibratedDriftProcessor.load_mass_spectrum(ms_path)
        if ms_df is not None:
            plot_full_mass_spectrum(
                ms_df, selected_protein, protein_masses, charge_ranges[selected_protein], selected_charge
            )

        # Autogenerate scale factors using autodetection
        if autogen:
            for prot in protein_names:
                mass = protein_masses[prot]
                min_c, max_c = charge_ranges[prot]
                protein_dir = os.path.join(tmpdir_name, prot)
                ms_path = os.path.join(protein_dir, "mass_spectrum.txt")
                ms_df = CalibratedDriftProcessor.load_mass_spectrum(ms_path)
                if ms_df is None or mass == 0:
                    continue
                
                # Get optimal charge ranges based on 1/n scaling
                optimal_ranges = get_optimal_charge_ranges(ms_df, mass, min_c, 20)
                
                # Process each charge state with its optimal range
                for c in range(int(min_c), int(max_c)+1):
                    if c not in optimal_ranges:
                        continue
                        
                    mz = (mass + PROTON_MASS * c) / c
                    mz_range = optimal_ranges[c]
                    
                    # Use the optimal range for this charge state
                    ms_df_window = ms_df[(ms_df["m/z"] >= mz_range[0]) & (ms_df["m/z"] <= mz_range[1])].copy()
                    ms_df_window["Smoothed"] = ms_df_window["Intensity"].rolling(
                        window=max(1, smoothing_window), center=True, min_periods=1
                    ).mean()
                    
                    # Use autodetection centered on theoretical m/z
                    auto_min, auto_max = autodetect_peak_and_range(ms_df_window, None, mz)
                    
                    # No need to check for overlaps since we're using non-overlapping ranges
                    area, _ = plot_and_integrate_with_unzoom(ms_df, mz, (auto_min, auto_max), smoothing_window, False)
                    if area is not None:
                        scale_ranges[(prot, c)] = (auto_min, auto_max)
                        scale_factors[(prot, c)] = area
            
            st.session_state["scale_ranges"] = scale_ranges
            st.session_state["scale_factors"] = scale_factors
            st.success("Scale factors autodetected and autogenerated for all proteins/charges.")

        # Manual selection for current protein/charge
        if os.path.exists(ms_path):
            ms_df = CalibratedDriftProcessor.load_mass_spectrum(ms_path)
            if ms_df is not None:
                mass = protein_masses[selected_protein]
                mz = (mass + PROTON_MASS * selected_charge) / selected_charge
                
                # Zoom control
                show_zoomed = st.session_state.get(f"show_zoomed_{selected_protein}_{selected_charge}", True)
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🔍 Toggle Zoom", key=f"zoom_{selected_protein}_{selected_charge}"):
                        st.session_state[f"show_zoomed_{selected_protein}_{selected_charge}"] = not show_zoomed
                        st.experimental_rerun()
                
                if show_zoomed:
                    mz_window_min = mz * 0.90
                    mz_window_max = mz * 1.10
                    ms_df_window = ms_df[(ms_df["m/z"] >= mz_window_min) & (ms_df["m/z"] <= mz_window_max)].copy()
                else:
                    ms_df_window = ms_df.copy()
                
                ms_df_window["Smoothed"] = ms_df_window["Intensity"].rolling(
                    window=max(1, smoothing_window), center=True, min_periods=1
                ).mean()
                
                mz_min = float(ms_df_window["m/z"].min())
                mz_max = float(ms_df_window["m/z"].max())
                
                # Use charge-specific window for autodetection centered on theoretical m/z
                default_min, default_max = autodetect_peak_and_range(ms_df_window, None, mz)
                
                current_range = scale_ranges.get(
                    (selected_protein, selected_charge),
                    (default_min, default_max)
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
                    key=f"slider_{selected_protein}_{selected_charge}"
                )
                
                st.write(f"Selected integration range: {selected_range[0]:.3f} - {selected_range[1]:.3f} m/z")
                
                area, range_outside_view = plot_and_integrate_with_unzoom(
                    ms_df, mz, selected_range, smoothing_window, show_zoomed
                )
                
                if range_outside_view and show_zoomed:
                    st.info("💡 Integration range extends beyond zoomed view. Use 'Toggle Zoom' to see full spectrum.")
                
                if area is not None:
                    scale_ranges[(selected_protein, selected_charge)] = selected_range
                    st.session_state["scale_ranges"] = scale_ranges
                    scale_factors[(selected_protein, selected_charge)] = area
                    st.session_state["scale_factors"] = scale_factors
                    st.write(f"Integration area (scale factor, above line): {area:.2e}")
                else:
                    st.write("No valid peak detected for this charge state.")

        # Tabulate scale factors for all proteins/charges
        if scale_factors:
            st.markdown("#### Scale Factors Table")
            rows = []
            for prot in protein_names:
                min_c, max_c = charge_ranges[prot]
                for c in range(int(min_c), int(max_c)+1):
                    factor = scale_factors.get((prot, c), None)
                    rng = scale_ranges.get((prot, c), ("-", "-"))
                    rows.append({
                        "Protein": prot,
                        "Charge": c,
                        "Integration Range (m/z)": f"{rng[0]:.3f} - {rng[1]:.3f}" if rng[0] != "-" else "-",
                        "Scale Factor": f"{factor:.2e}" if factor is not None else "-"
                    })
            st.dataframe(pd.DataFrame(rows))

        return scale_ranges, charge_ranges

    @staticmethod
    def show_processing_status(processed_files: int, matched_points: int, protein_count: int):
        st.markdown(f"""
        <div class="status-card success-card">
            <strong>🎉 Processing Complete!</strong><br>
            Processed <span class="metric-badge">{processed_files} files</span>
            with <span class="metric-badge">{matched_points} matched data points</span>
            for <span class="metric-badge">{protein_count} proteins</span>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_protein_card(protein_name: str, total_points: int, charge_states: int):
        st.markdown(f"""
        <div class="protein-card">
            <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">🧬 {protein_name}</h4>
            <p style="margin: 0; color: #64748b;">
                <span class="metric-badge">{total_points} calibrated points</span>
                <span class="metric-badge">{charge_states} charge states</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_next_steps():
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #667eea; margin-top: 0;">🎯 Next Steps</h4>
            <p>Your normalized and calibrated drift data is ready! Each CSV file contains:</p>
            <ul>
                <li><strong>Charge:</strong> Charge state</li>
                <li><strong>Drift:</strong> Drift time (seconds)</li>
                <li><strong>CCS:</strong> Collision cross-section</li>
                <li><strong>CCS Std.Dev.:</strong> Standard deviation</li>
                <li><strong>Normalized_Intensity:</strong> ATD intensity normalized to max=1</li>
                <li><strong>Scaled_Intensity:</strong> Normalized intensity × mass spectrum integration</li>
                <li><strong>m/z:</strong> Calculated m/z value</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- Main App ---
def main():
    styling.load_custom_css()
    UI.show_main_header()
    UI.show_info_card()

    # Step 1: Upload files
    drift_zip = UI.show_upload_section()
    cal_csvs = UI.show_calibration_upload()
    if not drift_zip or not cal_csvs:
        st.info("Upload both drift ZIP and calibration CSVs to continue.")
        return

    # Step 2: Get protein info
    protein_names = [file.name.replace(".csv", "") for file in cal_csvs]
    protein_masses = UI.get_protein_masses(protein_names)

    # Validate masses
    if any(mass == 0.0 for mass in protein_masses.values()):
        st.warning("Please enter a mass for every protein.")
        return

    # Step 3: Integration range and charge selection
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
        tmpdir.name, protein_names, protein_masses
    )

    # Step 4: Instrument settings
    instrument_type, inject_time = UI.show_instrument_settings()

    # Step 5: Process data
    if st.button("🚀 Process Normalized ATDs", type="primary"):
        with st.spinner("Processing normalized ATD data..."):
            result = CalibratedDriftProcessor.match_and_calibrate(
                drift_zip, cal_csvs, instrument_type, inject_time,
                charge_ranges, scale_ranges, protein_masses
            )

        if result.output_buffers:
            UI.show_processing_status(result.processed_files, result.matched_points, len(result.output_buffers))
            for filename, dfs in result.output_buffers.items():
                protein_name = filename.replace('.csv', '')
                total_points = sum(len(df) for df in dfs)
                UI.show_protein_card(protein_name, total_points, len(dfs))

            zip_buffer = CalibratedDriftProcessor.prepare_zip(result.output_buffers)
            st.download_button(
                label="📦 Download Normalized & Calibrated Data (ZIP)",
                data=zip_buffer,
                file_name="normalized_calibrated_drift_data.zip",
                mime="application/zip"
            )
            UI.show_next_steps()
        else:
            st.error("No matching data found. Please check your file formats and naming.")

if __name__ == "__main__":
    main()