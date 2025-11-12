"""
Streamlit page for calibrating drift files and scaling normalized ATDs.

This page allows users to:
1. Upload raw drift files (ZIP with ATD data + mass spectra)
2. Upload calibration CSV files from previous step
3. Configure protein masses and charge ranges
4. Set integration ranges for mass spectrum scaling
5. Process data to generate calibrated and scaled CCSDs

Uses the imspartacus.processing module for all core scientific processing.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from myutils import styling
from imspartacus.processing import (
    DriftCalibrationProcessor,
    get_automatic_range,
    calculate_theoretical_mz,
    plot_spectrum_with_integration,
    plot_full_spectrum_with_charge_states,
    PROTON_MASS
)


class UI:
    """Streamlit UI components for drift calibration and scaling."""
    
    @staticmethod
    def show_main_header():
        """Display the main page header."""
        st.markdown(
            """
            <div class="main-header">
                <h1>Calibrate Drift Files & Scale Normalized ATDs</h1>
                <p>Normalize ATD intensities to max=1, then match calibration data and scale using mass spectrum integration</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_info_card():
        """Display information about the page functionality."""
        st.markdown(
            """
            <div class="info-card">
                <p>Use this page to generate calibrated and scaled CCSDs for each protein using the processed 
                IMSCal<sup>1</sup> output files. This step completes the calibration process by normalizing ATD 
                intensities to a maximum of 1, then matching calibration data and scaling using mass spectrum 
                integration with baseline fitting.</p>
                <p><strong>What you'll need:</strong></p>
                <ul>
                    <li><strong>ZIP file containing raw drift files:</strong> Each protein folder should contain 
                    X.txt files (where X is the charge state) and a mass_spectrum.txt file</li>
                    <li><strong>CSV files from the 'Process Output Files' step:</strong> These contain the 
                    calibration data generated in the previous step</li>
                    <li><strong>Protein masses:</strong> Molecular mass (Da) for each protein</li>
                    <li><strong>Charge state range:</strong> Specify which charge states to include</li>
                    <li><strong>Integration ranges:</strong> Define m/z ranges for mass spectrum integration</li>
                </ul>
                <p><strong>Note:</strong> This step performs baseline fitting and integration for accurate scaling. 
                The output includes both normalized intensities (max=1) and scaled intensities (normalized × mass 
                spectrum scale factor).</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_upload_section():
        """Display upload section for drift files."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📁 Step 1: Upload Raw Drift Files</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return st.file_uploader(
            "Upload zipped folder of raw drift files", 
            type="zip",
            help="ZIP file should contain folders with X.txt files and mass_spectrum.txt",
            key="drift_zip"
        )
    
    @staticmethod
    def show_calibration_upload():
        """Display upload section for calibration CSV files."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📊 Step 2: Upload Calibration Data</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return st.file_uploader(
            "Upload the CSV files from the 'Process Output Files' page", 
            type="csv", 
            accept_multiple_files=True,
            help="Select all CSV files generated in the previous step",
            key="cal_csvs"
        )
    
    @staticmethod
    def get_instrument_settings():
        """Get instrument configuration from user."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">⚙️ Step 3: Instrument Settings</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            instrument_type = st.selectbox(
                "Instrument Type",
                ["Synapt", "Cyclic"],
                help="Synapt: standard TWIMS. Cyclic: requires injection time correction"
            )
        
        with col2:
            inject_time = 0.0
            if instrument_type == "Cyclic":
                inject_time = st.number_input(
                    "Injection Time (ms)",
                    min_value=0.0,
                    value=0.5,
                    step=0.1,
                    help="Time subtracted from drift times for cyclic IMS"
                )
        
        return instrument_type, inject_time
    
    @staticmethod
    def get_protein_masses(protein_names: List[str]) -> Dict[str, float]:
        """Get molecular masses for each protein."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">⚖️ Step 4: Enter Protein Masses</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        masses = {}
        cols = st.columns(min(3, len(protein_names)))
        for i, name in enumerate(protein_names):
            with cols[i % len(cols)]:
                mass = st.number_input(
                    f"Mass (Da) for {name}",
                    min_value=0.0,
                    value=0.0,
                    key=f"mass_{name}"
                )
                masses[name] = mass
        return masses
    
    @staticmethod
    def get_charge_ranges(protein_names: List[str]) -> Dict[str, Tuple[int, int]]:
        """Get charge state ranges for each protein."""
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">🔋 Step 5: Select Charge State Ranges</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        charge_ranges = {}
        for protein in protein_names:
            col1, col2 = st.columns(2)
            with col1:
                min_charge = st.number_input(
                    f"{protein} - Min charge",
                    min_value=1,
                    value=2,
                    key=f"min_charge_{protein}"
                )
            with col2:
                max_charge = st.number_input(
                    f"{protein} - Max charge",
                    min_value=min_charge,
                    value=min_charge + 2,
                    key=f"max_charge_{protein}"
                )
            charge_ranges[protein] = (min_charge, max_charge)
        
        return charge_ranges
    
    @staticmethod
    def configure_integration_ranges(
        protein_names: List[str],
        protein_masses: Dict[str, float],
        charge_ranges: Dict[str, Tuple[int, int]],
        ms_data: Dict[str, pd.DataFrame],
        use_max_intensity: bool = False
    ) -> Tuple[Dict[Tuple[str, int], Tuple[float, float]], pd.DataFrame]:
        """
        Configure integration ranges for mass spectrum scaling.
        
        Provides interactive UI for each protein/charge combination with:
        - Automatic range suggestion
        - Manual adjustment
        - Live preview with baseline fitting
        - Real-time scale factor calculation and display
        
        Returns:
            Tuple of (scale_ranges, scale_factors_df)
        """
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📐 Step 7: Set Integration Ranges</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        method_text = "maximum intensity" if use_max_intensity else "baseline-corrected integration"
        st.info(
            f"For each protein/charge combination, set the m/z integration range for mass spectrum scaling. "
            f"Scale factors will be calculated using {method_text}."
        )
        
        scale_ranges = {}
        scale_factor_data = []  # For live table
        
        for protein in protein_names:
            if protein not in ms_data or protein_masses.get(protein, 0) == 0:
                continue
            
            st.markdown(f"#### {protein}")
            ms_df = ms_data[protein]
            mass = protein_masses[protein]
            min_charge, max_charge = charge_ranges.get(protein, (2, 4))
            
            # Create tabs for each charge state
            charges = list(range(min_charge, max_charge + 1))
            tabs = st.tabs([f"Charge {c}+" for c in charges])
            
            for idx, charge in enumerate(charges):
                with tabs[idx]:
                    mz = calculate_theoretical_mz(mass, charge)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Auto-range suggestion
                        auto_percent = st.slider(
                            "Auto-range %",
                            min_value=1.0,
                            max_value=20.0,
                            value=5.0,
                            step=0.5,
                            key=f"auto_{protein}_{charge}"
                        )
                        auto_min, auto_max = get_automatic_range(mz, auto_percent)
                        
                        if st.button(f"Use auto-range", key=f"auto_btn_{protein}_{charge}"):
                            st.session_state[f"range_min_{protein}_{charge}"] = auto_min
                            st.session_state[f"range_max_{protein}_{charge}"] = auto_max
                        
                        # Manual range input
                        range_min = st.number_input(
                            "Min m/z",
                            value=st.session_state.get(f"range_min_{protein}_{charge}", auto_min),
                            key=f"range_min_{protein}_{charge}",
                            format="%.3f"
                        )
                        range_max = st.number_input(
                            "Max m/z",
                            value=st.session_state.get(f"range_max_{protein}_{charge}", auto_max),
                            key=f"range_max_{protein}_{charge}",
                            format="%.3f"
                        )
                        
                        scale_ranges[(protein, charge)] = (range_min, range_max)
                        
                        # Calculate scale factor in real-time
                        scale_factor, _ = DriftCalibrationProcessor.calculate_scale_factor(
                            ms_df,
                            protein,
                            charge,
                            mass,
                            {(protein, charge): (range_min, range_max)},
                            use_max_intensity=use_max_intensity,
                            smoothing_window=51
                        )
                        
                        # Show theoretical m/z and scale factor
                        st.metric("Theoretical m/z", f"{mz:.3f}")
                        if scale_factor is not None:
                            st.metric("Scale Factor", f"{scale_factor:.2e}")
                            # Store for table
                            scale_factor_data.append({
                                "Protein": protein,
                                "Charge": charge,
                                "m/z": f"{mz:.3f}",
                                "Range": f"{range_min:.3f} - {range_max:.3f}",
                                "Scale Factor": f"{scale_factor:.2e}"
                            })
                        else:
                            st.warning("⚠ Could not calculate scale factor")
                    
                    with col2:
                        # Plot with integration preview
                        show_zoomed = st.checkbox(
                            "Zoom to ±10%",
                            value=True,
                            key=f"zoom_{protein}_{charge}"
                        )
                        
                        area, range_outside, fig = plot_spectrum_with_integration(
                            ms_df,
                            mz,
                            (range_min, range_max),
                            smoothing_window=51,
                            show_zoomed=show_zoomed
                        )
                        
                        if fig:
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                        
                        # Display appropriate message based on method
                        if area is not None:
                            # Note: area variable contains integration result from plot
                            # but scale_factor uses the selected method
                            st.success(f"✓ Integration area: {area:.2e}")
                            if use_max_intensity:
                                st.info("ℹ️ Note: Scale factor uses maximum intensity, but plot shows integration area")
                        elif range_outside:
                            st.warning("⚠ Integration range extends beyond view. Toggle zoom to see full range.")
                        else:
                            st.warning("⚠ Integration range too small. Please expand the range.")
        
        # Create DataFrame of scale factors
        scale_factors_df = pd.DataFrame(scale_factor_data) if scale_factor_data else None
        
        return scale_ranges, scale_factors_df
    
    @staticmethod
    def show_processing_results(result, skipped_files: List[str]):
        """Display processing results."""
        st.markdown(
            f"""
            <div class="success-card">
                <strong>✅ Processing Complete!</strong><br>
                • Processed: <span class="metric-badge">{result.processed_files} files</span><br>
                • Matched: <span class="metric-badge">{result.matched_points} data points</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if skipped_files:
            with st.expander(f"⚠ Skipped {len(skipped_files)} files", expanded=False):
                for msg in skipped_files[:20]:
                    st.write(f"• {msg}")
                if len(skipped_files) > 20:
                    st.write(f"...and {len(skipped_files) - 20} more")
    
    @staticmethod
    def show_references():
        """Display references section."""
        st.markdown(
            """
            <div class="info-card">
                <h3>📚 References</h3>
                <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, 
                <em>International Journal of Mass Spectrometry,</em> 2023, 492, 117112.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def extract_protein_names_from_csvs(cal_csvs: List) -> List[str]:
    """Extract protein names from calibration CSV filenames."""
    return [f.name.replace(".csv", "") for f in cal_csvs]


def load_mass_spectra(drift_zip) -> Dict[str, pd.DataFrame]:
    """Load mass spectrum files from ZIP."""
    import zipfile
    import tempfile
    import os
    
    ms_data = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract ZIP
        zip_path = Path(tmpdir) / "drift.zip"
        with open(zip_path, "wb") as f:
            f.write(drift_zip.getvalue())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Find mass spectrum files
        for root, dirs, files in os.walk(tmpdir):
            if "mass_spectrum.txt" in files:
                protein_name = os.path.basename(root)
                ms_path = os.path.join(root, "mass_spectrum.txt")
                ms_df = DriftCalibrationProcessor.load_mass_spectrum(ms_path)
                if ms_df is not None:
                    ms_data[protein_name] = ms_df
    
    return ms_data


def main():
    """Main application logic."""
    # Load custom CSS
    styling.load_custom_css()
    
    # Show header and info
    UI.show_main_header()
    UI.show_info_card()
    
    # Step 1: Upload drift files
    drift_zip = UI.show_upload_section()
    if not drift_zip:
        UI.show_references()
        return
    
    # Step 2: Upload calibration CSVs
    cal_csvs = UI.show_calibration_upload()
    if not cal_csvs:
        st.warning("Please upload calibration CSV files to continue.")
        return
    
    # Extract protein names
    protein_names = extract_protein_names_from_csvs(cal_csvs)
    st.success(f"Found {len(protein_names)} protein(s): {', '.join(protein_names)}")
    
    # Step 3: Instrument settings
    instrument_type, inject_time = UI.get_instrument_settings()
    
    # Step 4: Protein masses
    protein_masses = UI.get_protein_masses(protein_names)
    if any(m == 0.0 for m in protein_masses.values()):
        st.warning("⚠ Please enter masses for all proteins.")
        return
    
    # Step 5: Charge ranges
    charge_ranges = UI.get_charge_ranges(protein_names)
    
    # Load mass spectra for integration range configuration
    with st.spinner("Loading mass spectra..."):
        ms_data = load_mass_spectra(drift_zip)
    
    if not ms_data:
        st.error("❌ No mass spectrum files found in ZIP. Each protein folder must contain mass_spectrum.txt")
        return
    
    # Step 5.5: Scaling method selection
    st.markdown(
        """
        <div class="section-card">
            <div class="section-header">📏 Step 6: Select Scaling Method</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    use_max_intensity = st.radio(
        "How should scale factors be calculated?",
        options=[False, True],
        format_func=lambda x: "Integration (baseline-corrected area)" if not x else "Maximum intensity in range",
        help="Integration provides more accurate quantification by fitting a baseline and integrating the area. "
             "Maximum intensity is faster and simpler but less accurate."
    )
    
    # Step 7: Configure integration ranges
    scale_ranges, scale_factors_df = UI.configure_integration_ranges(
        protein_names,
        protein_masses,
        charge_ranges,
        ms_data,
        use_max_intensity
    )
    
    if not scale_ranges:
        st.warning("⚠ Please configure at least one integration range.")
        return
    
    # Display scale factors table
    if scale_factors_df is not None and len(scale_factors_df) > 0:
        st.markdown("---")
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📊 Scale Factors Summary</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(
            scale_factors_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Processing section
    st.markdown("---")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-header">🚀 Process Data</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("🔄 Process All Data", type="primary"):
        with st.spinner("Processing..."):
            # Process using imspartacus library
            result, skipped_files = DriftCalibrationProcessor.match_and_calibrate(
                drift_zip=drift_zip,
                cal_csvs=cal_csvs,
                instrument_type=instrument_type,
                inject_time=inject_time,
                charge_ranges=charge_ranges,
                scale_ranges=scale_ranges,
                protein_masses=protein_masses,
                use_max_intensity=use_max_intensity
            )
            
            # Show results
            UI.show_processing_results(result, skipped_files)
            
            # Prepare download
            if result.output_buffers:
                zip_buffer = DriftCalibrationProcessor.prepare_zip(result.output_buffers)
                
                st.download_button(
                    label="📦 Download Calibrated Data (ZIP)",
                    data=zip_buffer,
                    file_name="calibrated_scaled_data.zip",
                    mime="application/zip"
                )
    
    # Show references
    UI.show_references()


if __name__ == "__main__":
    main()
