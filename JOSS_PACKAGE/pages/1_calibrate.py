"""
Streamlit page for calibrant data processing.

This is the refactored version that uses the imspartacus core library.
"""

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import io

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Import from existing utils
from myutils import import_tools
from myutils import styling

# Import from new core library
from imspartacus.calibration import (
    CalibrantDatabase,
    CalibrantProcessor,
    InstrumentParams,
    adjust_dataframe_drift_times,
    CALIBRANT_FOLDER_MAPPING
)
from imspartacus.io.writers import write_imscal_dat


@dataclass
class CalibrationParams:
    """UI-specific configuration parameters for calibration process."""
    velocity: float
    voltage: float
    pressure: float
    length: float
    calibrant_type: str
    data_type: str
    inject_time: float = 0.0
    min_r2: float = 0.9


class ResultsDisplayer:
    """Handles Streamlit display of processing results."""
    
    @staticmethod
    def display_dataframe_results(results_df: pd.DataFrame) -> None:
        """Display results dataframe in Streamlit."""
        if not results_df.empty:
            st.markdown('<h3 class="section-header">Gaussian Fit Results</h3>', unsafe_allow_html=True)
            st.dataframe(results_df)
        else:
            st.markdown(
                '<div class="warning-card">No valid calibrant data found that matches the database.</div>',
                unsafe_allow_html=True
            )
    
    @staticmethod
    def display_plots(all_measurements: List, min_r2: float) -> None:
        """
        Display all fitting plots.
        
        Args:
            all_measurements: List of CalibrantMeasurement objects (both successful and skipped)
            min_r2: R² threshold for color-coding plots
        """
        # Filter to only measurements with fit results
        measurements_with_fits = [m for m in all_measurements if m.fit_result is not None]
        
        if not measurements_with_fits:
            return
        
        n_plots = len(measurements_with_fits)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 4 * n_rows))
        
        for i, measurement in enumerate(measurements_with_fits):
            fit = measurement.fit_result
            plt.subplot(n_rows, n_cols, i + 1)
            plt.plot(fit.drift_time, fit.intensity, 'b.', label='Raw Data', markersize=3)
            plt.plot(fit.drift_time, fit.fitted_values, 'r-', label='Gaussian Fit', linewidth=1)
            
            # Color code the title based on R² value
            title_color = 'red' if fit.r_squared < min_r2 else 'black'
            filename = measurement.filename or f"charge_{measurement.charge_state}"
            plt.title(f'{filename}\nApex: {fit.apex:.2f}, R²: {fit.r_squared:.3f}', color=title_color)
            plt.xlabel('Drift Time')
            plt.ylabel('Intensity')
            plt.legend()
            plt.grid()
        
        plt.tight_layout()
        st.pyplot(plt)
    
    @staticmethod
    def display_skipped_entries(skipped_entries: List[str]) -> None:
        """Display skipped entries with warnings."""
        if not skipped_entries:
            return
            
        st.markdown('<h3 class="section-header">⚠️ Skipped Entries</h3>', unsafe_allow_html=True)
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.write("The following entries were skipped:")
        for entry in skipped_entries:
            st.write(f"• {entry}")
        st.markdown('</div>', unsafe_allow_html=True)


class FileDownloader:
    """Handles generation of download buttons for Streamlit."""
    
    @staticmethod
    def create_download_buttons(results_df: pd.DataFrame, params: CalibrationParams) -> None:
        """Create Streamlit download buttons for results."""
        if results_df.empty:
            st.markdown(
                '<div class="error-card">No valid results to download. Please check your data and database matching.</div>',
                unsafe_allow_html=True
            )
            return

        st.markdown('<div class="section-divider">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📥 Download Results</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # CSV download
            csv_filename = st.text_input(
                "CSV filename", 
                value="combined_gaussian_fit_results.csv", 
                key="csv_filename"
            )
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📊 Download Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name=csv_filename if csv_filename else "combined_gaussian_fit_results.csv",
                mime="text/csv"
            )

        with col2:
            # .dat download
            dat_filename = st.text_input(
                ".dat filename", 
                value="calibration_data.dat", 
                key="dat_filename"
            )
            
            # Adjust drift times if needed
            adjusted_df = results_df.copy()
            if params.data_type.lower() == "cyclic":
                instrument_params = InstrumentParams(
                    wave_velocity=params.velocity,
                    wave_height=params.voltage,
                    pressure=params.pressure,
                    drift_length=params.length,
                    instrument_type=params.data_type.lower(),
                    inject_time=params.inject_time
                )
                adjusted_df = adjust_dataframe_drift_times(adjusted_df, instrument_params)

            # Generate .dat content
            dat_content = write_imscal_dat(
                adjusted_df,
                velocity=params.velocity,
                voltage=params.voltage,
                pressure=params.pressure,
                length=params.length,
                output_path=None  # Return string instead of writing file
            )
            
            if dat_content:
                st.download_button(
                    label="📋 Download .dat File",
                    data=dat_content,
                    file_name=dat_filename if dat_filename else "calibration_data.dat",
                    mime="text/plain"
                )

        st.markdown('</div>', unsafe_allow_html=True)


class UIComponents:
    """Handles UI component creation and user input."""
    
    @staticmethod
    def display_folder_naming_table() -> None:
        """Display the calibrant folder naming reference table."""
        df = pd.DataFrame({
            'Protein': list(CALIBRANT_FOLDER_MAPPING.keys()),
            'Folder Name': list(CALIBRANT_FOLDER_MAPPING.values())
        })
        
        st.markdown('<h3 class="section-header">Calibrant Folder Naming</h3>', unsafe_allow_html=True)
        st.table(df)
    
    @staticmethod
    def get_calibration_parameters() -> CalibrationParams:
        """Get calibration parameters from user input."""
        st.markdown('<h3 class="section-header">⚗️ Calibration Parameters</h3>', unsafe_allow_html=True)
        st.markdown(
            'Most of the time you should calibrate with calibrant values obtained for the same '
            'drift gas as you used in your experiment, but sometimes you might not so the option is here.'
        )
        
        calibrant_type = st.selectbox(
            "Which values from the Bush database would you like to calibrate with?",
            options=["Helium", "Nitrogen"]
        )
        
        # R² threshold setting
        st.markdown("**Quality Control Settings**")
        min_r2 = st.number_input(
            "Minimum R² value for inclusion (entries below this will be skipped)",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Default is 0.9. Gaussian fits with R² below this threshold will be excluded from results but still shown in plots (in red)."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            velocity = st.number_input(
                "Enter wave velocity (m/s), multiplied by 0.75 if this is Cyclic data",
                min_value=0.0, value=281.0
            )
            voltage = st.number_input("Enter wave height (V)", min_value=0.0, value=20.0)
        
        with col2:
            pressure = st.number_input("Enter IMS pressure", min_value=0.0, value=1.63)
            length = st.number_input(
                "Enter drift cell length (0.25m for Synapt, 0.98m for Cyclic)",
                min_value=0.0, value=0.980
            )
        
        data_type = st.radio("Is this Cyclic or Synapt data?", options=["Cyclic", "Synapt"])
        
        inject_time = 0.0
        if data_type.lower() == "cyclic":
            inject_time = st.number_input("Enter inject time (ms)", min_value=0.0, value=0.0)
        
        return CalibrationParams(
            velocity=velocity,
            voltage=voltage,
            pressure=pressure,
            length=length,
            calibrant_type=calibrant_type,
            data_type=data_type,
            inject_time=inject_time,
            min_r2=min_r2
        )


def main():
    """Main Streamlit application function."""
    styling.load_custom_css()

    # Header
    st.markdown(
        '<div class="main-header">'
        '<h1>Process Calibrant Data</h1>'
        '<p>Fit ATDs of calibrants and generate reference files for IMSCal<sup>1</sup></p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Info card
    st.markdown("""
    <div class="info-card">
        <p>Use this page to fit the ATDs of your calibrants either from text files or CSV files generated using TWIMExtract<sup>2</sup> and generate a reference file for IMSCal<sup>1</sup> and/or a csv file of calibrant measured and literature arrival times. This is designed for use with denatured calibrants, so the fitting only allows for a single peak in each ATD - consider another tool if your ATDs are not gaussian. You will still be able to use subsequent tools.</p>
        <p>To start, make a folder for each calibrant you used. You should name these folders according to the table below (or they won't match the database file<sup>3</sup>). You can use either:</p>
        <ul>
            <li><strong>Text files (.txt):</strong> Create a text file for each charge state (called 'X.txt' where X is the charge state) and paste the corresponding ATD from MassLynx into each file. Remember to set the x-axis to ms not bins!</li>
            <li><strong>CSV files (.csv):</strong> From TWIMExtract<sup>2</sup> or similar tools, generate a CSV file for the ATD of each charge state and rename them 'X.csv' where X is the charge state. CSV files can contain comment lines starting with '#' which will be ignored.</li>
        </ul>
        <p>Save these files under their respective protein folder, zip the protein folders together, and upload below.</p>
        <p><strong>Quality Control:</strong> By default, entries with R² < 0.9 are excluded from results but shown in plots (colored red) for manual inspection. This threshold can be adjusted in the parameters section.</p>
    </div>
    """, unsafe_allow_html=True)

    # Folder naming table
    UIComponents.display_folder_naming_table()

    # File upload
    st.markdown('<h3 class="section-header">📁 Upload Calibrant Data</h3>', unsafe_allow_html=True)
    uploaded_zip_file = st.file_uploader(
        "Upload a ZIP file containing your calibrant folders",
        type="zip"
    )

    # Clear cache button
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()

    if uploaded_zip_file is None:
        st.markdown("""
        <div class="info-card">
            <h3>📚 References</h3>
            <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, <em> International Journal of Mass Spectrometry,</em>,2023, 492, 117112.</p>
            <p><sup>2</sup> S. E. Haynes, D. A. Polasky, S. M. Dixit, J. D. Majmudar, K. Neeson, B. T. Ruotolo and B. R. Martin, <em>Analytical Chemistry</em>, 2017, 89, 5669–5672.</p>
            <p><sup>3</sup> Bush, M. F., et al. <em>Journal of the American Society for Mass Spectrometry</em>, 2010, 21, 1003-1010.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        # Extract uploaded ZIP
        folders, temp_dir = import_tools.handle_zip_upload(uploaded_zip_file)
        
        # Load Bush database
        bush_df = import_tools.read_bush()
        if bush_df.empty:
            st.markdown(
                '<div class="error-card">Cannot proceed without the Bush calibrant database.</div>',
                unsafe_allow_html=True
            )
            return

        # Get user parameters
        params = UIComponents.get_calibration_parameters()

        # Initialize core library components
        db = CalibrantDatabase(bush_df)
        processor = CalibrantProcessor(db, min_r2=params.min_r2)
        displayer = ResultsDisplayer()

        # Process all folders
        all_results = []
        all_skipped = []
        all_measurements_for_plots = []  # For plotting (includes low R²)

        st.markdown('<h3 class="section-header">🔬 Processing Results</h3>', unsafe_allow_html=True)

        for folder in folders:
            st.markdown(
                f'<div class="form-section">Processing folder: <span class="metric-badge">{folder}</span></div>',
                unsafe_allow_html=True
            )
            folder_path = Path(temp_dir) / folder
            
            # Process folder using core library
            measurements, skipped = processor.process_folder(
                folder_path,
                folder,
                params.calibrant_type.lower()
            )
            
            # Convert successful measurements to DataFrame format
            if measurements:
                folder_df = pd.DataFrame([
                    {
                        'protein': m.protein,
                        'mass': m.mass,
                        'charge state': m.charge_state,
                        'drift time': m.drift_time,
                        'r2': m.r_squared,
                        'calibrant_value': m.ccs_literature
                    }
                    for m in measurements
                ])
                all_results.append(folder_df)
            
            # Collect all measurements for plotting (successful + low R²)
            all_measurements_for_plots.extend(measurements)
            all_measurements_for_plots.extend([item for item in skipped if not isinstance(item, str)])
            
            # Format skipped entries for display
            for item in skipped:
                if isinstance(item, str):
                    # It's an error message
                    all_skipped.append(f"{folder} - {item}")
                else:
                    # It's a CalibrantMeasurement with low R²
                    all_skipped.append(
                        f"{folder} charge {item.charge_state} - "
                        f"R² ({item.r_squared:.3f}) below threshold ({params.min_r2:.1f})"
                    )

        # Combine results
        combined_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

        # Display results
        displayer.display_dataframe_results(combined_results)
        displayer.display_plots(all_measurements_for_plots, params.min_r2)
        displayer.display_skipped_entries(all_skipped)

        # Download buttons
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        FileDownloader.create_download_buttons(combined_results, params)
        st.markdown('</div>', unsafe_allow_html=True)

        # References
        st.markdown("""
        <div class="info-card">
            <h3>📚 References</h3>
            <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, <em> International Journal of Mass Spectrometry,</em>,2023, 492, 117112.</p>
            <p><sup>2</sup> S. E. Haynes, D. A. Polasky, S. M. Dixit, J. D. Majmudar, K. Neeson, B. T. Ruotolo and B. R. Martin, <em>Analytical Chemistry</em>, 2017, 89, 5669–5672.</p>
            <p><sup>3</sup> Bush, M. F., et al. <em>Journal of the American Society for Mass Spectrometry</em>, 2010, 21, 1003-1010.</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(
            f'<div class="error-card">An error occurred during processing: {str(e)}</div>',
            unsafe_allow_html=True
        )
        # Print traceback for debugging
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
