from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import os
import io
import matplotlib.pyplot as plt
from pathlib import Path

from myutils import data_tools
from myutils import import_tools
from myutils import styling


@dataclass
class CalibrationParams:
    """Configuration parameters for calibration process."""
    velocity: float
    voltage: float
    pressure: float
    length: float
    calibrant_type: str
    data_type: str
    inject_time: float = 0.0


@dataclass
class ProcessingResult:
    """Container for processing results."""
    results_df: pd.DataFrame
    plots: List[Tuple]
    skipped_entries: List[str]


class CalibrantProcessor:
    """Handles calibrant data processing and fitting."""
    
    def __init__(self, bush_df: pd.DataFrame):
        self.bush_df = bush_df
        self.REQUIRED_COLUMNS = ['protein', 'mass', 'charge state', 'drift time', 'r2', 'calibrant_value']
    
    def _get_calibrant_column(self, calibrant_type: str) -> str:
        """Get the appropriate column name based on calibrant type."""
        return 'CCS_he' if calibrant_type.lower() == 'helium' else 'CCS_n2'
    
    def _load_data_from_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load drift time and intensity data from either txt or csv file."""
        if file_path.suffix.lower() == '.csv':
            # Handle CSV format with potential comments
            data_rows = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comment lines (starting with #)
                    if not line or line.startswith('#'):
                        continue
                    # Parse the data line
                    try:
                        values = line.split(',')
                        if len(values) >= 2:
                            drift_time = float(values[0])
                            intensity = float(values[1])
                            data_rows.append([drift_time, intensity])
                    except (ValueError, IndexError):
                        continue  # Skip malformed lines
            
            if not data_rows:
                raise ValueError("No valid data found in CSV file")
            
            data = np.array(data_rows)
            return data[:, 0], data[:, 1]
        
        else:  # .txt file
            # Use original method for text files
            data = np.loadtxt(file_path)
            return data[:, 0], data[:, 1]
    
    def _process_single_file(self, file_path: Path, folder_name: str, calibrant_column: str) -> Optional[Dict[str, Any]]:
        """Process a single data file and return result or None if failed."""
        try:
            # Load data using the appropriate method
            drift_time, intensity = self._load_data_from_file(file_path)
            
            # Perform gaussian fit
            params, r2, fitted_values = data_tools.fit_gaussian_with_retries(drift_time, intensity)
            
            if params is None:
                return None
                
            amp, apex, stddev = params
            charge_state = int(file_path.stem)
            
            # Look up calibrant data
            calibrant_row = self.bush_df[
                (self.bush_df['protein'] == folder_name) & 
                (self.bush_df['charge'] == charge_state)
            ]
            
            if calibrant_row.empty:
                return None
                
            calibrant_value = calibrant_row[calibrant_column].values[0]
            mass = calibrant_row['mass'].values[0]
            
            if pd.isna(calibrant_value):
                return None
                
            return {
                'data': [folder_name, mass, charge_state, apex, r2, calibrant_value],
                'plot': (drift_time, intensity, fitted_values, file_path.name, apex, r2)
            }
            
        except Exception as e:
            st.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _is_valid_data_file(self, file_path: Path) -> bool:
        """Check if file is a valid data file to process."""
        return (
            file_path.suffix in ['.txt', '.csv'] and 
            file_path.name[0].isdigit()
        )
    
    def process_folder(self, folder_name: str, folder_path: Path, calibrant_type: str) -> ProcessingResult:
        """Process all files in a folder and return results."""
        calibrant_column = self._get_calibrant_column(calibrant_type)
        
        results = []
        plots = []
        skipped_entries = []
        
        for file_path in folder_path.iterdir():
            if not self._is_valid_data_file(file_path):
                continue
                
            result = self._process_single_file(file_path, folder_name, calibrant_column)
            
            if result:
                results.append(result['data'])
                plots.append(result['plot'])
            else:
                charge_state = file_path.stem
                skipped_entries.append(
                    f"{folder_name} charge {charge_state} - "
                    f"Processing failed or no {calibrant_type.lower()} CCS value available"
                )
        
        results_df = pd.DataFrame(results, columns=self.REQUIRED_COLUMNS)
        return ProcessingResult(results_df, plots, skipped_entries)


class ResultsDisplayer:
    """Handles display of processing results."""
    
    @staticmethod
    def display_dataframe_results(results_df: pd.DataFrame) -> None:
        """Display results dataframe."""
        if not results_df.empty:
            st.markdown('<h3 class="section-header">Gaussian Fit Results</h3>', unsafe_allow_html=True)
            st.dataframe(results_df)
        else:
            st.markdown(
                '<div class="warning-card">No valid calibrant data found that matches the database.</div>',
                unsafe_allow_html=True
            )
    
    @staticmethod
    def display_plots(plots: List[Tuple]) -> None:
        """Display all fitting plots."""
        if not plots:
            return
            
        n_plots = len(plots)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 4 * n_rows))
        
        for i, (drift_time, intensity, fitted_values, filename, apex, r2) in enumerate(plots):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.plot(drift_time, intensity, 'b.', label='Raw Data', markersize=3)
            plt.plot(drift_time, fitted_values, 'r-', label='Gaussian Fit', linewidth=1)
            plt.title(f'{filename}\nApex: {apex:.2f}, R²: {r2:.3f}')
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
            
        st.markdown('<div class="section-divider">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚠️ Skipped Entries</h3>', unsafe_allow_html=True)
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.write("The following entries were skipped:")
        for entry in skipped_entries:
            st.write(f"• {entry}")
        st.markdown('</div></div>', unsafe_allow_html=True)


class FileGenerator:
    """Handles generation of output files."""
    
    @staticmethod
    def generate_dat_content(results_df: pd.DataFrame, params: CalibrationParams) -> Optional[str]:
        """Generate .dat file content from results."""
        if results_df.empty:
            return None
        
        header = (
            f"# length {params.length}\n"
            f"# velocity {params.velocity}\n"
            f"# voltage {params.voltage}\n"
            f"# pressure {params.pressure}\n"
        )
        
        content_lines = []
        for _, row in results_df.iterrows():
            protein = row['protein']
            charge_state = row['charge state']
            mass = row['mass']
            calibrant_value = row['calibrant_value'] * 100  # Convert to Ų
            drift_time = row['drift time']
            
            content_lines.append(
                f"{protein}_{charge_state} {mass} {charge_state} {calibrant_value} {drift_time}"
            )
        
        return header + "\n".join(content_lines)
    
    @staticmethod
    def create_download_buttons(results_df: pd.DataFrame, params: CalibrationParams) -> None:
        """Create download buttons for results."""
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
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📊 Download Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="combined_gaussian_fit_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Adjust drift times for cyclic data
            adjusted_df = results_df.copy()
            if params.data_type.lower() == "cyclic":
                adjusted_df['drift time'] = adjusted_df['drift time'] - params.inject_time
            
            # .dat file download
            dat_content = FileGenerator.generate_dat_content(adjusted_df, params)
            if dat_content:
                st.download_button(
                    label="📋 Download .dat File",
                    data=dat_content,
                    file_name="calibration_data.dat",
                    mime="text/plain"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)


class UIComponents:
    """Handles UI component creation and user input."""
    
    CALIBRANT_FOLDER_MAPPING = {
        'Denatured Myoglobin': 'myoglobin',
        'Denatured Cytochrome C': 'cytochromec',
        'Polyalanine Peptide of Length X': 'polyalanineX',
        'Denatured Ubiquitin': 'ubiquitin'
    }
    
    @classmethod
    def display_folder_naming_table(cls) -> None:
        """Display the calibrant folder naming reference table."""
        df = pd.DataFrame({
            'Protein': list(cls.CALIBRANT_FOLDER_MAPPING.keys()),
            'Folder Name': list(cls.CALIBRANT_FOLDER_MAPPING.values())
        })
        
        st.markdown('<h3 class="section-header">Calibrant Folder Naming</h3>', unsafe_allow_html=True)
        st.table(df)
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def get_calibration_parameters() -> CalibrationParams:
        """Get calibration parameters from user input."""
        st.markdown('<div class="section-divider">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚗️ Calibration Parameters</h3>', unsafe_allow_html=True)
        st.markdown(
            'Most of the time you should calibrate with calibrant values obtained for the same '
            'drift gas as you used in your experiment, but sometimes you might not so the option is here.'
        )
        
        calibrant_type = st.selectbox(
            "Which values from the Bush database would you like to calibrate with?",
            options=["Helium", "Nitrogen"]
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
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return CalibrationParams(
            velocity=velocity,
            voltage=voltage,
            pressure=pressure,
            length=length,
            calibrant_type=calibrant_type,
            data_type=data_type,
            inject_time=inject_time
        )


def main():
    """Main application function."""
    styling.load_custom_css()

    # Main header (light pink background, straight borders)
    st.markdown(
        '<div class="main-header" >'
        '<h1>Process Calibrant Data</h1>'
        '<p>Fit ATDs of calibrants and generate reference files for IMSCal</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Info card (blue background, blue borders)
    st.markdown("""
    <div class="info-card">
        <p>Use this page to fit the ATDs of your calibrants from text files or CSV files and generate a reference file for IMSCal and/or a csv file of calibrant measured and literature arrival times. This is designed for use with denatured calibrants, so the fitting only allows for a single peak in each ATD - consider another tool if your ATDs are not gaussian.</p>
        <p>To start, make a folder for each calibrant you used. You should name these folders according to the table below (or they won't match the bush database file). You can use either:</p>
        <ul>
            <li><strong>Text files (.txt):</strong> Create a text file for each charge state (called 'X.txt' where X is the charge state) and paste the corresponding ATD from MassLynx into each file. Remember to set the x-axis to ms not bins!</li>
            <li><strong>CSV files (.csv):</strong> From TWIMExtract or similar tools, generate a CSV file for the ATD of each charge state and rename them 'X.csv' where X is the charge state. CSV files can contain comment lines starting with '#' which will be ignored.</li>
        </ul>
        <p>Save these files under their respective protein folder, zip the protein folders together, and upload below.</p>
    </div>
    """, unsafe_allow_html=True)

    # Folder naming table
    UIComponents.display_folder_naming_table()

    # File upload section header
    st.markdown('<h3 class="section-header">📁 Upload Calibrant Data</h3>', unsafe_allow_html=True)
    uploaded_zip_file = st.file_uploader(
        "Upload a ZIP file containing your calibrant folders",
        type="zip"
    )

    # Clear cache button inside info card for consistent styling
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()

    if uploaded_zip_file is None:
        return

    try:
        folders, temp_dir = import_tools.handle_zip_upload(uploaded_zip_file)
        bush_df = import_tools.read_bush()

        if bush_df.empty:
            st.markdown(
                '<div class="error-card">Cannot proceed without the Bush calibrant database.</div>',
                unsafe_allow_html=True
            )
            return

        params = UIComponents.get_calibration_parameters()

        processor = CalibrantProcessor(bush_df)
        displayer = ResultsDisplayer()

        all_results = []
        all_plots = []
        all_skipped = []

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🔬 Processing Results</h3>', unsafe_allow_html=True)

        for folder in folders:
            st.markdown(
                f'<div class="form-section">Processing folder: <span class="metric-badge">{folder}</span></div>',
                unsafe_allow_html=True
            )
            folder_path = Path(temp_dir) / folder
            result = processor.process_folder(folder, folder_path, params.calibrant_type)

            all_results.append(result.results_df)
            all_plots.extend(result.plots)
            all_skipped.extend(result.skipped_entries)

        st.markdown('</div>', unsafe_allow_html=True)

        combined_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

        # Results table (styled as CSV)
        displayer.display_dataframe_results(combined_results)
        displayer.display_plots(all_plots)
        displayer.display_skipped_entries(all_skipped)

        # Download buttons in info card
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        FileGenerator.create_download_buttons(combined_results, params)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown(
            f'<div class="error-card">An error occurred during processing: {str(e)}</div>',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()