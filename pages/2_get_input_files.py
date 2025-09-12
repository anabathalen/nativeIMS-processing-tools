from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import os
import io
import zipfile
from pathlib import Path

from myutils import import_tools
from myutils import styling

# --- Data Classes ---
@dataclass
class InputParams:
    drift_mode: str
    inject_time: float
    sample_mass_map: Dict[str, float]

@dataclass
class InputProcessingResult:
    processed_files: Dict[str, List[str]]
    failed_files: Dict[str, List[str]]
    sample_paths: List[str]

# --- Processing Logic ---
class InputProcessor:
    def __init__(self, base_path: str, params: InputParams):
        self.base_path = base_path
        self.params = params

    def _is_valid_data_file(self, file_path: Path) -> bool:
        """Check if file is a valid data file to process."""
        if file_path.suffix.lower() == '.csv':
            # For CSV files, check if filename contains a charge state pattern
            filename_without_ext = file_path.stem
            
            import re
            patterns = [
                r'range_(\d+)\.txt',  # Matches "range_24.txt"
                r'range_(\d+)_',      # Matches "range_24_"
                r'_(\d+)\.txt_raw',   # Matches "_24.txt_raw"
                r'_(\d+)_raw$',       # Matches "_24_raw" at end
                r'_(\d+)$'            # Matches "_24" at end
            ]
            
            # Look for a numeric part that could be a charge state
            for pattern in patterns:
                if re.search(pattern, filename_without_ext):
                    return True
            return False
        else:
            # For .txt files, use original method
            return (
                file_path.suffix == '.txt' and 
                file_path.name[0].isdigit()
            )

    def _extract_charge_state(self, file_path: Path) -> Optional[int]:
        """Extract charge state from filename."""
        if file_path.suffix.lower() == '.csv':
            # For CSV files from TWIMExtract, extract charge state from end of filename
            filename_without_ext = file_path.stem
            
            import re
            patterns = [
                r'range_(\d+)\.txt',  # Matches "range_24.txt"
                r'range_(\d+)_',      # Matches "range_24_"
                r'_(\d+)\.txt_raw',   # Matches "_24.txt_raw"
                r'_(\d+)_raw$',       # Matches "_24_raw" at end
                r'_(\d+)$'            # Matches "_24" at end
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename_without_ext)
                if match:
                    return int(match.group(1))
            return None
        else:
            # For .txt files, use original method (filename starts with charge state)
            try:
                return int(file_path.stem)
            except ValueError:
                return None

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

    def process_sample_folder(self, folder_name: str) -> (str, List[str], List[str]):
        processed_files = []
        failed_files = []
        output_folder = os.path.join(self.base_path, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        sample_folder_path = os.path.join(self.base_path, folder_name)
        if not os.path.exists(sample_folder_path):
            return output_folder, processed_files, failed_files

        for filename in os.listdir(sample_folder_path):
            file_path = Path(os.path.join(sample_folder_path, filename))
            
            # Use the same validation system as calibrate.py
            if not self._is_valid_data_file(file_path):
                continue
                
            try:
                # Extract charge state using the same method as calibrate.py
                charge_state = self._extract_charge_state(file_path)
                if charge_state is None:
                    failed_files.append(f"{filename} - Could not extract charge state from filename")
                    continue
                
                # Load data using the same method as calibrate.py
                drift_time, intensity = self._load_data_from_file(file_path)
                
                if len(drift_time) == 0 or len(intensity) == 0:
                    failed_files.append(f"{filename} - No valid data found")
                    continue
                
                # Apply drift time corrections
                if self.params.drift_mode == "Cyclic" and self.params.inject_time is not None:
                    drift_time = drift_time - self.params.inject_time
                drift_time = np.maximum(drift_time, 0)
                
                # Create dataframe
                index = np.arange(len(drift_time))
                df = pd.DataFrame({
                    "index": index,
                    "mass": self.params.sample_mass_map[folder_name],
                    "charge": charge_state,
                    "intensity": intensity,
                    "drift_time": drift_time
                })
                
                # Save as .dat file
                dat_filename = f"input_{charge_state}.dat"
                dat_path = os.path.join(output_folder, dat_filename)
                df.to_csv(dat_path, sep=' ', index=False, header=False)
                processed_files.append(filename)
                
            except Exception as e:
                failed_files.append(f"{filename} - {str(e)}")
        
        return output_folder, processed_files, failed_files

    def process_all(self, sample_folders: List[str]) -> InputProcessingResult:
        all_processed = {}
        all_failed = {}
        all_paths = []
        for sample in sample_folders:
            path, processed, failed = self.process_sample_folder(sample)
            all_paths.append(path)
            all_processed[sample] = processed
            all_failed[sample] = failed
        return InputProcessingResult(all_processed, all_failed, all_paths)

# --- File Generation ---
class OutputGenerator:
    @staticmethod
    def generate_zip(sample_folders_paths: List[str]) -> io.BytesIO:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for sample_folder in sample_folders_paths:
                for root, _, files in os.walk(sample_folder):
                    for file in files:
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, os.path.dirname(sample_folder))
                        zipf.write(full_path, arcname=relative_path)
        return zip_buffer

# --- UI Components ---
class UI:
    @staticmethod
    def show_main_header():
        st.markdown(
            '<div class="main-header">'
            '<h1>Get IMSCal Input Files</h1>'
            '<p>Generate input files for IMSCal calibration from your sample data</p>'
            '</div>',
            unsafe_allow_html=True
        )

    @staticmethod
    def show_info_card():
        st.markdown("""
        <div class="info-card">
            <p>Use this page to generate input files for IMSCal<sup>1</sup> calibration from your sample data. To use IMSCal, you need a reference file and an input file. The reference file is your calibrant information (if you haven't got this yet, go to 'Process Calibrant Data'), and the input file is your data to be calibrated.</p>
            <p>Just as for the calibration, make a folder for each sample and within that add files for each charge state. You can use either:</p>
            <ul>
                <li><strong>Text files (.txt):</strong> Create a text file for each charge state (called 'X.txt' where X is the charge state) and paste the corresponding ATD from MassLynx into each file. Remember to set the x-axis to ms not bins!</li>
                <li><strong>CSV files (.csv):</strong> From TWIMExtract<sup>2</sup> or similar tools - the charge state will be automatically extracted from filenames like "DT_name_range_24.txt_raw.csv" where 24 is the charge state. CSV files can contain comment lines starting with '#' which will be ignored.</li>
            </ul>
            <p>Save these files under their respective sample folder, zip the folders together, and upload below.</p>
            <p><strong>Note:</strong> This step is not doing any fitting! All it does is generate an input for IMSCal, which will then convert ATDs to CCSDs.</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def get_uploaded_zip():
        st.markdown('<h3 class="section-header">📁 Upload Sample Data</h3>', unsafe_allow_html=True)
        return st.file_uploader("Upload ZIP containing sample protein folders", type="zip")

    @staticmethod
    def get_instrument_settings():
        drift_mode = st.radio("Which instrument did you use?", options=["Cyclic", "Synapt"])
        inject_time = None
        if drift_mode == "Cyclic":
            inject_time = st.number_input("Enter inject time to subtract (ms)", min_value=0.0, value=12.0)
        return drift_mode, inject_time

    @staticmethod
    def get_sample_masses(sample_folders: List[str]) -> Dict[str, float]:
        st.write("Enter the molecular mass (Da) for each sample protein:")
        sample_mass_map = {}
        cols = st.columns(min(3, len(sample_folders)))
        for i, sample in enumerate(sample_folders):
            with cols[i % len(cols)]:
                mass = st.number_input(f"Mass (Da) for {sample}", min_value=0.0, key=sample)
                sample_mass_map[sample] = mass
        return sample_mass_map

    @staticmethod
    def show_processing_results(result: InputProcessingResult):
        total_processed = sum(len(files) for files in result.processed_files.values())
        total_failed = sum(len(files) for files in result.failed_files.values())
        if total_processed > 0:
            st.markdown(
                f"""
                <div class="success-card">
                    ✅ <strong>Processing Complete!</strong><br>
                    • Successfully processed: <strong>{total_processed}</strong> files
                    {f'<br>• Failed to process: <strong>{total_failed}</strong> files' if total_failed > 0 else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="error-card">No files were successfully processed. Please check your data format and try again.</div>', unsafe_allow_html=True)
        with st.expander("📊 Detailed Processing Results"):
            for sample in result.processed_files:
                st.write(f"**{sample}:**")
                if result.processed_files[sample]:
                    st.write(f"  ✅ Processed: {', '.join(result.processed_files[sample])}")
                if result.failed_files[sample]:
                    st.write(f"  ❌ Failed: {', '.join(result.failed_files[sample])}")

    @staticmethod
    def show_download_button(zip_buffer: io.BytesIO):
        st.markdown('<h3 class="section-header">📥 Download Results</h3>', unsafe_allow_html=True)
        st.download_button(
            label="📦 Download All .dat Files (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="sample_dat_files.zip",
            mime="application/zip"
        )

# --- Main App ---
def main():
    styling.load_custom_css()
    UI.show_main_header()
    UI.show_info_card()
    uploaded_zip_file = UI.get_uploaded_zip()
    
    # Clear cache button inside info card for consistent styling
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()
    
    if uploaded_zip_file is None:
        # Add references section when no file is uploaded
        st.markdown("""
        <div class="info-card">
            <h3>📚 References</h3>
            <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, <em> International Journal of Mass Spectrometry,</em>,2023, 492, 117112.</p>
            <p><sup>2</sup> S. E. Haynes, D. A. Polasky, S. M. Dixit, J. D. Majmudar, K. Neeson, B. T. Ruotolo and B. R. Martin, <em>Analytical Chemistry</em>, 2017, 89, 5669–5672.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Use myutils.import_tools to handle ZIP extraction
    try:
        sample_folders, temp_dir = import_tools.handle_zip_upload(uploaded_zip_file)
    except Exception as e:
        st.markdown(f'<div class="error-card">Error extracting ZIP: {str(e)}</div>', unsafe_allow_html=True)
        return

    if not sample_folders:
        st.markdown('<div class="error-card">No folders found in the ZIP file.</div>', unsafe_allow_html=True)
        return

    drift_mode, inject_time = UI.get_instrument_settings()
    sample_mass_map = UI.get_sample_masses(sample_folders)
    if any(mass == 0.0 for mass in sample_mass_map.values()):
        st.markdown('<div class="warning-card">Please provide masses for all samples.</div>', unsafe_allow_html=True)
        return

    params = InputParams(drift_mode=drift_mode, inject_time=inject_time, sample_mass_map=sample_mass_map)
    processor = InputProcessor(base_path=temp_dir, params=params)
    result = processor.process_all(sample_folders)
    UI.show_processing_results(result)
    if sum(len(files) for files in result.processed_files.values()) > 0:
        zip_buffer = OutputGenerator.generate_zip(result.sample_paths)
        UI.show_download_button(zip_buffer)
    
    # Add references section at the end
    st.markdown("""
    <div class="info-card">
        <h3>📚 References</h3>
        <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, <em> International Journal of Mass Spectrometry,</em>,2023, 492, 117112.</p>
        <p><sup>2</sup> S. E. Haynes, D. A. Polasky, S. M. Dixit, J. D. Majmudar, K. Neeson, B. T. Ruotolo and B. R. Martin, <em>Analytical Chemistry</em>, 2017, 89, 5669–5672.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()