from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import streamlit as st
import zipfile
import tempfile
import os
from io import BytesIO, StringIO
from pathlib import Path

from myutils import styling, import_tools

# --- Data Classes ---
@dataclass
class ProteinOutput:
    protein_name: str
    dataframes: List[pd.DataFrame]

@dataclass
class OutputProcessingResult:
    protein_outputs: Dict[str, ProteinOutput]
    files_processed: int

# --- Processing Logic ---
class OutputProcessor:
    @staticmethod
    def extract_protein_data(zip_file: BytesIO) -> OutputProcessingResult:
        protein_outputs = {}
        files_processed = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.getvalue())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.startswith("output_") and file.endswith(".dat"):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, tmpdir)
                        parts = rel_path.split(os.sep)
                        if len(parts) < 2:
                            continue
                        protein_name = parts[0]
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        try:
                            cal_index = next(i for i, line in enumerate(lines) if line.strip() == "[CALIBRATED DATA]")
                            data_lines = lines[cal_index + 1:]
                        except StopIteration:
                            continue
                        try:
                            df = pd.read_csv(StringIO(''.join(data_lines)))
                            # Ensure Intensity is included
                            cols = ['Z', 'Drift', 'CCS', 'CCS Std.Dev.']
                            if 'Intensity' in df.columns:
                                cols.append('Intensity')
                            df = df[cols]
                            if protein_name not in protein_outputs:
                                protein_outputs[protein_name] = ProteinOutput(protein_name, [])
                            protein_outputs[protein_name].dataframes.append(df)
                            files_processed += 1
                        except Exception:
                            continue
        return OutputProcessingResult(protein_outputs, files_processed)

    @staticmethod
    def prepare_download(df: pd.DataFrame, protein_name: str) -> BytesIO:
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

# --- UI Components ---
class UI:
    @staticmethod
    def show_main_header():
        st.markdown("""
        <div class="main-header">
            <h1>Process Output Files</h1>
            <p>Get a CCS value for every timepoint for every charge state of every protein</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_info_card():
        st.markdown("""
        <div class="info-card">
            <p>Use this page to process the output files from IMSCal<sup>1</sup> and generate CCS values for every timepoint in the ATD for all charge states of your proteins. This step completes the calibration process by converting the calibrated drift times to collision cross-sections.</p>
            <p>This step is particularly useful when you have performed multiple experiments on the same protein (e.g., activated ion mobility experiments at different collision voltages). In such cases, you only need to calibrate once using IMSCal, then use this tool to process all your experimental conditions.</p>
            <p>To use this tool:</p>
            <ul>
                <li>Upload a ZIP file containing folders with your IMSCal output files</li>
                <li>Each folder should represent a different protein or experimental condition</li>
                <li>Files should be named <code>output_X.dat</code> where X is the charge state</li>
                <li>Files must contain a <code>[CALIBRATED DATA]</code> section from IMSCal</li>
            </ul>
            <p><strong>Note:</strong> This step does not perform any fitting - it simply extracts and organizes the calibrated data. After completing this step, proceed to 'Get Calibrated Scaled Data' to finish your analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_upload_section():
        st.markdown("""
        <div class="section-card">
            <div class="section-header">📁 Upload Your Data</div>
        </div>
        """, unsafe_allow_html=True)
        return st.file_uploader(
            "Upload a zipped folder containing your output files",
            type="zip",
            help="Select a ZIP file containing folders with output_X.dat files"
        )

    @staticmethod
    def show_processing_status(files_processed, protein_count):
        st.markdown(f"""
        <div class="success-card">
            <strong>✅ Processing Complete!</strong><br>
            Found data for <span class="metric-badge">{protein_count} proteins</span>
            from <span class="metric-badge">{files_processed} files</span>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_download_section():
        st.markdown("""
        <div class="section-card">
            <div class="section-header">📥 Download Processed Data</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_protein_card(protein_name, combined_df, dfs):
        st.markdown(f"""
        <div class="protein-card">
            <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">🧬 {protein_name}</h4>
            <p style="margin: 0; color: #64748b;">
                <span class="metric-badge">{len(combined_df)} data points</span>
                <span class="metric-badge">{len(dfs)} files combined</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_next_steps():
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #667eea; margin-top: 0;">📋 Next Steps</h4>
            <p>Your processed data is now ready for download. Each CSV file contains:</p>
            <ul>
                <li><strong>Z:</strong> Charge state</li>
                <li><strong>Drift:</strong> Drift time</li>
                <li><strong>CCS:</strong> Collision cross-section value</li>
                <li><strong>CCS Std.Dev.:</strong> Standard deviation</li>
                <li><strong>Intensity:</strong> Intensity value</li>
            </ul>
            <p><strong>Ready to continue?</strong> Go to 'Get Calibrated Scaled Data' to finish your analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_warning_card():
        st.markdown("""
        <div class="warning-card">
            <strong>⚠️ No Valid Data Found</strong><br>
            No valid output_X.dat files were found in the uploaded ZIP file.
            Please ensure your ZIP contains folders with properly formatted output files.
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_help_section():
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #667eea; margin-top: 0;">📖 Expected File Structure</h4>
            <p>Your ZIP file should contain:</p>
            <pre style="background: #f1f5f9; padding: 1rem; border-radius: 6px; font-size: 0.9rem;">
your_data.zip/
├── Protein1/
│   ├── output_1.dat
│   ├── output_2.dat
│   └── ...
├── Protein2/
│   ├── output_1.dat
│   └── ...
└── ...</pre>
            <p>Each output_X.dat file should contain a <code>[CALIBRATED DATA]</code> section.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Main App ---
def main():
    styling.load_custom_css()
    UI.show_main_header()
    UI.show_info_card()
    
    # Clear cache button inside info card for consistent styling
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()
    
    uploaded_zip = UI.show_upload_section()
    
    if not uploaded_zip:
        # Add references section when no file is uploaded
        st.markdown("""
        <div class="info-card">
            <h3>📚 References</h3>
            <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, <em> International Journal of Mass Spectrometry,</em>,2023, 492, 117112.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    result = OutputProcessor.extract_protein_data(uploaded_zip)
    if result.protein_outputs:
        UI.show_processing_status(result.files_processed, len(result.protein_outputs))
        UI.show_download_section()
        for protein_name, protein_output in result.protein_outputs.items():
            combined_df = pd.concat(protein_output.dataframes, ignore_index=True)
            UI.show_protein_card(protein_name, combined_df, protein_output.dataframes)
            buffer = OutputProcessor.prepare_download(combined_df, protein_name)
            st.download_button(
                label=f"📊 Download {protein_name}.csv",
                data=buffer,
                file_name=f"{protein_name}.csv",
                mime="text/csv",
                key=f"download_{protein_name}"
            )
        UI.show_next_steps()
    else:
        UI.show_warning_card()
        UI.show_help_section()
    
    # Add references section at the end
    st.markdown("""
    <div class="info-card">
        <h3>📚 References</h3>
        <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, <em> International Journal of Mass Spectrometry,</em>,2023, 492, 117112.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()