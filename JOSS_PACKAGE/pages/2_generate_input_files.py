"""
Streamlit page for generating IMSCal input files from sample data.

This page allows users to:
1. Upload a ZIP file containing sample folders
2. Configure instrument settings (drift mode, injection time)
3. Specify sample masses
4. Process data to generate .dat files for IMSCal software
5. Download the results as a ZIP archive

Uses the imspartacus.extraction module for all core processing logic.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from myutils import styling, import_tools
from imspartacus.extraction import InputProcessor, InputParams
from imspartacus.io.writers import generate_zip_archive


class UI:
    """Streamlit UI components for input file generation."""
    
    @staticmethod
    def show_main_header():
        """Display the main page header."""
        st.markdown(
            '<h1 class="main-header">🔬 Generate IMSCal Input Files</h1>',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def show_info_card():
        """Display information about the page functionality."""
        st.markdown(
            """
            <div class="info-card">
                <h3>📋 How to Use This Tool</h3>
                <ol>
                    <li><strong>Upload Data:</strong> Upload a ZIP file containing folders for each sample</li>
                    <li><strong>Configure Settings:</strong> Select drift mode and injection time</li>
                    <li><strong>Enter Masses:</strong> Provide molecular mass for each sample</li>
                    <li><strong>Process:</strong> The tool will generate .dat files for IMSCal<sup>1</sup></li>
                    <li><strong>Download:</strong> Get all processed files as a ZIP archive</li>
                </ol>
                <p><strong>Expected Format:</strong> Each folder should contain arrival time distribution 
                files (e.g., "range_24.txt") with charge state in filename.<sup>2</sup></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def get_uploaded_zip():
        """Get the uploaded ZIP file from user."""
        st.markdown('<h3 class="section-header">📂 Upload Sample Data</h3>', unsafe_allow_html=True)
        return st.file_uploader(
            "Upload a ZIP file containing sample folders",
            type=["zip"],
            help="Each folder should contain data files with charge states (e.g., range_24.txt)"
        )
    
    @staticmethod
    def get_instrument_settings():
        """Get instrument configuration from user."""
        st.markdown('<h3 class="section-header">⚙️ Instrument Settings</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            drift_mode = st.selectbox(
                "Instrument Type",
                ["Synapt", "Cyclic"],
                help="Synapt: standard TWIMS. Cyclic: requires injection time correction"
            )
        
        with col2:
            inject_time = 0.0
            if drift_mode == "Cyclic":
                inject_time = st.number_input(
                    "Injection Time (ms)",
                    min_value=0.0,
                    value=0.5,
                    step=0.1,
                    help="Time subtracted from drift times for cyclic IMS"
                )
        
        return drift_mode, inject_time
    
    @staticmethod
    def get_sample_masses(sample_folders):
        """Get molecular masses for each sample."""
        st.markdown('<h3 class="section-header">⚖️ Sample Masses</h3>', unsafe_allow_html=True)
        st.write("Enter the molecular mass (Da) for each sample:")
        
        sample_mass_map = {}
        for sample in sample_folders:
            mass = st.number_input(
                f"{sample}",
                min_value=0.0,
                value=0.0,
                step=100.0,
                key=f"mass_{sample}",
                help=f"Molecular mass of {sample} in Daltons"
            )
            sample_mass_map[sample] = mass
        
        return sample_mass_map
    
    @staticmethod
    def show_processing_results(result):
        """Display processing results."""
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
            st.markdown(
                '<div class="error-card">No files were successfully processed. Please check your data format and try again.</div>',
                unsafe_allow_html=True
            )
        
        # Show detailed results in expander
        with st.expander("📊 Detailed Processing Results", expanded=True):
            for sample in result.processed_files:
                st.write(f"**{sample}:**")
                if result.processed_files[sample]:
                    st.write(f"  ✅ Processed: {', '.join(result.processed_files[sample])}")
                if result.failed_files[sample]:
                    st.write(f"  ❌ Failed:")
                    for failed in result.failed_files[sample]:
                        st.write(f"     - {failed}")
    
    @staticmethod
    def show_download_button(zip_buffer):
        """Display download button for results."""
        st.markdown('<h3 class="section-header">📥 Download Results</h3>', unsafe_allow_html=True)
        st.download_button(
            label="📦 Download All .dat Files (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="sample_dat_files.zip",
            mime="application/zip"
        )
    
    @staticmethod
    def show_references():
        """Display references section."""
        st.markdown(
            """
            <div class="info-card">
                <h3>📚 References</h3>
                <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, 
                <em>International Journal of Mass Spectrometry,</em> 2023, 492, 117112.</p>
                <p><sup>2</sup> S. E. Haynes, D. A. Polasky, S. M. Dixit, J. D. Majmudar, K. Neeson, 
                B. T. Ruotolo and B. R. Martin, <em>Analytical Chemistry</em>, 2017, 89, 5669–5672.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    """Main application logic."""
    # Load custom CSS
    styling.load_custom_css()
    
    # Show header and info
    UI.show_main_header()
    UI.show_info_card()
    
    # Get uploaded file
    uploaded_zip_file = UI.get_uploaded_zip()
    
    # Clear cache button
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()
    
    # If no file uploaded, show references and exit
    if uploaded_zip_file is None:
        UI.show_references()
        return
    
    # Extract ZIP file
    try:
        sample_folders, temp_dir = import_tools.handle_zip_upload(uploaded_zip_file)
    except Exception as e:
        st.markdown(
            f'<div class="error-card">Error extracting ZIP: {str(e)}</div>',
            unsafe_allow_html=True
        )
        return
    
    if not sample_folders:
        st.markdown(
            '<div class="error-card">No folders found in the ZIP file.</div>',
            unsafe_allow_html=True
        )
        return
    
    # Get instrument settings
    drift_mode, inject_time = UI.get_instrument_settings()
    
    # Get sample masses
    sample_mass_map = UI.get_sample_masses(sample_folders)
    
    # Validate that all masses are provided
    if any(mass == 0.0 for mass in sample_mass_map.values()):
        st.markdown(
            '<div class="warning-card">⚠️ Please provide masses for all samples.</div>',
            unsafe_allow_html=True
        )
        return
    
    # Create processing parameters
    params = InputParams(
        drift_mode=drift_mode,
        inject_time=inject_time,
        sample_mass_map=sample_mass_map
    )
    
    # Process all samples using imspartacus library
    processor = InputProcessor(base_path=temp_dir, params=params)
    result = processor.process_all(sample_folders)
    
    # Show results
    UI.show_processing_results(result)
    
    # If any files were processed, offer download
    total_processed = sum(len(files) for files in result.processed_files.values())
    if total_processed > 0:
        zip_buffer = generate_zip_archive(result.sample_paths)
        UI.show_download_button(zip_buffer)
    
    # Show references at the end
    UI.show_references()


if __name__ == "__main__":
    main()
