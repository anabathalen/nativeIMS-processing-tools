"""
Generate TWIMExtract Range Files - Refactored
Generate range files for TWIMExtract based on protein mass and charge states.
"""

import sys
from pathlib import Path

# Add parent directory to path to import myutils
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import tempfile
from typing import Tuple

# Import from imspartacus package
from imspartacus.io import (
    RangeParameters,
    RangeFileGenerator,
    RangeFilePackager
)

# Import Streamlit UI helpers
from myutils import styling, import_tools

# Apply custom styling
styling.load_custom_css()


class RangeFileInterface:
    """Streamlit interface for range file generation."""
    
    @staticmethod
    def show_header():
        """Display page header."""
        st.markdown(
            '<div class="main-header">'
            '<h1>Generate TWIMExtract Range Files</h1>'
            '<p>Generate range files for TWIMExtract based on protein mass and charge states</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div class="info-card">
            <p>Use this page to generate range files for TWIMExtract<sup>1</sup> analysis. Range files define the m/z, retention time, and drift time windows for data extraction.</p>
            <p><strong>How it works:</strong></p>
            <ul>
                <li>Enter your protein mass (accurate average mass) and desired charge state range</li>
                <li>Specify the m/z window size (total range around the calculated m/z)</li>
                <li>Set retention time and drift time ranges for your experiment (if you used direct injection, typically this will be the whole range)</li>
                <li>Choose a name for your output folder</li>
                <li>The tool calculates m/z values using: <strong>m/z = (mass + charge) / charge</strong></li>
                <li>Each range file covers: <strong>calculated_mz ± (range_size/2)</strong></li>
            </ul>
            <p>The generated files can be used directly in TWIMExtract for automated data extraction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def get_protein_parameters() -> Tuple[float, float, Tuple[int, int], str]:
        """Get protein mass, m/z range size, charge range, and folder name from user.
        
        Returns:
            Tuple of (mass, mz_range_size, charge_range, folder_name)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🧬 Protein Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            mass = st.number_input(
                "Protein Mass (Da)", 
                min_value=100.0, 
                value=15000.0, 
                step=100.0,
                help="Enter the molecular mass of your protein in Daltons"
            )
            
            mz_range_size = st.number_input(
                "m/z Range Size", 
                min_value=1.0, 
                value=50.0, 
                step=1.0,
                help="Total m/z window size (will be split equally around calculated m/z)"
            )
            
            folder_name = st.text_input(
                "Output Folder Name", 
                value="protein_ranges",
                help="Name for the folder containing the range files"
            )
        
        with col2:
            min_charge = st.number_input(
                "Minimum Charge State", 
                min_value=1, 
                value=10, 
                step=1,
                help="Lowest charge state to generate"
            )
            
            max_charge = st.number_input(
                "Maximum Charge State", 
                min_value=1, 
                value=20, 
                step=1,
                help="Highest charge state to generate"
            )
        
        # Validate charge range
        if min_charge > max_charge:
            st.error("Minimum charge state must be less than or equal to maximum charge state")
            st.markdown('</div>', unsafe_allow_html=True)
            return mass, mz_range_size, (min_charge, min_charge), folder_name
        
        # Validate folder name
        if not folder_name.strip():
            st.error("Please enter a folder name")
            folder_name = "protein_ranges"
        
        charge_range = (min_charge, max_charge)
        
        # Show preview using the generator
        st.markdown("**Preview of charge states and m/z values:**")
        temp_params = RangeParameters(
            mass=mass,
            mz_range_size=mz_range_size,
            charge_range=charge_range,
            rt_start=0.0,
            rt_end=100.0,
            dt_start=1,
            dt_end=200,
            folder_name=folder_name.strip()
        )
        temp_generator = RangeFileGenerator(temp_params)
        preview_data = temp_generator.generate_preview_data()
        st.table(preview_data)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return mass, mz_range_size, charge_range, folder_name.strip()
    
    @staticmethod
    def get_experimental_parameters() -> Tuple[float, float, int, int]:
        """Get retention time and drift time parameters.
        
        Returns:
            Tuple of (rt_start, rt_end, dt_start, dt_end)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚙️ Experimental Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Retention Time Range**")
            rt_start = st.number_input(
                "RT Start (minutes)", 
                min_value=0.0, 
                value=0.0, 
                step=0.1,
                help="Start of retention time window"
            )
            
            rt_end = st.number_input(
                "RT End (minutes)", 
                min_value=0.1, 
                value=100.0, 
                step=0.1,
                help="End of retention time window"
            )
        
        with col2:
            st.write("**Drift Time Range**")
            dt_start = st.number_input(
                "DT Start (bins)", 
                min_value=1, 
                value=1, 
                step=1,
                help="Start of drift time window in bins"
            )
            
            dt_end = st.number_input(
                "DT End (bins)", 
                min_value=2, 
                value=200, 
                step=1,
                help="End of drift time window in bins"
            )
        
        # Validate ranges
        if rt_start >= rt_end:
            st.error("RT Start must be less than RT End")
        
        if dt_start >= dt_end:
            st.error("DT Start must be less than DT End")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return rt_start, rt_end, dt_start, dt_end
    
    @staticmethod
    def show_generation_results(result, params: RangeParameters):
        """Display the results of range file generation.
        
        Args:
            result: RangeFileResult object
            params: RangeParameters object
        """
        st.markdown(
            f"""
            <div class="success-card">
                ✅ <strong>Generation Complete!</strong><br>
                • Generated <strong>{len(result.generated_files)}</strong> range files<br>
                • Charge states: <strong>{params.charge_range[0]}+ to {params.charge_range[1]}+</strong><br>
                • m/z range: <strong>{params.mz_range_size} Da</strong> window per charge state<br>
                • Output folder: <strong>{params.folder_name}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show detailed results
        with st.expander("📊 Detailed Range File Information"):
            for charge in result.charge_states:
                mz = result.mz_values[charge]
                half_range = params.mz_range_size / 2.0
                st.write(f"**{params.folder_name}/range_{charge}.txt:**")
                st.write(f"  • Charge: {charge}+")
                st.write(f"  • Calculated m/z: {mz:.3f}")
                st.write(f"  • m/z range: {mz - half_range:.1f} - {mz + half_range:.1f}")
    
    @staticmethod
    def show_download_section(zip_buffer, params: RangeParameters):
        """Show download button for the generated range files.
        
        Args:
            zip_buffer: BytesIO buffer containing ZIP file
            params: RangeParameters object
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📥 Download Range Files</h3>', unsafe_allow_html=True)
        
        filename = RangeFilePackager.get_zip_filename(params.folder_name)
        
        st.download_button(
            label="📦 Download Range Files (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=filename,
            mime="application/zip",
            help="Download ZIP file containing all generated range files"
        )
        
        st.info("💡 **Next steps:** Extract the ZIP file and use the range files in TWIMExtract for automated data extraction.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def show_references():
        """Display references section."""
        st.markdown("""
        <div class="info-card">
            <h3>📚 References</h3>
            <p><sup>1</sup> S. E. Haynes, D. A. Polasky, J. D. Majmudar, S. M. Dixit, B. T. Ruotolo and B. R. Martin, <em>Analytical Chemistry</em>, 2017, 89, 5669–5672.</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Show header
    RangeFileInterface.show_header()
    
    # Clear cache button
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()
    
    # Step 1: Get protein parameters
    mass, mz_range_size, charge_range, folder_name = RangeFileInterface.get_protein_parameters()
    
    if charge_range[0] > charge_range[1]:
        st.stop()
    
    # Step 2: Get experimental parameters
    rt_start, rt_end, dt_start, dt_end = RangeFileInterface.get_experimental_parameters()
    
    if rt_start >= rt_end or dt_start >= dt_end:
        st.warning("Please check your experimental parameter ranges.")
        st.stop()
    
    # Create parameters object
    params = RangeParameters(
        mass=mass,
        mz_range_size=mz_range_size,
        charge_range=charge_range,
        rt_start=rt_start,
        rt_end=rt_end,
        dt_start=dt_start,
        dt_end=dt_end,
        folder_name=folder_name
    )
    
    # Step 3: Generate button
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">🚀 Generate Files</h3>', unsafe_allow_html=True)
    
    if st.button("🚀 Generate Range Files", type="primary"):
        with st.spinner("Generating range files..."):
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate range files
                generator = RangeFileGenerator(params)
                result = generator.generate_all_files(temp_dir)
                
                # Show results
                RangeFileInterface.show_generation_results(result, params)
                
                # Generate ZIP
                zip_buffer = RangeFilePackager.create_zip(temp_dir, result, folder_name)
                
                # Show download button
                RangeFileInterface.show_download_section(zip_buffer, params)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add references section
    RangeFileInterface.show_references()


if __name__ == "__main__":
    main()
