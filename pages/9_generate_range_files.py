from dataclasses import dataclass
from typing import List, Dict, Tuple
import streamlit as st
import os
import io
import zipfile
import tempfile

from myutils import import_tools
from myutils import styling

# --- Data Classes ---
@dataclass
class RangeParams:
    mass: float
    mz_range_size: float
    charge_range: Tuple[int, int]
    rt_start: float
    rt_end: float
    dt_start: int
    dt_end: int
    folder_name: str

@dataclass
class RangeFileResult:
    generated_files: List[str]
    charge_states: List[int]
    mz_values: Dict[int, float]

# --- Processing Logic ---
class RangeFileGenerator:
    def __init__(self, params: RangeParams):
        self.params = params

    def calculate_mz(self, charge: int) -> float:
        """Calculate m/z for given charge state using (mass + charge)/charge"""
        return (self.params.mass + charge) / charge

    def generate_range_content(self, charge: int) -> str:
        """Generate the content for a single range file"""
        mz = self.calculate_mz(charge)
        half_range = self.params.mz_range_size / 2.0
        
        mz_start = mz - half_range
        mz_end = mz + half_range
        
        content = f"""MZ_start: {mz_start:.1f}
MZ_end: {mz_end:.1f}
RT_start_(minutes): {self.params.rt_start:.1f}
RT_end_(minutes): {self.params.rt_end:.1f}
DT_start_(bins): {self.params.dt_start}
DT_end_(bins): {self.params.dt_end}"""
        
        return content

    def generate_all_files(self, output_dir: str) -> RangeFileResult:
        """Generate all range files for the specified charge range"""
        generated_files = []
        charge_states = []
        mz_values = {}
        
        min_charge, max_charge = self.params.charge_range
        
        for charge in range(min_charge, max_charge + 1):
            filename = f"range_{charge}.txt"
            filepath = os.path.join(output_dir, filename)
            
            content = self.generate_range_content(charge)
            mz = self.calculate_mz(charge)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            generated_files.append(filename)
            charge_states.append(charge)
            mz_values[charge] = mz
        
        return RangeFileResult(generated_files, charge_states, mz_values)

# --- File Generation ---
class OutputGenerator:
    @staticmethod
    def generate_zip(output_dir: str, result: RangeFileResult, folder_name: str) -> io.BytesIO:
        """Generate ZIP file containing all range files"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for filename in result.generated_files:
                filepath = os.path.join(output_dir, filename)
                # Add files to a folder inside the ZIP
                zipf.write(filepath, os.path.join(folder_name, filename))
        
        zip_buffer.seek(0)
        return zip_buffer

# --- UI Components ---
class UI:
    @staticmethod
    def show_main_header():
        st.markdown(
            '<div class="main-header">'
            '<h1>Generate TWIMExtract Range Files</h1>'
            '<p>Generate range files for TWIMExtract based on protein mass and charge states</p>'
            '</div>',
            unsafe_allow_html=True
        )

    @staticmethod
    def show_info_card():
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
        """Get protein mass, m/z range size, charge range, and folder name from user"""
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
            return mass, mz_range_size, (min_charge, min_charge), folder_name
        
        # Validate folder name
        if not folder_name.strip():
            st.error("Please enter a folder name")
            folder_name = "protein_ranges"
        
        charge_range = (min_charge, max_charge)
        
        # Show preview of what will be generated
        st.markdown("**Preview of charge states and m/z values:**")
        preview_data = []
        for charge in range(min_charge, min(max_charge + 1, min_charge + 5)):  # Show first 5
            mz = (mass + charge) / charge
            half_range = mz_range_size / 2.0
            preview_data.append({
                "Charge": f"{charge}+",
                "m/z": f"{mz:.2f}",
                "Range": f"{mz - half_range:.1f} - {mz + half_range:.1f}"
            })
        
        if max_charge - min_charge > 4:
            preview_data.append({
                "Charge": "...",
                "m/z": "...",
                "Range": f"(+{max_charge - min_charge - 4} more)"
            })
        
        st.table(preview_data)
        
        return mass, mz_range_size, charge_range, folder_name.strip()

    @staticmethod
    def get_experimental_parameters() -> Tuple[float, float, int, int]:
        """Get retention time and drift time parameters"""
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
        
        return rt_start, rt_end, dt_start, dt_end

    @staticmethod
    def show_generation_results(result: RangeFileResult, params: RangeParams):
        """Display the results of range file generation"""
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
    def show_download_button(zip_buffer: io.BytesIO, params: RangeParams):
        """Show download button for the generated range files"""
        st.markdown('<h3 class="section-header">📥 Download Range Files</h3>', unsafe_allow_html=True)
        
        filename = f"{params.folder_name}_range_files.zip"
        
        st.download_button(
            label="📦 Download Range Files (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=filename,
            mime="application/zip",
            help="Download ZIP file containing all generated range files"
        )
        
        st.info("💡 **Next steps:** Extract the ZIP file and use the range files in TWIMExtract for automated data extraction.")

# --- Main App ---
def main():
    styling.load_custom_css()
    UI.show_main_header()
    UI.show_info_card()
    
    # Clear cache button
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()
    
    # Get protein parameters
    mass, mz_range_size, charge_range, folder_name = UI.get_protein_parameters()
    
    if charge_range[0] > charge_range[1]:
        st.stop()
    
    # Get experimental parameters
    rt_start, rt_end, dt_start, dt_end = UI.get_experimental_parameters()
    
    if rt_start >= rt_end or dt_start >= dt_end:
        st.warning("Please check your experimental parameter ranges.")
        st.stop()
    
    # Create parameters object
    params = RangeParams(
        mass=mass,
        mz_range_size=mz_range_size,
        charge_range=charge_range,
        rt_start=rt_start,
        rt_end=rt_end,
        dt_start=dt_start,
        dt_end=dt_end,
        folder_name=folder_name
    )
    
    # Generate button
    if st.button("🚀 Generate Range Files", type="primary"):
        with st.spinner("Generating range files..."):
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate range files
                generator = RangeFileGenerator(params)
                result = generator.generate_all_files(temp_dir)
                
                # Show results
                UI.show_generation_results(result, params)
                
                # Generate ZIP
                zip_buffer = OutputGenerator.generate_zip(temp_dir, result, folder_name)
                
                # Show download button
                UI.show_download_button(zip_buffer, params)
    
    # Add references section
    st.markdown("""
    <div class="info-card">
        <h3>📚 References</h3>
        <p><sup>1</sup> S. E. Haynes, D. A. Polasky, J. D. Majmudar, S. M. Dixit, B. T. Ruotolo and B. R. Martin, <em>Analytical Chemistry</em>, 2017, 89, 5669–5672.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
