"""
ESI Prot - Protein Charge State Deconvolution - Refactored
ESIprot 1.1 - License: GPLv3 - Robert Winkler, 2009-2017
"""

import sys
from pathlib import Path

# Add parent directory to path to import myutils
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from typing import List

# Import from imspartacus package
from imspartacus.processing import (
    ESIProtCalculator,
    ESIProtDataExporter,
    DeconvolutionResult
)

# Import Streamlit UI helpers
from myutils import styling

# Page configuration
st.set_page_config(page_title="ESI Prot", page_icon="🧬", layout="wide")

# Apply custom styling
styling.load_custom_css()


class ESIProtInterface:
    """Streamlit interface for ESIProt calculations."""
    
    @staticmethod
    def show_header():
        """Display page header."""
        st.markdown("""
        <div class="main-header">
            <h1>🧬 ESI Prot - Protein Charge State Deconvolution</h1>
            <p>ESIprot 1.1 - License: GPLv3 - Robert Winkler, 2009-2017</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <strong>ℹ️ About:</strong> This is a web implementation of ESI Prot by Robert Winkler. 
            It deconvolutes protein charge states from electrospray ionization mass spectrometry data 
            and calculates m/z values for given masses using the original algorithm.
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state for m/z values."""
        if 'mz_values' not in st.session_state:
            st.session_state.mz_values = [1091.7, 1182.5, 1290.0, 1418.8, 1576.3, 1773.3, 0.0, 0.0, 0.0]
    
    @staticmethod
    def get_mz_inputs() -> List[float]:
        """Get m/z input values from user.
        
        Returns:
            List of 9 m/z values (floats)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📊 Input Parameters</h3>', unsafe_allow_html=True)
        
        st.markdown("**Peaks from spectrum**")
        
        # Create input fields for m/z values
        mz_inputs = []
        for i in range(9):
            default_value = float(st.session_state.mz_values[i])
            
            mz_val = st.number_input(
                f"m/z ({i+1}):",
                min_value=0.0,
                max_value=None,
                value=default_value,
                step=0.1,
                format="%.4f",
                key=f"mz_{i+1}",
                help=f"Enter the m/z value for peak {i+1} (use 0 to skip)"
            )
            mz_inputs.append(float(mz_val))
        
        st.markdown('</div>', unsafe_allow_html=True)
        return mz_inputs
    
    @staticmethod
    def show_action_buttons() -> tuple:
        """Show action buttons for deconvolution.
        
        Returns:
            Tuple of (calculate_clicked, clear_clicked)
        """
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            calculate_clicked = st.button("🧮 Calculate MW", type="primary")
        
        with col_btn2:
            clear_clicked = st.button("🗑️ Clear m/z values")
        
        return calculate_clicked, clear_clicked
    
    @staticmethod
    def show_deconvolution_results(result: DeconvolutionResult, mz_inputs: List[float]):
        """Display deconvolution results.
        
        Args:
            result: DeconvolutionResult object
            mz_inputs: Original m/z input values
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📈 Results</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-card">
            <strong>✅ Calculation completed successfully!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Create results table
        display_data = []
        for i in range(9):
            mz_val = mz_inputs[i]
            if mz_val > 0:
                # Find this m/z in results
                found_idx = -1
                for j, result_mz in enumerate(result.mz_values):
                    if abs(result_mz - mz_val) < 0.0001:
                        found_idx = j
                        break
                
                if found_idx >= 0:
                    display_data.append({
                        'm/z': f"{mz_val:.4f}",
                        'Charge (+)': result.charge_states[found_idx],
                        'MW [Da]': f"{result.molecular_weights[found_idx]:.2f}",
                        'Error [Da]': f"{result.errors[found_idx]:.4f}"
                    })
        
        if display_data:
            results_df = pd.DataFrame(display_data)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Final results
        st.markdown('<h3 class="section-header">🎯 Final Results</h3>', unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"""
            <div class="protein-card">
                <strong>Deconvoluted MW [Da]:</strong><br>
                <span class="metric-badge">{result.average_mw:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            st.markdown(f"""
            <div class="protein-card">
                <strong>Standard deviation [Da]:</strong><br>
                <span class="metric-badge">{result.stdev:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Download button
        download_data = ESIProtDataExporter.to_dict_list(result, mz_inputs)
        download_df = pd.DataFrame(download_data)
        csv = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="esiprot_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Detailed output
        with st.expander("📋 Detailed Calculation Results (ESIProt Format)"):
            st.markdown("**FINAL RESULTS**")
            st.markdown("*" * 79)
            detail_text = ESIProtDataExporter.to_esiprot_format(result)
            st.text(detail_text)
            
            st.markdown("**CSV Download Preview:**")
            st.dataframe(download_df, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def show_deconvolution_placeholder():
        """Show placeholder when no calculation performed."""
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📈 Results</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <strong>📊 Ready for calculation</strong><br>
            Enter your m/z values and click "Calculate MW" to perform deconvolution.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def get_mass_calculation_inputs() -> tuple:
        """Get inputs for m/z calculation from known mass.
        
        Returns:
            Tuple of (molecular_weight, charge_min, charge_max)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚖️ Mass to m/z Calculation</h3>', unsafe_allow_html=True)
        
        st.markdown("**Known molecular weight**")
        
        molecular_weight = st.number_input(
            "Molecular Weight [Da]:",
            min_value=0.0,
            value=15000.0,
            step=1.0,
            format="%.2f",
            help="Enter the known molecular weight of your protein"
        )
        
        st.markdown("**Charge state range**")
        
        col_charge1, col_charge2 = st.columns(2)
        with col_charge1:
            charge_min = st.number_input(
                "Min charge (+):",
                min_value=1,
                value=5,
                step=1,
                help="Minimum charge state to calculate"
            )
        
        with col_charge2:
            charge_max = st.number_input(
                "Max charge (+):",
                min_value=1,
                value=25,
                step=1,
                help="Maximum charge state to calculate"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        return molecular_weight, charge_min, charge_max
    
    @staticmethod
    def show_mz_calculation_results(calculations, molecular_weight, charge_min, charge_max):
        """Display m/z calculation results.
        
        Args:
            calculations: List of MZCalculation objects
            molecular_weight: Input molecular weight
            charge_min: Minimum charge state
            charge_max: Maximum charge state
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🎯 Calculated m/z Values</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-card">
            <strong>✅ Calculation completed successfully!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Create calculation results table
        calc_data = ESIProtDataExporter.mz_calculations_to_dict_list(calculations)
        calc_df = pd.DataFrame(calc_data)
        st.dataframe(calc_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = calc_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Calculations as CSV",
            data=csv,
            file_name="calculated_mz_values.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Summary metrics
        st.markdown('<h3 class="section-header">📊 Summary</h3>', unsafe_allow_html=True)
        
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.markdown(f"""
            <div class="protein-card">
                <strong>Input Mass [Da]:</strong><br>
                <span class="metric-badge">{molecular_weight:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_sum2:
            st.markdown(f"""
            <div class="protein-card">
                <strong>Charge States:</strong><br>
                <span class="metric-badge">{charge_min} - {charge_max}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def show_mz_calculation_placeholder():
        """Show placeholder for m/z calculation tab."""
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🎯 Calculated m/z Values</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <strong>🧮 Ready for calculation</strong><br>
            Enter a molecular weight and charge range to calculate m/z values.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def show_help_section():
        """Display help and usage information."""
        with st.expander("ℹ️ Help & Usage - Official ESIProt Documentation"):
            st.markdown("""
            **ESI Prot** - Web implementation of the original ESIProt tool by Robert Winkler.
            
            ## 🔬 Deconvolution Mode
            **Purpose:** Calculate molecular weight from observed m/z values using the exact ESIProt algorithm
            
            **Algorithm Details:**
            - Tests consecutive decreasing charge states (e.g., 10+, 9+, 8+, 7+...)
            - Uses precise hydrogen mass: 1.00794 - 0.0005485799 Da (IUPAC 2005)
            - Formula: `MW = (m/z × charge) - (charge × H)`
            - Finds charge assignment with minimum standard deviation
            
            **How to use:**
            1. Enter m/z values for up to 9 peaks (use 0 to skip)
            2. Click "Calculate MW" for automatic deconvolution
            3. Review results showing charge states, masses, and errors
            4. Download results as CSV

            ## 📊 m/z Calculation Mode
            **Purpose:** Calculate m/z values for a known molecular weight

            **Formula:** `m/z = (MW + (charge × H)) / charge`
            
            **How to use:**
            1. Enter known molecular weight
            2. Set charge range to calculate
            3. Click "Calculate m/z values"
            4. Download calculations as CSV

            ## Reference
            **Original ESIProt:** Robert Winkler, 2009-2017, GPLv3 License
            
            **Atomic weights:** IUPAC Technical Report (2006)  
            M. E. Wieser: Pure Appl. Chem., Vol. 78, No. 11. (2006), pp. 2051-2066.
            
            **Notes:**
            - This implementation uses the exact same algorithm as the original ESIProt
            - Results should be identical to the desktop version
            - Minimum 2 m/z values required for deconvolution
            """)


def main():
    """Main application function."""
    # Initialize session state
    ESIProtInterface.initialize_session_state()
    
    # Show header
    ESIProtInterface.show_header()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔬 Deconvolution", "📊 m/z Calculation"])
    
    # TAB 1: Deconvolution
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Get m/z inputs
            mz_inputs = ESIProtInterface.get_mz_inputs()
            
            # Action buttons
            calculate_clicked, clear_clicked = ESIProtInterface.show_action_buttons()
            
            # Handle clear button
            if clear_clicked:
                st.session_state.mz_values = [0.0] * 9
                for i in range(9):
                    if f"mz_{i+1}" in st.session_state:
                        del st.session_state[f"mz_{i+1}"]
                st.rerun()
            
            # Handle calculate button
            if calculate_clicked:
                st.session_state.calculate = True
        
        with col2:
            # Show results or placeholder
            if hasattr(st.session_state, 'calculate') and st.session_state.calculate:
                result, error_msg = ESIProtCalculator.deconvolute(mz_inputs)
                
                if error_msg:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<h3 class="section-header">📈 Results</h3>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="error-card">
                        <strong>❌ Error:</strong> {error_msg}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                elif result:
                    ESIProtInterface.show_deconvolution_results(result, mz_inputs)
                
                st.session_state.calculate = False
            else:
                ESIProtInterface.show_deconvolution_placeholder()
    
    # TAB 2: m/z Calculation
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Get calculation inputs
            molecular_weight, charge_min, charge_max = ESIProtInterface.get_mass_calculation_inputs()
            
            # Calculate button
            if st.button("🧮 Calculate m/z values", type="primary"):
                st.session_state.calculate_mz = True
        
        with col2:
            # Show results or placeholder
            if hasattr(st.session_state, 'calculate_mz') and st.session_state.calculate_mz:
                if charge_max < charge_min:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<h3 class="section-header">🎯 Calculated m/z Values</h3>', unsafe_allow_html=True)
                    st.markdown("""
                    <div class="error-card">
                        <strong>❌ Error:</strong> Maximum charge must be greater than or equal to minimum charge.
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    calculations = ESIProtCalculator.calculate_mz_from_mass(
                        molecular_weight, charge_min, charge_max
                    )
                    ESIProtInterface.show_mz_calculation_results(
                        calculations, molecular_weight, charge_min, charge_max
                    )
                
                st.session_state.calculate_mz = False
            else:
                ESIProtInterface.show_mz_calculation_placeholder()
    
    # Section divider
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Help section
    ESIProtInterface.show_help_section()


if __name__ == "__main__":
    main()
