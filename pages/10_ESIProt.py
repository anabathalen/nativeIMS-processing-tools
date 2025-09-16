import streamlit as st
import numpy as np
import pandas as pd
from math import sqrt, pow
import sys
import os
from io import StringIO

# Add the parent directory to the path to import myutils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myutils.styling import load_custom_css

st.set_page_config(page_title="ESI Prot", page_icon="🧬", layout="wide")

# Load custom CSS
load_custom_css()

def calc_mw(mz_values):
    """Calculate molecular weight from ESI charge state data - EXACT ESIProt algorithm"""
    
    # Filter out zero values but keep track of original positions
    mz = []
    original_indices = []
    for i, mz_val in enumerate(mz_values):
        if mz_val > 0:
            mz.append(mz_val)
            original_indices.append(i)
    
    n_peaks = len(mz)
    
    if n_peaks < 2:
        return None, "Please enter at least 2 m/z values for calculation."
    
    # EXACT hydrogen mass from original ESIProt
    hydrogen = 1.00794 - 0.0005485799
    
    # Initialize variables exactly as in original
    stdev_champion = 1000
    average_champion = 1000000
    charge_champion = [0] * 9
    mw_champion = [0] * 9
    error_champion = [0] * 9
    
    # Charge range - using original ESIProt logic
    charge_min = 1
    charge_max = 100
    
    for chargecount_1 in range(charge_min, charge_max + 1):
        # Initialize arrays for this iteration
        mw = [0] * 9
        error = [0] * 9
        charge = [0] * 9
        
        # Set initial charge for first peak
        charge[0] = chargecount_1
        
        # Set consecutive decreasing charges for subsequent peaks
        for i in range(1, 9):
            charge[i] = charge[i-1] - 1
        
        # Calculate molecular weights and average
        sum_mw = 0
        nulls = 0
        
        for i in range(9):
            if i < len(mz):
                # Calculate MW using exact ESIProt formula
                mw[i] = (mz[i] * charge[i]) - (charge[i] * hydrogen)
                sum_mw += mw[i]
            else:
                mw[i] = 0
                nulls += 1
        
        notnulls = 9 - nulls
        if notnulls == 0:
            continue
            
        average = sum_mw / notnulls
        
        # Calculate errors and standard deviation exactly as in original
        errorsquaresum = 0
        for i in range(notnulls):
            error[i] = mw[i] - average
            errorsquare = pow(error[i], 2)
            errorsquaresum += errorsquare
        
        # Calculate standard deviation exactly as in original
        if notnulls > 1:
            stdev = sqrt(errorsquaresum * pow((notnulls - 1), -1))
        else:
            stdev = 0
        
        # Check if this is the best result (lowest standard deviation)
        if stdev < stdev_champion:
            stdev_champion = stdev
            average_champion = average
            
            # Fill in the values for the best approximation
            for z in range(9):
                charge_champion[z] = charge[z]
                mw_champion[z] = mw[z]
                if mw_champion[z] == 0:
                    charge_champion[z] = 0
                error_champion[z] = error[z]
    
    # Prepare results in the format expected by the UI
    results = {
        'mz': mz[:n_peaks],
        'charges': charge_champion[:n_peaks],
        'mw': mw_champion[:n_peaks],
        'errors': error_champion[:n_peaks],
        'average': average_champion,
        'stdev': stdev_champion,
        'original_indices': original_indices
    }
    
    return results, None

def calc_mz_values(mass, charge_min, charge_max):
    """Calculate m/z values for a given mass and charge range"""
    # EXACT hydrogen mass from original ESIProt
    hydrogen = 1.00794 - 0.0005485799
    
    calculations = []
    for charge in range(charge_min, charge_max + 1):
        # Use exact reverse formula: m/z = (MW + (charge * H)) / charge
        mz = (mass + (charge * hydrogen)) / charge
        calculations.append({
            'charge': charge,
            'mz': mz
        })
    
    return calculations

def create_download_data(results, mz_inputs):
    """Create CSV data for download"""
    if not results:
        return None
    
    # Prepare data for CSV
    csv_data = []
    
    # Add individual charge state results using original indices
    for i, orig_idx in enumerate(results['original_indices']):
        if i < len(results['mz']):
            csv_data.append({
                'm/z': f"{results['mz'][i]:.4f}",
                'charge (z)': results['charges'][i],
                'mass (da)': f"{results['mw'][i]:.2f}",
                'error (da)': f"{results['errors'][i]:.4f}"
            })
    
    # Add final deconvoluted mass row
    csv_data.append({
        'm/z': "Final Result",
        'charge (z)': "N/A", 
        'mass (da)': f"{results['average']:.2f}",
        'error (da)': f"{results['stdev']:.4f}"
    })
    
    return pd.DataFrame(csv_data)

def create_calculation_download_data(calculations):
    """Create CSV data for calculated m/z values"""
    csv_data = []
    
    for calc in calculations:
        csv_data.append({
            'charge (z)': calc['charge'],
            'm/z': f"{calc['mz']:.4f}"
        })
    
    return pd.DataFrame(csv_data)

def main():
    # Initialize session state for m/z values with proper float type
    if 'mz_values' not in st.session_state:
        st.session_state.mz_values = [1091.7, 1182.5, 1290.0, 1418.8, 1576.3, 1773.3, 0.0, 0.0, 0.0]
    
    # Main header with custom styling
    st.markdown("""
    <div class="main-header">
        <h1>🧬 ESI Prot - Protein Charge State Deconvolution</h1>
        <p>ESIprot 1.1 - License: GPLv3 - Robert Winkler, 2009-2017</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Information card
    st.markdown("""
    <div class="info-card">
        <strong>ℹ️ About:</strong> This is a web implementation of ESI Prot by Robert Winkler. 
        It deconvolutes protein charge states from electrospray ionization mass spectrometry data 
        and calculates m/z values for given masses using the original algorithm.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🔬 Deconvolution", "📊 m/z Calculation"])
    
    with tab1:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">📊 Input Parameters</div>', unsafe_allow_html=True)
            
            # Form section for inputs
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown("**Peaks from spectrum**")
            
            # Create input fields for m/z values
            mz_inputs = []
            for i in range(9):
                # Ensure all values are floats
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
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🧮 Calculate MW", type="primary"):
                    st.session_state.calculate = True
            
            with col_btn2:
                if st.button("🗑️ Clear m/z values"):
                    # Clear all m/z values properly
                    st.session_state.mz_values = [0.0] * 9
                    # Also clear the individual session state keys
                    for i in range(9):
                        if f"mz_{i+1}" in st.session_state:
                            del st.session_state[f"mz_{i+1}"]
                    st.rerun()
        
        with col2:
            st.markdown('<div class="section-header">📈 Results</div>', unsafe_allow_html=True)
            
            # Perform calculation if button was pressed
            if hasattr(st.session_state, 'calculate') and st.session_state.calculate:
                results, error_msg = calc_mw(mz_inputs)
                
                if error_msg:
                    st.markdown(f"""
                    <div class="error-card">
                        <strong>❌ Error:</strong> {error_msg}
                    </div>
                    """, unsafe_allow_html=True)
                
                elif results:
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Calculation completed successfully!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create results table
                    display_data = []
                    
                    # Fill in results for all input m/z values
                    for i in range(9):
                        mz_val = mz_inputs[i]
                        if mz_val > 0:
                            # Find this m/z in results
                            found_idx = -1
                            for j, result_mz in enumerate(results['mz']):
                                if abs(result_mz - mz_val) < 0.0001:
                                    found_idx = j
                                    break
                        
                            if found_idx >= 0:
                                display_data.append({
                                    'm/z': f"{mz_val:.4f}",
                                    'Charge (+)': results['charges'][found_idx],
                                    'MW [Da]': f"{results['mw'][found_idx]:.2f}",
                                    'Error [Da]': f"{results['errors'][found_idx]:.4f}"
                                })
                    
                    if display_data:
                        results_df = pd.DataFrame(display_data)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Final results in metric badges
                    st.markdown('<div class="section-header">🎯 Final Results</div>', unsafe_allow_html=True)
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown(f"""
                        <div class="protein-card">
                            <strong>Deconvoluted MW [Da]:</strong><br>
                            <span class="metric-badge">{results['average']:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown(f"""
                        <div class="protein-card">
                            <strong>Standard deviation [Da]:</strong><br>
                            <span class="metric-badge">{results['stdev']:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download button
                    download_df = create_download_data(results, mz_inputs)
                    if download_df is not None:
                        csv = download_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="esiprot_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Detailed output - showing the exact ESIProt output format
                    with st.expander("📋 Detailed Calculation Results (ESIProt Format)"):
                        st.markdown("**FINAL RESULTS**")
                        st.markdown("*" * 79)
                        detail_text = ""
                        for i, (mz_val, charge, mw, error) in enumerate(zip(results['mz'], results['charges'], results['mw'], results['errors'])):
                            detail_text += f"m/z: {mz_val:.1f} charge: {charge}+ MW [Da]: {mw:.2f} Error [Da]: {error:.2f}\n"
                        
                        detail_text += f"Deconvoluted MW [Da]: {results['average']:.2f} Standard deviation [Da]: {results['stdev']:.2f}"
                        
                        st.text(detail_text)
                        
                        # Show download data preview
                        st.markdown("**CSV Download Preview:**")
                        st.dataframe(download_df, use_container_width=True, hide_index=True)
                
                # Reset calculation flag
                st.session_state.calculate = False
            
            else:
                # Show placeholder when no calculation has been performed
                st.markdown("""
                <div class="info-card">
                    <strong>📊 Ready for calculation</strong><br>
                    Enter your m/z values and click "Calculate MW" to perform deconvolution.
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">⚖️ Mass to m/z Calculation</div>', unsafe_allow_html=True)

            # Form section for calculation inputs
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
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
                charge_min_calc = st.number_input(
                    "Min charge (+):",
                    min_value=1,
                    value=5,
                    step=1,
                    help="Minimum charge state to calculate"
                )
            
            with col_charge2:
                charge_max_calc = st.number_input(
                    "Max charge (+):",
                    min_value=1,
                    value=25,
                    step=1,
                    help="Maximum charge state to calculate"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            # Calculation button
            if st.button("🧮 Calculate m/z values", type="primary"):
                st.session_state.calculate_mz = True
        
        with col2:
            st.markdown('<div class="section-header">🎯 Calculated m/z Values</div>', unsafe_allow_html=True)

            # Perform calculation if button was pressed
            if hasattr(st.session_state, 'calculate_mz') and st.session_state.calculate_mz:
                if charge_max_calc < charge_min_calc:
                    st.markdown("""
                    <div class="error-card">
                        <strong>❌ Error:</strong> Maximum charge must be greater than or equal to minimum charge.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    calculations = calc_mz_values(molecular_weight, charge_min_calc, charge_max_calc)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Calculation completed successfully!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create calculation results table
                    calc_data = []
                    for calc in calculations:
                        calc_data.append({
                            'Charge (+)': calc['charge'],
                            'Calculated m/z': f"{calc['mz']:.4f}"
                        })
                    
                    calc_df = pd.DataFrame(calc_data)
                    st.dataframe(calc_df, use_container_width=True, hide_index=True)
                    
                    # Download button for calculations
                    calc_download_df = create_calculation_download_data(calculations)
                    if calc_download_df is not None:
                        csv = calc_download_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Calculations as CSV",
                            data=csv,
                            file_name="calculated_mz_values.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Summary metrics
                    st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)
                    
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
                            <span class="metric-badge">{charge_min_calc} - {charge_max_calc}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Reset calculation flag
                st.session_state.calculate_mz = False
            
            else:
                # Show placeholder when no calculation has been performed
                st.markdown("""
                <div class="info-card">
                    <strong>🧮 Ready for calculation</strong><br>
                    Enter a molecular weight and charge range to calculate m/z values.
                </div>
                """, unsafe_allow_html=True)
    
    # Section divider
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Information section
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

if __name__ == "__main__":
    main()