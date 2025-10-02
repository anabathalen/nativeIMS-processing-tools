import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
except ImportError:
    st.error("Please install scikit-learn: pip install scikit-learn")
    st.stop()
import io
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

# Set page config
st.set_page_config(page_title="DTIMS Data Calibration", layout="wide")

# Buffer gas properties
BUFFER_GASES = {
    "Helium": {"mass": 4.002602, "symbol": "He"},  # Da
    "Nitrogen": {"mass": 28.014, "symbol": "N₂"}   # Da
}

# Physical constants
@dataclass
class PhysicalConstants:
    k_B: float = 1.380649e-23      # Boltzmann constant (J/K)
    e: float = 1.602176634e-19     # Elementary charge (C)
    N_A: float = 6.02214076e23     # Avogadro's number
    da_to_kg: float = 1.66054e-27  # Dalton to kg conversion

CONSTANTS = PhysicalConstants()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .parameter-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .results-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">DTIMS Data Calibration Tool</div>', unsafe_allow_html=True)

def parse_dtims_csv(file) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[List[str]]]:
    """Parse the DTIMS CSV file and extract data"""
    try:
        content = file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        # Extract header information
        range_files = lines[0].split(',')[1:]  # Skip first empty cell
        raw_files = lines[1].split(',')[1:]   # Skip first empty cell
        
        # Parse data into DataFrame
        data_rows = []
        for line in lines[2:]:  # Skip first 2 header lines
            values = line.split(',')
            if len(values) >= 6:  # Ensure we have all columns
                data_rows.append([float(v) if v else 0 for v in values])
        
        # Create DataFrame
        columns = ['Time'] + [f'File_{i+1}' for i in range(len(raw_files))]
        df = pd.DataFrame(data_rows, columns=columns)
        
        return df, raw_files, range_files
    
    except Exception as e:
        st.error(f"Error parsing CSV file: {str(e)}")
        return None, None, None

def find_max_drift_time(df: pd.DataFrame, column: str) -> Tuple[Optional[float], Optional[float]]:
    """Find the drift time with maximum intensity for a given column"""
    if column in df.columns:
        max_idx = df[column].idxmax()
        return df.loc[max_idx, 'Time'], df.loc[max_idx, column]
    return None, None

def calculate_reduced_mass(mass_analyte: float, mass_buffer: float) -> float:
    """Calculate reduced mass in kg"""
    mass_analyte_kg = mass_analyte * CONSTANTS.da_to_kg
    mass_buffer_kg = mass_buffer * CONSTANTS.da_to_kg
    return (mass_analyte_kg * mass_buffer_kg) / (mass_analyte_kg + mass_buffer_kg)

def calculate_ccs_mason_schamp(
    drift_time: float, 
    voltage: float, 
    temperature: float, 
    pressure: float, 
    mass_analyte: float, 
    mass_buffer: float,
    charge: int = 1, 
    length: float = 25.05
) -> float:
    """
    Calculate CCS using the Mason-Schamp equation
    
    Parameters:
    - drift_time: drift time in ms
    - voltage: voltage in V
    - temperature: temperature in K
    - pressure: pressure in mbar
    - mass_analyte: mass of analyte in Da
    - mass_buffer: mass of buffer gas in Da
    - charge: charge state
    - length: drift tube length in cm (default 25.05 cm for Synapt)
    
    Returns:
    - CCS in Å²
    """
    # Convert units
    drift_time_s = drift_time * 1e-3  # ms to s
    length_m = length * 1e-2          # cm to m
    pressure_Pa = pressure * 100      # mbar to Pa
    
    # Calculate reduced mass
    reduced_mass = calculate_reduced_mass(mass_analyte, mass_buffer)
    
    # Calculate mobility
    mobility = length_m / (voltage * drift_time_s)  # m²/(V·s)
    
    # Calculate number density of buffer gas
    n_gas = pressure_Pa / (CONSTANTS.k_B * temperature)  # molecules/m³
    
    # Mason-Schamp equation
    ccs_m2 = (
        (3 * CONSTANTS.e * charge) / (16 * n_gas) * 
        np.sqrt(2 * np.pi / (reduced_mass * CONSTANTS.k_B * temperature)) * 
        (1 / mobility)
    )
    
    # Convert to Å²
    return ccs_m2 * 1e20

def calculate_true_voltage(
    helium_cell_dc: float, 
    bias: float, 
    transfer_dc_entrance: float, 
    helium_exit_dc: float
) -> float:
    """Calculate true voltage from experimental parameters"""
    return (helium_cell_dc + bias) - (transfer_dc_entrance + helium_exit_dc)

def perform_linear_regression(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
    """Perform linear regression and return results"""
    reg = LinearRegression()
    reg.fit(x_data.reshape(-1, 1), y_data)
    
    y_pred = reg.predict(x_data.reshape(-1, 1))
    r2 = r2_score(y_data, y_pred)
    
    return {
        'gradient': reg.coef_[0],
        'intercept': reg.intercept_,
        'r2': r2,
        'y_pred': y_pred
    }

def create_calibration_plot(valid_data: pd.DataFrame, regression_results: Dict[str, float]) -> plt.Figure:
    """Create calibration plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(valid_data['Voltage_Inverse'], valid_data['Max_Drift_Time'], 
              color='blue', s=100, alpha=0.7, label='Data Points')
    
    # Plot regression line
    x_line = np.linspace(valid_data['Voltage_Inverse'].min(), 
                        valid_data['Voltage_Inverse'].max(), 100)
    y_line = regression_results['gradient'] * x_line + regression_results['intercept']
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'Fit: y = {regression_results["gradient"]:.6f}x + {regression_results["intercept"]:.6f}')
    
    ax.set_xlabel('1/Voltage (V⁻¹)')
    ax.set_ylabel('Drift Time (ms)')
    ax.set_title(f'DTIMS Calibration Plot (R² = {regression_results["r2"]:.6f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_ccs_voltage_plot(ccs_data: pd.DataFrame) -> plt.Figure:
    """Create CCS vs Voltage plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(ccs_data['Voltage (V)'], ccs_data['CCS (Å²)'], 
              color='red', s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add connecting line
    sorted_data = ccs_data.sort_values('Voltage (V)')
    ax.plot(sorted_data['Voltage (V)'], sorted_data['CCS (Å²)'], 
           'r--', alpha=0.6, linewidth=1)
    
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('CCS (Å²)')
    ax.set_title('CCS vs Voltage (Maximum Intensity Points)')
    ax.grid(True, alpha=0.3)
    
    # Annotate points with file names
    for _, row in ccs_data.iterrows():
        ax.annotate(row['File'], 
                   (row['Voltage (V)'], row['CCS (Å²)']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    return fig

# Initialize session state
session_defaults = {
    'data_uploaded': False,
    'parameters_set': False
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# File upload section
st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload DTIMS CSV file", type=['csv'])

if uploaded_file is not None:
    df, raw_files, range_files = parse_dtims_csv(uploaded_file)
    
    if df is not None:
        st.session_state.data_uploaded = True
        st.session_state.df = df
        st.session_state.raw_files = raw_files
        st.session_state.range_files = range_files
        
        st.success(f"File uploaded successfully! Found {len(raw_files)} raw files.")
        
        # Display basic info about the data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Files", len(raw_files))
        with col2:
            st.metric("Number of Data Points", len(df))
        with col3:
            st.metric("Time Range", f"{df['Time'].min():.2f} - {df['Time'].max():.2f}")

# Parameter input section
if st.session_state.data_uploaded:
    st.markdown('<div class="section-header">⚙️ Experimental Parameters</div>', unsafe_allow_html=True)
    
    # Global parameters
    st.markdown("### Global Experimental Conditions")
    col1, col2 = st.columns(2)
    
    with col1:
        pressure = st.number_input("Pressure (mbar)", value=3.0, format="%.3f")
        temperature = st.number_input("Temperature (K)", value=298.0, format="%.1f")
        pusher_time = st.number_input("Pusher Time (µs)", value=100.0, format="%.1f")
        
        # Buffer gas selection
        buffer_gas = st.selectbox(
            "Buffer Gas", 
            options=list(BUFFER_GASES.keys()),
            index=0,  # Default to Helium
            help="Select the buffer gas used in your experiments"
        )
    
    with col2:
        transfer_dc_entrance = st.number_input("Transfer DC Entrance (V)", value=0.0, format="%.1f")
        helium_exit_dc = st.number_input("Helium Exit DC (V)", value=0.0, format="%.1f")
        
        # Show selected buffer gas properties
        st.info(f"**{buffer_gas}** selected\n\n"
                f"Symbol: {BUFFER_GASES[buffer_gas]['symbol']}\n\n"
                f"Mass: {BUFFER_GASES[buffer_gas]['mass']} Da")
    
    # File-specific parameters
    st.markdown("### File-Specific Parameters")
    st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
    
    # Create input fields for each file
    file_params = {}
    
    for i, raw_file in enumerate(st.session_state.raw_files):
        st.markdown(f"**File {i+1}: {raw_file}**")
        col1, col2 = st.columns(2)
        
        with col1:
            helium_cell_dc = st.number_input(
                f"Helium Cell DC (V)", 
                key=f"helium_dc_{i}",
                value=0.0,
                format="%.1f"
            )
        
        with col2:
            bias = st.number_input(
                f"Bias (V)", 
                key=f"bias_{i}",
                value=0.0,
                format="%.1f"
            )
        
        file_params[f'File_{i+1}'] = {
            'helium_cell_dc': helium_cell_dc,
            'bias': bias,
            'raw_file': raw_file
        }
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyte mass input
    st.markdown("### Analyte Information")
    st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
    mass_analyte = st.number_input(
        "**Analyte Mass (Da)**", 
        value=0.0, 
        min_value=0.0,
        format="%.4f", 
        help="Enter the mass of your analyte in Daltons (Da). This is required for CCS calculations using the Mason-Schamp equation."
    )
    
    if mass_analyte <= 0:
        st.warning("⚠️ Please enter a valid analyte mass (> 0 Da) to proceed with CCS calculations.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process data button
    process_disabled = mass_analyte <= 0
    
    if process_disabled:
        st.info("👆 Please enter the analyte mass above to enable data processing.")
    
    if st.button("🔬 Process Data and Calculate Calibration", type="primary", disabled=process_disabled):
        # Find maximum drift times and intensities
        results_data = []
        
        for file_col, params in file_params.items():
            max_drift, max_intensity = find_max_drift_time(st.session_state.df, file_col)
            
            if max_drift is not None:
                true_voltage = calculate_true_voltage(
                    params['helium_cell_dc'], params['bias'],
                    transfer_dc_entrance, helium_exit_dc
                )
                voltage_inverse = 1 / true_voltage if true_voltage != 0 else np.nan
                
                results_data.append({
                    'File': params['raw_file'],
                    'Column': file_col,
                    'Helium_Cell_DC': params['helium_cell_dc'],
                    'Bias': params['bias'],
                    'Max_Drift_Time': max_drift,
                    'Max_Intensity': max_intensity,
                    'True_Voltage': true_voltage,
                    'Voltage_Inverse': voltage_inverse
                })
        
        results_df = pd.DataFrame(results_data)
        
        # Filter out invalid data
        valid_data = results_df.dropna(subset=['Voltage_Inverse', 'Max_Drift_Time'])
        
        if len(valid_data) >= 2:
            # Perform linear regression
            regression_results = perform_linear_regression(
                valid_data['Voltage_Inverse'].values,
                valid_data['Max_Drift_Time'].values
            )
            
            # Store calibration results
            st.session_state.calibration_results = {
                **regression_results,
                'results_df': results_df,
                'valid_data': valid_data
            }
            
            # Display results
            st.markdown('<div class="section-header">📊 Calibration Results</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Gradient (td)", f"{regression_results['gradient']:.6f}")
            with col2:
                st.metric("Intercept (t0)", f"{regression_results['intercept']:.6f}")
            with col3:
                st.metric("R² Value", f"{regression_results['r2']:.6f}")
            
            # Create and display calibration plot
            fig = create_calibration_plot(valid_data, regression_results)
            st.pyplot(fig)
            
            # Display detailed results table
            st.markdown("### Detailed Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Calculate CCS values for ALL drift times and voltages
            st.markdown('<div class="section-header">🧮 CCS Calculations (All Data Points)</div>', unsafe_allow_html=True)
            
            # CCS calculation parameters
            col1, col2 = st.columns(2)
            with col1:
                charge_state = st.number_input("Charge State", value=1, min_value=1)
            with col2:
                st.info(f"Using Mason-Schamp equation with:\n"
                        f"- Buffer gas: {buffer_gas} ({BUFFER_GASES[buffer_gas]['mass']} Da)\n"
                        f"- Drift tube length: 25.05 cm")
            
            # Calculate CCS for ALL drift times across all files and voltages
            all_ccs_data = []
            buffer_mass = BUFFER_GASES[buffer_gas]['mass']
            
            for file_col, params in file_params.items():
                # Get all non-zero drift times and intensities for this file
                file_data = st.session_state.df[st.session_state.df[file_col] > 0].copy()
                
                if not file_data.empty:
                    true_voltage = calculate_true_voltage(
                        params['helium_cell_dc'], params['bias'],
                        transfer_dc_entrance, helium_exit_dc
                    )
                    
                    for _, row in file_data.iterrows():
                        drift_time = row['Time']
                        intensity = row[file_col]
                        
                        # Calculate CCS using Mason-Schamp equation
                        ccs_value = calculate_ccs_mason_schamp(
                            drift_time=drift_time,
                            voltage=abs(true_voltage),
                            temperature=temperature,
                            pressure=pressure,
                            mass_analyte=mass_analyte,
                            mass_buffer=buffer_mass,
                            charge=charge_state
                        )
                        
                        all_ccs_data.append({
                            'Charge': charge_state,
                            'Drift': drift_time,
                            'CCS': ccs_value,
                            'True_Voltage': true_voltage,
                            'Intensity': intensity,
                            'File': params['raw_file']
                        })
            
            # Create comprehensive CCS DataFrame
            comprehensive_ccs_df = pd.DataFrame(all_ccs_data)
            
            if not comprehensive_ccs_df.empty:
                # Remove any invalid CCS values
                comprehensive_ccs_df = comprehensive_ccs_df.dropna(subset=['CCS'])
                comprehensive_ccs_df = comprehensive_ccs_df[np.isfinite(comprehensive_ccs_df['CCS'])]
                
                st.markdown("### Complete CCS Dataset")
                st.dataframe(comprehensive_ccs_df, use_container_width=True)
                
                # Create output DataFrame
                output_df = comprehensive_ccs_df[['Charge', 'Drift', 'CCS', 'True_Voltage', 'Intensity']].copy()
                
                # Create download button
                csv_buffer = io.StringIO()
                output_df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Complete Calibrated Data (CSV)",
                    data=csv_string,
                    file_name=f"dtims_complete_calibrated_data_{buffer_gas.lower()}.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                st.markdown("### Summary Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Data Points", len(output_df))
                with col2:
                    st.metric("Average CCS (Å²)", f"{output_df['CCS'].mean():.2f}")
                
                # Create compact table for max intensity CCS at each voltage
                st.markdown("### CCS at Maximum Intensity (by Voltage)")
                max_intensity_data = []
                
                for file_col, params in file_params.items():
                    file_data = st.session_state.df[st.session_state.df[file_col] > 0]
                    if not file_data.empty:
                        max_idx = file_data[file_col].idxmax()
                        max_drift = file_data.loc[max_idx, 'Time']
                        max_intensity = file_data.loc[max_idx, file_col]
                        
                        true_voltage = calculate_true_voltage(
                            params['helium_cell_dc'], params['bias'],
                            transfer_dc_entrance, helium_exit_dc
                        )
                        
                        ccs_max = calculate_ccs_mason_schamp(
                            drift_time=max_drift,
                            voltage=abs(true_voltage),
                            temperature=temperature,
                            pressure=pressure,
                            mass_analyte=mass_analyte,
                            mass_buffer=buffer_mass,
                            charge=charge_state
                        )
                        
                        max_intensity_data.append({
                            'File': params['raw_file'],
                            'Voltage (V)': true_voltage,
                            'Max Drift Time (ms)': max_drift,
                            'Max Intensity': max_intensity,
                            'CCS (Å²)': ccs_max
                        })
                
                max_intensity_df = pd.DataFrame(max_intensity_data)
                
                if not max_intensity_df.empty:
                    st.dataframe(max_intensity_df, use_container_width=True)
                    
                    # Plot CCS vs Voltage
                    fig = create_ccs_voltage_plot(max_intensity_df)
                    st.pyplot(fig)
                
                # Show file-wise summary
                st.markdown("### File-wise Summary")
                file_summary = comprehensive_ccs_df.groupby('File').agg({
                    'CCS': ['count', 'mean'],
                    'True_Voltage': 'first',
                    'Intensity': 'sum'
                }).round(2)
                
                file_summary.columns = ['Data_Points', 'Mean_CCS', 'True_Voltage', 'Total_Intensity']
                st.dataframe(file_summary, use_container_width=True)
        
        else:
            st.error("Not enough valid data points for calibration. Need at least 2 data points.")

# Information section
st.markdown('<div class="section-header">ℹ️ Information</div>', unsafe_allow_html=True)

with st.expander("How to use this tool"):
    st.markdown("""
    1. **Upload your DTIMS CSV file** - The file should contain time series data with multiple columns for different experimental conditions
    2. **Set global parameters** - Enter the experimental conditions (pressure, temperature, pusher time, etc.)
    3. **Select buffer gas** - Choose between Helium and Nitrogen
    4. **Set file-specific parameters** - For each raw file, enter the Helium Cell DC and Bias values
    5. **Enter analyte mass** - Required for CCS calculations
    6. **Process the data** - The tool will:
       - Find the drift time with maximum intensity for each file (for calibration plot)
       - Calculate the true voltage: (Helium Cell DC + Bias) - (Transfer DC Entrance + Helium Exit DC)
       - Plot drift time vs. 1/voltage and fit a linear regression for calibration verification
       - Calculate CCS for ALL drift times using the Mason-Schamp equation
       - Generate a downloadable CSV with complete calibrated data (Charge, Drift, CCS, True_Voltage, Intensity)
    """)

with st.expander("About the calculations"):
    st.markdown("""
    **True Voltage Calculation:**
    ```
    True Voltage = (Helium Cell DC + Bias) - (Transfer DC Entrance + Helium Exit DC)
    ```
    
    **Mason-Schamp CCS Calculation:**
    ```
    CCS = (3 × e × z) / (16 × N₀) × √(2π / (μ × k_B × T)) × (1 / K₀)
    ```
    
    Where:
    - `e` = elementary charge (1.602 × 10⁻¹⁹ C)
    - `z` = charge state
    - `N₀` = number density of buffer gas
    - `μ` = reduced mass of analyte-buffer gas system
    - `k_B` = Boltzmann constant (1.381 × 10⁻²³ J/K)
    - `T` = temperature (K)
    - `K₀` = mobility = L/(V×t_d)
    - `L` = drift tube length (25.05 cm for Synapt)
    
    **Reduced Mass Calculation:**
    ```
    μ = (m_analyte × m_buffer) / (m_analyte + m_buffer)
    ```
    
    **Supported Buffer Gases:**
    - Helium (He): 4.002602 Da
    - Nitrogen (N₂): 28.014 Da
    """)

with st.expander("Buffer Gas Selection"):
    st.markdown("""
    **Helium (He):**
    - Mass: 4.002602 Da
    - Most commonly used buffer gas in DTIMS
    - Provides high resolution and sensitivity
    
    **Nitrogen (N₂):**
    - Mass: 28.014 Da
    - Alternative buffer gas
    - Different collision cross-section behavior compared to helium
    
    The tool automatically uses the correct mass in the Mason-Schamp equation based on your selection.
    """)