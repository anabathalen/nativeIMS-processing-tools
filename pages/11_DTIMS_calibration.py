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
    pi: float = np.pi

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
    length: float = 25.05,
    return_mobility: bool = False
) -> float | Tuple[float, float]:
    """
    Calculate CCS (Å²) using the Mason–Schamp equation (correct form).
    
    Ω = (3 z e)/(16 N K) * sqrt( (2 π)/( μ k_B T ) )
    
    Where:
      N = P / (k_B T)
      K = L^2 / (V * t_d)
    
    Inputs:
      drift_time : ms
      voltage    : V
      temperature: K
      pressure   : mbar
      length     : cm (drift region length)
    
    Returns:
      CCS in Å² (or (CCS, mobility) if return_mobility=True)
    """
    if (drift_time is None or voltage is None or
        drift_time <= 0 or voltage <= 0 or
        temperature <= 0 or pressure <= 0 or
        mass_analyte <= 0 or mass_buffer <= 0):
        return (np.nan, np.nan) if return_mobility else np.nan

    # Unit conversions
    t_s = drift_time * 1e-3          # ms -> s
    L_m = length * 1e-2              # cm -> m
    P_Pa = pressure * 100.0          # mbar -> Pa

    # Reduced mass (kg)
    mu = calculate_reduced_mass(mass_analyte, mass_buffer)

    # Number density N (m^-3)
    N = P_Pa / (CONSTANTS.k_B * temperature)

    # Mobility (m^2 / (V·s))
    K = (L_m * L_m) / (voltage * t_s)

    if K <= 0 or not np.isfinite(K):
        return (np.nan, np.nan) if return_mobility else np.nan

    # Mason–Schamp (correct square root term)
    prefactor = (3.0 * charge * CONSTANTS.e) / (16.0 * N * K)
    sqrt_term = np.sqrt((2.0 * CONSTANTS.pi) / (mu * CONSTANTS.k_B * temperature))
    ccs_m2 = prefactor * sqrt_term  # m^2

    ccs_A2 = ccs_m2 * 1e20  # m^2 -> Å^2

    return (ccs_A2, K) if return_mobility else ccs_A2

def calculate_true_voltage(
    helium_cell_dc: float, 
    bias: float, 
    transfer_dc_entrance: float, 
    helium_exit_dc: float
) -> float:
    """Calculate true voltage from experimental parameters"""
    return (helium_cell_dc + bias) - (transfer_dc_entrance + helium_exit_dc)

def perform_linear_regression(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
    """Linear regression: y vs x"""
    reg = LinearRegression()
    reg.fit(x_data.reshape(-1, 1), y_data)
    y_pred = reg.predict(x_data.reshape(-1, 1))
    r2 = r2_score(y_data, y_pred)
    return {
        'gradient': reg.coef_[0],   # slope (Δy / Δx)
        'intercept': reg.intercept_,
        'r2': r2,
        'y_pred': y_pred
    }

def create_calibration_plot(valid_data: pd.DataFrame, regression_results: Dict[str, float]) -> plt.Figure:
    """
    Plot: x = Drift Time (ms), y = 1/Voltage (V⁻¹)
    Regression performed on (Drift Time -> 1/Voltage)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter
    ax.scatter(valid_data['Max_Drift_Time'],
               valid_data['Voltage_Inverse'],
               color='blue', s=100, alpha=0.75, edgecolors='black', linewidth=0.5,
               label='Data Points')

    # Fit line
    x_line = np.linspace(valid_data['Max_Drift_Time'].min(),
                         valid_data['Max_Drift_Time'].max(), 200)
    y_line = regression_results['gradient'] * x_line + regression_results['intercept']
    ax.plot(x_line, y_line, 'r-', lw=2,
            label=f'Fit: y = {regression_results["gradient"]:.6g}x + {regression_results["intercept"]:.6g}')

    ax.set_xlabel('Drift Time (ms)')
    ax.set_ylabel('1 / Voltage (V⁻¹)')
    ax.set_title(f'Calibration: 1/V vs Drift Time (R² = {regression_results["r2"]:.6f})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
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
        results_data = []
        zero_voltage_files = []

        for file_col, params in file_params.items():
            max_drift, max_intensity = find_max_drift_time(st.session_state.df, file_col)
            if max_drift is not None:
                true_voltage = calculate_true_voltage(
                    params['helium_cell_dc'], params['bias'],
                    transfer_dc_entrance, helium_exit_dc
                )
                if true_voltage == 0:
                    zero_voltage_files.append(params['raw_file'])
                    voltage_inverse = np.nan
                else:
                    voltage_inverse = 1.0 / true_voltage

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

        if zero_voltage_files:
            st.warning("These files have zero true voltage (CCS cannot be computed, values set NaN): " +
                       ", ".join(zero_voltage_files))

        results_df = pd.DataFrame(results_data)
        valid_data = results_df.dropna(subset=['Voltage_Inverse', 'Max_Drift_Time'])

        if len(valid_data) >= 2:
            # NOTE: Regression now: y = 1/V, x = Drift Time
            regression_results = perform_linear_regression(
                valid_data['Max_Drift_Time'].values,
                valid_data['Voltage_Inverse'].values
            )

            st.session_state.calibration_results = {
                **regression_results,
                'results_df': results_df,
                'valid_data': valid_data
            }

            st.markdown('<div class="section-header">📊 Calibration Results</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Slope (Δ(1/V)/Δt)", f"{regression_results['gradient']:.6g}")
            with col2:
                st.metric("Intercept", f"{regression_results['intercept']:.6g}")
            with col3:
                st.metric("R²", f"{regression_results['r2']:.6f}")

            fig = create_calibration_plot(valid_data, regression_results)
            st.pyplot(fig)

            st.markdown("### Drift Time / Voltage Summary (Apex Points)")
            st.dataframe(results_df, use_container_width=True)

            # =========================
            # CCS CALCULATIONS SECTION
            # =========================
            st.markdown('<div class="section-header">🧮 Apex CCS Values</div>', unsafe_allow_html=True)
            colA, colB = st.columns(2)
            with colA:
                charge_state = st.number_input("Charge State", value=1, min_value=1)
            with colB:
                st.info("Apex CCS uses each file's maximum intensity drift time and true voltage.")

            buffer_mass = BUFFER_GASES[buffer_gas]['mass']

            # Compute CCS for each apex (max drift time per file)
            apex_ccs_rows = []
            for _, row in results_df.iterrows():
                td = row['Max_Drift_Time']
                V = row['True_Voltage']
                if np.isfinite(td) and V is not None and V > 0:
                    apex_ccs = calculate_ccs_mason_schamp(
                        drift_time=td,
                        voltage=abs(V),
                        temperature=temperature,
                        pressure=pressure,
                        mass_analyte=mass_analyte,
                        mass_buffer=buffer_mass,
                        charge=charge_state
                    )
                else:
                    apex_ccs = np.nan

                apex_ccs_rows.append({
                    'File': row['File'],
                    'Max_Drift_Time (ms)': td,
                    'True_Voltage (V)': V,
                    '1/Voltage (V⁻¹)': (1.0 / V) if V not in (0, None) else np.nan,
                    'Apex_Intensity': row['Max_Intensity'],
                    'Helium_Cell_DC': row['Helium_Cell_DC'],
                    'Bias': row['Bias'],
                    'Charge': charge_state,
                    'Apex_CCS (Å²)': apex_ccs
                })

            apex_ccs_df = pd.DataFrame(apex_ccs_rows)
            st.dataframe(apex_ccs_df, use_container_width=True)

            # Download apex CCS table
            apex_csv = apex_ccs_df.to_csv(index=False)
            st.download_button(
                "📥 Download Apex CCS Table (CSV)",
                data=apex_csv,
                file_name=f"dtims_apex_ccs_{buffer_gas.lower()}.csv",
                mime="text/csv"
            )

            # =========================
            # FULL DATASET CCS
            # =========================
            st.markdown('<div class="section-header">📈 Full Dataset CCS Conversion</div>', unsafe_allow_html=True)
            st.caption("All drift times for every file converted to CCS (long format).")

            all_ccs_data = []
            skipped_ccs = 0

            # Iterate each file (column) and compute CCS for every non-zero intensity drift time
            for file_col, params in file_params.items():
                file_signal = st.session_state.df[['Time', file_col]].copy()
                file_signal = file_signal[file_signal[file_col] > 0]  # keep positive intensities
                if file_signal.empty:
                    continue

                true_voltage = calculate_true_voltage(
                    params['helium_cell_dc'], params['bias'],
                    transfer_dc_entrance, helium_exit_dc
                )

                if true_voltage is None or true_voltage <= 0:
                    skipped_ccs += len(file_signal)
                    continue

                inv_voltage = 1.0 / true_voltage if true_voltage != 0 else np.nan

                for _, r in file_signal.iterrows():
                    td = r['Time']
                    intensity = r[file_col]

                    if td <= 0:
                        ccs_val = np.nan
                    else:
                        ccs_val = calculate_ccs_mason_schamp(
                            drift_time=td,
                            voltage=abs(true_voltage),
                            temperature=temperature,
                            pressure=pressure,
                            mass_analyte=mass_analyte,
                            mass_buffer=buffer_mass,
                            charge=charge_state
                        )

                    all_ccs_data.append({
                        'File': params['raw_file'],
                        'Time (ms)': td,
                        'Intensity': intensity,
                        'True_Voltage (V)': true_voltage,
                        '1/Voltage (V⁻¹)': inv_voltage,
                        'Helium_Cell_DC': params['helium_cell_dc'],
                        'Bias': params['bias'],
                        'Charge': charge_state,
                        'CCS (Å²)': ccs_val
                    })

            if skipped_ccs > 0:
                st.warning(f"Skipped {skipped_ccs} drift points (non-positive true voltage).")

            full_ccs_df = pd.DataFrame(all_ccs_data)

            if full_ccs_df.empty:
                st.error("No CCS values computed (check voltages and intensities).")
            else:
                st.dataframe(full_ccs_df, use_container_width=True, height=400)

                # Download full CCS dataset
                full_csv = full_ccs_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Full CCS Dataset (CSV)",
                    data=full_csv,
                    file_name=f"dtims_full_ccs_{buffer_gas.lower()}.csv",
                    mime="text/csv"
                )
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
       - Show both td vs 1/V (traditional) and td vs V (direct) relationships
       - Calculate CCS for ALL drift times using the corrected Mason-Schamp equation
       - Generate a downloadable CSV with complete calibrated data
    """)

with st.expander("About the calculations"):
    st.markdown("""
    **True Voltage Calculation:**
    ```
    True Voltage = (Helium Cell DC + Bias) - (Transfer DC Entrance + Helium Exit DC)
    ```
    
    **Corrected Mason-Schamp CCS Calculation:**
    ```
    CCS = (3 * e * z)/(16 * N) * sqrt(2 * π * μ * k_B * T) * (1 / K)
    ```
    
    Where:
    - `e` = elementary charge (1.602 × 10⁻¹⁹ C)
    - `z` = charge state
    - `N` = number density of buffer gas = P/(kT)
    - `μ` = reduced mass of analyte-buffer gas system
    - `k` = Boltzmann constant (1.381 × 10⁻²³ J/K)
    - `T` = temperature (K)
    - `K` = mobility = L²/(V×td) [corrected formula]
    - `L` = drift tube length (25.05 cm for Synapt)
    
    **Key Correction:**
    - Mobility: K₀ = L²/(V×td) instead of L/(V×td)
    - This ensures the proper linear relationship: td ∝ V
    
    **Reduced Mass Calculation:**
    ```
    μ = (m_analyte × m_buffer) / (m_analyte + m_buffer)
    ```
    
    **Supported Buffer Gases:**
    - Helium (He): 4.002602 Da
    - Nitrogen (N₂): 28.014 Da
    """)

with st.expander("Linear Relationships"):
    st.markdown("""
    **Expected Linear Relationships:**
    
    1. **td vs 1/V (Traditional Calibration):**
       - From the relationship: td = (L²/K₀) × (1/V) + t₀
       - Should give a straight line when plotting td vs 1/V
       - Slope = L²/K₀, Intercept = t₀
    
    2. **td vs V (Direct Relationship):**
       - For constant mobility: td ∝ 1/V, so td vs V should be approximately linear for small voltage ranges
       - Useful for verification of instrumental stability
    
    The tool now shows both relationships to help verify your calibration.
    """)