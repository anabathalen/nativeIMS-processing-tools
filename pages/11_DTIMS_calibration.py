import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import math
import io
from typing import Tuple, Optional, Dict, List

# Set page config
st.set_page_config(page_title="SYNAPT G2 DTIMS Data Processing", layout="wide")

# Set page config
st.set_page_config(page_title="SYNAPT G2 DTIMS Data Processing", layout="wide")

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

st.markdown('<div class="main-header">SYNAPT G2 DTIMS Data Processing</div>', unsafe_allow_html=True)

# Helper functions from the notebook
def get_arrival(get_scans, pusher):
    """Convert scans to arrival time"""
    scans = get_scans
    arrival = []
    pusher = pusher
    for i in scans:
        arrival.append(i*0.000001*pusher)
    return arrival

def normalise(a):
    """Normalize intensity data"""
    data_norm = []
    for i in a:
        data_norm.append(i/np.max(a))
    return data_norm

def get_scans(y):
    """Generate scan numbers from data length"""
    n = len(y)
    scans = []
    for i in range(n):
        scans.append(i+1)
    return scans

def CCS(z, V, mass, td, t0, Tavg, Pavg):
    """Calculate CCS for given drift time - SYNAPT G2 formula"""
    # Define constants
    e = 1.6E-19
    T = Tavg
    P = Pavg
    L = 0.255  # Length in meters for SYNAPT G2
    kb = 1.38E-23
    pi = math.pi
    m1 = mass
    m2 = 6.64E-27  # Helium mass in kg
    
    # Calculate CCS for given drift time
    top = 3*z*e*math.sqrt(2*pi*T*kb)*V*(td - t0)
    bot = 16*P*(L**2)*math.sqrt((m1*m2)/(m1+m2))
    ans = top/bot
    scaled = ans*100000000000000000000  # Scale to Angstrom^2
    return scaled

def plot_ATD(df, ax=None):
    """Plot arrival time distribution"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()
    
    # Take columns from data set
    x = df['Arrival Time (ms)'].tolist()
    y = df['Intensity'].tolist()
    
    # Smooth the data
    xnew = np.linspace(np.min(x), np.max(x), 1500)
    f = interp1d(x, y)
    y_smooth = f(xnew)
    line, = ax.plot(xnew, y_smooth, 'r-')
    
    # Find maximum
    ymax = np.max(y)
    xpos = y.index(ymax)
    xmax = x[xpos]
    time = xmax
    
    # Add annotation
    ax.annotate(s=round(xmax, 2), xy=(xmax, ymax), xytext=(xmax+0.5, ymax*0.9),
                arrowprops=dict(facecolor='orange'))
    
    ax.set_xlim(np.min(x), np.max(x))
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Arrival time (ms)')
    ax.set_title(df.name if hasattr(df, 'name') else 'ATD')
    
    return time, fig

def plot_CCSD(df2, ax=None):
    """Make plot of CCSD"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()
    
    # Take columns from data set
    x = df2.iloc[:, 1].tolist()
    y = df2.iloc[:, 2].tolist()
    
    # Smooth the data
    xnew = np.linspace(np.min(x), np.max(x), 1500)
    f = interp1d(x, y)
    y_smooth = f(xnew)
    line, = ax.plot(xnew, y_smooth, 'r-')
    
    # Find maximum
    ymax = np.max(y)
    xpos = y.index(ymax)
    xmax = x[xpos]
    apex = round(xmax, 2)
    
    # Add annotation
    ax.annotate(s=apex, xy=(xmax, ymax), xytext=(xmax+100, ymax-0.1),
                arrowprops=dict(facecolor='orange'))
    
    ax.set_ylim(0)
    ax.set_xlim(np.min(x), np.max(x))
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(u"CCS (\u212B\u00B2)")
    ax.set_title(df2.name if hasattr(df2, 'name') else 'CCSD')
    
    return apex, fig

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
            # Linear regression: td vs 1/V (standard DTIMS calibration method)
            # y = 1/V, x = Drift Time
            # Expected relationship: td = (L²/K₀)(1/V) + t₀
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
            
            # Calculate mobility from calibration slope
            # Relationship: 1/V = (K/L²)·td + intercept
            # So: K = slope × L²
            # BUT: slope is in V⁻¹/ms, need to convert to V⁻¹/s first
            length_cm = 25.05
            length_m = length_cm * 1e-2
            slope_V_per_ms = regression_results['gradient']  # V⁻¹/ms
            slope_V_per_s = slope_V_per_ms * 1000.0  # V⁻¹/s (convert ms to s)
            mobility_K = slope_V_per_s * (length_m ** 2)  # m²/(V·s)
            
            # Convert to reduced mobility K₀ (at standard conditions: 273.15 K, 101325 Pa)
            # K₀ = K × (P/P₀) × (T₀/T)
            T0 = 273.15  # K (standard temperature)
            P0 = 101325.0  # Pa (standard pressure, 1 atm)
            P_Pa = pressure * 100.0  # convert mbar to Pa
            reduced_mobility_K0 = mobility_K * (P_Pa / P0) * (T0 / temperature)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Slope (Δ(1/V)/Δt)", f"{regression_results['gradient']:.6g} V⁻¹/ms")
            with col2:
                st.metric("Intercept", f"{regression_results['intercept']:.6g} V⁻¹")
            with col3:
                st.metric("R²", f"{regression_results['r2']:.6f}")
            with col4:
                st.metric("Reduced Mobility K₀", f"{reduced_mobility_K0:.6e} m²/(V·s)")

            st.info(f"**Mobility from calibration**: K = slope × L² = ({slope_V_per_ms:.6g} V⁻¹/ms × 1000) × ({length_m:.4f} m)² = {mobility_K:.6e} m²/(V·s)\n\n" +
                   f"**Reduced mobility**: K₀ = K × (P/P₀) × (T₀/T) = {mobility_K:.6e} × ({P_Pa}/{P0}) × ({T0}/{temperature}) = {reduced_mobility_K0:.6e} m²/(V·s)\n\n" +
                   f"**For CCS calculation**: Using experimental K with experimental T and P (correct approach)")

            fig = create_calibration_plot(valid_data, regression_results)
            st.pyplot(fig)

            st.markdown("### Drift Time / Voltage Summary (Apex Points)")
            st.dataframe(results_df, use_container_width=True)

            # =========================
            # CCS CALCULATIONS SECTION
            # =========================
            st.markdown('<div class="section-header">🧮 CCS Calculation</div>', unsafe_allow_html=True)
            st.info("**Important**: CCS is calculated using the experimental mobility K from the calibration, " +
                   "combined with experimental temperature and pressure in the Mason-Schamp equation: Ω = (3ze)/(16NK) × sqrt(2π/(μk_BT))")
            
            colA, colB = st.columns(2)
            with colA:
                charge_state = st.number_input("Charge State", value=1, min_value=1)
            with colB:
                st.metric("Using Mobility K", f"{mobility_K:.6e} m²/(V·s)")
                st.caption(f"(K₀ = {reduced_mobility_K0:.6e} m²/(V·s) for reference)")

            buffer_mass = BUFFER_GASES[buffer_gas]['mass']

            # Calculate CCS using the experimental mobility from calibration
            # IMPORTANT: Use experimental T and P with experimental K
            # This is the correct approach - don't mix reduced mobility with standard conditions
            ccs_value = calculate_ccs_from_mobility(
                mobility_K=mobility_K,  # Use experimental K, not K₀
                temperature=temperature,  # Use experimental temperature
                pressure=pressure,  # Use experimental pressure (in mbar)
                mass_analyte=mass_analyte,
                mass_buffer=buffer_mass,
                charge=charge_state
            )

            st.markdown('<div class="results-box">', unsafe_allow_html=True)
            st.markdown(f"### **Collision Cross Section (CCS)**")
            st.markdown(f"# {ccs_value:.2f} Ų")
            st.markdown('</div>', unsafe_allow_html=True)

            # Show apex table with calculated CCS (same for all since using calibration K)
            apex_ccs_rows = []
            for _, row in results_df.iterrows():
                apex_ccs_rows.append({
                    'File': row['File'],
                    'Max_Drift_Time (ms)': row['Max_Drift_Time'],
                    'True_Voltage (V)': row['True_Voltage'],
                    '1/Voltage (V⁻¹)': row['Voltage_Inverse'],
                    'Apex_Intensity': row['Max_Intensity'],
                    'Helium_Cell_DC': row['Helium_Cell_DC'],
                    'Bias': row['Bias'],
                    'Charge': charge_state,
                    'CCS (Å²)': ccs_value  # Same CCS for all - from calibration
                })

            apex_ccs_df = pd.DataFrame(apex_ccs_rows)
            st.markdown("### Calibration Data with CCS")
            st.dataframe(apex_ccs_df, use_container_width=True)

            # Download apex CCS table
            apex_csv = apex_ccs_df.to_csv(index=False)
            st.download_button(
                "📥 Download Calibration Data with CCS (CSV)",
                data=apex_csv,
                file_name=f"dtims_calibration_ccs_{buffer_gas.lower()}.csv",
                mime="text/csv"
            )

            # =========================
            # FULL DATASET - Apply CCS to all drift times
            # =========================
            st.markdown('<div class="section-header">📈 Full Dataset with CCS</div>', unsafe_allow_html=True)
            st.caption(f"All drift times for every file. CCS = {ccs_value:.2f} Ų (same for all, from calibration mobility).")

            all_ccs_data = []

            # Iterate each file (column) and add CCS column (same value for all)
            for file_col, params in file_params.items():
                file_signal = st.session_state.df[['Time', file_col]].copy()
                file_signal = file_signal[file_signal[file_col] > 0]  # keep positive intensities
                if file_signal.empty:
                    continue

                true_voltage = calculate_true_voltage(
                    params['helium_cell_dc'], params['bias'],
                    transfer_dc_entrance, helium_exit_dc
                )

                inv_voltage = 1.0 / true_voltage if (true_voltage and true_voltage != 0) else np.nan

                for _, r in file_signal.iterrows():
                    td = r['Time']
                    intensity = r[file_col]

                    all_ccs_data.append({
                        'File': params['raw_file'],
                        'Time (ms)': td,
                        'Intensity': intensity,
                        'True_Voltage (V)': true_voltage,
                        '1/Voltage (V⁻¹)': inv_voltage,
                        'Helium_Cell_DC': params['helium_cell_dc'],
                        'Bias': params['bias'],
                        'Charge': charge_state,
                        'CCS (Å²)': ccs_value  # Same CCS for all - from calibration
                    })

            full_ccs_df = pd.DataFrame(all_ccs_data)

            if full_ccs_df.empty:
                st.error("No data to display.")
            else:
                st.dataframe(full_ccs_df, use_container_width=True, height=400)

                # Download full CCS dataset
                full_csv = full_ccs_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Full Dataset with CCS (CSV)",
                    data=full_csv,
                    file_name=f"dtims_full_data_ccs_{buffer_gas.lower()}.csv",
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
       - Find the drift time with maximum intensity for each file (apex points)
       - Calculate the true voltage: (Helium Cell DC + Bias) - (|helium_exit_dc| + |transfer_dc_entrance|)
       - Plot the standard DTIMS calibration: td vs 1/V (linear relationship)
       - Calculate CCS for apex points and the full dataset using the Mason-Schamp equation
       - Generate downloadable CSV files with complete calibrated data
    """)

with st.expander("About the calculations"):
    st.markdown("""
    **True Voltage Calculation:**
    ```
    Set Voltage = Helium Cell DC + Bias
    Exit Sum = |Helium Exit DC| + |Transfer DC Entrance|
    True Voltage = Set Voltage - Exit Sum
    ```
    
    **Mason-Schamp CCS Calculation:**
    ```
    CCS (Ω) = (3·z·e)/(16·N·K) × sqrt(2π/(μ·kB·T))
    
    Where:
    - N = P/(kB·T) is the number density at experimental conditions
    - K is the mobility from calibration at experimental conditions
    ```
    
    Where:
    - `z` = charge state (integer)
    - `e` = elementary charge (1.602176634 × 10⁻¹⁹ C)
    - `N` = number density of buffer gas = P/(kB·T) [particles/m³]
    - `K` = mobility from calibration = (slope × L²) [m²/(V·s)]
    - `μ` = reduced mass of ion-buffer system [kg]
    - `kB` = Boltzmann constant (1.380649 × 10⁻²³ J/K)
    - `T` = experimental temperature [K]
    - `P` = experimental pressure [Pa]
    - `L` = drift tube length (25.05 cm for Synapt G2)
    - slope = gradient from calibration plot (V⁻¹/s)
    
    **Reduced Mass Calculation:**
    ```
    μ = (m_ion × m_buffer) / (m_ion + m_buffer)
    ```
    (masses in kg)
    
    **Supported Buffer Gases:**
    - Helium (He): 4.002602 Da
    - Nitrogen (N₂): 28.014 Da
    
    **Important Notes:**
    - The calibration yields mobility K at experimental T and P
    - This K is used directly in Mason-Schamp equation with experimental T and P
    - The reduced mobility K₀ is calculated for reference/comparison purposes
    - CCS is reported in Ų (square Angstroms)
    """)

with st.expander("Linear Relationships"):
    st.markdown("""
    **DTIMS Calibration Theory:**
    
    The standard DTIMS calibration uses the **td vs 1/V** relationship:
    
    From the mobility equation:
    - K = L²/(V·td)  [corrected mobility formula]
    - Rearranging: td = (L²/K)·(1/V)
    
    **Expected Linear Relationship:**
    - Plot: td (x-axis) vs 1/V (y-axis)
    - Should give a straight line: 1/V = m·td + c
    - Slope m = K/L² (experimental mobility divided by length squared)
    - Intercept c is related to any dead time (t₀)
    - From slope: K = m × L² (mobility at experimental conditions)
    
    **Why this method?**
    - Standard approach in DTIMS calibration
    - Directly validates the linear relationship between drift time and inverse voltage
    - Slope provides information about ion mobility
    - Good linearity (high R²) confirms proper experimental conditions
    """)