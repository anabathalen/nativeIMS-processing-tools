"""
Fit Data - Refactored
======================

This page provides peak fitting functionality for CCSD data using the imspartacus package.
Supports multiple peak types, baseline correction, parameter constraints, and comprehensive
result analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path to import myutils
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import trapezoid
import warnings
warnings.filterwarnings('ignore')

# Import from imspartacus package
from imspartacus.fitting import (
    # Peak functions
    multi_peak_function,
    get_params_per_peak,
    get_parameter_names,
    # Classes
    PeakDetector,
    ParameterEstimator,
    ParameterManager,
    FittingEngine,
    DataProcessor,
    ResultAnalyzer,
    CCSDDataProcessor
)

from myutils import styling


# --- UI Components ---
class FitDataUI:
    """UI components for the fit data page."""
    
    @staticmethod
    def show_main_header():
        """Display main page header."""
        st.set_page_config(page_title="Fit Data", page_icon="📊", layout="wide")
        styling.load_custom_css()
        
        st.markdown("""
        <div class="main-header">
            <h1>📊 Peak Fitting Analysis</h1>
            <p>Perform comprehensive peak fitting on CCSD data with Origin-style analysis. 
            Supports multiple peak types, baseline correction, and parameter constraints.</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_peak_detection_controls():
        """Show peak detection controls."""
        with st.sidebar.expander("🔍 Peak Detection", expanded=True):
            st.markdown("""
            <div class="section-header">🔍 Peak Detection</div>
            """, unsafe_allow_html=True)
            
            auto_detect = st.checkbox("Auto-detect peaks", value=True)
            
            if auto_detect:
                st.markdown("**Detection Parameters**")
                min_height = st.slider("Min Height (%)", 1, 50, 5)
                min_prominence = st.slider("Min Prominence (%)", 1, 20, 2)
                min_distance = st.slider("Min Distance (%)", 1, 20, 5)
                smoothing = st.slider("Smoothing Points", 0, 20, 5)
                
                detection_params = {
                    'min_height_percent': min_height,
                    'min_prominence_percent': min_prominence,
                    'min_distance_percent': min_distance,
                    'smoothing_points': smoothing
                }
            else:
                detection_params = None
                
        return auto_detect, detection_params
    
    @staticmethod
    def show_fitting_options():
        """Show fitting options controls."""
        with st.sidebar.expander("⚙️ Fitting Options", expanded=True):
            st.markdown("""
            <div class="section-header">⚙️ Fitting Options</div>
            """, unsafe_allow_html=True)
            
            peak_type = st.selectbox(
                "Peak Type",
                ["Gaussian", "Lorentzian", "Voigt", "BiGaussian", "EMG"]
            )
            
            baseline_type = st.selectbox(
                "Baseline Correction",
                ["None", "Linear", "Polynomial", "Exponential"]
            )
            
            poly_degree = 2
            if baseline_type == "Polynomial":
                poly_degree = st.slider("Polynomial Degree", 2, 5, 2)
            
            fit_method = st.selectbox(
                "Fitting Method",
                ["Levenberg-Marquardt", "Global"]
            )
            
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
            
            tolerance = st.select_slider(
                "Tolerance",
                options=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
                value=1e-8,
                format_func=lambda x: f"{x:.0e}"
            )
            
            use_weights = st.checkbox("Use weighted fitting", value=False)
            
        return {
            'peak_type': peak_type,
            'baseline_type': baseline_type,
            'poly_degree': poly_degree,
            'fit_method': fit_method,
            'max_iterations': max_iterations,
            'tolerance': tolerance,
            'use_weights': use_weights
        }
    
    @staticmethod
    def show_preprocessing_options():
        """Show data preprocessing options."""
        with st.sidebar.expander("🔧 Preprocessing", expanded=False):
            st.markdown("""
            <div class="section-header">🔧 Preprocessing</div>
            """, unsafe_allow_html=True)
            
            smooth_data = st.checkbox("Smooth data", value=False)
            
            if smooth_data:
                smooth_method = st.selectbox(
                    "Smoothing Method",
                    ["Savitzky-Golay", "Moving Average"]
                )
                window_size = st.slider("Window Size", 3, 21, 5, step=2)
                poly_order = 2
                if smooth_method == "Savitzky-Golay":
                    poly_order = st.slider("Polynomial Order", 1, 5, 2)
            else:
                smooth_method = "Savitzky-Golay"
                window_size = 5
                poly_order = 2
                
        return {
            'smooth_data': smooth_data,
            'smooth_method': smooth_method,
            'window_size': window_size,
            'poly_order': poly_order
        }
    
    @staticmethod
    def display_peak_table(peak_info):
        """Display detected peaks in a table."""
        st.markdown("### 🎯 Detected Peaks")
        
        peak_data = []
        for i, peak in enumerate(peak_info):
            peak_data.append({
                'Peak': i + 1,
                'CCS': f"{peak['x']:.2f}",
                'Intensity': f"{peak['y']:.2f}",
                'Width (FWHM)': f"{peak.get('width_half', 0):.2f}",
                'Prominence': f"{peak.get('prominence', 0):.2f}",
                'Area (Est.)': f"{peak.get('area_estimate', 0):.2f}"
            })
        
        st.dataframe(pd.DataFrame(peak_data), use_container_width=True)
    
    @staticmethod
    def display_fit_statistics(result, peak_stats):
        """Display fitting statistics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R²", f"{result['r_squared']:.4f}")
        with col2:
            st.metric("Adjusted R²", f"{result['adj_r_squared']:.4f}")
        with col3:
            st.metric("RMSE", f"{result['rmse']:.4f}")
        with col4:
            st.metric("Reduced χ²", f"{result['reduced_chi_squared']:.4f}")
        
        # Additional statistics in expander
        with st.expander("📊 Additional Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**AIC:** {result['aic']:.2f}")
                st.write(f"**BIC:** {result['bic']:.2f}")
            with col2:
                st.write(f"**Reduced χ²:** {result['reduced_chi_squared']:.2f}")
                st.write(f"**Fit Success:** {'✅ Yes' if result['success'] else '❌ No'}")
        
        # Peak statistics table
        if peak_stats:
            st.markdown("""
            <div class="section-card">
                <div class="section-header">📈 Peak Statistics</div>
            </div>
            """, unsafe_allow_html=True)
            peak_df = pd.DataFrame(peak_stats)
            st.dataframe(peak_df, use_container_width=True)
    
    @staticmethod
    def create_fit_plot(x_data, y_data, result, baseline=None, show_components=True):
        """Create interactive plot of fit results."""
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name='Data',
            marker=dict(size=4, color='rgba(0,0,0,0.5)')
        ))
        
        # Baseline
        if baseline is not None:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=baseline,
                mode='lines',
                name='Baseline',
                line=dict(color='gray', dash='dash')
            ))
        
        # Fitted curve
        fig.add_trace(go.Scatter(
            x=x_data,
            y=result['fitted_curve'],
            mode='lines',
            name='Fitted Curve',
            line=dict(color='red', width=2)
        ))
        
        # Individual peak components
        if show_components and 'peak_components' in result:
            colors = px.colors.qualitative.Set2
            for i, component in enumerate(result['peak_components']):
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=component,
                    mode='lines',
                    name=f'Peak {i+1}',
                    line=dict(color=colors[i % len(colors)], dash='dot'),
                    opacity=0.7
                ))
        
        # Residuals (as separate subplot would be better, but keeping simple)
        fig.add_trace(go.Scatter(
            x=x_data,
            y=result['residuals'],
            mode='lines',
            name='Residuals',
            line=dict(color='green', width=1),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Peak Fitting Results',
            xaxis_title='CCS (Ų)',
            yaxis_title='Intensity',
            yaxis2=dict(
                title='Residuals',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.15
            )
        )
        
        return fig
    
    @staticmethod
    def show_parameter_editor(param_manager, x_data, y_data):
        """Show interactive parameter editor."""
        st.markdown("### 🎚️ Parameter Editor")
        
        param_names = get_parameter_names(param_manager.peak_type)
        
        # Create tabs for each peak
        n_peaks = param_manager.n_peaks
        if n_peaks == 0:
            st.info("No peaks to edit")
            return
        
        tabs = st.tabs([f"Peak {i+1}" for i in range(n_peaks)])
        
        for peak_idx, tab in enumerate(tabs):
            with tab:
                cols = st.columns(len(param_names) + 1)
                
                for param_idx, param_name in enumerate(param_names):
                    with cols[param_idx]:
                        global_idx = peak_idx * param_manager.params_per_peak + param_idx
                        current_value = param_manager.parameters[global_idx]
                        
                        # Parameter value input
                        new_value = st.number_input(
                            param_name,
                            value=float(current_value),
                            format="%.4f",
                            key=f"param_{peak_idx}_{param_idx}"
                        )
                        
                        # Update if changed
                        if new_value != current_value:
                            param_manager.update_parameter(peak_idx, param_idx, new_value)
                        
                        # Fix/unfix checkbox
                        is_fixed = param_manager.is_parameter_fixed(peak_idx, param_idx)
                        fixed = st.checkbox(
                            "Fix",
                            value=is_fixed,
                            key=f"fix_{peak_idx}_{param_idx}"
                        )
                        
                        if fixed != is_fixed:
                            param_manager.fix_parameter(peak_idx, param_idx, fixed)
                
                # Delete peak button
                with cols[-1]:
                    st.write("")  # Spacing
                    st.write("")
                    if st.button("🗑️ Delete", key=f"delete_{peak_idx}"):
                        param_manager.delete_peak(peak_idx)
                        st.rerun()


def perform_fitting(x_data, y_data, fitting_options, detection_params=None,
                   preprocessing_options=None, manual_peaks=None):
    """
    Perform complete fitting workflow.
    
    Parameters
    ----------
    x_data : ndarray
        CCS values
    y_data : ndarray
        Intensity values
    fitting_options : dict
        Fitting configuration options
    detection_params : dict, optional
        Peak detection parameters
    preprocessing_options : dict, optional
        Data preprocessing options
    manual_peaks : list, optional
        Manually specified peak positions
        
    Returns
    -------
    dict
        Complete fitting results including parameters, statistics, and components
    """
    # Initialize processor
    processor = DataProcessor()
    
    # Preprocessing
    if preprocessing_options and preprocessing_options['smooth_data']:
        y_processed = processor.smooth_data(
            x_data, y_data,
            method=preprocessing_options['smooth_method'],
            window_size=preprocessing_options['window_size'],
            poly_order=preprocessing_options['poly_order']
        )
    else:
        y_processed = y_data.copy()
    
    # Baseline subtraction
    y_corrected, baseline = processor.subtract_baseline(
        x_data, y_processed,
        method=fitting_options['baseline_type'],
        poly_degree=fitting_options.get('poly_degree', 2)
    )
    
    # Peak detection
    if detection_params:
        detector = PeakDetector()
        peak_info = detector.find_peaks_origin_style(
            x_data, y_corrected,
            min_height_percent=detection_params['min_height_percent'],
            min_prominence_percent=detection_params['min_prominence_percent'],
            min_distance_percent=detection_params['min_distance_percent'],
            smoothing_points=detection_params['smoothing_points']
        )
    elif manual_peaks:
        # Create peak info from manual positions
        peak_info = []
        for center in manual_peaks:
            center_idx = np.argmin(np.abs(x_data - center))
            peak_info.append({
                'index': center_idx,
                'x': center,
                'y': y_corrected[center_idx],
                'prominence': y_corrected[center_idx],
                'width_half': (x_data.max() - x_data.min()) / 20,
                'width_base': (x_data.max() - x_data.min()) / 10,
                'area_estimate': y_corrected[center_idx] * (x_data.max() - x_data.min()) / 20
            })
    else:
        st.error("Either enable auto-detection or specify manual peak positions")
        return None
    
    if not peak_info:
        st.warning("No peaks detected. Try adjusting detection parameters.")
        return None
    
    # Parameter estimation
    estimator = ParameterEstimator()
    initial_params = estimator.estimate_parameters(
        x_data, y_corrected, peak_info, fitting_options['peak_type']
    )
    
    # Create parameter manager
    x_range = (x_data.min(), x_data.max())
    param_manager = ParameterManager(
        fitting_options['peak_type'],
        initial_params,
        x_range
    )
    
    # Setup fitting engine
    engine = FittingEngine()
    engine.set_fitting_options(
        peak_type=fitting_options['peak_type'],
        baseline_type="None",  # Already subtracted
        fit_method=fitting_options['fit_method'],
        max_iterations=fitting_options['max_iterations'],
        tolerance=fitting_options['tolerance'],
        use_weights=fitting_options['use_weights']
    )
    engine.set_parameter_manager(param_manager)
    
    # Perform fitting
    weights = None
    if fitting_options['use_weights']:
        weights = processor.calculate_weights(y_corrected)
    
    result = engine.fit_peaks(x_data, y_corrected, initial_params, weights=weights)
    
    if not result['success']:
        st.error(f"Fitting failed: {result.get('message', 'Unknown error')}")
        return None
    
    # Analyze results
    analyzer = ResultAnalyzer()
    peak_stats = analyzer.calculate_peak_statistics(
        x_data,
        y_corrected,
        result['fitted_curve'],
        result['parameters'],
        fitting_options['peak_type']
    )
    
    # Add peak components to result
    n_peaks = len(peak_info)
    params_per_peak = get_params_per_peak(fitting_options['peak_type'])
    peak_components = []
    
    for i in range(n_peaks):
        start_idx = i * params_per_peak
        end_idx = (i + 1) * params_per_peak
        peak_params = result['parameters'][start_idx:end_idx]
        
        component = multi_peak_function(
            x_data,
            fitting_options['peak_type'],
            *peak_params
        )
        peak_components.append(component)
    
    result['peak_components'] = peak_components
    result['baseline'] = baseline
    result['peak_info'] = peak_info
    result['y_corrected'] = y_corrected
    
    return result, param_manager, peak_stats


def main():
    """Main Streamlit application."""
    FitDataUI.show_main_header()
    
    # Initialize session state
    if 'all_charge_results' not in st.session_state:
        st.session_state['all_charge_results'] = {}
    if 'parameter_manager' not in st.session_state:
        st.session_state['parameter_manager'] = None
    
    # File upload
    st.markdown("""
    <div class="section-card">
        <div class="section-header">📁 Step 1: Upload Calibrated CSV File</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload calibrated CSV file", type=['csv'])
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to get started")
        st.markdown("""
        ### Required CSV Format:
        - **Charge**: Charge state values
        - **CCS**: Collision Cross Section values
        - **Scaled_Intensity**: Intensity values
        - **Drift**: Drift time values (for export)
        - **m/z**: Mass-to-charge ratio (for export)
        """)
        return
    
    try:
        # Load and validate data
        df = pd.read_csv(uploaded_file)
        
        # Ensure numeric types and clean data
        if 'CCS' in df.columns:
            df['CCS'] = pd.to_numeric(df['CCS'].astype(str).str.replace(',', ''), errors='coerce')
        if 'Scaled_Intensity' in df.columns:
            df['Scaled_Intensity'] = pd.to_numeric(df['Scaled_Intensity'], errors='coerce')
        if 'Charge' in df.columns:
            df['Charge'] = pd.to_numeric(df['Charge'], errors='coerce', downcast='integer')
        
        # Drop invalid rows
        df = df.dropna(subset=['Charge', 'CCS', 'Scaled_Intensity'])
        
        required_cols = ['Charge', 'CCS', 'Scaled_Intensity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return
        
        if df.empty:
            st.error("The uploaded file is empty")
            return
        
        # CCS range controls
        ccs_min_raw = float(df['CCS'].min())
        ccs_max_raw = float(df['CCS'].max())
        
        with st.sidebar.expander("🧪 CCS Range / Export Settings", expanded=True):
            if 'fit_ccs_min' not in st.session_state:
                st.session_state['fit_ccs_min'] = ccs_min_raw
            if 'fit_ccs_max' not in st.session_state:
                st.session_state['fit_ccs_max'] = ccs_max_raw
            
            fit_ccs_min = st.number_input(
                "Min CCS to fit",
                min_value=ccs_min_raw,
                max_value=ccs_max_raw,
                value=float(st.session_state['fit_ccs_min']),
                step=max((ccs_max_raw - ccs_min_raw) / 1000, 0.01)
            )
            fit_ccs_max = st.number_input(
                "Max CCS to fit",
                min_value=fit_ccs_min,
                max_value=ccs_max_raw,
                value=float(st.session_state['fit_ccs_max']),
                step=max((ccs_max_raw - ccs_min_raw) / 1000, 0.01)
            )
            
            st.session_state['fit_ccs_min'] = fit_ccs_min
            st.session_state['fit_ccs_max'] = fit_ccs_max
            
            export_points = st.number_input(
                "Export points per charge",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Number of evenly spaced CCS points for each charge in exported fitted data",
                key="export_points_per_charge"
            )
        
        # Apply CCS cropping
        df = df[(df['CCS'] >= st.session_state['fit_ccs_min']) & 
                (df['CCS'] <= st.session_state['fit_ccs_max'])]
        
        if df.empty:
            st.error("No data remain after applying the selected CCS range.")
            return
        
        # Sidebar controls
        st.sidebar.markdown("### 🎛️ Analysis Configuration")
        
        charges = sorted(df['Charge'].unique())
        mode = st.sidebar.radio("Analysis Mode", ["Individual Charge State", "Summed Data"])
        
        # Show saved results summary
        if st.session_state['all_charge_results']:
            st.sidebar.markdown("### 💾 Saved Results")
            for charge in sorted(st.session_state['all_charge_results'].keys()):
                result_data = st.session_state['all_charge_results'][charge]
                n_peaks = len(result_data['peak_stats'])
                r_squared = result_data['fit_result']['r_squared']
                st.sidebar.markdown(f"**Charge {charge}**: {n_peaks} peaks (R² = {r_squared:.3f})")
            
            # Clear all results button
            if st.sidebar.button("🗑️ Clear All Saved Results"):
                st.session_state['all_charge_results'] = {}
                st.rerun()
        
        # Get analysis data
        if mode == "Individual Charge State":
            selected_charge = st.sidebar.selectbox("Select Charge State", charges)
            plot_data = df[df['Charge'] == selected_charge].copy().sort_values('CCS')
            data_label = f"Charge {selected_charge}"
        else:
            plot_data = CCSDDataProcessor.create_summed_data(df)
            data_label = "Summed Data"
            selected_charge = None
        
        plot_data = plot_data[plot_data['Scaled_Intensity'] > 0]
        if len(plot_data) == 0:
            st.error("No data points with positive intensity found")
            return
        
        x_data = plot_data['CCS'].values
        y_data = plot_data['Scaled_Intensity'].values
        
        # Get UI options
        auto_detect, detection_params = FitDataUI.show_peak_detection_controls()
        fitting_options = FitDataUI.show_fitting_options()
        preprocessing_options = FitDataUI.show_preprocessing_options()
        
        # Store options in session state
        st.session_state['fitting_options'] = fitting_options
        st.session_state['data_label'] = data_label
        
        # Main content area
        st.markdown("""
        <div class="section-card">
            <div class="section-header">📊 Peak Fitting</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Analysis: {data_label}**")
        
        # Perform fitting button
        if st.button("🚀 Perform Fitting", type="primary"):
            with st.spinner("Fitting peaks..."):
                result = perform_fitting(
                    x_data, y_data,
                    fitting_options,
                    detection_params=detection_params if auto_detect else None,
                    preprocessing_options=preprocessing_options
                )
                
                if result:
                    fit_result, param_manager, peak_stats = result
                    st.session_state['fit_result'] = fit_result
                    st.session_state['parameter_manager'] = param_manager
                    st.session_state['peak_stats'] = peak_stats
                    st.success("✅ Fitting completed successfully!")
        
        # Display results if available
        if 'fit_result' in st.session_state:
            result = st.session_state['fit_result']
            peak_stats = st.session_state.get('peak_stats', [])
            
            # Display statistics
            st.markdown("""
            <div class="section-card">
                <div class="section-header">📊 Fit Statistics</div>
            </div>
            """, unsafe_allow_html=True)
            FitDataUI.display_fit_statistics(result, peak_stats)
            
            # Display plot
            st.markdown("""
            <div class="section-card">
                <div class="section-header">📈 Fit Visualization</div>
            </div>
            """, unsafe_allow_html=True)
            show_components = st.checkbox("Show individual peak components", value=True)
            fig = FitDataUI.create_fit_plot(
                x_data, y_data, result,
                baseline=result.get('baseline'),
                show_components=show_components
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Parameter editor
            if st.session_state['parameter_manager']:
                with st.expander("🎚️ Edit Parameters & Re-fit"):
                    FitDataUI.show_parameter_editor(
                        st.session_state['parameter_manager'],
                        x_data,
                        y_data
                    )
                    
                    if st.button("🔄 Re-fit with Edited Parameters"):
                        with st.spinner("Re-fitting..."):
                            # Get updated parameters
                            param_manager = st.session_state['parameter_manager']
                            
                            # Setup engine
                            engine = FittingEngine()
                            engine.set_fitting_options(
                                peak_type=fitting_options['peak_type'],
                                baseline_type="None",
                                fit_method=fitting_options['fit_method'],
                                max_iterations=fitting_options['max_iterations'],
                                tolerance=fitting_options['tolerance'],
                                use_weights=fitting_options['use_weights']
                            )
                            engine.set_parameter_manager(param_manager)
                            
                            # Re-fit
                            weights = None
                            if fitting_options['use_weights']:
                                processor = DataProcessor()
                                weights = processor.calculate_weights(result['y_corrected'])
                            
                            new_result = engine.fit_peaks(
                                x_data,
                                result['y_corrected'],
                                param_manager.parameters,
                                weights=weights
                            )
                            
                            if new_result['success']:
                                # Update result
                                analyzer = ResultAnalyzer()
                                peak_stats = analyzer.calculate_peak_statistics(
                                    x_data,
                                    result['y_corrected'],
                                    new_result['fitted_curve'],
                                    new_result['parameters'],
                                    fitting_options['peak_type']
                                )
                                
                                # Add components
                                n_peaks = param_manager.n_peaks
                                params_per_peak = get_params_per_peak(fitting_options['peak_type'])
                                peak_components = []
                                
                                for i in range(n_peaks):
                                    start_idx = i * params_per_peak
                                    end_idx = (i + 1) * params_per_peak
                                    peak_params = new_result['parameters'][start_idx:end_idx]
                                    
                                    component = multi_peak_function(
                                        x_data,
                                        fitting_options['peak_type'],
                                        *peak_params
                                    )
                                    peak_components.append(component)
                                
                                new_result['peak_components'] = peak_components
                                new_result['baseline'] = result['baseline']
                                new_result['peak_info'] = result['peak_info']
                                new_result['y_corrected'] = result['y_corrected']
                                
                                st.session_state['fit_result'] = new_result
                                st.session_state['peak_stats'] = peak_stats
                                st.success("✅ Re-fitting completed!")
                                st.rerun()
                            else:
                                st.error("Re-fitting failed")
            
            # Save results for charge state
            if mode == "Individual Charge State":
                if st.sidebar.button(f"💾 Save Results for Charge {selected_charge}", type="primary"):
                    st.session_state['all_charge_results'][selected_charge] = {
                        'fit_result': st.session_state['fit_result'].copy(),
                        'peak_stats': st.session_state.get('peak_stats', []),
                        'fitting_options': st.session_state['fitting_options'].copy(),
                        'parameter_manager': st.session_state['parameter_manager'],
                        'data_info': {
                            'charge': selected_charge,
                            'n_points': len(plot_data),
                            'ccs_range': (plot_data['CCS'].min(), plot_data['CCS'].max())
                        }
                    }
                    st.sidebar.success(f"✅ Saved results for Charge {selected_charge}")
                    st.rerun()
            
            # Export options
            st.markdown("""
            <div class="section-card">
                <div class="section-header">💾 Export Results</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export parameters
                params_per_peak = get_params_per_peak(fitting_options['peak_type'])
                param_names = get_parameter_names(fitting_options['peak_type'])
                
                param_data = []
                for i in range(len(peak_stats)):
                    peak_params = {
                        'Peak': i + 1
                    }
                    for j, param_name in enumerate(param_names):
                        peak_params[param_name] = result['parameters'][i * params_per_peak + j]
                    param_data.append(peak_params)
                
                param_df = pd.DataFrame(param_data)
                csv_params = param_df.to_csv(index=False)
                st.download_button(
                    "📊 Download Parameters",
                    csv_params,
                    f"fit_parameters_{data_label.lower().replace(' ', '_')}.csv",
                    "text/csv"
                )
            
            with col2:
                # Export fit data
                fit_data = pd.DataFrame({
                    'CCS': x_data,
                    'Original': y_data,
                    'Fitted': result['fitted_curve'],
                    'Residuals': result['residuals']
                })
                
                if result.get('baseline') is not None:
                    fit_data['Baseline'] = result['baseline']
                
                csv_fit = fit_data.to_csv(index=False)
                st.download_button(
                    "📈 Download Fit Data",
                    csv_fit,
                    f"fit_data_{data_label.lower().replace(' ', '_')}.csv",
                    "text/csv"
                )
            
            with col3:
                # Export high-resolution fit
                export_points_target = int(st.session_state.get('export_points_per_charge', 1000))
                x_hr = np.linspace(x_data.min(), x_data.max(), max(len(x_data), export_points_target))
                y_hr = multi_peak_function(x_hr, fitting_options['peak_type'], *result['parameters'])
                
                hr_df = pd.DataFrame({
                    'CCS': x_hr,
                    'Fitted': y_hr
                })
                csv_fit_hr = hr_df.to_csv(index=False)
                st.download_button(
                    "📈 Download High-Res Fit",
                    csv_fit_hr,
                    f"fit_highres_{data_label.lower().replace(' ', '_')}.csv",
                    "text/csv"
                )
        
        # Display detected peaks if available
        if 'fit_result' in st.session_state and 'peak_info' in st.session_state['fit_result']:
            with st.expander("🎯 Detected Peaks"):
                FitDataUI.display_peak_table(st.session_state['fit_result']['peak_info'])
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
