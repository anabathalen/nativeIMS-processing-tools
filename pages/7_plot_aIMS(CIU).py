import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import seaborn as sns
from io import BytesIO
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import styling utilities if available
try:
    from myutils import styling
except ImportError:
    styling = None

@dataclass
class ORIGAMISettings:
    """ORIGAMI-style configuration settings"""
    # Grid and interpolation
    grid_resolution: int = 200
    interpolation_method: str = 'cubic'
    
    # Normalization and processing
    normalize_data: bool = True
    normalization_mode: str = 'Maximum'  # Maximum, Logarithmic, Natural log, Square root
    
    # Smoothing
    apply_smoothing: bool = False
    smoothing_type: str = 'gaussian'  # gaussian, savgol
    gaussian_sigma: float = 2.0
    savgol_window: int = 11
    savgol_polyorder: int = 3
    
    # Noise reduction
    noise_threshold: float = 0.0
    apply_intensity_threshold: bool = False
    intensity_min_threshold: float = 0.0
    intensity_max_threshold: float = 1.0
    
    # Plot appearance
    colormap: str = 'viridis'
    show_colorbar: bool = True
    colorbar_position: str = 'right'
    colorbar_width: float = 5.0
    colorbar_pad: float = 0.05
    
    # Labels and fonts
    font_size: int = 12
    label_weight: bool = False
    tick_size: int = 10
    
    # Figure
    figure_size: float = 8.0
    dpi: int = 300

class ORIGAMIDataProcessor:
    """Exact replication of ORIGAMI's data processing methods"""
    
    @staticmethod
    def load_twim_extract(file) -> pd.DataFrame:
        """Load TWIMExtract file with ORIGAMI-style parsing"""
        try:
            file.seek(0)
            content = file.read().decode('utf-8').strip()
            lines = content.split('\n')
            
            # Find collision voltage line
            cv_line_idx = None
            for i, line in enumerate(lines):
                if line.startswith('$TrapCV:'):
                    cv_line_idx = i
                    break
            
            if cv_line_idx is None:
                raise ValueError("Could not find $TrapCV: line in TWIMExtract file")
            
            # Extract collision voltages
            cv_line = lines[cv_line_idx]
            cv_values = [float(x.strip()) for x in cv_line.split(',')[1:] if x.strip()]
            
            # Parse data starting after CV line
            data_lines = lines[cv_line_idx + 1:]
            data_rows = []
            
            for line in data_lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= len(cv_values) + 1:
                        try:
                            drift_time = float(parts[0])
                            intensities = []
                            for i in range(1, len(cv_values) + 1):
                                try:
                                    intensity = float(parts[i]) if parts[i].strip() else 0.0
                                    intensities.append(intensity)
                                except (ValueError, IndexError):
                                    intensities.append(0.0)
                            data_rows.append([drift_time] + intensities)
                        except (ValueError, IndexError):
                            continue
            
            if not data_rows:
                raise ValueError("No valid data found in TWIMExtract file")
            
            # Create DataFrame
            columns = ['Drift Time (ms)'] + [f'CV_{cv:.0f}V' for cv in cv_values]
            df = pd.DataFrame(data_rows, columns=columns)
            
            # Clean data
            df = df.dropna(subset=['Drift Time (ms)'])
            df = df[df['Drift Time (ms)'] > 0]
            
            # Store CV values as metadata
            df.cv_values = cv_values
            
            st.success(f"✅ Loaded TWIMExtract: {len(df)} drift time points, {len(cv_values)} CV values")
            st.info(f"📏 Drift time range: {df['Drift Time (ms)'].min():.2f} - {df['Drift Time (ms)'].max():.2f} ms")
            st.info(f"⚡ CV range: {min(cv_values):.0f} - {max(cv_values):.0f} V")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading TWIMExtract file: {str(e)}")
    
    @staticmethod
    def validate_and_clean_calibration(cal_data: pd.DataFrame) -> pd.DataFrame:
        """Validate calibration data for physical consistency"""
        original_count = len(cal_data)
        
        # Sort by drift time
        cal_data = cal_data.sort_values("Drift").reset_index(drop=True)
        
        # Find minimum drift time and corresponding CCS
        min_drift_idx = cal_data["Drift"].idxmin()
        min_drift_ccs = cal_data.loc[min_drift_idx, "CCS"]
        
        # Check if any points have larger CCS with smaller drift times
        # This is physically inconsistent - smaller drift times should give smaller CCS
        problematic_mask = (cal_data["Drift"] <= cal_data.loc[min_drift_idx, "Drift"]) & (cal_data["CCS"] > min_drift_ccs)
        
        if problematic_mask.any():
            problematic_count = problematic_mask.sum()
            st.warning(f"🔍 Found {problematic_count} physically inconsistent calibration points")
            
            # Show details of problematic points
            problematic_points = cal_data[problematic_mask]
            st.info("❌ Removing points with CCS > min_CCS but Drift ≤ min_Drift:")
            for _, row in problematic_points.iterrows():
                st.info(f"   • Drift: {row['Drift']:.3f} ms, CCS: {row['CCS']:.1f} Å² (vs min: {min_drift_ccs:.1f} Å²)")
            
            # Remove problematic points
            cal_data = cal_data[~problematic_mask].reset_index(drop=True)
        
        # Additional check: ensure monotonic relationship (optional strict check)
        # Remove any points where CCS decreases with increasing drift time
        drift_sorted = cal_data.sort_values("Drift")
        ccs_values = drift_sorted["CCS"].values
        
        # Find where CCS decreases (non-monotonic behavior)
        non_monotonic_mask = np.diff(ccs_values) < 0
        
        if non_monotonic_mask.any():
            non_monotonic_count = non_monotonic_mask.sum()
            st.warning(f"🔍 Found {non_monotonic_count} non-monotonic transitions in calibration")
            
            # Keep only points that maintain monotonic increase
            keep_indices = [0]  # Always keep first point
            for i in range(1, len(drift_sorted)):
                if drift_sorted.iloc[i]["CCS"] >= drift_sorted.iloc[keep_indices[-1]]["CCS"]:
                    keep_indices.append(i)
            
            cal_data = drift_sorted.iloc[keep_indices].reset_index(drop=True)
            removed_non_monotonic = len(drift_sorted) - len(keep_indices)
            if removed_non_monotonic > 0:
                st.info(f"🧹 Removed {removed_non_monotonic} points to ensure monotonic CCS vs Drift relationship")
        
        # Final validation
        if len(cal_data) < 3:
            raise ValueError("Insufficient calibration points remaining after validation (need at least 3)")
        
        # Check final relationship
        correlation = cal_data["Drift"].corr(cal_data["CCS"])
        if correlation < 0.8:
            st.warning(f"⚠️ Low correlation between Drift and CCS (r={correlation:.3f}). Check calibration data quality.")
        
        removed_count = original_count - len(cal_data)
        if removed_count > 0:
            st.success(f"✅ Calibration validation complete: removed {removed_count}/{original_count} inconsistent points")
            st.info(f"📊 Final calibration: {len(cal_data)} points, correlation r={correlation:.3f}")
        else:
            st.success(f"✅ Calibration validation complete: all {original_count} points are physically consistent")
        
        return cal_data
    
    @staticmethod
    def load_calibration_data(file, charge_state: int) -> pd.DataFrame:
        """Load calibration data with ORIGAMI-style filtering and validation"""
        try:
            file.seek(0)
            cal_df = pd.read_csv(file)
            
            # Check required columns
            required_cols = ["Z", "Drift", "CCS"]
            missing_cols = [col for col in required_cols if col not in cal_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Filter by charge state
            cal_data = cal_df[cal_df["Z"] == charge_state].copy()
            
            if cal_data.empty:
                available_z = sorted(cal_df["Z"].unique())
                raise ValueError(f"No data for Z={charge_state}. Available: {available_z}")
            
            # Convert drift times to milliseconds if needed
            if cal_data["Drift"].max() < 1:  # Assuming seconds if max < 1
                cal_data["Drift"] = cal_data["Drift"] * 1000
                st.info("🔄 Converted drift times from seconds to milliseconds")
            
            st.info(f"📥 Raw calibration: {len(cal_data)} points for Z={charge_state}")
            st.info(f"📏 Raw drift range: {cal_data['Drift'].min():.2f} - {cal_data['Drift'].max():.2f} ms")
            st.info(f"🎯 Raw CCS range: {cal_data['CCS'].min():.1f} - {cal_data['CCS'].max():.1f} Å²")
            
            # Validate and clean calibration data
            cal_data = ORIGAMIDataProcessor.validate_and_clean_calibration(cal_data)
            
            # Sort final data
            cal_data = cal_data.sort_values("Drift").reset_index(drop=True)
            
            st.success(f"✅ Final calibration: {len(cal_data)} validated points")
            st.info(f"📏 Final drift range: {cal_data['Drift'].min():.2f} - {cal_data['Drift'].max():.2f} ms")
            st.info(f"🎯 Final CCS range: {cal_data['CCS'].min():.1f} - {cal_data['CCS'].max():.1f} Å²")
            
            return cal_data
            
        except Exception as e:
            raise ValueError(f"Error loading calibration data: {str(e)}")
    
    @staticmethod
    def calibrate_twim_data(twim_df: pd.DataFrame, cal_data: pd.DataFrame, 
                           inject_time: float = 0.0) -> np.ndarray:
        """Calibrate TWIM data using ORIGAMI-style interpolation"""
        calibrated_data = []
        
        drift_times = twim_df['Drift Time (ms)'].values
        cv_values = twim_df.cv_values
        
        # Apply injection time correction for cyclic data
        corrected_drift_times = drift_times - inject_time
        
        # Filter valid drift times
        valid_mask = corrected_drift_times > 0
        corrected_drift_times = corrected_drift_times[valid_mask]
        
        # Check calibration range
        cal_min, cal_max = cal_data['Drift'].min(), cal_data['Drift'].max()
        
        st.info(f"🔧 Processing {len(corrected_drift_times)} valid drift time points")
        st.info(f"📏 Calibration range: {cal_min:.2f} - {cal_max:.2f} ms")
        
        # Count points outside calibration range
        outside_range = ((corrected_drift_times < cal_min) | (corrected_drift_times > cal_max)).sum()
        if outside_range > 0:
            st.warning(f"⚠️ {outside_range} drift time points outside calibration range will be skipped")
        
        valid_points = 0
        for i, dt in enumerate(drift_times):
            if not valid_mask[i]:
                continue
                
            corrected_dt = corrected_drift_times[valid_points]
            
            # Skip if outside calibration range
            if corrected_dt < cal_min or corrected_dt > cal_max:
                valid_points += 1
                continue
            
            # Interpolate CCS value using validated calibration
            ccs_value = np.interp(corrected_dt, cal_data['Drift'].values, cal_data['CCS'].values)
            
            # Get intensities for this drift time
            intensities = twim_df.iloc[i, 1:].values
            
            # Add data point for each CV
            for j, (cv, intensity) in enumerate(zip(cv_values, intensities)):
                if not pd.isna(intensity) and intensity > 0:  # Only include positive intensities
                    calibrated_data.append([ccs_value, corrected_dt, cv, intensity])
            
            valid_points += 1
        
        if not calibrated_data:
            raise ValueError("No valid calibrated data points generated")
        
        result = np.array(calibrated_data)
        st.success(f"✅ Calibrated {len(result):,} data points")
        st.info(f"🎯 CCS range: {result[:, 0].min():.1f} - {result[:, 0].max():.1f} Å²")
        st.info(f"⚡ CV range: {result[:, 2].min():.1f} - {result[:, 2].max():.1f} V")
        st.info(f"💡 Intensity range: {result[:, 3].min():.0f} - {result[:, 3].max():.0f}")
        
        return result

class ORIGAMIVisualizer:
    """ORIGAMI-style visualization methods"""
    
    @staticmethod
    def normalize_2D(input_data: np.ndarray, mode: str = 'Maximum') -> np.ndarray:
        """Exact replication of ORIGAMI's normalize_2D function"""
        input_data = np.nan_to_num(input_data)
        input_data[input_data < 0] = 0
        
        if mode == "Maximum":
            # Column-wise normalization (each CV column to its maximum)
            norm_data = input_data.copy().astype(np.float64)
            for col in range(norm_data.shape[1]):
                col_max = np.max(norm_data[:, col])
                if col_max > 0:
                    norm_data[:, col] = norm_data[:, col] / col_max
                    
        elif mode == 'Logarithmic':
            norm_data = np.log10(np.clip(input_data.astype(np.float64), a_min=1e-10, a_max=None))
            
        elif mode == 'Natural log':
            norm_data = np.log(np.clip(input_data.astype(np.float64), a_min=1e-10, a_max=None))
            
        elif mode == 'Square root':
            norm_data = np.sqrt(np.clip(input_data.astype(np.float64), a_min=0, a_max=None))
        
        else:
            norm_data = input_data.copy()
        
        norm_data[norm_data < 0] = 0
        return norm_data
    
    @staticmethod
    def smooth_gaussian_2D(input_data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """ORIGAMI's Gaussian smoothing"""
        if input_data is None or len(input_data) == 0:
            return input_data
        
        if sigma < 0:
            sigma = 1
            st.warning("Sigma too low, reset to 1")
        
        data_out = gaussian_filter(input_data, sigma=sigma, order=0)
        data_out[data_out < 0] = 0
        return data_out
    
    @staticmethod
    def smooth_savgol_2D(input_data: np.ndarray, poly_order: int = 2, window_size: int = 5) -> np.ndarray:
        """ORIGAMI's Savitzky-Golay smoothing"""
        if input_data is None or len(input_data) == 0:
            return input_data
        
        if poly_order <= 0:
            poly_order = 2
            st.warning("Polynomial order too small, reset to 2")
        
        if window_size <= poly_order:
            window_size = poly_order + 1
            st.warning(f"Window size too small, reset to {window_size}")
        
        if window_size % 2 == 0:
            window_size += 1
            st.info("Window size was even, made odd")
        
        data_out = savgol_filter(input_data, polyorder=poly_order, window_length=window_size, axis=0)
        data_out[data_out < 0] = 0
        return data_out
    
    @staticmethod
    def remove_noise_2D(input_data: np.ndarray, threshold: float = 0) -> np.ndarray:
        """ORIGAMI's noise removal"""
        if threshold > np.max(input_data) or threshold < 0:
            st.warning(f"Threshold too high (max={np.max(input_data):.0f}), reset to 0")
            threshold = 0
        
        input_data[input_data <= threshold] = 0
        return input_data
    
    @staticmethod
    def adjust_min_max_intensity(input_data: np.ndarray, min_threshold: float = 0.0, 
                                max_threshold: float = 1.0) -> np.ndarray:
        """ORIGAMI's intensity thresholding"""
        if min_threshold > max_threshold:
            st.warning("Min > Max threshold, swapping values")
            min_threshold, max_threshold = max_threshold, min_threshold
        
        if min_threshold == max_threshold:
            st.warning("Min and Max thresholds are equal")
            return input_data
        
        data_max = np.max(input_data)
        min_thresh = min_threshold * data_max
        max_thresh = max_threshold * data_max
        
        input_data[input_data <= min_thresh] = 0
        input_data[input_data >= max_thresh] = data_max
        
        return input_data
    
    @staticmethod
    def create_2D_grid(data: np.ndarray, cv_range: Tuple[float, float], 
                      ccs_range: Tuple[float, float], resolution: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create interpolated 2D grid using ORIGAMI method"""
        # Extract data columns: CCS, Drift, CV, Intensity
        cv_vals = data[:, 2]
        ccs_vals = data[:, 0]
        intensity_vals = data[:, 3]
        
        # Create regular grid
        cv_grid = np.linspace(cv_range[0], cv_range[1], resolution)
        ccs_grid = np.linspace(ccs_range[0], ccs_range[1], resolution)
        CV_grid, CCS_grid = np.meshgrid(cv_grid, ccs_grid)
        
        # Interpolate intensity onto grid
        try:
            Z = griddata(
                (cv_vals, ccs_vals), 
                intensity_vals, 
                (CV_grid, CCS_grid), 
                method='cubic', 
                fill_value=0
            )
        except:
            st.warning("Cubic interpolation failed, using linear")
            Z = griddata(
                (cv_vals, ccs_vals), 
                intensity_vals, 
                (CV_grid, CCS_grid), 
                method='linear', 
                fill_value=0
            )
        
        Z = np.nan_to_num(Z, nan=0.0)
        Z[Z < 0] = 0
        
        return CV_grid, CCS_grid, Z
    
    def create_ciu_heatmap(self, data: np.ndarray, settings: ORIGAMISettings) -> plt.Figure:
        """Create CIU heatmap using ORIGAMI methods"""
        # Determine data ranges
        cv_range = (data[:, 2].min(), data[:, 2].max())
        ccs_range = (data[:, 0].min(), data[:, 0].max())
        
        st.info(f"📊 Creating CIU heatmap:")
        st.info(f"   • CV range: {cv_range[0]:.1f} - {cv_range[1]:.1f} V")
        st.info(f"   • CCS range: {ccs_range[0]:.1f} - {ccs_range[1]:.1f} Å²")
        st.info(f"   • Data points: {len(data):,}")
        
        # Create interpolated grid
        X, Y, Z = self.create_2D_grid(data, cv_range, ccs_range, settings.grid_resolution)
        
        # Apply ORIGAMI processing
        if settings.normalize_data:
            Z = self.normalize_2D(Z, mode=settings.normalization_mode)
            st.info(f"✅ Applied {settings.normalization_mode} normalization")
        
        if settings.apply_smoothing:
            if settings.smoothing_type == 'gaussian':
                Z = self.smooth_gaussian_2D(Z, sigma=settings.gaussian_sigma)
                st.info(f"✅ Applied Gaussian smoothing (σ={settings.gaussian_sigma})")
            else:
                Z = self.smooth_savgol_2D(Z, poly_order=settings.savgol_polyorder, 
                                        window_size=settings.savgol_window)
                st.info(f"✅ Applied Savitzky-Golay smoothing")
        
        if settings.noise_threshold > 0:
            Z = self.remove_noise_2D(Z, threshold=settings.noise_threshold)
            st.info(f"✅ Removed noise below {settings.noise_threshold}")
        
        if settings.apply_intensity_threshold:
            Z = self.adjust_min_max_intensity(Z, settings.intensity_min_threshold, 
                                            settings.intensity_max_threshold)
            st.info("✅ Applied intensity thresholding")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(settings.figure_size, settings.figure_size), 
                              dpi=settings.dpi)
        fig.patch.set_facecolor('white')
        
        # Plot heatmap
        im = ax.pcolormesh(X, Y, Z, cmap=settings.colormap, shading='auto')
        
        # Set labels and formatting
        ax.set_xlabel('Collision Voltage (V)', fontsize=settings.font_size, 
                     fontweight='bold' if settings.label_weight else 'normal')
        ax.set_ylabel('CCS (Å²)', fontsize=settings.font_size, 
                     fontweight='bold' if settings.label_weight else 'normal')
        
        ax.tick_params(labelsize=settings.tick_size)
        
        # Add colorbar
        if settings.show_colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=settings.colorbar_pad)
            intensity_label = "Normalized Intensity" if settings.normalize_data else "Intensity"
            cbar.set_label(intensity_label, fontsize=settings.font_size)
            cbar.ax.tick_params(labelsize=settings.tick_size)
        
        plt.tight_layout()
        
        st.success("✅ CIU heatmap created successfully")
        return fig
    
    def create_stacked_plot(self, data: np.ndarray, settings: ORIGAMISettings) -> plt.Figure:
        """Create stacked CCS distribution plot"""
        # Get unique CV values
        unique_cvs = np.unique(data[:, 2])
        unique_cvs = unique_cvs[::2]  # Take every other CV for clarity
        
        fig, ax = plt.subplots(figsize=(settings.figure_size, settings.figure_size * 1.2), 
                              dpi=settings.dpi)
        fig.patch.set_facecolor('white')
        
        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_cvs)))
        
        offset = 0
        max_intensity = 0
        
        for i, cv in enumerate(unique_cvs):
            # Extract data for this CV
            cv_mask = np.abs(data[:, 2] - cv) < 0.5  # Small tolerance for CV matching
            cv_data = data[cv_mask]
            
            if len(cv_data) == 0:
                continue
            
            # Sort by CCS
            sort_idx = np.argsort(cv_data[:, 0])
            ccs_vals = cv_data[sort_idx, 0]
            intensities = cv_data[sort_idx, 3]
            
            # Normalize intensities
            if np.max(intensities) > 0:
                intensities = intensities / np.max(intensities)
            
            max_intensity = max(max_intensity, np.max(intensities))
            
            # Plot trace
            y_vals = intensities + offset
            ax.plot(ccs_vals, y_vals, color=colors[i], linewidth=1.5, label=f'{cv:.0f}V')
            ax.fill_between(ccs_vals, offset, y_vals, color=colors[i], alpha=0.3)
            
            offset += 1.5  # Spacing between traces
        
        ax.set_xlabel('CCS (Å²)', fontsize=settings.font_size)
        ax.set_ylabel('Collision Voltage (V)', fontsize=settings.font_size)
        ax.set_title('Stacked CCS Distributions', fontsize=settings.font_size + 2)
        
        plt.tight_layout()
        return fig

class ORIGAMIInterface:
    """Streamlit interface for ORIGAMI-style CIU analysis"""
    
    def __init__(self):
        self.processor = ORIGAMIDataProcessor()
        self.visualizer = ORIGAMIVisualizer()
    
    def show_header(self):
        """Display application header"""
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;">
            <h1 style="color: white; margin: 0;">🔬 ORIGAMI-Style CIU Analysis</h1>
            <p style="color: white; margin: 10px 0 0 0; font-size: 18px;">Process TWIMExtract files with ORIGAMI methods</p>
        </div>
        """, unsafe_allow_html=True)
    
    def file_upload_section(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Handle file uploads"""
        st.header("📁 File Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("TWIMExtract Data")
            twim_file = st.file_uploader(
                "Upload TWIMExtract CSV", 
                type="csv", 
                help="TWIMExtract output file with collision voltage data"
            )
        
        with col2:
            st.subheader("Calibration Data")
            cal_file = st.file_uploader(
                "Upload Calibration CSV", 
                type="csv", 
                help="Calibration file with Z, Drift, CCS columns"
            )
        
        return twim_file, cal_file
    
    def processing_settings_section(self) -> Dict[str, Any]:
        """Get processing settings"""
        st.header("⚙️ Processing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Parameters")
            charge_state = st.number_input("Charge State (Z)", min_value=1, max_value=50, value=10)
            instrument_type = st.selectbox("Instrument Type", ["Synapt", "Cyclic IMS"])
            
            inject_time = 0.0
            if instrument_type == "Cyclic IMS":
                inject_time = st.number_input("Injection Time (ms)", min_value=0.0, value=0.0, step=0.1)
        
        with col2:
            st.subheader("Grid Settings")
            grid_resolution = st.number_input("Grid Resolution", min_value=50, max_value=500, value=200)
            interpolation = st.selectbox("Interpolation Method", ["cubic", "linear", "nearest"])
        
        return {
            "charge_state": charge_state,
            "instrument_type": instrument_type,
            "inject_time": inject_time,
            "grid_resolution": grid_resolution,
            "interpolation_method": interpolation
        }
    
    def visualization_settings_section(self) -> ORIGAMISettings:
        """Get visualization settings"""
        st.header("🎨 Visualization Settings")
        
        # Create tabs for different setting categories
        tab1, tab2, tab3, tab4 = st.tabs(["🎨 Appearance", "📐 Processing", "🔧 Advanced", "📊 Plot Type"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Colors & Style")
                colormap = st.selectbox("Colormap", [
                    "viridis", "plasma", "inferno", "cividis", "coolwarm", 
                    "Blues", "Greens", "Reds", "YlOrRd", "hot"
                ])
                show_colorbar = st.checkbox("Show Colorbar", value=True)
                
            with col2:
                st.subheader("Fonts & Size")
                font_size = st.number_input("Font Size", min_value=8, max_value=20, value=12)
                figure_size = st.number_input("Figure Size", min_value=4.0, max_value=15.0, value=8.0, step=0.5)
                dpi = st.number_input("Resolution (DPI)", min_value=100, max_value=600, value=300, step=50)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Normalization")
                normalize_data = st.checkbox("Normalize Data", value=True)
                normalization_mode = st.selectbox("Normalization Mode", [
                    "Maximum", "Logarithmic", "Natural log", "Square root"
                ])
                
                st.subheader("Smoothing")
                apply_smoothing = st.checkbox("Apply Smoothing", value=False)
                if apply_smoothing:
                    smoothing_type = st.selectbox("Smoothing Type", ["gaussian", "savgol"])
                    if smoothing_type == "gaussian":
                        gaussian_sigma = st.number_input("Gaussian Sigma", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
                        savgol_window = 11
                        savgol_polyorder = 3
                    else:
                        gaussian_sigma = 2.0
                        savgol_window = st.number_input("Window Size", min_value=3, max_value=21, value=11, step=2)
                        savgol_polyorder = st.number_input("Polynomial Order", min_value=1, max_value=6, value=3)
                else:
                    smoothing_type = "gaussian"
                    gaussian_sigma = 2.0
                    savgol_window = 11
                    savgol_polyorder = 3
            
            with col2:
                st.subheader("Noise Reduction")
                noise_threshold = st.number_input("Noise Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
                
                apply_intensity_threshold = st.checkbox("Apply Intensity Thresholding", value=False)
                if apply_intensity_threshold:
                    intensity_min_threshold = st.number_input("Min Intensity (%)", min_value=0.0, max_value=100.0, value=0.0) / 100
                    intensity_max_threshold = st.number_input("Max Intensity (%)", min_value=0.0, max_value=100.0, value=100.0) / 100
                else:
                    intensity_min_threshold = 0.0
                    intensity_max_threshold = 1.0
        
        with tab3:
            st.subheader("Advanced Settings")
            grid_resolution = st.number_input("Grid Resolution (Advanced)", min_value=50, max_value=1000, value=200)
            colorbar_width = st.number_input("Colorbar Width (%)", min_value=1.0, max_value=10.0, value=5.0)
            colorbar_pad = st.number_input("Colorbar Padding", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
        
        with tab4:
            plot_type = st.radio("Plot Type", ["CIU Heatmap", "Stacked CCS Distributions"])
        
        # Create settings object
        settings = ORIGAMISettings(
            grid_resolution=grid_resolution,
            interpolation_method="cubic",
            normalize_data=normalize_data,
            normalization_mode=normalization_mode,
            apply_smoothing=apply_smoothing,
            smoothing_type=smoothing_type,
            gaussian_sigma=gaussian_sigma,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
            noise_threshold=noise_threshold,
            apply_intensity_threshold=apply_intensity_threshold,
            intensity_min_threshold=intensity_min_threshold,
            intensity_max_threshold=intensity_max_threshold,
            colormap=colormap,
            show_colorbar=show_colorbar,
            colorbar_width=colorbar_width,
            colorbar_pad=colorbar_pad,
            font_size=font_size,
            figure_size=figure_size,
            dpi=dpi
        )
        
        return settings, plot_type
    
    def download_section(self, fig: plt.Figure, data: np.ndarray):
        """Provide download options"""
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            filename = st.text_input("Base filename", value="ciu_analysis", help="Enter filename without extension")
        
        with col2:
            st.write("")  # Spacing
            st.write("")
        
        if filename:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # PNG download
                img_png = BytesIO()
                fig.savefig(img_png, format='png', bbox_inches='tight', dpi=300)
                img_png.seek(0)
                st.download_button(
                    "📊 Download PNG",
                    data=img_png,
                    file_name=f"{filename}.png",
                    mime="image/png"
                )
            
            with col2:
                # SVG download
                img_svg = BytesIO()
                fig.savefig(img_svg, format='svg', bbox_inches='tight')
                img_svg.seek(0)
                st.download_button(
                    "📐 Download SVG",
                    data=img_svg,
                    file_name=f"{filename}.svg",
                    mime="image/svg+xml"
                )
            
            with col3:
                # Data CSV download
                df_data = pd.DataFrame(data, columns=["CCS (Å²)", "Drift Time (ms)", "CV (V)", "Intensity"])
                csv_data = df_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📄 Download CSV",
                    data=csv_data,
                    file_name=f"{filename}_data.csv",
                    mime="text/csv"
                )

def main():
    """Main application function"""
    st.set_page_config(
        page_title="ORIGAMI-Style CIU Analysis",
        page_icon="🔬",
        layout="wide"
    )
    
    # Load custom styling if available
    if styling:
        styling.load_custom_css()
    
    # Initialize interface
    interface = ORIGAMIInterface()
    interface.show_header()
    
    # File upload section
    twim_file, cal_file = interface.file_upload_section()
    
    if twim_file and cal_file:
        # Processing settings
        proc_settings = interface.processing_settings_section()
        
        # Process data button
        if st.button("🔄 Process Data", type="primary", use_container_width=True):
            try:
                with st.spinner("Processing data with ORIGAMI methods..."):
                    # Load files
                    twim_df = interface.processor.load_twim_extract(twim_file)
                    cal_data = interface.processor.load_calibration_data(cal_file, proc_settings["charge_state"])
                    
                    # Calibrate data
                    calibrated_data = interface.processor.calibrate_twim_data(
                        twim_df, cal_data, proc_settings["inject_time"]
                    )
                    
                    # Store in session state
                    st.session_state["calibrated_data"] = calibrated_data
                    st.session_state["proc_settings"] = proc_settings
                    
                    st.success("✅ Data processed successfully!")
                    
                    # Show data preview
                    st.subheader("📊 Data Preview")
                    preview_df = pd.DataFrame(
                        calibrated_data[:100], 
                        columns=["CCS (Å²)", "Drift Time (ms)", "CV (V)", "Intensity"]
                    )
                    st.dataframe(preview_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
    
    # Visualization section
    if "calibrated_data" in st.session_state:
        data = st.session_state["calibrated_data"]
        
        st.header("📊 Visualization")
        
        # Visualization settings
        viz_settings, plot_type = interface.visualization_settings_section()
        
        # Generate plot button
        if st.button("🎨 Generate Plot", type="primary", use_container_width=True):
            try:
                with st.spinner("Creating ORIGAMI-style visualization..."):
                    if plot_type == "CIU Heatmap":
                        fig = interface.visualizer.create_ciu_heatmap(data, viz_settings)
                    else:
                        fig = interface.visualizer.create_stacked_plot(data, viz_settings)
                    
                    st.pyplot(fig)
                    st.session_state["current_figure"] = fig
                    
            except Exception as e:
                st.error(f"❌ Error creating visualization: {str(e)}")
        
        # Download section
        if "current_figure" in st.session_state:
            interface.download_section(st.session_state["current_figure"], data)
    
    else:
        # Instructions
        st.info("👆 Please upload your TWIMExtract and calibration files to get started.")
        
        with st.expander("ℹ️ How to Use This Tool"):
            st.markdown("""
            ### ORIGAMI-Style CIU Analysis Tool
            
            This tool replicates the data processing and visualization methods from ORIGAMI (Lukasz G. Migas) 
            for analyzing Collision-Induced Unfolding (CIU) data from TWIMExtract files.
            
            **Steps:**
            1. **Upload Files**: TWIMExtract CSV and calibration CSV files
            2. **Set Parameters**: Choose charge state, instrument type, and processing options
            3. **Process Data**: Calibrate drift times to CCS values using ORIGAMI methods
            4. **Visualize**: Create publication-ready CIU heatmaps or stacked plots
            5. **Download**: Export plots and data in multiple formats
            
            **Features:**
            - **Automatic calibration validation**: Removes physically inconsistent points
            - **Exact replication of ORIGAMI's normalize_2D function**
            - **ORIGAMI-style smoothing and noise reduction**
            - **Multiple normalization modes** (Maximum, Logarithmic, etc.)
            - **High-quality publication-ready plots**
            - **Support for both Synapt and Cyclic IMS data**
            
            **New in this version:**
            - 🔍 **Calibration validation**: Automatically checks and removes points where:
              - Small drift times have large CCS values (physically inconsistent)
              - Non-monotonic CCS vs drift time relationships
            - 📊 **Enhanced feedback**: Shows which points are removed and why
            - ✅ **Quality metrics**: Reports correlation coefficients and data quality
            
            **File Formats:**
            - **TWIMExtract**: CSV file with drift times and collision voltage data
            - **Calibration**: CSV file with Z, Drift, CCS columns
            """)

if __name__ == "__main__":
    main()