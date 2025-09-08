import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
import matplotlib.colors as mcolors
import seaborn as sns
from io import BytesIO
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter
from enum import Enum
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import pickle

from myutils import styling, import_tools

# Enums for better type safety
class InstrumentType(Enum):
    SYNAPT = "Synapt"
    CYCLIC = "Cyclic"

class SmoothingType(Enum):
    SAVGOL = "savgol"
    GAUSSIAN = "gaussian"

class InterpolationMethod(Enum):
    CUBIC = "cubic"
    LINEAR = "linear"
    NEAREST = "nearest"

class PlotType(Enum):
    HEATMAP = "heatmap"
    STACKED = "stacked"

class ColorMapType(Enum):
    STANDARD = "Standard"
    COLORBLIND = "Seaborn Colorblind"

@dataclass
class AppConfig:
    """Centralized application configuration"""
    MAX_FILE_SIZE_MB: int = 100
    DEFAULT_GRID_RESOLUTION: int = 200
    DEFAULT_DPI: int = 300
    CACHE_SIZE: int = 32
    CHUNK_SIZE: int = 10000

@dataclass
class ProcessingSettings:
    """Configuration for CIU data processing"""
    data_type: InstrumentType
    charge_state: int
    inject_time: Optional[float] = None

@dataclass
class HeatmapSettings:
    """Configuration for heatmap visualization"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    normalize_data: bool
    interpolation_method: str
    grid_resolution: int
    apply_smoothing: bool
    smoothing_type: str = 'savgol'
    window_length: int = 11
    poly_order: int = 3
    gaussian_sigma: float = 2.0
    noise_threshold: float = 0.0
    apply_intensity_threshold: bool = False
    intensity_min_threshold: float = 0.0
    intensity_max_threshold: float = 1.0
    custom_cmap: Optional[Any] = None
    color_map: Optional[str] = None
    font_size: int = 12
    figure_size: int = 10
    dpi: int = 300
    show_colorbar: bool = True
    colorbar_shrink: float = 0.8
    colorbar_aspect: int = 20
    x_values: List[float] = None
    x_labels: List[str] = None
    y_values: List[float] = None
    y_labels: List[str] = None
    reference_line_color: str = 'black'
    stacked_offset_mode: str = 'auto'
    stacked_offset_value: float = 1.0
    stacked_line_width: float = 1.5
    stacked_fill_alpha: float = 0.3
    stacked_show_labels: bool = True
    stacked_label_frequency: int = 1
    stacked_line_color_mode: str = 'gradient'
    stacked_single_color: str = '#1f77b4'
    show_grid: bool = False
    
    def __post_init__(self):
        if self.x_values is None:
            self.x_values = []
        if self.x_labels is None:
            self.x_labels = []
        if self.y_values is None:
            self.y_values = []
        if self.y_labels is None:
            self.y_labels = []

# Context managers for better resource management
@contextmanager
def streamlit_spinner(message: str):
    """Context manager for Streamlit spinners with error handling"""
    with st.spinner(message):
        try:
            yield
        except Exception as e:
            st.error(f"❌ {message} failed: {str(e)}")
            raise

@contextmanager
def matplotlib_figure(figsize: Tuple[int, int], dpi: int):
    """Context manager for matplotlib figures to ensure cleanup"""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    try:
        yield fig, ax
    finally:
        plt.close(fig)

class CIUDataProcessor:
    """Handles CIU data processing and calibration"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
    
    @staticmethod
    def load_twim_extract(file) -> pd.DataFrame:
        """Load TWIM extract file with automatic header detection"""
        try:
            first_row = file.readline().decode("utf-8")
            file.seek(0)
            
            if first_row.startswith("#"):
                df = pd.read_csv(file, header=2)
            else:
                df = pd.read_csv(file)
            
            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            df = df.dropna(subset=[df.columns[0]])
            df.columns = ['Drift Time'] + list(df.columns[1:])
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading TWIM extract file: {str(e)}")
    
    @staticmethod
    def load_calibration_data(file, charge_state: int) -> pd.DataFrame:
        """Load calibration data - subtract 12ms from drift times before conversion"""
        try:
            cal_df = pd.read_csv(file)
            cal_data = cal_df[cal_df["Z"] == charge_state].copy()
            
            if cal_data.empty:
                raise ValueError(f"No calibration data found for charge state {charge_state}")
            
            required_cols = ["Drift", "CCS"]
            missing_cols = [col for col in required_cols if col not in cal_data.columns]
            if missing_cols:
                raise ValueError(f"Calibration data missing required columns: {missing_cols}")
            
            # Subtract 12ms before converting to milliseconds
            cal_data["Drift (ms)"] = (cal_data["Drift"] - 0.012) * 1000
            return cal_data
            
        except Exception as e:
            raise ValueError(f"Error loading calibration data: {str(e)}")
    
    @staticmethod
    def apply_drift_correction(df: pd.DataFrame, settings: ProcessingSettings) -> pd.DataFrame:
        """Apply instrument-specific drift time corrections"""
        df_corrected = df.copy()
        
        if settings.data_type == InstrumentType.CYCLIC and settings.inject_time is not None:
            df_corrected["Drift Time"] = df_corrected["Drift Time"] - settings.inject_time
        
        return df_corrected
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _get_calibration_lookup(cal_data_hash: str, cal_data_pickle: bytes) -> Dict[float, float]:
        """Cache calibration lookups to avoid repeated calculations"""
        cal_data = pickle.loads(cal_data_pickle)
        
        lookup = {}
        for _, row in cal_data.iterrows():
            drift_rounded = round(row["Drift (ms)"], 4)
            lookup[drift_rounded] = row["CCS"]
        return lookup
    
    def calibrate_data(self, twim_df: pd.DataFrame, cal_data: pd.DataFrame, inject_time: float = 0.0) -> np.ndarray:
        """Calibrate TWIM data - subtract inject time from each drift time, then interpolate CCS"""
        calibrated_data = []
        
        drift_times = twim_df["Drift Time"]
        collision_voltages = twim_df.columns[1:]
        
        # Sort calibration data by drift time for interpolation
        cal_data_sorted = cal_data.sort_values("Drift (ms)")
        cal_drift_times = cal_data_sorted["Drift (ms)"].values
        cal_ccs_values = cal_data_sorted["CCS"].values
        
        for idx, drift_time in enumerate(drift_times):
            if pd.isna(drift_time):
                continue
            
            # Step 1: Convert drift time to milliseconds and subtract inject time
            corrected_drift_time = drift_time - inject_time
            
            intensities = twim_df.iloc[idx, 1:].values
            
            # Step 2: Look up corresponding CCS value using linear interpolation
            if corrected_drift_time <= cal_drift_times.min():
                # Extrapolate using first calibration point
                ccs_value = cal_ccs_values[0]
            elif corrected_drift_time >= cal_drift_times.max():
                # Extrapolate using last calibration point
                ccs_value = cal_ccs_values[-1]
            else:
                # Linear interpolation between the two nearest calibration points
                ccs_value = np.interp(corrected_drift_time, cal_drift_times, cal_ccs_values)
            
            # Step 3: Create calibrated data points for each CV
            for col_idx, intensity in enumerate(intensities):
                if not pd.isna(intensity) and intensity > 0:
                    cv = collision_voltages[col_idx]
                    try:
                        cv_float = float(cv)
                        calibrated_data.append([ccs_value, corrected_drift_time, cv_float, intensity])
                    except (ValueError, TypeError):
                        continue
        
        if not calibrated_data:
            raise ValueError("No valid calibrated data points generated")
        
        return np.array(calibrated_data)

class CIUVisualization:
    """Handles CIU heatmap visualization"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
    
    @staticmethod
    def create_colorblind_cmap(color_name: str):
        """Create a colormap from white to seaborn colorblind color"""
        colors = sns.color_palette("colorblind")
        color_dict = {
            'pink': colors[6], 'blue': colors[0], 'orange': colors[1], 'green': colors[2],
            'red': colors[3], 'purple': colors[4], 'brown': colors[5], 'gray': colors[7],
            'olive': colors[8], 'cyan': colors[9]
        }
        selected_color = color_dict.get(color_name, colors[6])
        return mcolors.LinearSegmentedColormap.from_list(
            f'white_to_{color_name}', ['white', selected_color], N=256
        )
    
    @staticmethod
    def filter_data_by_range(data: np.ndarray, settings: HeatmapSettings) -> np.ndarray:
        """Filter data to specified axis ranges using boolean indexing"""
        mask = ((data[:, 2] >= settings.x_min) & (data[:, 2] <= settings.x_max) & 
               (data[:, 0] >= settings.y_min) & (data[:, 0] <= settings.y_max))
        
        if not np.any(mask):
            raise ValueError("No data points in the specified range")
        
        return data[mask]
    
    @staticmethod
    def normalize_2D_origami(inputData: np.ndarray, mode: str = 'Maximum') -> np.ndarray:
        """EXACT copy of ORIGAMI's normalize_2D function - no debug output"""
        inputData = np.nan_to_num(inputData)
        
        if mode == "Maximum":
            normData = normalize(inputData.astype(np.float64), axis=0, norm='max')
        elif mode == 'Logarithmic':
            normData = np.log10(inputData.astype(np.float64))
        elif mode == 'Natural log':
            normData = np.log(inputData.astype(np.float64))
        elif mode == 'Square root':
            normData = np.sqrt(inputData.astype(np.float64))
        elif mode == 'Least Abs Deviation':
            normData = normalize(inputData.astype(np.float64), axis=0, norm='l1')
        elif mode == 'Least Squares':
            normData = normalize(inputData.astype(np.float64), axis=0, norm='l2')
        
        return normData
    
    def create_interpolation_grid_origami(self, data: np.ndarray, settings: HeatmapSettings) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create interpolated intensity grid EXACTLY like ORIGAMI - conditional normalization"""
        filtered_data = self.filter_data_by_range(data, settings)
        
        grid_x = np.linspace(settings.x_min, settings.x_max, num=settings.grid_resolution)
        grid_y = np.linspace(settings.y_min, settings.y_max, num=settings.grid_resolution)
        X, Y = np.meshgrid(grid_x, grid_y)
        
        Z = griddata(
            (filtered_data[:, 2], filtered_data[:, 0]),
            filtered_data[:, 3],
            (X, Y),
            method=settings.interpolation_method,
            fill_value=0
        )
        
        Z = np.nan_to_num(Z, nan=0.0)
        
        # Apply ORIGAMI normalization only if requested
        if settings.normalize_data:
            Z = self.normalize_2D_origami(Z, mode='Maximum')
        
        return X, Y, Z, filtered_data
    
    def create_interpolation_grid_origami_stacked(self, data: np.ndarray, settings: HeatmapSettings) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create interpolated intensity grid for stacked plots - ALWAYS NORMALIZED"""
        filtered_data = self.filter_data_by_range(data, settings)
        
        grid_x = np.linspace(settings.x_min, settings.x_max, num=settings.grid_resolution)
        grid_y = np.linspace(settings.y_min, settings.y_max, num=settings.grid_resolution)
        X, Y = np.meshgrid(grid_x, grid_y)
        
        Z = griddata(
            (filtered_data[:, 2], filtered_data[:, 0]),
            filtered_data[:, 3],
            (X, Y),
            method=settings.interpolation_method,
            fill_value=0
        )
        
        Z = np.nan_to_num(Z, nan=0.0)
        
        # ALWAYS apply ORIGAMI normalization for stacked plots
        Z = self.normalize_2D_origami(Z, mode='Maximum')
        
        return X, Y, Z, filtered_data

    def _setup_plot_appearance(self, ax, settings: HeatmapSettings) -> None:
        """Configure plot styling and appearance"""
        ax.set_xlabel("Collision Voltage (V)", fontsize=settings.font_size, 
                     fontweight='normal', color='black')
        ax.set_ylabel("CCS (Å²)", fontsize=settings.font_size, 
                     fontweight='normal', color='black')
        ax.tick_params(labelsize=settings.font_size * 0.9, colors='black', width=1.2)
        
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)
        
        ax.set_facecolor('white')
    
    def _add_colorbar(self, fig, ax, c, settings: HeatmapSettings) -> None:
        """Add colorbar to the plot"""
        if not settings.show_colorbar:
            return
            
        cbar = plt.colorbar(c, ax=ax, shrink=settings.colorbar_shrink, 
                          aspect=settings.colorbar_aspect)
        intensity_label = "Normalized Intensity" if settings.normalize_data else "Intensity"
        cbar.set_label(intensity_label, fontsize=settings.font_size, 
                      fontweight='normal', color='black')
        cbar.ax.tick_params(labelsize=settings.font_size * 0.9, colors='black')
    
    def _add_reference_lines(self, ax, settings: HeatmapSettings):
        """Add reference lines with labels positioned to the right"""
        for x_val, x_label in zip(settings.x_values, settings.x_labels):
            ax.axvline(x=x_val, color=settings.reference_line_color, linestyle='--', linewidth=1, alpha=0.8)
            ax.text(x_val + (settings.x_max - settings.x_min) * 0.01, settings.y_max * 0.95, x_label, 
                   color=settings.reference_line_color, va='top', ha='left', fontsize=settings.font_size)
        
        for y_val, y_label in zip(settings.y_values, settings.y_labels):
            ax.axhline(y=y_val, color=settings.reference_line_color, linestyle='--', linewidth=1, alpha=0.8)
            ax.text(settings.x_max * 0.98, y_val, y_label, 
                   color=settings.reference_line_color, va='center', ha='right', fontsize=settings.font_size)

    @staticmethod
    def adjust_min_max_intensity(inputData: np.ndarray, min_threshold: float = 0.0, max_threshold: float = 1.0) -> np.ndarray:
        """EXACT copy of ORIGAMI's adjust_min_max_intensity function"""
        if min_threshold > max_threshold:
            st.warning("Minimum threshold is larger than the maximum. Values were reversed.")
            min_threshold, max_threshold = max_threshold, min_threshold
        
        if min_threshold == max_threshold:
            st.warning("Minimum and maximum thresholds are the same.")
            return inputData
        
        data_max = np.max(inputData)
        min_threshold = min_threshold * data_max
        max_threshold = max_threshold * data_max
        
        inputData[inputData <= min_threshold] = 0
        inputData[inputData >= max_threshold] = data_max
        
        return inputData
    
    @staticmethod
    def remove_noise_2D(inputData: np.ndarray, threshold: float = 0) -> np.ndarray:
        """EXACT copy of ORIGAMI's remove_noise_2D function"""
        if (threshold > np.max(inputData)) or (threshold < 0):
            st.warning(f"Threshold value was too high - the maximum value is {np.max(inputData)}. Value was reset to 0.")
            threshold = 0
        elif threshold == 0.0:
            pass
        elif (threshold < (np.max(inputData)/10000)):
            if (threshold > 1) or (threshold <= 0):
                threshold = 0
            st.warning(f"Threshold value was too low - the maximum value is {np.max(inputData)}. Value was reset to 0.")
            threshold = 0
        else:
            threshold = threshold
              
        inputData[inputData <= threshold] = 0
        return inputData
    
    @staticmethod
    def smooth_gaussian_2D(inputData: np.ndarray, sigma: float = 2) -> np.ndarray:
        """EXACT copy of ORIGAMI's smooth_gaussian_2D function"""
        if inputData is None or len(inputData) == 0:
            return None
        if sigma < 0:
            st.warning("Value of sigma is too low. Value was reset to 1")
            sigma = 1
        
        dataOut = gaussian_filter(inputData, sigma=sigma, order=0)
        dataOut[dataOut < 0] = 0
        return dataOut
    
    @staticmethod
    def smooth_savgol_2D(inputData: np.ndarray, polyOrder: int = 2, windowSize: int = 5) -> np.ndarray:
        """EXACT copy of ORIGAMI's smooth_savgol_2D function"""
        if inputData is None or len(inputData) == 0:
            return None
        
        if (polyOrder <= 0):
            st.warning("Polynomial order is too small. Value was reset to 2")
            polyOrder = 2   
        
        if windowSize is None:
            windowSize = polyOrder + 1
        elif (windowSize % 2) and (windowSize > polyOrder):
            windowSize = windowSize
        elif windowSize <= polyOrder:
            st.warning(f"Window size was smaller than the polynomial order. Value was reset to {polyOrder + 1}")
            windowSize = polyOrder + 1
        else:
            st.info('Window size is even. Adding 1 to make it odd.')
            windowSize = windowSize + 1
              
        dataOut = savgol_filter(inputData, polyorder=polyOrder, window_length=windowSize, axis=0)
        dataOut[dataOut < 0] = 0
        return dataOut
    
    def apply_smoothing_origami(self, Z: np.ndarray, settings: HeatmapSettings) -> np.ndarray:
        """Apply ORIGAMI-style smoothing options - NO ADDITIONAL PREPROCESSING"""
        if not settings.apply_smoothing:
            return Z
        
        if settings.smoothing_type == 'gaussian':
            Z_smooth = self.smooth_gaussian_2D(Z, sigma=settings.gaussian_sigma)
        else:  # savgol
            Z_smooth = self.smooth_savgol_2D(Z, polyOrder=settings.poly_order, windowSize=settings.window_length)
        
        if settings.noise_threshold > 0:
            Z_smooth = self.remove_noise_2D(Z_smooth, threshold=settings.noise_threshold)
        
        if settings.apply_intensity_threshold:
            Z_smooth = self.adjust_min_max_intensity(Z_smooth, 
                                                   min_threshold=settings.intensity_min_threshold, 
                                                   max_threshold=settings.intensity_max_threshold)
        
        return Z_smooth
    
    def create_stacked_ccsd_plot_origami(self, data: np.ndarray, settings: HeatmapSettings) -> Tuple[plt.Figure, np.ndarray]:
        """Create stacked CCSD plot - FIXED Y-axis offset calculation"""
        X, Y, Z, filtered_data = self.create_interpolation_grid_origami_stacked(data, settings)
        Z = self.apply_smoothing_origami(Z, settings)
        
        # Get actual CV values from data (ORIGAMI approach)
        actual_cv_values = np.unique(filtered_data[:, 2])
        actual_cv_values = actual_cv_values[(actual_cv_values >= settings.x_min) & 
                                          (actual_cv_values <= settings.x_max)]
        actual_cv_values = np.sort(actual_cv_values)
        
        ccs_values = np.linspace(settings.y_min, settings.y_max, num=settings.grid_resolution)
        
        fig, ax = plt.subplots(figsize=(settings.figure_size, settings.figure_size * 0.75), dpi=settings.dpi)
        fig.patch.set_facecolor('white')
        
        # Select CVs to plot based on frequency
        cv_step = settings.stacked_label_frequency
        cv_indices_to_plot = range(0, len(actual_cv_values), cv_step)
        selected_cvs = [actual_cv_values[i] for i in cv_indices_to_plot]
        n_traces = len(selected_cvs)
        
        # Calculate base trace height (height of one normalized trace)
        ccs_range = settings.y_max - settings.y_min
        base_trace_height = ccs_range / 10  # Default height for one trace
        
        # Calculate Y-axis offset based on user input
        if settings.stacked_offset_mode == 'auto':
            # ORIGAMI auto: space traces evenly across the range
            vertical_spacing = ccs_range / max(1, n_traces) if n_traces > 0 else ccs_range * 0.1
        elif settings.stacked_offset_mode == 'percentage':
            # Percentage of the base trace height
            vertical_spacing = base_trace_height * (settings.stacked_offset_value / 100)
        else:  # manual
            # Direct multiplier of base trace height
            # If user sets 0.5, next trace appears halfway up the previous one
            vertical_spacing = base_trace_height * settings.stacked_offset_value

        # Each trace height is always the base height regardless of offset
        max_trace_height = base_trace_height

        # Generate colors
        if settings.stacked_line_color_mode == 'single':
            colors = [settings.stacked_single_color] * n_traces
        elif settings.stacked_line_color_mode == 'gradient':
            if settings.custom_cmap is not None:
                colors = [settings.custom_cmap(i / max(1, n_traces-1)) for i in range(n_traces)]
            else:
                cmap = plt.cm.get_cmap(settings.color_map or 'viridis')
                colors = [cmap(i / max(1, n_traces-1)) for i in range(n_traces)]
        else:  # colorblind_cycle
            cb_colors = sns.color_palette("colorblind", n_colors=min(n_traces, 10))
            colors = [cb_colors[i % len(cb_colors)] for i in range(n_traces)]
        
        y_tick_positions = []
        y_tick_labels = []
        
        # Start from bottom of CCS range and stack upward
        y_start = settings.y_min
        
        for plot_index, cv_value in enumerate(selected_cvs):
            # Find corresponding intensity data
            grid_cv_values = np.linspace(settings.x_min, settings.x_max, num=settings.grid_resolution)
            closest_grid_idx = np.argmin(np.abs(grid_cv_values - cv_value))
            
            intensity_data = Z[:, closest_grid_idx].copy()
            
            if np.max(intensity_data) <= 0:
                continue
            
            # REMOVED: Individual trace normalization and baseline correction
            # Data is already normalized by ORIGAMI at the grid level
            
            # Y-axis positioning - each trace moves up by vertical_spacing
            base_y_position = y_start + (plot_index * vertical_spacing)
            
            # Scale intensities to consistent height (they're already 0-1 normalized)
            scaled_intensity = intensity_data * max_trace_height
            
            y_coordinates = base_y_position + scaled_intensity
            baseline_y = np.full_like(ccs_values, base_y_position)
            
            color = colors[plot_index % len(colors)]
            
            ax.plot(ccs_values, y_coordinates, 
                   color=color, 
                   linewidth=settings.stacked_line_width, 
                   alpha=0.9)
            
            ax.fill_between(ccs_values, baseline_y, y_coordinates,
                          color=color, alpha=settings.stacked_fill_alpha)
            
            y_tick_positions.append(base_y_position)
            y_tick_labels.append(f"{cv_value:.0f}")
        
        # ORIGAMI axis configuration - FIXED CV label handling
        ax.set_xlabel("CCS (Å²)", fontsize=settings.font_size, fontweight='normal', color='black')
        
        # Y-axis label depends on whether CV labels are shown
        if settings.stacked_show_labels:
            ax.set_ylabel("Collision Voltage (V)", fontsize=settings.font_size, fontweight='normal', color='black')
        else:
            ax.set_ylabel("Normalised Intensity", fontsize=settings.font_size, fontweight='normal', color='black')
        
        ax.set_xlim(settings.y_min, settings.y_max)  # CCS range on X-axis
        
        # Y-axis limits - show all traces with some padding
        if y_tick_positions:
            y_min_plot = min(y_tick_positions) - vertical_spacing * 0.1
            y_max_plot = max(y_tick_positions) + max_trace_height + vertical_spacing * 0.1
            
            ax.set_ylim(y_min_plot, y_max_plot)
            
            # FIXED: Properly handle CV labels checkbox
            if settings.stacked_show_labels:
                ax.set_yticks(y_tick_positions)
                ax.set_yticklabels(y_tick_labels)
            else:
                # Remove all Y-axis ticks and labels when CV labels are disabled
                ax.set_yticks([])
                ax.set_yticklabels([])

        # ORIGAMI styling
        for spine_name in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color('black')
            ax.spines[spine_name].set_linewidth(1.5)
        
        ax.tick_params(axis='x', labelsize=settings.font_size * 0.9, colors='black', width=1.2)
        ax.tick_params(axis='y', labelsize=settings.font_size * 0.9, colors='black', width=1.2)
        
        if hasattr(settings, 'show_grid') and settings.show_grid:
            ax.grid(True, alpha=0.1, axis='x', linestyle='-', linewidth=0.5)
        
        ax.set_facecolor('white')
        plt.tight_layout()
        
        return fig, filtered_data

    def create_heatmap(self, data: np.ndarray, settings: HeatmapSettings, plot_type: str = "heatmap") -> Tuple[plt.Figure, np.ndarray]:
        """Create CIU visualization with specified settings"""
        if plot_type == "stacked":
            return self.create_stacked_ccsd_plot_origami(data, settings)
        
        X, Y, Z, filtered_data = self.create_interpolation_grid_origami(data, settings)
        Z = self.apply_smoothing_origami(Z, settings)
        
        fig, ax = plt.subplots(figsize=(settings.figure_size, settings.figure_size), dpi=settings.dpi)
        fig.patch.set_facecolor('white')
        
        cmap = settings.custom_cmap if settings.custom_cmap is not None else settings.color_map
        c = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
        
        self._setup_plot_appearance(ax, settings)
        self._add_colorbar(fig, ax, c, settings)
        self._add_reference_lines(ax, settings)
        
        plt.tight_layout()
        return fig, filtered_data

class CIUInterface:
    """Handles Streamlit UI components"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
    
    @staticmethod
    def show_main_header():
        """Display main page header"""
        st.markdown("""
        <div class="main-header">
            <h1>📊 Plot aIMS/CIU Heatmaps</h1>
            <p>Generate publication-ready CIU heatmaps from TWIMExtract data with ORIGAMI-style processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_upload_section() -> Tuple[Optional[Any], Optional[Any]]:
        """Handle file uploads"""
        st.header("📁 File Upload")
        
        twim_extract_file = st.file_uploader(
            "TWIM Extract CSV", 
            type="csv", 
            help="Upload your TWIMExtract output file"
        )
        calibration_file = st.file_uploader(
            "Calibration CSV", 
            type="csv", 
            help="Upload your calibration file with CCS values"
        )
        
        return twim_extract_file, calibration_file
    
    @staticmethod
    def get_processing_settings() -> ProcessingSettings:
        """Get processing configuration from user"""
        st.header("⚙️ Processing Settings")
        
        data_type_str = st.selectbox("Instrument Type", ["Synapt", "Cyclic"])
        data_type = InstrumentType.SYNAPT if data_type_str == "Synapt" else InstrumentType.CYCLIC
        charge_state = st.number_input("Charge State (Z)", min_value=1, max_value=100, value=1)
        
        inject_time = None
        if data_type == InstrumentType.CYCLIC:
            inject_time = st.number_input("Injection Time (ms)", min_value=0.0, value=0.0, step=0.1)
        
        return ProcessingSettings(
            data_type=data_type,
            charge_state=charge_state,
            inject_time=inject_time
        )
    
    @staticmethod
    def get_heatmap_settings(data_range: Dict[str, float]) -> Tuple[HeatmapSettings, str]:
        """Get heatmap visualization settings"""
        plot_type = st.radio("Visualization Type", 
                           ["Heatmap", "Stacked CCSDs"], 
                           help="Choose between traditional CIU heatmap or stacked CCSD plot")
        
        if plot_type == "Stacked CCSDs":
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 Appearance", "📐 Data Processing", "📊 Stacked Options", "📏 Axis Settings", "📍 Annotations"])
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["🎨 Appearance", "📐 Data Processing", "📏 Axis Settings", "📍 Annotations"])
        
        settings_dict = {}
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Color Settings")
                colormap_type = st.selectbox("Colormap Type", ["Standard", "Seaborn Colorblind"])
                
                if colormap_type == "Standard":
                    color_map = st.selectbox("Color Map", [
                        "viridis", "plasma", "inferno", "cividis", "coolwarm", "magma", 
                        "Blues", "Greens", "Purples", "Oranges", "Reds", "jet"
                    ])
                    custom_cmap = None
                else:
                    colorblind_options = ['pink', 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'gray', 'olive', 'cyan']
                    colorblind_color = st.selectbox("Colorblind Color", colorblind_options)
                    custom_cmap = CIUVisualization.create_colorblind_cmap(colorblind_color)
                    color_map = None
                
                settings_dict.update({'custom_cmap': custom_cmap, 'color_map': color_map})
            
            with col2:
                st.subheader("Plot Settings")
                font_size = st.number_input("Font Size", min_value=6, max_value=30, value=12)
                figure_size = st.number_input("Figure Size (inches)", min_value=4, max_value=20, value=10)
                dpi = st.number_input("Resolution (DPI)", min_value=50, max_value=1000, value=300, step=50)
                
                if plot_type == "Heatmap":
                    st.subheader("Colorbar Settings")
                    show_colorbar = st.checkbox("Show colorbar", value=True)
                    colorbar_shrink = 0.8
                    colorbar_aspect = 20
                    if show_colorbar:
                        colorbar_shrink = st.number_input("Colorbar size", min_value=0.3, max_value=1.0, value=0.8, step=0.05)
                        colorbar_aspect = st.number_input("Colorbar aspect ratio", min_value=10, max_value=50, value=20)
                else:
                    show_colorbar = False
                    colorbar_shrink = 0.8
                    colorbar_aspect = 20
                    
                    st.subheader("Grid Settings")
                    show_grid = st.checkbox("Show subtle grid", value=False)
                    settings_dict.update({'show_grid': show_grid})
                
                settings_dict.update({
                    'font_size': font_size, 'figure_size': figure_size, 'dpi': dpi,
                    'show_colorbar': show_colorbar, 'colorbar_shrink': colorbar_shrink, 'colorbar_aspect': colorbar_aspect
                })
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Interpolation")
                interpolation_method = st.selectbox("Method", ["cubic", "linear", "nearest"])
                grid_resolution = st.number_input("Grid Resolution", min_value=50, max_value=500, value=200, step=10)
                
                if plot_type == "Heatmap":
                    # ORIGAMI-style normalization option for heatmaps only
                    normalize_data = st.checkbox("Normalize data (ORIGAMI-style)", value=True, 
                                                help="Apply ORIGAMI Maximum normalization to each CV column")
                    if normalize_data:
                        st.info("✅ ORIGAMI normalization: each CV column max = 1")
                    else:
                        st.info("ℹ️ Raw intensity values preserved")
                else:
                    # Stacked plots always normalized
                    normalize_data = True
                    st.info("✅ ORIGAMI normalization always applied for stacked plots")
                
                settings_dict.update({
                    'interpolation_method': interpolation_method, 
                    'grid_resolution': grid_resolution, 
                    'normalize_data': normalize_data
                })
            
            with col2:
                st.subheader("Smoothing (ORIGAMI-style)")
                apply_smoothing = st.checkbox("Apply smoothing")
                
                smoothing_type = 'savgol'
                window_length = 11
                poly_order = 3
                gaussian_sigma = 2.0
                noise_threshold = 0.0
                apply_intensity_threshold = False
                intensity_min_threshold = 0.0
                intensity_max_threshold = 1.0
                
                if apply_smoothing:
                    smoothing_type = st.selectbox("Smoothing Type", ["savgol", "gaussian"])
                    
                    if smoothing_type == "savgol":
                        window_length = st.number_input("Window Length", min_value=3, max_value=51, value=11, step=2)
                        poly_order = st.number_input("Polynomial Order", min_value=1, max_value=6, value=3)
                    else:  # gaussian
                        gaussian_sigma = st.number_input("Gaussian Sigma", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
                
                st.subheader("Post-processing")
                noise_threshold = st.number_input("Noise Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                                                help="Remove values below this threshold")
                
                apply_intensity_threshold = st.checkbox("Apply intensity thresholding")
                if apply_intensity_threshold:
                    intensity_min_threshold = st.number_input("Min Intensity Threshold (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0) / 100
                    intensity_max_threshold = st.number_input("Max Intensity Threshold (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0) / 100
                
                settings_dict.update({
                    'apply_smoothing': apply_smoothing, 'smoothing_type': smoothing_type,
                    'window_length': window_length, 'poly_order': poly_order, 'gaussian_sigma': gaussian_sigma,
                    'noise_threshold': noise_threshold, 'apply_intensity_threshold': apply_intensity_threshold,
                    'intensity_min_threshold': intensity_min_threshold, 'intensity_max_threshold': intensity_max_threshold
                })
        
        # Stacked plot specific options
        if plot_type == "Stacked CCSDs":
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Trace Selection")
                    stacked_label_frequency = st.number_input("Trace Frequency", 
                                                min_value=1, max_value=20, value=1, step=1,
                                                help="Show every Nth trace (1 = all traces, 2 = every other trace, etc.)")

                    st.subheader("Y-axis Offset Settings")
                    stacked_offset_mode = st.selectbox("Offset Mode", 
                     ["auto", "percentage", "manual"],
                     help="auto: even spacing, percentage: % of trace height, manual: multiplier of trace height")

                    stacked_offset_value = 1.0
                    if stacked_offset_mode == "manual":
                        stacked_offset_value = st.number_input("Offset Multiplier", 
                         min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                         help="Multiplier of trace height (0.5 = halfway overlap, 1.0 = no overlap, 2.0 = double spacing)")
                    elif stacked_offset_mode == "percentage":
                        stacked_offset_value = st.number_input("Offset Percentage", 
                         min_value=10.0, max_value=500.0, value=100.0, step=10.0,
                         help="Percentage of trace height (50% = halfway overlap, 100% = no overlap)")
                    else:  # auto
                        st.info("Auto mode: traces evenly spaced across Y-axis range")

                with col2:
                    st.subheader("Trace Appearance")
                    stacked_line_width = st.number_input("Line Width", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
                    stacked_fill_alpha = st.number_input("Fill Transparency", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

                    st.subheader("Color Settings")
                    stacked_line_color_mode = st.selectbox("Color Mode", 
                                             ["gradient", "single", "colorblind_cycle"],
                                             help="gradient: color map, single: one color, colorblind_cycle: distinct colors")

                    stacked_single_color = '#1f77b4'  # Default blue
                    if stacked_line_color_mode == "single":
                        stacked_single_color = st.color_picker("Single Color", value='#1f77b4')
                    elif stacked_line_color_mode == "gradient":
                        st.info("Gradient colors will use the selected colormap from Appearance tab")
                    else:  # colorblind_cycle
                        st.info("Using colorblind-friendly color cycle")

                    st.subheader("Labels")
                    stacked_show_labels = st.checkbox("Show CV labels on Y-axis", value=True)
                    
                    # Add info about automatic normalization
                    st.info("✅ ORIGAMI grid normalization automatically applied to all traces")

            # Update settings_dict to remove the normalization settings:
            settings_dict.update({
                'stacked_label_frequency': stacked_label_frequency,
                'stacked_offset_mode': stacked_offset_mode,
                'stacked_offset_value': stacked_offset_value,
                'stacked_line_width': stacked_line_width,
                'stacked_fill_alpha': stacked_fill_alpha,
                'stacked_line_color_mode': stacked_line_color_mode,
                'stacked_single_color': stacked_single_color,
                'stacked_show_labels': stacked_show_labels
                # REMOVED: stacked_normalize_individual and stacked_baseline_correction
            })

            axis_tab = tab4
            annotation_tab = tab5
        else:
            # Set default values for stacked settings when not in stacked mode
            settings_dict.update({
                'stacked_label_frequency': 1,
                'stacked_offset_mode': 'auto',
                'stacked_offset_value': 1.0,
                'stacked_line_width': 1.5,
                'stacked_fill_alpha': 0.3,
                'stacked_line_color_mode': 'gradient',
                'stacked_single_color': '#1f77b4',
                'stacked_show_labels': True
                # REMOVED: stacked_normalize_individual and stacked_baseline_correction defaults
            })
            axis_tab = tab3
            annotation_tab = tab4
        
        with axis_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Collision Voltage Range")
                x_min, x_max = st.slider("CV Range", 
                                        data_range['x_min'], data_range['x_max'], 
                                        (data_range['x_min'], data_range['x_max']))
            
            with col2:
                st.subheader("CCS Range")
                y_min, y_max = st.slider("CCS Range", 
                                        data_range['y_min'], data_range['y_max'], 
                                        (data_range['y_min'], data_range['y_max']))
            
            settings_dict.update({'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})
        
        with annotation_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Reference Line Settings")
                reference_line_color = st.selectbox("Reference Line Color", ["black", "white"])
                settings_dict.update({'reference_line_color': reference_line_color})
                
                st.subheader("Vertical Reference Lines")
                num_x_lines = st.slider("Number of vertical lines", 0, 5, 0)
                x_values, x_labels = [], []
                for i in range(num_x_lines):
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        value = st.number_input(f"X-value {i+1}", min_value=x_min, max_value=x_max, 
                                              value=(x_min + x_max)/2, key=f"x_val_{i}")
                    with subcol2:
                        label = st.text_input(f"X-label {i+1}", value=f"Line {i+1}", key=f"x_label_{i}")
                    x_values.append(value)
                    x_labels.append(label)
            
            with col2:
                st.write("")
                st.write("")
                st.subheader("Horizontal Reference Lines")
                num_y_lines = st.slider("Number of horizontal lines", 0, 5, 0)
                y_values, y_labels = [], []
                for i in range(num_y_lines):
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        value = st.number_input(f"Y-value {i+1}", min_value=y_min, max_value=y_max, 
                                              value=(y_min + y_max)/2, key=f"y_val_{i}")
                    with subcol2:
                        label = st.text_input(f"Y-label {i+1}", value=f"Line {i+1}", key=f"y_label_{i}")
                    y_values.append(value)
                    y_labels.append(label)
            
            settings_dict.update({'x_values': x_values, 'x_labels': x_labels, 'y_values': y_values, 'y_labels': y_labels})
        
        plot_type_code = "stacked" if plot_type == "Stacked CCSDs" else "heatmap"
        return HeatmapSettings(**settings_dict), plot_type_code

    @staticmethod
    def show_download_options(dpi: int):
        """Show download buttons for generated plots"""
        if "current_figure" in st.session_state and "processed_data" in st.session_state:
            st.subheader("💾 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = pd.DataFrame(st.session_state["processed_data"], 
                                      columns=["CCS", "Drift Time", "Collision Voltage", "Intensity"])
                csv = csv_data.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download CSV", data=csv, file_name="ciu_data.csv", mime="text/csv")
            
            with col2:
                img_png = BytesIO()
                st.session_state["current_figure"].savefig(img_png, format='png', bbox_inches="tight", dpi=dpi)
                img_png.seek(0)
                st.download_button("🖼️ Download PNG", data=img_png, file_name="ciu_heatmap.png", mime="image/png")
            
            with col3:
                img_svg = BytesIO()
                st.session_state["current_figure"].savefig(img_svg, format='svg', bbox_inches="tight")
                img_svg.seek(0)
                st.download_button("📐 Download SVG", data=img_svg, file_name="ciu_heatmap.svg", mime="image/svg+xml")
    
    @staticmethod
    def show_instructions():
        """Show usage instructions"""
        st.info("👆 Please upload your TWIM Extract and calibration files in the sidebar to get started.")
        
        with st.expander("ℹ️ How to Use This Tool"):
            st.markdown("""
            ### Step-by-Step Instructions:
            
            1. **Upload Files**: 
               - Upload your TWIM Extract CSV file
               - Upload your calibration CSV file with CCS values
            
            2. **Configure Processing**:
               - Select your instrument type (Synapt or Cyclic)
               - Enter the charge state (Z) for your analysis
               - For Cyclic data, specify the injection time
            
            3. **Process Data**: Click "Process Data" to calibrate your measurements
            
            4. **Customize Visualization**:
               - **Appearance**: Choose colors, fonts, and plot size
               - **Data Processing**: Set interpolation, ORIGAMI normalization, and smoothing
               - **Axis Settings**: Define the CV and CCS ranges to display
               - **Annotations**: Add reference lines with labels
            
            5. **Generate & Download**: Create your heatmap and download in multiple formats
            
            ### Features:
            - ✅ ORIGAMI-style data processing and normalization
            - ✅ Origin-style professional heatmaps
            - ✅ Colorblind-friendly palettes
            - ✅ Advanced smoothing and interpolation
            - ✅ Stacked CCSD visualization option
            - ✅ Multiple export formats (PNG, SVG, CSV)
            """)

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Plot aIMS/CIU Heatmaps",
        page_icon="📊",
        layout="wide"
    )
    
    styling.load_custom_css()
    
    # Initialize components with shared config
    config = AppConfig()
    interface = CIUInterface(config)
    processor = CIUDataProcessor(config)
    visualizer = CIUVisualization(config)
    
    interface.show_main_header()
    
    if st.button("🧹 Clear Cache & Restart App"):
        import_tools.clear_cache()
    
    with st.sidebar:
        twim_extract_file, calibration_file = interface.show_upload_section()
        
        if twim_extract_file and calibration_file:
            processing_settings = interface.get_processing_settings()
            
            if st.button("🔄 Process Data", type="primary"):
                try:
                    with streamlit_spinner("Processing data..."):
                        twim_df = processor.load_twim_extract(twim_extract_file)
                        cal_data = processor.load_calibration_data(
                            calibration_file, 
                            processing_settings.charge_state
                        )
                        
                        twim_df = processor.apply_drift_correction(twim_df, processing_settings)
                        
                        # Pass inject_time to calibration
                        inject_time = processing_settings.inject_time if processing_settings.inject_time is not None else 0.0
                        calibrated_array = processor.calibrate_data(twim_df, cal_data, inject_time)
                        
                        st.session_state["calibrated_array"] = calibrated_array
                        st.session_state["processing_settings"] = processing_settings
                        
                        st.success("✅ Data processed successfully!")
                        
                        df_preview = pd.DataFrame(calibrated_array[:100], 
                                                columns=["CCS", "Drift Time", "CV", "Intensity"])
                        st.dataframe(df_preview, height=200)
                        st.info(f"Processed {len(calibrated_array):,} data points")
                        
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
    
    if "calibrated_array" in st.session_state:
        calibrated_array = st.session_state["calibrated_array"]
        
        data_range = {
            'x_min': float(calibrated_array[:, 2].min()),
            'x_max': float(calibrated_array[:, 2].max()),
            'y_min': float(calibrated_array[:, 0].min()),
            'y_max': float(calibrated_array[:, 0].max())
        }
        
        st.subheader("📊 Visualization Settings")
        heatmap_settings, plot_type = interface.get_heatmap_settings(data_range)
        
        # REMOVE any diagnostic calls here - go straight to visualization
        button_text = "🎨 Generate Stacked CCSDs" if plot_type == "stacked" else "🎨 Generate CIU Heatmap"
        if st.button(button_text, type="primary", use_container_width=True):
            try:
                with streamlit_spinner("Generating visualization..."):
                    fig, processed_data = visualizer.create_heatmap(calibrated_array, heatmap_settings, plot_type)
                    
                    st.pyplot(fig)
                    
                    st.session_state["current_figure"] = fig
                    st.session_state["processed_data"] = processed_data
                    
            except Exception as e:
                st.error(f"❌ Error generating visualization: {str(e)}")
        
        interface.show_download_options(heatmap_settings.dpi)
        
    else:
        interface.show_instructions()

if __name__ == "__main__":
    main()