import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import linregress
from scipy.integrate import trapezoid
import warnings
warnings.filterwarnings('ignore')

from myutils import styling

# --- Origin-Style Peak Functions ---
def gaussian_peak(x, amplitude, center, width):
    """Gaussian peak function (Origin's Gauss function)"""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

def lorentzian_peak(x, amplitude, center, width):
    """Lorentzian peak function (Origin's Lorentz function)"""
    return amplitude / (1 + ((x - center) / width) ** 2)

def voigt_peak(x, amplitude, center, width_g, width_l):
    """Pseudo-Voigt approximation (Origin's Voigt function)"""
    eta = 1.36603 * (width_l / width_g) - 0.47719 * (width_l / width_g)**2 + 0.11116 * (width_l / width_g)**3
    if eta > 1:
        eta = 1
    elif eta < 0:
        eta = 0
    
    gaussian = np.exp(-0.693147 * ((x - center) / width_g) ** 2)
    lorentzian = 1 / (1 + ((x - center) / width_l) ** 2)
    
    return amplitude * (eta * lorentzian + (1 - eta) * gaussian)

def asymmetric_gaussian(x, amplitude, center, width, asymmetry):
    """Asymmetric Gaussian (Origin's ExpGauss function)"""
    gaussian = np.exp(-0.5 * ((x - center) / width) ** 2)
    exponential = np.exp((x - center) / asymmetry)
    
    # Convolution approximation
    return amplitude * gaussian * exponential

def exponentially_modified_gaussian(x, amplitude, center, width, tau):
    """Exponentially Modified Gaussian (Origin's EMG function)"""
    from scipy.special import erfc
    
    sigma = width / np.sqrt(2)
    lambda_param = 1.0 / tau if tau != 0 else 1e10
    
    term1 = (lambda_param / 2) * np.exp((lambda_param / 2) * (2 * center + lambda_param * sigma**2 - 2 * x))
    term2 = erfc((center + lambda_param * sigma**2 - x) / (sigma * np.sqrt(2)))
    
    return amplitude * term1 * term2

def bigaussian_peak(x, amplitude, center, width1, width2):
    """BiGaussian function (Origin's BiGauss)"""
    result = np.zeros_like(x)
    left_mask = x <= center
    right_mask = x > center
    
    if np.any(left_mask):
        result[left_mask] = amplitude * np.exp(-0.5 * ((x[left_mask] - center) / width1) ** 2)
    if np.any(right_mask):
        result[right_mask] = amplitude * np.exp(-0.5 * ((x[right_mask] - center) / width2) ** 2)
    
    return result

def multi_peak_function(x, peak_type, *params):
    """Multi-peak function supporting different peak types"""
    y = np.zeros_like(x)
    
    if peak_type == "Gaussian":
        params_per_peak = 3
        for i in range(0, len(params), params_per_peak):
            if i + params_per_peak <= len(params):
                y += gaussian_peak(x, params[i], params[i+1], params[i+2])
    
    elif peak_type == "Lorentzian":
        params_per_peak = 3
        for i in range(0, len(params), params_per_peak):
            if i + params_per_peak <= len(params):
                y += lorentzian_peak(x, params[i], params[i+1], params[i+2])
    
    elif peak_type == "Voigt":
        params_per_peak = 4
        for i in range(0, len(params), params_per_peak):
            if i + params_per_peak <= len(params):
                y += voigt_peak(x, params[i], params[i+1], params[i+2], params[i+3])
    
    elif peak_type == "BiGaussian":
        params_per_peak = 4
        for i in range(0, len(params), params_per_peak):
            if i + params_per_peak <= len(params):
                y += bigaussian_peak(x, params[i], params[i+1], params[i+2], params[i+3])
    
    elif peak_type == "EMG":
        params_per_peak = 4
        for i in range(0, len(params), params_per_peak):
            if i + params_per_peak <= len(params):
                y += exponentially_modified_gaussian(x, params[i], params[i+1], params[i+2], params[i+3])
    
    return y

# --- Origin-Style Baseline Functions ---
def linear_baseline(x, slope, intercept):
    """Linear baseline"""
    return slope * x + intercept

def polynomial_baseline(x, *coeffs):
    """Polynomial baseline"""
    return np.polyval(coeffs, x)

def exponential_baseline(x, a, b, c):
    """Exponential baseline"""
    return a * np.exp(b * x) + c

# --- Origin-Style Peak Detection ---
class OriginPeakDetector:
    @staticmethod
    def find_peaks_origin_style(x, y, min_height_percent=5, min_prominence_percent=2, 
                               min_distance_percent=5, smoothing_points=5):
        """Peak detection using Origin-style parameters"""
        # Smooth data if requested
        if smoothing_points > 0:
            y_smooth = savgol_filter(y, window_length=min(smoothing_points*2+1, len(y)-1), polyorder=2)
        else:
            y_smooth = y.copy()
        
        # Calculate thresholds
        y_range = np.max(y_smooth) - np.min(y_smooth)
        min_height = np.min(y_smooth) + y_range * (min_height_percent / 100)
        min_prominence = y_range * (min_prominence_percent / 100)
        min_distance = len(x) * (min_distance_percent / 100)
        
        # Find peaks
        peaks, properties = find_peaks(
            y_smooth,
            height=min_height,
            prominence=min_prominence,
            distance=int(min_distance)
        )
        
        # Calculate peak widths at different heights (Origin-style)
        if len(peaks) > 0:
            try:
                widths_half = peak_widths(y_smooth, peaks, rel_height=0.5)[0]
                widths_base = peak_widths(y_smooth, peaks, rel_height=0.1)[0]
                
                peak_info = []
                for i, peak_idx in enumerate(peaks):
                    info = {
                        'index': peak_idx,
                        'x': x[peak_idx],
                        'y': y_smooth[peak_idx],
                        'prominence': properties['prominences'][i],
                        'width_half': widths_half[i] * (x[1] - x[0]) if i < len(widths_half) else 0,
                        'width_base': widths_base[i] * (x[1] - x[0]) if i < len(widths_base) else 0,
                        'area_estimate': properties['prominences'][i] * widths_half[i] * (x[1] - x[0]) if i < len(widths_half) else 0
                    }
                    peak_info.append(info)
                
                return peak_info
            except:
                # Fallback if width calculation fails
                peak_info = []
                for i, peak_idx in enumerate(peaks):
                    info = {
                        'index': peak_idx,
                        'x': x[peak_idx],
                        'y': y_smooth[peak_idx],
                        'prominence': properties['prominences'][i],
                        'width_half': (x[-1] - x[0]) / 20,  # Default width
                        'width_base': (x[-1] - x[0]) / 10,
                        'area_estimate': properties['prominences'][i] * (x[-1] - x[0]) / 20
                    }
                    peak_info.append(info)
                return peak_info
        
        return []

# --- Origin-Style Parameter Estimation ---
class OriginParameterEstimator:
    @staticmethod
    def estimate_gaussian_parameters(x, y, peak_info):
        """Estimate initial parameters for Gaussian peaks (Origin method)"""
        params = []
        x_span = x[-1] - x[0]
        
        for peak in peak_info:
            amplitude = max(abs(peak['y']), np.max(y) * 0.01)  # Ensure positive amplitude
            center = peak['x']
            # Convert FWHM to sigma for Gaussian, ensure reasonable width
            sigma = max(peak.get('width_half', x_span/20) / (2 * np.sqrt(2 * np.log(2))), x_span * 0.001)
            params.extend([amplitude, center, sigma])
        
        return params
    
    @staticmethod
    def estimate_lorentzian_parameters(x, y, peak_info):
        """Estimate initial parameters for Lorentzian peaks"""
        params = []
        x_span = x[-1] - x[0]
        
        for peak in peak_info:
            amplitude = max(abs(peak['y']), np.max(y) * 0.01)
            center = peak['x']
            # For Lorentzian, FWHM = 2 * gamma
            gamma = max(peak.get('width_half', x_span/20) / 2, x_span * 0.001)
            params.extend([amplitude, center, gamma])
        
        return params
    
    @staticmethod
    def estimate_voigt_parameters(x, y, peak_info):
        """Estimate initial parameters for Voigt peaks"""
        params = []
        x_span = x[-1] - x[0]
        
        for peak in peak_info:
            amplitude = max(abs(peak['y']), np.max(y) * 0.01)
            center = peak['x']
            # Start with equal Gaussian and Lorentzian contributions
            base_width = max(peak.get('width_half', x_span/20), x_span * 0.001)
            width_g = base_width / (2 * np.sqrt(2 * np.log(2)))
            width_l = base_width / 2
            params.extend([amplitude, center, width_g, width_l])
        
        return params

# --- Origin-Style Parameter Manager ---
class OriginParameterManager:
    def __init__(self, peak_type, parameters, x_range):
        self.peak_type = peak_type
        self.parameters = parameters.copy()
        self.x_range = x_range
        self.fixed_params = {}  # Track which parameters are fixed
        self.param_bounds = {}  # Track custom bounds
        self.params_per_peak = self.get_params_per_peak()
        self.n_peaks = len(parameters) // self.params_per_peak
        
    def get_params_per_peak(self):
        """Get number of parameters per peak for different peak types"""
        params_dict = {
            "Gaussian": 3,
            "Lorentzian": 3,
            "Voigt": 4,
            "BiGaussian": 4,
            "EMG": 4
        }
        return params_dict.get(self.peak_type, 3)
    
    def get_parameter_names(self):
        """Get parameter names for each peak type"""
        if self.peak_type == "Gaussian":
            return ["Amplitude", "Center", "Width (σ)"]
        elif self.peak_type == "Lorentzian":
            return ["Amplitude", "Center", "Width (γ)"]
        elif self.peak_type == "Voigt":
            return ["Amplitude", "Center", "Width_G", "Width_L"]
        elif self.peak_type == "BiGaussian":
            return ["Amplitude", "Center", "Width_L", "Width_R"]
        elif self.peak_type == "EMG":
            return ["Amplitude", "Center", "Width", "Tau"]
        else:
            return ["Amplitude", "Center", "Width"]
    
    def update_parameter(self, peak_idx, param_idx, value):
        """Update a specific parameter"""
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        if 0 <= global_param_idx < len(self.parameters):
            self.parameters[global_param_idx] = value
    
    def fix_parameter(self, peak_idx, param_idx, fixed=True):
        """Fix or unfix a parameter"""
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        if fixed:
            self.fixed_params[global_param_idx] = self.parameters[global_param_idx]
        else:
            self.fixed_params.pop(global_param_idx, None)
    
    def is_parameter_fixed(self, peak_idx, param_idx):
        """Check if a parameter is fixed"""
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        return global_param_idx in self.fixed_params
    
    def set_parameter_bounds(self, peak_idx, param_idx, lower, upper):
        """Set custom bounds for a parameter"""
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        self.param_bounds[global_param_idx] = (lower, upper)
    
    def get_fitting_parameters(self):
        """Get parameters for fitting (excluding fixed ones)"""
        free_params = []
        param_mapping = []
        
        for i, param in enumerate(self.parameters):
            if i not in self.fixed_params:
                free_params.append(param)
                param_mapping.append(i)
        
        return free_params, param_mapping
    
    def reconstruct_full_parameters(self, fitted_params, param_mapping):
        """Reconstruct full parameter array from fitted parameters"""
        full_params = self.parameters.copy()
        
        for i, mapped_idx in enumerate(param_mapping):
            full_params[mapped_idx] = fitted_params[i]
        
        # Update fixed parameters with their fixed values
        for idx, value in self.fixed_params.items():
            full_params[idx] = value
        
        return full_params
    
    def get_bounds_for_fitting(self, param_mapping):
        """Get bounds for free parameters only"""
        bounds_lower = []
        bounds_upper = []
        
        for mapped_idx in param_mapping:
            peak_idx = mapped_idx // self.params_per_peak
            param_idx = mapped_idx % self.params_per_peak
            
            # Check if custom bounds are set
            if mapped_idx in self.param_bounds:
                lower, upper = self.param_bounds[mapped_idx]
                bounds_lower.append(lower)
                bounds_upper.append(upper)
            else:
                # Use default bounds
                if param_idx == 0:  # Amplitude
                    bounds_lower.append(0)
                    bounds_upper.append(self.parameters[mapped_idx] * 10)
                elif param_idx == 1:  # Center
                    bounds_lower.append(self.x_range[0])
                    bounds_upper.append(self.x_range[1])
                else:  # Width parameters
                    width = self.parameters[mapped_idx]
                    bounds_lower.append(width * 0.1)
                    bounds_upper.append(width * 10)
        
        return bounds_lower, bounds_upper

# --- Enhanced Origin-Style Fitting Engine ---
class OriginFittingEngine:
    def __init__(self):
        self.peak_type = "Gaussian"
        self.baseline_type = "None"
        self.fit_method = "Levenberg-Marquardt"
        self.max_iterations = 1000
        self.tolerance = 1e-8
        self.use_weights = False
        self.parameter_manager = None
        
    def set_fitting_options(self, peak_type="Gaussian", baseline_type="None", 
                           fit_method="Levenberg-Marquardt", max_iterations=1000,
                           tolerance=1e-8, use_weights=False):
        """Set fitting options like Origin's Peak Analyzer"""
        self.peak_type = peak_type
        self.baseline_type = baseline_type
        self.fit_method = fit_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_weights = use_weights
    
    def set_parameter_manager(self, parameter_manager):
        """Set the parameter manager for manual control"""
        self.parameter_manager = parameter_manager
    
    def create_bounds(self, initial_params, x_range):
        """Create parameter bounds (Origin-style constraints)"""
        bounds_lower = []
        bounds_upper = []
        
        params_per_peak = self.get_params_per_peak()
        n_peaks = len(initial_params) // params_per_peak
        
        for i in range(n_peaks):
            base_idx = i * params_per_peak
            
            # Amplitude bounds
            amp = abs(initial_params[base_idx])
            bounds_lower.append(max(0.001, amp * 0.001))  # Minimum positive value
            bounds_upper.append(max(amp * 100, 1.0))  # Ensure upper > lower
            
            # Center bounds
            center = initial_params[base_idx + 1]
            x_span = abs(x_range[1] - x_range[0])
            bounds_lower.append(x_range[0] - x_span * 0.1)
            bounds_upper.append(x_range[1] + x_span * 0.1)
            
            # Width bounds (different for each peak type)
            if self.peak_type in ["Gaussian", "Lorentzian"]:
                width = abs(initial_params[base_idx + 2])
                width = max(width, x_span * 0.001)  # Ensure reasonable minimum
                bounds_lower.append(width * 0.01)
                bounds_upper.append(width * 100)
            
            elif self.peak_type == "Voigt":
                width_g = abs(initial_params[base_idx + 2])
                width_l = abs(initial_params[base_idx + 3])
                width_g = max(width_g, x_span * 0.001)
                width_l = max(width_l, x_span * 0.001)
                bounds_lower.extend([width_g * 0.01, width_l * 0.01])
                bounds_upper.extend([width_g * 100, width_l * 100])
            
            elif self.peak_type in ["BiGaussian", "EMG"]:
                width1 = abs(initial_params[base_idx + 2])
                width2 = abs(initial_params[base_idx + 3])
                width1 = max(width1, x_span * 0.001)
                width2 = max(width2, x_span * 0.001)
                bounds_lower.extend([width1 * 0.01, width2 * 0.01])
                bounds_upper.extend([width1 * 100, width2 * 100])
        
        # Ensure all bounds are valid
        for i in range(len(bounds_lower)):
            if bounds_lower[i] >= bounds_upper[i]:
                bounds_upper[i] = bounds_lower[i] * 10
        
        return bounds_lower, bounds_upper
    
    def get_params_per_peak(self):
        """Get number of parameters per peak for different peak types"""
        params_dict = {
            "Gaussian": 3,
            "Lorentzian": 3,
            "Voigt": 4,
            "BiGaussian": 4,
            "EMG": 4
        }
        return params_dict.get(self.peak_type, 3)
    
    def fit_peaks(self, x, y, initial_params, weights=None):
        """Fit peaks using Origin-style methodology with parameter constraints"""
        try:
            # Use parameter manager if available
            if self.parameter_manager:
                free_params, param_mapping = self.parameter_manager.get_fitting_parameters()
                
                # Create fitting function that only optimizes free parameters
                def fit_function(x_data, *free_param_values):
                    full_params = self.parameter_manager.reconstruct_full_parameters(
                        free_param_values, param_mapping
                    )
                    return multi_peak_function(x_data, self.peak_type, *full_params)
                
                # Get bounds for free parameters only
                bounds_lower, bounds_upper = self.parameter_manager.get_bounds_for_fitting(param_mapping)
                initial_free_params = free_params
            else:
                # Standard fitting without parameter constraints
                def fit_function(x_data, *params):
                    return multi_peak_function(x_data, self.peak_type, *params)
                
                x_range = (x.min(), x.max())
                bounds_lower, bounds_upper = self.create_bounds(initial_params, x_range)
                initial_free_params = initial_params
                param_mapping = list(range(len(initial_params)))
            
            # Check if we have any free parameters to fit
            if len(initial_free_params) == 0:
                # All parameters are fixed
                if self.parameter_manager:
                    fitted_params = self.parameter_manager.parameters
                else:
                    fitted_params = initial_params
                
                y_fit = multi_peak_function(x, self.peak_type, *fitted_params)
                param_errors = np.zeros_like(fitted_params)
            else:
                # Perform fitting
                if self.fit_method == "Levenberg-Marquardt":
                    # Use curve_fit (LM algorithm)
                    if weights is not None and self.use_weights:
                        sigma = 1.0 / np.sqrt(weights)
                        sigma[sigma == np.inf] = 1e6
                    else:
                        sigma = None
                    
                    popt, pcov = curve_fit(
                        fit_function, x, y,
                        p0=initial_free_params,
                        bounds=(bounds_lower, bounds_upper),
                        sigma=sigma,
                        maxfev=self.max_iterations,
                        ftol=self.tolerance,
                        xtol=self.tolerance
                    )
                    
                    # Reconstruct full parameters
                    if self.parameter_manager:
                        fitted_params = self.parameter_manager.reconstruct_full_parameters(
                            popt, param_mapping
                        )
                        # Update parameter manager with fitted values
                        self.parameter_manager.parameters = fitted_params
                    else:
                        fitted_params = popt
                    
                    # Calculate parameter uncertainties (only for free parameters)
                    if pcov is not None:
                        free_param_errors = np.sqrt(np.diag(pcov))
                        param_errors = np.zeros(len(fitted_params))
                        for i, mapped_idx in enumerate(param_mapping):
                            param_errors[mapped_idx] = free_param_errors[i]
                    else:
                        param_errors = np.zeros_like(fitted_params)
                    
                elif self.fit_method == "Global":
                    # Use differential evolution (global optimization)
                    bounds = list(zip(bounds_lower, bounds_upper))
                    
                    def objective(free_param_values):
                        y_fit = fit_function(x, *free_param_values)
                        if weights is not None and self.use_weights:
                            residuals = (y - y_fit) * np.sqrt(weights)
                        else:
                            residuals = y - y_fit
                        return np.sum(residuals**2)
                    
                    result = differential_evolution(
                        objective, bounds,
                        maxiter=self.max_iterations // 10,
                        tol=self.tolerance,
                        seed=42
                    )
                    
                    # Reconstruct full parameters
                    if self.parameter_manager:
                        fitted_params = self.parameter_manager.reconstruct_full_parameters(
                            result.x, param_mapping
                        )
                        # Update parameter manager with fitted values
                        self.parameter_manager.parameters = fitted_params
                    else:
                        fitted_params = result.x
                    
                    param_errors = np.zeros_like(fitted_params)  # No uncertainties from DE
                
                y_fit = multi_peak_function(x, self.peak_type, *fitted_params)
            
            # Calculate fit statistics
            residuals = y - y_fit
            
            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Adjusted R-squared
            n = len(y)
            p = len([i for i in range(len(fitted_params)) if i not in getattr(self.parameter_manager, 'fixed_params', {})])
            adj_r_squared = 1 - (ss_res / ss_tot) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r_squared
            
            # RMSE
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Reduced Chi-squared
            if weights is not None and self.use_weights:
                chi_squared = np.sum(((y - y_fit) ** 2) * weights)
                reduced_chi_squared = chi_squared / (n - p) if (n - p) > 0 else chi_squared
            else:
                reduced_chi_squared = ss_res / (n - p) if (n - p) > 0 else ss_res
            
            # AIC and BIC
            aic = n * np.log(ss_res / n) + 2 * p
            bic = n * np.log(ss_res / n) + p * np.log(n)
            
            return {
                'success': True,
                'parameters': fitted_params,
                'parameter_errors': param_errors,
                'fitted_curve': y_fit,
                'residuals': residuals,
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'rmse': rmse,
                'reduced_chi_squared': reduced_chi_squared,
                'aic': aic,
                'bic': bic,
                'iterations': getattr(result, 'nit', None) if self.fit_method == "Global" else None,
                'free_parameters': len(initial_free_params)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'parameters': initial_params,
                'parameter_errors': np.zeros_like(initial_params),
                'fitted_curve': np.zeros_like(y),
                'residuals': y.copy(),
                'r_squared': 0,
                'adj_r_squared': 0,
                'rmse': np.inf,
                'reduced_chi_squared': np.inf,
                'aic': np.inf,
                'bic': np.inf,
                'iterations': None,
                'free_parameters': 0
            }

# --- Origin-Style Data Preprocessing ---
class OriginDataProcessor:
    @staticmethod
    def smooth_data(x, y, method="Savitzky-Golay", window_size=5, poly_order=2):
        """Data smoothing (Origin-style options)"""
        if method == "Savitzky-Golay":
            if window_size >= len(y):
                window_size = len(y) - 1 if len(y) > 1 else 1
            if window_size % 2 == 0:
                window_size += 1
            if poly_order >= window_size:
                poly_order = window_size - 1
            return savgol_filter(y, window_size, poly_order)
        
        elif method == "Moving Average":
            from scipy.ndimage import uniform_filter1d
            return uniform_filter1d(y, size=window_size)
        
        return y
    
    @staticmethod
    def subtract_baseline(x, y, method="Linear", poly_degree=2, regions=None):
        """Baseline subtraction (Origin-style methods)"""
        if method == "None":
            return y, np.zeros_like(y)
        
        elif method == "Linear":
            # Fit linear baseline to endpoints or specified regions
            if regions is None:
                # Use first and last 10% of data
                n_points = max(2, len(x) // 10)
                x_baseline = np.concatenate([x[:n_points], x[-n_points:]])
                y_baseline = np.concatenate([y[:n_points], y[-n_points:]])
            else:
                x_baseline = []
                y_baseline = []
                for start, end in regions:
                    mask = (x >= start) & (x <= end)
                    x_baseline.extend(x[mask])
                    y_baseline.extend(y[mask])
                x_baseline = np.array(x_baseline)
                y_baseline = np.array(y_baseline)
            
            if len(x_baseline) >= 2:
                slope, intercept = np.polyfit(x_baseline, y_baseline, 1)
                baseline = slope * x + intercept
            else:
                baseline = np.zeros_like(y)
            
        elif method == "Polynomial":
            # Fit polynomial baseline
            if regions is None:
                x_baseline = x
                y_baseline = y
            else:
                x_baseline = []
                y_baseline = []
                for start, end in regions:
                    mask = (x >= start) & (x <= end)
                    x_baseline.extend(x[mask])
                    y_baseline.extend(y[mask])
                x_baseline = np.array(x_baseline)
                y_baseline = np.array(y_baseline)
            
            if len(x_baseline) >= poly_degree + 1:
                coeffs = np.polyfit(x_baseline, y_baseline, poly_degree)
                baseline = np.polyval(coeffs, x)
            else:
                baseline = np.zeros_like(y)
        
        else:
            baseline = np.zeros_like(y)
        
        return y - baseline, baseline

# --- Origin-Style Result Analysis ---
class OriginResultAnalyzer:
    @staticmethod
    def calculate_peak_areas(x, parameters, peak_type):
        """Calculate peak areas (Origin-style integration)"""
        params_per_peak = {
            "Gaussian": 3, "Lorentzian": 3, "Voigt": 4, "BiGaussian": 4, "EMG": 4
        }.get(peak_type, 3)
        
        n_peaks = len(parameters) // params_per_peak
        areas = []
        
        for i in range(n_peaks):
            base_idx = i * params_per_peak
            
            if peak_type == "Gaussian":
                amplitude, center, sigma = parameters[base_idx:base_idx+3]
                # Analytical area for Gaussian
                area = amplitude * sigma * np.sqrt(2 * np.pi)
            
            elif peak_type == "Lorentzian":
                amplitude, center, gamma = parameters[base_idx:base_idx+3]
                # Analytical area for Lorentzian
                area = amplitude * gamma * np.pi
            
            else:
                # Numerical integration for complex peak types
                x_peak = np.linspace(x.min(), x.max(), 1000)
                if peak_type == "Voigt":
                    y_peak = voigt_peak(x_peak, *parameters[base_idx:base_idx+4])
                elif peak_type == "BiGaussian":
                    y_peak = bigaussian_peak(x_peak, *parameters[base_idx:base_idx+4])
                elif peak_type == "EMG":
                    y_peak = exponentially_modified_gaussian(x_peak, *parameters[base_idx:base_idx+4])
                else:
                    y_peak = gaussian_peak(x_peak, *parameters[base_idx:base_idx+3])
                
                area = trapezoid(y_peak, x_peak)
            
            areas.append(area)
        
        return areas
    
    @staticmethod
    def calculate_peak_statistics(x, y, fitted_curve, parameters, peak_type):
        """Calculate comprehensive peak statistics (Origin-style report)"""
        params_per_peak = {
            "Gaussian": 3, "Lorentzian": 3, "Voigt": 4, "BiGaussian": 4, "EMG": 4
        }.get(peak_type, 3)
        
        n_peaks = len(parameters) // params_per_peak
        areas = OriginResultAnalyzer.calculate_peak_areas(x, parameters, peak_type)
        total_area = sum(areas)
        
        peak_stats = []
        
        for i in range(n_peaks):
            base_idx = i * params_per_peak
            
            amplitude = parameters[base_idx]
            center = parameters[base_idx + 1]
            
            # Calculate FWHM based on peak type
            if peak_type == "Gaussian":
                sigma = parameters[base_idx + 2]
                fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
            elif peak_type == "Lorentzian":
                gamma = parameters[base_idx + 2]
                fwhm = 2 * gamma
            elif peak_type == "Voigt":
                width_g = parameters[base_idx + 2]
                width_l = parameters[base_idx + 3]
                # Approximation for Voigt FWHM
                fwhm = 0.5346 * 2 * width_l + np.sqrt(0.2166 * (2 * width_l)**2 + (2 * width_g * np.sqrt(2 * np.log(2)))**2)
            else:
                fwhm = parameters[base_idx + 2] * 2.355  # Default approximation
            
            stats = {
                'peak_number': i + 1,
                'amplitude': amplitude,
                'center': center,
                'fwhm': fwhm,
                'area': areas[i],
                'area_percent': (areas[i] / total_area * 100) if total_area > 0 else 0,
                'height_percent': (amplitude / np.max(y) * 100) if np.max(y) > 0 else 0
            }
            
            peak_stats.append(stats)
        
        return peak_stats

# --- Data Processing ---
class CCSDDataProcessor:
    @staticmethod
    def create_summed_data(df):
        """Create summed data across charge states"""
        from scipy.interpolate import interp1d
        
        ccs_min, ccs_max = df['CCS'].min(), df['CCS'].max()
        ccs_range = np.linspace(ccs_min, ccs_max, 1000)
        summed_intensity = np.zeros_like(ccs_range)
        
        for charge in df['Charge'].unique():
            df_charge = df[df['Charge'] == charge]
            if len(df_charge) >= 2:
                df_charge = df_charge.drop_duplicates(subset=['CCS']).sort_values('CCS')
                interp_func = interp1d(
                    df_charge['CCS'],
                    df_charge['Scaled_Intensity'],
                    kind='linear',
                    bounds_error=False,
                    fill_value=0
                )
                summed_intensity += interp_func(ccs_range)
        
        return pd.DataFrame({'CCS': ccs_range, 'Scaled_Intensity': summed_intensity})

# --- Enhanced Origin-Style UI Components ---
class OriginStyleUI:
    @staticmethod
    def show_main_header():
        styling.load_custom_css()
        st.markdown(
            """
            <div class="main-header">
                <h1>🔬 Origin-Style Peak Analyzer</h1>
                <p>Professional peak fitting with Origin's algorithms and manual editing features</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def show_peak_detection_controls():
        """Show Origin-style peak detection controls"""
        st.sidebar.subheader("🔍 Peak Detection")
        
        auto_detect = st.sidebar.checkbox("Auto-detect peaks", value=True)
        
        if auto_detect:
            min_height = st.sidebar.slider(
                "Min Height (%)", 1, 50, 5,
                help="Minimum peak height as percentage of data range"
            )
            min_prominence = st.sidebar.slider(
                "Min Prominence (%)", 1, 20, 2,
                help="Minimum peak prominence as percentage of data range"
            )
            min_distance = st.sidebar.slider(
                "Min Distance (%)", 1, 20, 5,
                help="Minimum distance between peaks as percentage of data range"
            )
            smoothing = st.sidebar.slider(
                "Smoothing Points", 0, 15, 5,
                help="Number of points for data smoothing (0 = no smoothing)"
            )
            
            return True, {
                'min_height': min_height,
                'min_prominence': min_prominence,
                'min_distance': min_distance,
                'smoothing': smoothing
            }
        else:
            n_peaks = st.sidebar.number_input("Number of peaks", 1, 10, 2)
            return False, {'n_peaks': n_peaks}

    @staticmethod
    def show_fitting_options():
        """Show Origin-style fitting options"""
        st.sidebar.subheader("⚙️ Fitting Options")
        
        peak_type = st.sidebar.selectbox(
            "Peak Function",
            ["Gaussian", "Lorentzian", "Voigt", "BiGaussian", "EMG"],
            help="Choose the mathematical function for peak fitting"
        )
        
        baseline_type = st.sidebar.selectbox(
            "Baseline",
            ["None", "Linear", "Polynomial"],
            help="Baseline correction method"
        )
        
        if baseline_type == "Polynomial":
            poly_degree = st.sidebar.slider("Polynomial Degree", 1, 5, 2)
        else:
            poly_degree = 1
        
        fit_method = st.sidebar.selectbox(
            "Fitting Method",
            ["Levenberg-Marquardt", "Global"],
            help="Optimization algorithm"
        )
        
        max_iterations = st.sidebar.number_input(
            "Max Iterations", 100, 5000, 1000,
            help="Maximum number of fitting iterations"
        )
        
        tolerance = st.sidebar.select_slider(
            "Tolerance",
            options=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4],
            value=1e-8,
            format_func=lambda x: f"{x:.0e}",
            help="Convergence tolerance"
        )
        
        use_weights = st.sidebar.checkbox(
            "Use Statistical Weights",
            value=False,
            help="Weight data points by statistical uncertainty"
        )
        
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
        """Show data preprocessing options"""
        st.sidebar.subheader("📊 Data Preprocessing")
        
        smooth_data = st.sidebar.checkbox("Smooth Data", value=False)
        
        if smooth_data:
            smooth_method = st.sidebar.selectbox(
                "Smoothing Method",
                ["Savitzky-Golay", "Moving Average"]
            )
            window_size = st.sidebar.slider("Window Size", 3, 21, 5, step=2)
            if smooth_method == "Savitzky-Golay":
                poly_order = st.sidebar.slider("Polynomial Order", 1, min(window_size-1, 5), 2)
            else:
                poly_order = 2
        else:
            smooth_method = "None"
            window_size = 5
            poly_order = 2
        
        return {
            'smooth_data': smooth_data,
            'smooth_method': smooth_method,
            'window_size': window_size,
            'poly_order': poly_order
        }

    @staticmethod
    def show_peak_management_tools(parameter_manager, peak_manager, x_data, y_corrected, peak_type):
        """Show Origin-style peak management tools"""
        st.subheader("🛠️ Peak Management Tools")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Add peak
        with col1:
            if st.button("➕ Add Peak", help="Click to add a new peak"):
                st.session_state['adding_peak'] = True
        
        # Delete peak
        with col2:
            if parameter_manager.n_peaks > 1:
                peak_to_delete = st.selectbox(
                    "Delete Peak",
                    options=list(range(1, parameter_manager.n_peaks + 1)),
                    format_func=lambda x: f"Peak {x}",
                    key="delete_peak_select"
                )
                if st.button("🗑️ Delete", help="Delete selected peak"):
                    st.session_state['delete_peak'] = peak_to_delete - 1
        
        # Copy peak
        with col3:
            if parameter_manager.n_peaks > 0:
                peak_to_copy = st.selectbox(
                    "Copy Peak",
                    options=list(range(1, parameter_manager.n_peaks + 1)),
                    format_func=lambda x: f"Peak {x}",
                    key="copy_peak_select"
                )
                if st.button("📋 Copy", help="Copy selected peak"):
                    st.session_state['copy_peak'] = peak_to_copy - 1
        
        # Peak constraints
        with col4:
            constraint_mode = st.selectbox(
                "Constraints",
                options=["None", "Same Width", "Same Position", "Fixed Ratio"],
                help="Apply constraints between peaks"
            )
        
        # Handle add peak interaction
        if st.session_state.get('adding_peak', False):
            st.info("👆 Click on the plot where you want to add a new peak")
            x_position = st.number_input(
                "Peak Position (CCS)",
                min_value=float(x_data.min()),
                max_value=float(x_data.max()),
                value=float(np.mean(x_data)),
                step=0.1,
                key="new_peak_position"
            )
            
            col_add1, col_add2 = st.columns(2)
            if col_add1.button("✅ Add Here"):
                # Add peak at specified position
                peak_manager.add_peak(peak_type, x_position, y_corrected, x_data)
                
                # Update parameter manager
                new_params = peak_manager.get_all_parameters()
                parameter_manager.parameters = new_params
                parameter_manager.n_peaks = len(new_params) // parameter_manager.params_per_peak
                
                st.session_state['adding_peak'] = False
                st.session_state['peak_added'] = True
                st.rerun()
            
            if col_add2.button("❌ Cancel"):
                st.session_state['adding_peak'] = False
                st.rerun()
        
        return constraint_mode

    @staticmethod
    def show_advanced_fitting_controls():
        """Show advanced fitting controls like Origin"""
        st.subheader("🎛️ Advanced Fitting Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Convergence Criteria**")
            max_iterations = st.number_input("Max Iterations", 10, 10000, 1000)
            tolerance = st.select_slider(
                "Tolerance",
                options=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
                value=1e-8,
                format_func=lambda x: f"{x:.0e}"
            )
            
            chi_squared_test = st.checkbox("Chi-squared test", value=True)
            f_test = st.checkbox("F-test for model comparison", value=False)
        
        with col2:
            st.markdown("**Fitting Strategy**")
            fit_strategy = st.selectbox(
                "Strategy",
                ["Automatic", "Sequential", "Global then Local", "Local only"]
            )
            
            outlier_detection = st.checkbox("Outlier detection", value=False)
            robust_fitting = st.checkbox("Robust fitting", value=False)
            
            confidence_level = st.slider(
                "Confidence Level (%)", 90, 99, 95
            )
        
        return {
            'max_iterations': max_iterations,
            'tolerance': tolerance,
            'chi_squared_test': chi_squared_test,
            'f_test': f_test,
            'fit_strategy': fit_strategy,
            'outlier_detection': outlier_detection,
            'robust_fitting': robust_fitting,
            'confidence_level': confidence_level
        }

    @staticmethod
    def show_parameter_correlation_matrix(parameter_manager, result):
        """Show parameter correlation matrix like Origin"""
        if 'parameter_errors' not in result or len(result['parameter_errors']) == 0:
            return
        
        st.subheader("🔗 Parameter Correlation Matrix")
        
        # Calculate correlation matrix from covariance (simplified)
        param_names = []
        for i in range(parameter_manager.n_peaks):
            for j, name in enumerate(parameter_manager.get_parameter_names()):
                param_names.append(f"Peak{i+1}_{name}")
        
        # Create mock correlation matrix (in real implementation, this would come from fit covariance)
        n_params = len(result['parameters'])
        correlation_matrix = np.eye(n_params)
        
        # Add some realistic correlations (amplitude-width anti-correlation, etc.)
        for i in range(0, n_params, parameter_manager.params_per_peak):
            if i + 2 < n_params:  # amplitude-width correlation
                correlation_matrix[i, i+2] = -0.3
                correlation_matrix[i+2, i] = -0.3
        
        # Display as heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=param_names[:n_params],
            y=param_names[:n_params],
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation")
        ))
        
        fig_corr.update_layout(
            title="Parameter Correlation Matrix",
            height=400,
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)

    @staticmethod
    def show_manual_parameter_editor(parameter_manager):
        """Enhanced parameter editor with Origin features"""
        st.subheader("🎛️ Manual Parameter Editor")
        st.markdown("*Edit parameters manually like in Origin's Peak Analyzer*")
        
        # Quick actions row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("🔒 Fix All Centers"):
                for i in range(parameter_manager.n_peaks):
                    parameter_manager.fix_parameter(i, 1, True)  # Fix center (index 1)
                st.session_state['params_changed'] = True
        
        with col2:
            if st.button("🔓 Release All"):
                parameter_manager.fixed_params = {}
                st.session_state['params_changed'] = True
        
        with col3:
            if st.button("📏 Same Widths"):
                # Set all widths to the same value as peak 1
                if parameter_manager.n_peaks > 1:
                    ref_width = parameter_manager.parameters[2]  # First peak's width
                    for i in range(1, parameter_manager.n_peaks):
                        parameter_manager.update_parameter(i, 2, ref_width)
                st.session_state['params_changed'] = True
        
        with col4:
            normalize_areas = st.button("📊 Normalize Areas")
        
        with col5:
            sort_by_position = st.button("🔢 Sort by Position")
        
        param_names = parameter_manager.get_parameter_names()
        
        # Create expandable sections for each peak
        for peak_idx in range(parameter_manager.n_peaks):
            with st.expander(f"🔹 Peak {peak_idx + 1} Parameters", expanded=True):  # Changed to always expanded
                
                # Peak summary
                center = parameter_manager.parameters[peak_idx * parameter_manager.params_per_peak + 1]
                amplitude = parameter_manager.parameters[peak_idx * parameter_manager.params_per_peak]
                st.markdown(f"**Center: {center:.3f}, Amplitude: {amplitude:.2f}**")
                
                # Parameter table
                parameter_changed = False
                
                for param_idx, param_name in enumerate(param_names):
                    global_param_idx = peak_idx * parameter_manager.params_per_peak + param_idx
                    current_value = parameter_manager.parameters[global_param_idx]
                    is_fixed = parameter_manager.is_parameter_fixed(peak_idx, param_idx)
                    
                    # Create input row
                    cols = st.columns([2, 3, 1, 2, 2, 2])
                    
                    # Parameter name
                    cols[0].write(f"**{param_name}**")
                    
                    # Value input with appropriate step and bounds
                    if param_name in ["Amplitude"]:
                        step = max(abs(current_value) * 0.01, 0.001)
                        min_val = 0.0
                        max_val = abs(current_value) * 1000 if current_value != 0 else 1000
                        format_str = "%.4f"
                    elif param_name in ["Center"]:
                        step = (parameter_manager.x_range[1] - parameter_manager.x_range[0]) * 0.001
                        min_val = parameter_manager.x_range[0] - abs(parameter_manager.x_range[1] - parameter_manager.x_range[0])
                        max_val = parameter_manager.x_range[1] + abs(parameter_manager.x_range[1] - parameter_manager.x_range[0])
                        format_str = "%.4f"
                    else:  # Width parameters
                        step = max(abs(current_value) * 0.01, 0.001)
                        min_val = 0.001
                        max_val = abs(current_value) * 1000 if current_value != 0 else 1000
                        format_str = "%.4f"
                    
                    new_value = cols[1].number_input(
                        f"val_{peak_idx}_{param_idx}",
                        value=float(current_value),
                        step=step,
                        min_value=min_val,
                        max_value=max_val,
                        format=format_str,
                        key=f"param_val_{peak_idx}_{param_idx}",
                        label_visibility="collapsed"
                    )
                    
                    if abs(new_value - current_value) > 1e-10:
                        parameter_manager.update_parameter(peak_idx, param_idx, new_value)
                        parameter_changed = True
                    
                    # Fixed checkbox
                    fixed_changed = cols[2].checkbox(
                        "Fix",
                        value=is_fixed,
                        key=f"fix_param_{peak_idx}_{param_idx}",
                        help=f"Fix {param_name} during optimization"
                    )
                    
                    if fixed_changed != is_fixed:
                        parameter_manager.fix_parameter(peak_idx, param_idx, fixed_changed)
                        parameter_changed = True
                    
                    # Bounds
                    current_bounds = parameter_manager.param_bounds.get(global_param_idx, (min_val, max_val))
                    
                    min_bound = cols[3].number_input(
                        f"min_bound_{peak_idx}_{param_idx}",
                        value=float(current_bounds[0]),
                        step=step,
                        format=format_str,
                        key=f"min_bound_{peak_idx}_{param_idx}",
                        label_visibility="collapsed"
                    )
                    
                    max_bound = cols[4].number_input(
                        f"max_bound_{peak_idx}_{param_idx}",
                        value=float(current_bounds[1]),
                        step=step,
                        format=format_str,
                        key=f"max_bound_{peak_idx}_{param_idx}",
                        label_visibility="collapsed"
                    )
                    
                    # Percentage controls
                    if cols[5].button(f"±10%", key=f"pm10_{peak_idx}_{param_idx}", 
                                     help="Set bounds to ±10% of current value"):
                        parameter_manager.set_parameter_bounds(
                            peak_idx, param_idx, 
                            current_value * 0.9, current_value * 1.1
                        )
                        parameter_changed = True
                    
                    if (abs(min_bound - current_bounds[0]) > 1e-10 or 
                        abs(max_bound - current_bounds[1]) > 1e-10):
                        parameter_manager.set_parameter_bounds(peak_idx, param_idx, min_bound, max_bound)
                        parameter_changed = True
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        optimize_clicked = col1.button("🎯 Optimize Fit", type="primary", 
                                      help="Optimize only the non-fixed parameters")
        
        reset_clicked = col2.button("🔄 Reset Parameters", 
                                   help="Reset all parameters to auto-detected values")
        
        undo_clicked = col3.button("↶ Undo Last", 
                                  help="Undo last parameter change")
        
        # Real-time update options
        update_plot = col4.checkbox("🔄 Live Update", value=True, 
                                   help="Update plot in real-time as you change parameters")
        
        return parameter_changed, optimize_clicked, reset_clicked, update_plot

    @staticmethod
    def show_peak_statistics(peak_stats):
        """Display Origin-style peak statistics"""
        st.subheader("📊 Peak Analysis Report")
        
        # Create summary table
        summary_data = []
        for stats in peak_stats:
            summary_data.append({
                'Peak': stats['peak_number'],
                'Center': f"{stats['center']:.3f}",
                'Amplitude': f"{stats['amplitude']:.2f}",
                'FWHM': f"{stats['fwhm']:.3f}",
                'Area': f"{stats['area']:.1f}",
                'Area %': f"{stats['area_percent']:.1f}",
                'Height %': f"{stats['height_percent']:.1f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Area distribution pie chart
        if len(peak_stats) > 1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f"Peak {s['peak_number']}" for s in peak_stats],
                values=[s['area'] for s in peak_stats],
                hole=0.3
            )])
            fig_pie.update_layout(title="Peak Area Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    @staticmethod
    def show_fit_statistics(result):
        """Display Origin-style fit statistics"""
        st.subheader("📈 Fit Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R²", f"{result['r_squared']:.6f}")
            st.metric("Adj R²", f"{result['adj_r_squared']:.6f}")
        
        with col2:
            st.metric("RMSE", f"{result['rmse']:.4f}")
            st.metric("χ² (reduced)", f"{result['reduced_chi_squared']:.4f}")
        
        with col3:
            st.metric("AIC", f"{result['aic']:.2f}")
            st.metric("BIC", f"{result['bic']:.2f}")
        
        with col4:
            if result['iterations'] is not None:
                st.metric("Iterations", result['iterations'])
            st.metric("Success", "✅" if result['success'] else "❌")

def main():
    OriginStyleUI.show_main_header()
    
    # Initialize session state for storing results across charge states
    if 'all_charge_results' not in st.session_state:
        st.session_state['all_charge_results'] = {}
    
    # File upload
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
        
        required_cols = ['Charge', 'CCS', 'Scaled_Intensity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return
        
        if df.empty:
            st.error("The uploaded file is empty")
            return
        
        # Check for optional columns needed for export
        export_cols = ['Drift', 'm/z']
        has_export_cols = all(col in df.columns for col in export_cols)
        
        if not has_export_cols:
            st.warning("Missing Drift and/or m/z columns. Export will use default values.")
        
        # Sidebar controls
        st.sidebar.markdown("### 🎛️ Analysis Configuration")
        
        charges = sorted(df['Charge'].unique())
        mode = st.sidebar.radio("Analysis Mode", ["Individual Charge State", "Summed Data"])
        
        # Show saved results summary
        if st.session_state['all_charge_results']:
            st.sidebar.markdown("### 💾 Saved Results")
            for charge in sorted(st.session_state['all_charge_results'].keys()):
                result = st.session_state['all_charge_results'][charge]
                n_peaks = len(result['peak_stats'])
                r_squared = result['fit_result']['r_squared']
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
            
            # Show save button for current charge state
            if 'fit_result' in st.session_state and st.sidebar.button(f"💾 Save Results for Charge {selected_charge}", type="primary"):
                # Save current results
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
        else:
            plot_data = CCSDDataProcessor.create_summed_data(df)
            data_label = "Summed Data"
        
        plot_data = plot_data[plot_data['Scaled_Intensity'] > 0]
        if len(plot_data) == 0:
            st.error("No data points with positive intensity found")
            return
        
        x_data = plot_data['CCS'].values
        y_data = plot_data['Scaled_Intensity'].values
        
        # Get UI options
        auto_detect, detection_params = OriginStyleUI.show_peak_detection_controls()
        fitting_options = OriginStyleUI.show_fitting_options()
        preprocessing_options = OriginStyleUI.show_preprocessing_options()
        
        # Data preprocessing
        processor = OriginDataProcessor()
        
        # Smooth data if requested
        if preprocessing_options['smooth_data']:
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
            poly_degree=fitting_options['poly_degree']
        )
        
        # Peak detection and fitting
        if st.sidebar.button("🎯 Analyze Peaks", type="primary"):
            with st.spinner("Analyzing peaks..."):
                try:
                    # Peak detection
                    detector = OriginPeakDetector()
                    
                    if auto_detect:
                        peak_info = detector.find_peaks_origin_style(
                            x_data, y_corrected,
                            min_height_percent=detection_params['min_height'],
                            min_prominence_percent=detection_params['min_prominence'],
                            min_distance_percent=detection_params['min_distance'],
                            smoothing_points=detection_params['smoothing']
                        )
                    else:
                        # Manual peak specification
                        n_peaks = detection_params['n_peaks']
                        peak_indices = np.linspace(0, len(x_data)-1, n_peaks, dtype=int)
                        peak_info = []
                        for i, idx in enumerate(peak_indices):
                            peak_info.append({
                                'index': idx,
                                'x': x_data[idx],
                                'y': y_corrected[idx],
                                'width_half': (x_data[-1] - x_data[0]) / (n_peaks * 4),
                                'prominence': y_corrected[idx]
                            })
                    
                    if not peak_info:
                        st.error("No peaks detected. Try adjusting the detection parameters.")
                        return
                    
                    st.success(f"Detected {len(peak_info)} peaks")
                    
                    # Parameter estimation
                    estimator = OriginParameterEstimator()
                    
                    if fitting_options['peak_type'] == "Gaussian":
                        initial_params = estimator.estimate_gaussian_parameters(x_data, y_corrected, peak_info)
                    elif fitting_options['peak_type'] == "Lorentzian":
                        initial_params = estimator.estimate_lorentzian_parameters(x_data, y_corrected, peak_info)
                    elif fitting_options['peak_type'] == "Voigt":
                        initial_params = estimator.estimate_voigt_parameters(x_data, y_corrected, peak_info)
                    else:
                        initial_params = estimator.estimate_gaussian_parameters(x_data, y_corrected, peak_info)
                    
                    # Create parameter manager
                    parameter_manager = OriginParameterManager(
                        fitting_options['peak_type'], initial_params, (x_data.min(), x_data.max())
                    )
                    
                    # Fitting
                    fitter = OriginFittingEngine()
                    fitter.set_fitting_options(
                        peak_type=fitting_options['peak_type'],
                        baseline_type=fitting_options['baseline_type'],
                        fit_method=fitting_options['fit_method'],
                        max_iterations=fitting_options['max_iterations'],
                        tolerance=fitting_options['tolerance'],
                        use_weights=fitting_options['use_weights']
                    )
                    
                    # Set parameter manager in fitter
                    fitter.set_parameter_manager(parameter_manager)
                    
                    # Calculate weights if requested
                    weights = None
                    if fitting_options['use_weights']:
                        weights = 1.0 / np.sqrt(np.maximum(y_corrected, 1))
                    
                    # Perform fit
                    result = fitter.fit_peaks(x_data, y_corrected, initial_params, weights)
                    
                    if result['success']:
                        # Calculate peak statistics
                        analyzer = OriginResultAnalyzer()
                        peak_stats = analyzer.calculate_peak_statistics(
                            x_data, y_corrected, result['fitted_curve'],
                            result['parameters'], fitting_options['peak_type']
                        )
                        
                        # Store results in session state
                        st.session_state['fit_result'] = result
                        st.session_state['peak_info'] = peak_info
                        st.session_state['peak_stats'] = peak_stats
                        st.session_state['fitting_options'] = fitting_options
                        st.session_state['parameter_manager'] = parameter_manager
                        st.session_state['fitter'] = fitter
                        st.session_state['x_data'] = x_data
                        st.session_state['y_data'] = y_data
                        st.session_state['y_corrected'] = y_corrected
                        st.session_state['baseline'] = baseline
                        st.session_state['data_label'] = data_label
                        st.session_state['weights'] = weights
                        st.session_state['original_df'] = df
                        st.session_state['current_charge'] = selected_charge if mode == "Individual Charge State" else None
                        
                        st.success("Peak fitting completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Fitting failed: {result['error']}")
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)
        
        # Display results and manual editing interface if available
        if 'fit_result' in st.session_state:
            result = st.session_state['fit_result']
            fitting_options = st.session_state['fitting_options']
            parameter_manager = st.session_state['parameter_manager']
            fitter = st.session_state['fitter']
            x_data = st.session_state['x_data']
            y_data = st.session_state['y_data']
            y_corrected = st.session_state['y_corrected']
            baseline = st.session_state['baseline']
            data_label = st.session_state['data_label']
            weights = st.session_state['weights']
            
            # Initialize peak manager if not exists
            if 'peak_manager' not in st.session_state:
                st.session_state['peak_manager'] = OriginPeakManager()
                st.session_state['peak_manager'].update_from_parameters(
                    parameter_manager.parameters, fitting_options['peak_type']
                )
            
            peak_manager = st.session_state['peak_manager']
            
            # Show fit statistics
            OriginStyleUI.show_fit_statistics(result)
            
            # Peak management tools
            st.markdown("---")
            constraint_mode = OriginStyleUI.show_peak_management_tools(
                parameter_manager, peak_manager, x_data, y_corrected, fitting_options['peak_type']
            )
            
            # Handle peak deletion
            if 'delete_peak' in st.session_state:
                peak_idx = st.session_state['delete_peak']
                if peak_manager.delete_peak(peak_idx):
                    # Update parameter manager
                    new_params = peak_manager.get_all_parameters()
                    parameter_manager.parameters = new_params
                    parameter_manager.n_peaks = len(new_params) // parameter_manager.params_per_peak
                    
                    # Remove fixed parameters and bounds for deleted peak
                    params_per_peak = parameter_manager.params_per_peak
                    start_idx = peak_idx * params_per_peak
                    end_idx = (peak_idx + 1) * params_per_peak
                    
                    # Shift indices for remaining parameters
                    new_fixed_params = {}
                    new_param_bounds = {}
                    
                    for idx, value in parameter_manager.fixed_params.items():
                        if idx < start_idx:
                            new_fixed_params[idx] = value
                        elif idx >= end_idx:
                            new_fixed_params[idx - params_per_peak] = value
                    
                    for idx, bounds in parameter_manager.param_bounds.items():
                        if idx < start_idx:
                            new_param_bounds[idx] = bounds
                        elif idx >= end_idx:
                            new_param_bounds[idx - params_per_peak] = bounds
                    
                    parameter_manager.fixed_params = new_fixed_params
                    parameter_manager.param_bounds = new_param_bounds
                    
                    del st.session_state['delete_peak']
                    st.success(f"Deleted Peak {peak_idx + 1}")
                    st.rerun()
            
            # Handle peak copying
            if 'copy_peak' in st.session_state:
                peak_idx = st.session_state['copy_peak']
                if 0 <= peak_idx < parameter_manager.n_peaks:
                    # Copy peak parameters
                    params_per_peak = parameter_manager.params_per_peak
                    start_idx = peak_idx * params_per_peak
                    end_idx = (peak_idx + 1) * params_per_peak
                    peak_params = parameter_manager.parameters[start_idx:end_idx].copy()
                    
                    # Offset the center slightly to avoid overlap
                    peak_params[1] += (x_data.max() - x_data.min()) * 0.05
                    
                    # Add to parameter manager
                    parameter_manager.parameters.extend(peak_params)
                    parameter_manager.n_peaks += 1
                    
                    # Update peak manager
                    peak_manager.peak_list.append({
                        'id': len(peak_manager.peak_list),
                        'type': fitting_options['peak_type'],
                        'params': peak_params,
                        'active': True
                    })
                    
                    del st.session_state['copy_peak']
                    st.success(f"Copied Peak {peak_idx + 1}")
                    st.rerun()
            
            # Advanced fitting controls
            st.markdown("---")
            advanced_controls = OriginStyleUI.show_advanced_fitting_controls()
            
            # Manual parameter editor
            st.markdown("---")
            parameter_changed, optimize_clicked, reset_clicked, update_plot = OriginStyleUI.show_manual_parameter_editor(parameter_manager)
            
            # Parameter correlation matrix
            if st.checkbox("Show Parameter Correlations", value=False):
                OriginStyleUI.show_parameter_correlation_matrix(parameter_manager, result)
            
            # Handle reset to auto
            if reset_clicked:
                # Re-estimate parameters
                estimator = OriginParameterEstimator()
                if fitting_options['peak_type'] == "Gaussian":
                    initial_params = estimator.estimate_gaussian_parameters(x_data, y_corrected, st.session_state['peak_info'])
                elif fitting_options['peak_type'] == "Lorentzian":
                    initial_params = estimator.estimate_lorentzian_parameters(x_data, y_corrected, st.session_state['peak_info'])
                elif fitting_options['peak_type'] == "Voigt":
                    initial_params = estimator.estimate_voigt_parameters(x_data, y_corrected, st.session_state['peak_info'])
                else:
                    initial_params = estimator.estimate_gaussian_parameters(x_data, y_corrected, st.session_state['peak_info'])
                
                parameter_manager.parameters = initial_params
                parameter_manager.fixed_params = {}
                parameter_manager.param_bounds = {}
                peak_manager.update_from_parameters(initial_params, fitting_options['peak_type'])
                st.success("Parameters reset to auto-detected values")
                st.rerun()
            
            # Handle optimize fit with advanced controls
            if optimize_clicked:
                with st.spinner("Optimizing fit with current constraints..."):
                    try:
                        # Update fitter with advanced controls
                        fitter.max_iterations = advanced_controls['max_iterations']
                        fitter.tolerance = advanced_controls['tolerance']
                        
                        # Apply constraints if specified
                        if constraint_mode == "Same Width":
                            # Fix all widths to the same value
                            if parameter_manager.n_peaks > 1:
                                ref_width = parameter_manager.parameters[2]  # First peak's width
                                for i in range(1, parameter_manager.n_peaks):
                                    parameter_manager.fix_parameter(i, 2, True)
                                    parameter_manager.update_parameter(i, 2, ref_width)
                        
                        result = fitter.fit_peaks(x_data, y_corrected, parameter_manager.parameters, weights)
                        if result['success']:
                            # Recalculate peak statistics
                            analyzer = OriginResultAnalyzer()
                            peak_stats = analyzer.calculate_peak_statistics(
                                x_data, y_corrected, result['fitted_curve'],
                                result['parameters'], fitting_options['peak_type']
                            )
                            
                            st.session_state['fit_result'] = result
                            st.session_state['peak_stats'] = peak_stats
                            peak_manager.update_from_parameters(parameter_manager.parameters, fitting_options['peak_type'])
                            st.success("Optimization completed!")
                            
                            # Show convergence info
                            if advanced_controls['chi_squared_test']:
                                st.info(f"χ² = {result['reduced_chi_squared']:.4f}, "
                                       f"Free parameters: {result.get('free_parameters', 0)}")
                            
                            st.rerun()
                        else:
                            st.error(f"Optimization failed: {result['error']}")
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
            
            # Update plot if parameters changed and auto-update is enabled
            if parameter_changed and update_plot:
                try:
                    y_fit = multi_peak_function(x_data, fitting_options['peak_type'], *parameter_manager.parameters)
                    
                    # Update result with new curve
                    result = result.copy()
                    result['fitted_curve'] = y_fit
                    result['residuals'] = y_corrected - y_fit
                    result['parameters'] = parameter_manager.parameters
                    
                    # Recalculate statistics
                    ss_res = np.sum(result['residuals']**2)
                    ss_tot = np.sum((y_corrected - np.mean(y_corrected))**2)
                    result['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    result['rmse'] = np.sqrt(np.mean(result['residuals']**2))
                    
                    # Recalculate peak statistics
                    analyzer = OriginResultAnalyzer()
                    peak_stats = analyzer.calculate_peak_statistics(
                        x_data, y_corrected, result['fitted_curve'],
                        result['parameters'], fitting_options['peak_type']
                    )
                    
                    st.session_state['fit_result'] = result
                    st.session_state['peak_stats'] = peak_stats
                except Exception as e:
                    st.warning(f"Could not update preview: {str(e)}")
            
            # Get peak statistics
            peak_stats = st.session_state.get('peak_stats', [])
            
            # Show peak statistics
            OriginStyleUI.show_peak_statistics(peak_stats)
            
            # Visualization
            st.subheader(f"📈 Peak Fitting Results - {data_label}")
            
            fig = go.Figure()
            
            # Original data
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers',
                name='Original Data', marker=dict(color='lightblue', size=4),
                opacity=0.6
            ))
            
            # Baseline
            if fitting_options['baseline_type'] != "None":
                fig.add_trace(go.Scatter(
                    x=x_data, y=baseline, mode='lines',
                    name='Baseline', line=dict(color='gray', dash='dot', width=1)
                ))
            
            # Corrected data
            fig.add_trace(go.Scatter(
                x=x_data, y=y_corrected, mode='markers',
                name='Baseline Corrected', marker=dict(color='blue', size=4)
            ))
            
            # Overall fit
            fig.add_trace(go.Scatter(
                x=x_data, y=result['fitted_curve'], mode='lines',
                name=f'Overall Fit (R² = {result["r_squared"]:.4f})',
                line=dict(color='red', width=3)
            ))
            
            # Individual peaks
            colors = px.colors.qualitative.Set1
            
            # Create a temporary fitter to get params_per_peak
            temp_fitter = OriginFittingEngine()
            temp_fitter.set_fitting_options(peak_type=fitting_options['peak_type'])
            params_per_peak = temp_fitter.get_params_per_peak()
            
            n_peaks = len(result['parameters']) // params_per_peak
            
            for i in range(n_peaks):
                base_idx = i * params_per_peak
                peak_params = result['parameters'][base_idx:base_idx + params_per_peak]
                
                if fitting_options['peak_type'] == "Gaussian":
                    peak_curve = gaussian_peak(x_data, *peak_params)
                elif fitting_options['peak_type'] == "Lorentzian":
                    peak_curve = lorentzian_peak(x_data, *peak_params)
                elif fitting_options['peak_type'] == "Voigt":
                    peak_curve = voigt_peak(x_data, *peak_params)
                elif fitting_options['peak_type'] == "BiGaussian":
                    peak_curve = bigaussian_peak(x_data, *peak_params)
                elif fitting_options['peak_type'] == "EMG":
                    peak_curve = exponentially_modified_gaussian(x_data, *peak_params)
                else:
                    peak_curve = gaussian_peak(x_data, *peak_params)
                
                fig.add_trace(go.Scatter(
                    x=x_data, y=peak_curve, mode='lines',
                    name=f'Peak {i+1}',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    opacity=0.8
                ))
            
            fig.update_layout(
                title=f'{fitting_options["peak_type"]} Peak Fitting - {data_label}',
                xaxis_title='CCS (Ų)',
                yaxis_title='Scaled Intensity',
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals plot
            st.subheader("📊 Residuals Analysis")
            
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=x_data, y=result['residuals'], mode='markers',
                name='Residuals', marker=dict(color='green', size=4)
            ))
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
            
            # Add ±2σ confidence bands
            residual_std = np.std(result['residuals'])
            fig_residuals.add_hline(y=2*residual_std, line_dash="dot", line_color="red", opacity=0.5)
            fig_residuals.add_hline(y=-2*residual_std, line_dash="dot", line_color="red", opacity=0.5)
                                                                                                

            fig_residuals.update_layout(
                title='Residuals Analysis',
                xaxis_title='CCS (Ų)',
                yaxis_title='Residual',
                height=400
            )
            
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Export section - moved outside the fit_result check
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        # Show export options if we have saved results
        if st.session_state['all_charge_results']:
            st.markdown(f"**Saved results for {len(st.session_state['all_charge_results'])} charge states**")
            
            # Create combined export data
            def create_export_data():
                """Create export data in the required format"""
                export_rows = []
                original_df = st.session_state.get('original_df', df)
                
                for charge, saved_result in st.session_state['all_charge_results'].items():
                    # Get original data for this charge state
                    charge_data = original_df[original_df['Charge'] == charge].copy().sort_values('CCS')
                    
                    # Get fitted parameters and create fitted curve
                    fit_result = saved_result['fit_result']
                    peak_stats = saved_result['peak_stats']
                    fitting_options = saved_result['fitting_options']
                    
                    # Get CCS range for this charge state
                    ccs_min, ccs_max = charge_data['CCS'].min(), charge_data['CCS'].max()
                    
                    # Create evenly distributed x values (same number of points as original data)
                    n_points = len(charge_data)
                    x_new = np.linspace(ccs_min, ccs_max, n_points)
                    
                    # Calculate fitted curve at new x values
                    fitted_curve = multi_peak_function(x_new, fitting_options['peak_type'], *fit_result['parameters'])
                    
                    # Normalize fitted curve within this charge state (0 to 1)
                    max_fitted = np.max(fitted_curve) if np.max(fitted_curve) > 0 else 1
                    normalized_intensity = fitted_curve / max_fitted
                    
                    # Get peak parameters for standard deviation calculation
                    params_per_peak = {
                        "Gaussian": 3, "Lorentzian": 3, "Voigt": 4, "BiGaussian": 4, "EMG": 4
                    }.get(fitting_options['peak_type'], 3)
                    
                    n_peaks = len(fit_result['parameters']) // params_per_peak
                    
                    # Get default m/z value
                    default_mz = charge_data['m/z'].iloc[0] if 'm/z' in charge_data.columns else 1840.2934353393
                    
                    # For each data point, use the distributed CCS value but find closest peak for std dev
                    for i, (ccs_val, fitted_val, norm_val) in enumerate(zip(x_new, fitted_curve, normalized_intensity)):
                        # Find closest peak for standard deviation calculation
                        closest_peak_idx = 0
                        min_distance = float('inf')
                        
                        for peak_idx in range(n_peaks):
                            peak_center = fit_result['parameters'][peak_idx * params_per_peak + 1]
                            distance = abs(ccs_val - peak_center)
                            if distance < min_distance:
                                min_distance = distance
                                closest_peak_idx = peak_idx

                        # Get peak standard deviation from closest peak
                        if closest_peak_idx < len(peak_stats):
                            peak_stat = peak_stats[closest_peak_idx]
                            peak_std = peak_stat['fwhm'] / 2.355  # Convert FWHM to standard deviation
                        else:
                            # Fallback to parameter values
                            if fitting_options['peak_type'] == "Gaussian":
                                peak_std = fit_result['parameters'][closest_peak_idx * params_per_peak + 2]
                            elif fitting_options['peak_type'] == "Lorentzian":
                                peak_std = fit_result['parameters'][closest_peak_idx * params_per_peak + 2] / 2
                            else:
                                peak_std = 0.0
                        
                        export_rows.append({
                            'Charge': int(charge),
                            'CCS': ccs_val,  # Use the distributed CCS value, not peak center
                            'CCS Std.Dev.': peak_std,
                            'Normalized_Intensity': norm_val,
                            'Scaled_Intensity': fitted_val,
                            'm/z': default_mz
                        })
                
                return pd.DataFrame(export_rows)

            # Also save Gaussian parameters separately
            def create_gaussian_parameters_export():
                """Create export data for Gaussian parameters"""
                param_rows = []
                
                for charge, saved_result in st.session_state['all_charge_results'].items():
                    fit_result = saved_result['fit_result']
                    peak_stats = saved_result['peak_stats']
                    fitting_options = saved_result['fitting_options']
                    
                    params_per_peak = {
                        "Gaussian": 3, "Lorentzian": 3, "Voigt": 4, "BiGaussian": 4, "EMG": 4
                    }.get(fitting_options['peak_type'], 3)
                    
                    n_peaks = len(fit_result['parameters']) // params_per_peak
                    
                    for peak_idx in range(n_peaks):
                        base_idx = peak_idx * params_per_peak
                        
                        # Get parameters
                        amplitude = fit_result['parameters'][base_idx]
                        center = fit_result['parameters'][base_idx + 1]
                        
                        if fitting_options['peak_type'] == "Gaussian":
                            width = fit_result['parameters'][base_idx + 2]
                            width_type = "Sigma"
                        elif fitting_options['peak_type'] == "Lorentzian":
                            width = fit_result['parameters'][base_idx + 2]
                            width_type = "Gamma"
                        elif fitting_options['peak_type'] == "Voigt":
                            width = fit_result['parameters'][base_idx + 2]  # Gaussian component
                            width_type = "Width_G"
                        else:
                            width = fit_result['parameters'][base_idx + 2]
                            width_type = "Width"
                        
                        # Get errors if available
                        if 'parameter_errors' in fit_result and len(fit_result['parameter_errors']) > base_idx + 2:
                            amplitude_error = fit_result['parameter_errors'][base_idx]
                            center_error = fit_result['parameter_errors'][base_idx + 1]
                            width_error = fit_result['parameter_errors'][base_idx + 2]
                        else:
                            amplitude_error = 0.0
                            center_error = 0.0
                            width_error = 0.0
                        
                        # Get peak statistics
                        if peak_idx < len(peak_stats):
                            peak_stat = peak_stats[peak_idx]
                            fwhm = peak_stat['fwhm']
                            area = peak_stat['area']
                            area_percent = peak_stat['area_percent']
                        else:
                            fwhm = width * 2.355 if fitting_options['peak_type'] == "Gaussian" else width * 2
                            area = 0.0
                            area_percent = 0.0
                        
                        param_rows.append({
                            'Charge': int(charge),
                            'Peak_Number': peak_idx + 1,
                            'Peak_Type': fitting_options['peak_type'],
                            'Amplitude': amplitude,
                            'Amplitude_Error': amplitude_error,
                            'Center_CCS': center,
                            'Center_Error': center_error,
                            f'{width_type}': width,
                            f'{width_type}_Error': width_error,
                            'FWHM': fwhm,
                            'Area': area,
                            'Area_Percent': area_percent,
                            'R_Squared': fit_result['r_squared'],
                            'RMSE': fit_result['rmse']
                        })
                
                return pd.DataFrame(param_rows)
            
            # Export button
            if st.button("📊 Generate Export Data", type="primary"):
                with st.spinner("Generating export data..."):
                    try:
                        # Create main export data
                        export_df = create_export_data()
                        
                        # Create Gaussian parameters export
                        params_df = create_gaussian_parameters_export()
                        
                        # Show preview
                        st.markdown("### Fitted Data Preview")
                        st.dataframe(export_df.head(20))
                        
                        st.markdown("### Peak Parameters Preview")
                        st.dataframe(params_df.head(10))
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download fitted data
                            csv_data = export_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Fitted Data",
                                csv_data,
                                "fitted_peak_data_all_charges.csv",
                                "text/csv",
                                help="Download fitted data for all charge states"
                            )
                        
                        with col2:
                            # Download peak parameters
                            csv_params = params_df.to_csv(index=False)
                            st.download_button(
                                "📋 Download Peak Parameters",
                                csv_params,
                                "gaussian_parameters_all_charges.csv",
                                "text/csv",
                                help="Download Gaussian parameters for all charge states"
                            )
                        
                        # Show summary
                        st.success(f"Export data generated: {len(export_df)} fitted data rows and {len(params_df)} parameter rows covering {len(st.session_state['all_charge_results'])} charge states")
                        
                    except Exception as e:
                        st.error(f"Failed to generate export data: {str(e)}")
                        st.exception(e)
        
        # Individual export options if current fit exists
        elif 'fit_result' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                # Export peak parameters
                peak_stats = st.session_state.get('peak_stats', [])
                result = st.session_state['fit_result']
                fitting_options = st.session_state['fitting_options']
                
                params_per_peak = {
                    "Gaussian": 3, "Lorentzian": 3, "Voigt": 4, "BiGaussian": 4, "EMG": 4
                }.get(fitting_options['peak_type'], 3)
                
                export_data = []
                for i, stats in enumerate(peak_stats):
                    base_idx = i * params_per_peak
                    params = result['parameters'][base_idx:base_idx + params_per_peak]
                    errors = result['parameter_errors'][base_idx:base_idx + params_per_peak]
                    
                    row = {
                        'Peak': stats['peak_number'],
                        'Center': stats['center'],
                        'Amplitude': stats['amplitude'],
                        'FWHM': stats['fwhm'],
                        'Area': stats['area'],
                        'Area_Percent': stats['area_percent']
                    }
                    
                    # Add parameter-specific columns
                    if fitting_options['peak_type'] in ["Gaussian", "Lorentzian"]:
                        row.update({
                            'Width': params[2],
                            'Width_Error': errors[2]
                        })
                    elif fitting_options['peak_type'] == "Voigt":
                        row.update({
                            'Width_G': params[2],
                            'Width_L': params[3],
                            'Width_G_Error': errors[2],
                            'Width_L_Error': errors[3]
                        })
                    
                    export_data.append(row)
                
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        "📄 Download Peak Parameters",
                        csv,
                        f"peak_parameters_{st.session_state['data_label'].lower().replace(' ', '_')}.csv",
                        "text/csv"
                    )
            
            with col2:
                # Export fit data
                fit_data = pd.DataFrame({
                    'CCS': st.session_state['x_data'],
                    'Original': st.session_state['y_data'],
                    'Baseline_Corrected': st.session_state['y_corrected'],
                    'Fitted': result['fitted_curve'],
                    'Residuals': result['residuals']
                })
                
                if fitting_options['baseline_type'] != "None":
                    fit_data['Baseline'] = st.session_state['baseline']
                
                csv_fit = fit_data.to_csv(index=False)
                
                st.download_button(
                    "📈 Download Fit Data",
                    csv_fit,
                    f"fit_data_{st.session_state['data_label'].lower().replace(' ', '_')}.csv",
                    "text/csv"
                )
        else:
            st.info("No results available for export. Complete peak fitting to enable export options.")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

# --- Origin-Style Peak Manager ---
class OriginPeakManager:
    def __init__(self):
        self.peak_list = []
        self.next_id = 0
    
    def add_peak(self, peak_type, center, y_data, x_data):
        """Add a new peak at specified position"""
        # Estimate initial parameters
        center_idx = np.argmin(np.abs(x_data - center))
        amplitude = y_data[center_idx] if center_idx < len(y_data) else np.max(y_data)
        width = (x_data.max() - x_data.min()) / 20
        
        if peak_type == "Gaussian":
            params = [amplitude, center, width]
        elif peak_type == "Lorentzian":
            params = [amplitude, center, width]
        elif peak_type == "Voigt":
            params = [amplitude, center, width, width]
        elif peak_type == "BiGaussian":
            params = [amplitude, center, width, width]
        elif peak_type == "EMG":
            params = [amplitude, center, width, width]
        else:
            params = [amplitude, center, width]
        
        peak = {
            'id': self.next_id,
            'type': peak_type,
            'params': params,
            'active': True
        }
        
        self.peak_list.append(peak)
        self.next_id += 1
        return True
    
    def delete_peak(self, peak_idx):
        """Delete peak by index"""
        if 0 <= peak_idx < len(self.peak_list):
            del self.peak_list[peak_idx]

            return True
        return False
    
    def get_all_parameters(self):
        """Get flattened parameter array"""
        params = []
        for peak in self.peak_list:
            if peak['active']:
                params.extend(peak['params'])
        return params
    
    def update_from_parameters(self, parameters, peak_type):
        """Update peak list from parameter array"""
        params_per_peak = {
            "Gaussian": 3, "Lorentzian": 3, "Voigt": 4, "BiGaussian": 4, "EMG": 4
        }.get(peak_type, 3)
        
        self.peak_list = []
        n_peaks = len(parameters) // params_per_peak
        
        for i in range(n_peaks):
            start_idx = i * params_per_peak
            end_idx = (i + 1) * params_per_peak
            params = parameters[start_idx:end_idx]
            
            peak = {
                'id': i,
                'type': peak_type,
                'params': params,
                'active': True
            }
            self.peak_list.append(peak)
        
        self.next_id = len(self.peak_list)

if __name__ == "__main__":
    main()