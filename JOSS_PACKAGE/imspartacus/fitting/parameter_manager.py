"""Parameter manager module for constrained curve fitting.

Manages peak parameters, bounds, and constraints for fitting.
"""

import numpy as np


class ParameterManager:
    """Manages parameters with bounds and constraints for peak fitting."""
    
    def __init__(self, peak_type, parameters, x_range):
        """Initialize parameter manager.
        
        Args:
            peak_type: Type of peak ("Gaussian", "Lorentzian", "Voigt", "BiGaussian", "EMG")
            parameters: Flat array of initial parameters
            x_range: Tuple of (x_min, x_max) for data range
        """
        self.peak_type = peak_type
        self.parameters = np.array(parameters).copy()
        self.x_range = x_range
        self.fixed_params = {}  # Track which parameters are fixed
        self.param_bounds = {}  # Track custom bounds
        self.params_per_peak = self.get_params_per_peak()
        self.n_peaks = len(parameters) // self.params_per_peak
        
    def get_params_per_peak(self):
        """Get number of parameters per peak for different peak types.
        
        Returns:
            Number of parameters per peak
        """
        params_dict = {
            "Gaussian": 3,
            "Lorentzian": 3,
            "Voigt": 4,
            "BiGaussian": 4,
            "EMG": 4
        }
        return params_dict.get(self.peak_type, 3)
    
    def get_parameter_names(self):
        """Get parameter names for each peak type.
        
        Returns:
            List of parameter names
        """
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
        """Update a specific parameter.
        
        Args:
            peak_idx: Peak index (0-based)
            param_idx: Parameter index within peak (0-based)
            value: New parameter value
        """
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        if 0 <= global_param_idx < len(self.parameters):
            self.parameters[global_param_idx] = value
    
    def fix_parameter(self, peak_idx, param_idx, fixed=True):
        """Fix or unfix a parameter.
        
        Args:
            peak_idx: Peak index (0-based)
            param_idx: Parameter index within peak (0-based)
            fixed: True to fix, False to unfix
        """
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        if fixed:
            self.fixed_params[global_param_idx] = self.parameters[global_param_idx]
        else:
            self.fixed_params.pop(global_param_idx, None)
    
    def is_parameter_fixed(self, peak_idx, param_idx):
        """Check if a parameter is fixed.
        
        Args:
            peak_idx: Peak index (0-based)
            param_idx: Parameter index within peak (0-based)
            
        Returns:
            True if parameter is fixed
        """
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        return global_param_idx in self.fixed_params
    
    def set_parameter_bounds(self, peak_idx, param_idx, lower, upper):
        """Set custom bounds for a parameter.
        
        Args:
            peak_idx: Peak index (0-based)
            param_idx: Parameter index within peak (0-based)
            lower: Lower bound
            upper: Upper bound
        """
        global_param_idx = peak_idx * self.params_per_peak + param_idx
        self.param_bounds[global_param_idx] = (lower, upper)
    
    def get_fitting_parameters(self):
        """Get parameters for fitting (excluding fixed ones).
        
        Returns:
            Tuple of (free_params, param_mapping) where:
                - free_params: Array of parameters to fit
                - param_mapping: Indices mapping free params to full parameter array
        """
        free_params = []
        param_mapping = []
        
        for i, param in enumerate(self.parameters):
            if i not in self.fixed_params:
                free_params.append(param)
                param_mapping.append(i)
        
        return np.array(free_params), param_mapping
    
    def reconstruct_full_parameters(self, fitted_params, param_mapping):
        """Reconstruct full parameter array from fitted parameters.
        
        Args:
            fitted_params: Array of fitted parameter values
            param_mapping: Indices mapping fitted params to full array
            
        Returns:
            Full parameter array with fitted and fixed values
        """
        full_params = self.parameters.copy()
        
        for i, mapped_idx in enumerate(param_mapping):
            full_params[mapped_idx] = fitted_params[i]
        
        # Update fixed parameters with their fixed values
        for idx, value in self.fixed_params.items():
            full_params[idx] = value
        
        return full_params
    
    def get_bounds_for_fitting(self, param_mapping):
        """Get bounds for free parameters only.
        
        Args:
            param_mapping: Indices mapping free params to full array
            
        Returns:
            Tuple of (bounds_lower, bounds_upper) arrays
        """
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
                    amp = abs(self.parameters[mapped_idx])
                    bounds_lower.append(0)
                    bounds_upper.append(max(amp * 10, 1.0))
                elif param_idx == 1:  # Center
                    bounds_lower.append(self.x_range[0])
                    bounds_upper.append(self.x_range[1])
                else:  # Width parameters
                    width = abs(self.parameters[mapped_idx])
                    bounds_lower.append(max(width * 0.1, 1e-6))
                    bounds_upper.append(width * 10)
        
        return np.array(bounds_lower), np.array(bounds_upper)
