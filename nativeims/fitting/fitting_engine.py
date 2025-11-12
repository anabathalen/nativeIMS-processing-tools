"""
Fitting Engine Module
=====================

This module provides the core fitting engine for peak analysis, implementing
Origin-style curve fitting methodologies with support for parameter constraints,
multiple optimization algorithms, and comprehensive fit statistics.

Classes
-------
FittingEngine
    Main fitting engine for peak curve fitting with constraint handling
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution

from .peak_functions import multi_peak_function


class FittingEngine:
    """
    Main fitting engine for peak curve fitting.
    
    This class implements Origin-style peak fitting with support for:
    - Multiple peak types (Gaussian, Lorentzian, Voigt, BiGaussian, EMG)
    - Parameter constraints (fixed parameters, custom bounds)
    - Multiple optimization algorithms (Levenberg-Marquardt, Global)
    - Weighted fitting
    - Comprehensive fit statistics (R², adjusted R², RMSE, AIC, BIC, χ²)
    
    The engine integrates with ParameterManager to handle complex parameter
    constraints and fixed parameters during optimization.
    
    Attributes
    ----------
    peak_type : str
        Type of peak function to use ("Gaussian", "Lorentzian", "Voigt",
        "BiGaussian", or "EMG")
    baseline_type : str
        Type of baseline correction ("None", "Linear", "Polynomial", "Exponential")
    fit_method : str
        Optimization method ("Levenberg-Marquardt" or "Global")
    max_iterations : int
        Maximum number of fitting iterations
    tolerance : float
        Convergence tolerance
    use_weights : bool
        Whether to use weighted fitting
    parameter_manager : ParameterManager or None
        Manager for parameter constraints
        
    Methods
    -------
    set_fitting_options(peak_type, baseline_type, fit_method, max_iterations, 
                       tolerance, use_weights)
        Configure all fitting options
    set_parameter_manager(parameter_manager)
        Set the parameter manager for constraint handling
    fit_peaks(x, y, initial_params, weights=None)
        Perform peak fitting with current settings
    create_bounds(initial_params, x_range)
        Create default parameter bounds
    get_params_per_peak()
        Get number of parameters per peak for current peak type
        
    Examples
    --------
    Basic fitting:
    
    >>> from nativeims.fitting import FittingEngine
    >>> engine = FittingEngine()
    >>> engine.set_fitting_options(peak_type="Gaussian", fit_method="Levenberg-Marquardt")
    >>> initial_params = [100, 5.0, 0.5]  # amplitude, center, width
    >>> result = engine.fit_peaks(x_data, y_data, initial_params)
    >>> if result['success']:
    ...     fitted_params = result['parameters']
    ...     r_squared = result['r_squared']
    
    With parameter constraints:
    
    >>> from nativeims.fitting import FittingEngine, ParameterManager
    >>> engine = FittingEngine()
    >>> param_manager = ParameterManager("Gaussian", initial_params, x_range=(0, 10))
    >>> param_manager.fix_parameter(1)  # Fix peak center
    >>> engine.set_parameter_manager(param_manager)
    >>> result = engine.fit_peaks(x_data, y_data, initial_params)
    """
    
    def __init__(self):
        """Initialize fitting engine with default settings."""
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
        """
        Set fitting options like Origin's Peak Analyzer.
        
        Parameters
        ----------
        peak_type : str, optional
            Peak function type (default: "Gaussian")
        baseline_type : str, optional
            Baseline correction type (default: "None")
        fit_method : str, optional
            Optimization algorithm (default: "Levenberg-Marquardt")
        max_iterations : int, optional
            Maximum iterations (default: 1000)
        tolerance : float, optional
            Convergence tolerance (default: 1e-8)
        use_weights : bool, optional
            Enable weighted fitting (default: False)
        """
        self.peak_type = peak_type
        self.baseline_type = baseline_type
        self.fit_method = fit_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_weights = use_weights
    
    def set_parameter_manager(self, parameter_manager):
        """
        Set the parameter manager for manual control.
        
        Parameters
        ----------
        parameter_manager : ParameterManager
            Parameter manager instance for constraint handling
        """
        self.parameter_manager = parameter_manager
    
    def create_bounds(self, initial_params, x_range):
        """
        Create parameter bounds (Origin-style constraints).
        
        Parameters
        ----------
        initial_params : array_like
            Initial parameter values
        x_range : tuple of float
            (min_x, max_x) data range
            
        Returns
        -------
        bounds_lower : list of float
            Lower bounds for each parameter
        bounds_upper : list of float
            Upper bounds for each parameter
        """
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
        """
        Get number of parameters per peak for different peak types.
        
        Returns
        -------
        int
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
    
    def fit_peaks(self, x, y, initial_params, weights=None):
        """
        Fit peaks using Origin-style methodology with parameter constraints.
        
        Parameters
        ----------
        x : array_like
            X-axis data (e.g., m/z or CCS values)
        y : array_like
            Y-axis data (intensities)
        initial_params : array_like
            Initial parameter guesses
        weights : array_like, optional
            Weights for weighted fitting
            
        Returns
        -------
        dict
            Fitting results containing:
            - success : bool - Whether fit succeeded
            - parameters : ndarray - Fitted parameter values
            - parameter_errors : ndarray - Parameter uncertainties
            - fitted_curve : ndarray - Fitted y values
            - residuals : ndarray - Fit residuals (y - y_fit)
            - r_squared : float - Coefficient of determination
            - adj_r_squared : float - Adjusted R²
            - rmse : float - Root mean square error
            - reduced_chi_squared : float - Reduced χ² statistic
            - aic : float - Akaike Information Criterion
            - bic : float - Bayesian Information Criterion
            - iterations : int or None - Number of iterations (for global method)
            - free_parameters : int - Number of free (unfixed) parameters
            - error : str - Error message (if success=False)
        """
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
