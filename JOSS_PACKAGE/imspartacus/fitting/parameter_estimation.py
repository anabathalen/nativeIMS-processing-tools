"""Parameter estimation module for curve fitting.

Provides Origin-style initial parameter estimation for various peak types.
"""

import numpy as np


class ParameterEstimator:
    """Estimates initial parameters for different peak types."""
    
    @staticmethod
    def estimate_gaussian_parameters(x, y, peak_info):
        """Estimate initial parameters for Gaussian peaks (Origin method).
        
        Args:
            x: X-axis data
            y: Y-axis data
            peak_info: List of peak information dictionaries from peak detection
            
        Returns:
            Flat list of parameters [amplitude1, center1, width1, amplitude2, ...]
        """
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        
        for peak in peak_info:
            amplitude = max(abs(peak['y']), np.max(y) * 0.01)  # Ensure positive amplitude
            center = peak['x']
            # Convert FWHM to sigma for Gaussian, ensure reasonable width
            fwhm = peak.get('width_half', x_span / 20)
            sigma = max(fwhm / (2 * np.sqrt(2 * np.log(2))), x_span * 0.001)
            params.extend([amplitude, center, sigma])
        
        return params
    
    @staticmethod
    def estimate_lorentzian_parameters(x, y, peak_info):
        """Estimate initial parameters for Lorentzian peaks.
        
        Args:
            x: X-axis data
            y: Y-axis data
            peak_info: List of peak information dictionaries from peak detection
            
        Returns:
            Flat list of parameters [amplitude1, center1, width1, amplitude2, ...]
        """
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        
        for peak in peak_info:
            amplitude = max(abs(peak['y']), np.max(y) * 0.01)
            center = peak['x']
            # For Lorentzian, FWHM = 2 * gamma
            fwhm = peak.get('width_half', x_span / 20)
            gamma = max(fwhm / 2, x_span * 0.001)
            params.extend([amplitude, center, gamma])
        
        return params
    
    @staticmethod
    def estimate_voigt_parameters(x, y, peak_info):
        """Estimate initial parameters for Voigt peaks.
        
        Args:
            x: X-axis data
            y: Y-axis data
            peak_info: List of peak information dictionaries from peak detection
            
        Returns:
            Flat list of parameters [amplitude1, center1, width_g1, width_l1, amplitude2, ...]
        """
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        
        for peak in peak_info:
            amplitude = max(abs(peak['y']), np.max(y) * 0.01)
            center = peak['x']
            # Start with equal Gaussian and Lorentzian contributions
            base_width = max(peak.get('width_half', x_span / 20), x_span * 0.001)
            width_g = base_width / (2 * np.sqrt(2 * np.log(2)))
            width_l = base_width / 2
            params.extend([amplitude, center, width_g, width_l])
        
        return params
    
    @staticmethod
    def estimate_bigaussian_parameters(x, y, peak_info):
        """Estimate initial parameters for BiGaussian peaks.
        
        Args:
            x: X-axis data
            y: Y-axis data
            peak_info: List of peak information dictionaries from peak detection
            
        Returns:
            Flat list of parameters [amplitude1, center1, width_left1, width_right1, ...]
        """
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        
        for peak in peak_info:
            amplitude = max(abs(peak['y']), np.max(y) * 0.01)
            center = peak['x']
            # Start with symmetric widths
            fwhm = peak.get('width_half', x_span / 20)
            sigma = max(fwhm / (2 * np.sqrt(2 * np.log(2))), x_span * 0.001)
            params.extend([amplitude, center, sigma, sigma])
        
        return params
    
    @staticmethod
    def estimate_emg_parameters(x, y, peak_info):
        """Estimate initial parameters for EMG (Exponentially Modified Gaussian) peaks.
        
        Args:
            x: X-axis data
            y: Y-axis data
            peak_info: List of peak information dictionaries from peak detection
            
        Returns:
            Flat list of parameters [amplitude1, center1, width1, tau1, ...]
        """
        params = []
        x_span = x[-1] - x[0] if len(x) > 1 else 1.0
        
        for peak in peak_info:
            amplitude = max(abs(peak['y']), np.max(y) * 0.01)
            center = peak['x']
            fwhm = peak.get('width_half', x_span / 20)
            sigma = max(fwhm / (2 * np.sqrt(2 * np.log(2))), x_span * 0.001)
            # Initial tau (exponential decay) set to width
            tau = sigma
            params.extend([amplitude, center, sigma, tau])
        
        return params
    
    @staticmethod
    def estimate_parameters(x, y, peak_info, peak_type):
        """Estimate parameters for any peak type.
        
        Args:
            x: X-axis data
            y: Y-axis data
            peak_info: List of peak information dictionaries from peak detection
            peak_type: Type of peak ("Gaussian", "Lorentzian", "Voigt", "BiGaussian", "EMG")
            
        Returns:
            Flat list of initial parameters
        """
        estimators = {
            "Gaussian": ParameterEstimator.estimate_gaussian_parameters,
            "Lorentzian": ParameterEstimator.estimate_lorentzian_parameters,
            "Voigt": ParameterEstimator.estimate_voigt_parameters,
            "BiGaussian": ParameterEstimator.estimate_bigaussian_parameters,
            "EMG": ParameterEstimator.estimate_emg_parameters
        }
        
        estimator = estimators.get(peak_type, ParameterEstimator.estimate_gaussian_parameters)
        return estimator(x, y, peak_info)
