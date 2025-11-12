"""Peak function definitions for curve fitting.

This module provides various peak shape functions commonly used in
spectroscopic data analysis, similar to Origin Pro's peak fitting.
"""

import numpy as np
from scipy.special import erfc


def gaussian_peak(x, amplitude, center, width):
    """Gaussian peak function.
    
    Args:
        x: Independent variable
        amplitude: Peak height
        center: Peak center position
        width: Peak width (standard deviation)
        
    Returns:
        Array of y values
    """
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def lorentzian_peak(x, amplitude, center, width):
    """Lorentzian peak function.
    
    Args:
        x: Independent variable
        amplitude: Peak height
        center: Peak center position
        width: Peak width (HWHM)
        
    Returns:
        Array of y values
    """
    return amplitude / (1 + ((x - center) / width) ** 2)


def voigt_peak(x, amplitude, center, width_g, width_l):
    """Pseudo-Voigt approximation.
    
    Combines Gaussian and Lorentzian profiles with mixing parameter eta.
    
    Args:
        x: Independent variable
        amplitude: Peak height
        center: Peak center position
        width_g: Gaussian width
        width_l: Lorentzian width
        
    Returns:
        Array of y values
    """
    eta = 1.36603 * (width_l / width_g) - 0.47719 * (width_l / width_g)**2 + 0.11116 * (width_l / width_g)**3
    eta = np.clip(eta, 0, 1)
    
    gaussian = np.exp(-0.693147 * ((x - center) / width_g) ** 2)
    lorentzian = 1 / (1 + ((x - center) / width_l) ** 2)
    
    return amplitude * (eta * lorentzian + (1 - eta) * gaussian)


def bigaussian_peak(x, amplitude, center, width1, width2):
    """Bi-Gaussian function with different widths on each side.
    
    Args:
        x: Independent variable
        amplitude: Peak height
        center: Peak center position
        width1: Left-side width
        width2: Right-side width
        
    Returns:
        Array of y values
    """
    result = np.zeros_like(x)
    left_mask = x <= center
    right_mask = x > center
    
    if np.any(left_mask):
        result[left_mask] = amplitude * np.exp(-0.5 * ((x[left_mask] - center) / width1) ** 2)
    if np.any(right_mask):
        result[right_mask] = amplitude * np.exp(-0.5 * ((x[right_mask] - center) / width2) ** 2)
    
    return result


def exponentially_modified_gaussian(x, amplitude, center, width, tau):
    """Exponentially Modified Gaussian (EMG).
    
    Convolution of Gaussian with exponential decay, useful for tailing peaks.
    
    Args:
        x: Independent variable
        amplitude: Peak height
        center: Gaussian center position
        width: Gaussian width
        tau: Exponential time constant
        
    Returns:
        Array of y values
    """
    sigma = width / np.sqrt(2)
    lambda_param = 1.0 / tau if tau != 0 else 1e10
    
    term1 = (lambda_param / 2) * np.exp((lambda_param / 2) * (2 * center + lambda_param * sigma**2 - 2 * x))
    term2 = erfc((center + lambda_param * sigma**2 - x) / (sigma * np.sqrt(2)))
    
    return amplitude * term1 * term2


def asymmetric_gaussian(x, amplitude, center, width, asymmetry):
    """Asymmetric Gaussian function.
    
    Args:
        x: Independent variable
        amplitude: Peak height
        center: Peak center position
        width: Peak width
        asymmetry: Asymmetry parameter
        
    Returns:
        Array of y values
    """
    gaussian = np.exp(-0.5 * ((x - center) / width) ** 2)
    exponential = np.exp((x - center) / asymmetry)
    
    return amplitude * gaussian * exponential


def multi_peak_function(x, peak_type, *params):
    """Multi-peak function supporting different peak types.
    
    Args:
        x: Independent variable
        peak_type: Type of peak function ("Gaussian", "Lorentzian", "Voigt", "BiGaussian", "EMG")
        *params: Flattened array of parameters for all peaks
        
    Returns:
        Sum of all peak contributions
    """
    y = np.zeros_like(x)
    
    params_per_peak_map = {
        "Gaussian": 3,
        "Lorentzian": 3,
        "Voigt": 4,
        "BiGaussian": 4,
        "EMG": 4
    }
    
    params_per_peak = params_per_peak_map.get(peak_type, 3)
    
    peak_functions = {
        "Gaussian": gaussian_peak,
        "Lorentzian": lorentzian_peak,
        "Voigt": voigt_peak,
        "BiGaussian": bigaussian_peak,
        "EMG": exponentially_modified_gaussian
    }
    
    peak_func = peak_functions.get(peak_type)
    if peak_func is None:
        return y
    
    for i in range(0, len(params), params_per_peak):
        if i + params_per_peak <= len(params):
            y += peak_func(x, *params[i:i+params_per_peak])
    
    return y


def get_params_per_peak(peak_type: str) -> int:
    """Get number of parameters required for a peak type.
    
    Args:
        peak_type: Type of peak function
        
    Returns:
        Number of parameters per peak
    """
    params_map = {
        "Gaussian": 3,
        "Lorentzian": 3,
        "Voigt": 4,
        "BiGaussian": 4,
        "EMG": 4
    }
    return params_map.get(peak_type, 3)


def get_parameter_names(peak_type: str) -> list[str]:
    """Get parameter names for a peak type.
    
    Args:
        peak_type: Type of peak function
        
    Returns:
        List of parameter names
    """
    names_map = {
        "Gaussian": ["Amplitude", "Center", "Width"],
        "Lorentzian": ["Amplitude", "Center", "Width"],
        "Voigt": ["Amplitude", "Center", "Width_G", "Width_L"],
        "BiGaussian": ["Amplitude", "Center", "Width1", "Width2"],
        "EMG": ["Amplitude", "Center", "Width", "Tau"]
    }
    return names_map.get(peak_type, ["Amplitude", "Center", "Width"])
