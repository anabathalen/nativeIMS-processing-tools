"""Baseline correction functions for curve fitting."""

import numpy as np


def linear_baseline(x, slope, intercept):
    """Linear baseline function.
    
    Args:
        x: Independent variable
        slope: Linear slope
        intercept: Y-intercept
        
    Returns:
        Array of baseline values
    """
    return slope * x + intercept


def polynomial_baseline(x, *coeffs):
    """Polynomial baseline function.
    
    Args:
        x: Independent variable
        *coeffs: Polynomial coefficients (highest degree first)
        
    Returns:
        Array of baseline values
    """
    return np.polyval(coeffs, x)


def exponential_baseline(x, a, b, c):
    """Exponential baseline function.
    
    Args:
        x: Independent variable
        a: Amplitude coefficient
        b: Exponential rate
        c: Offset
        
    Returns:
        Array of baseline values
    """
    return a * np.exp(b * x) + c
