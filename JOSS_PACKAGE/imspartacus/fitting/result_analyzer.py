"""
Result Analyzer Module
======================

This module provides comprehensive peak analysis tools for fitted results,
including area calculation and peak statistics generation.

Classes
-------
ResultAnalyzer
    Static methods for calculating peak areas and comprehensive statistics
"""

import numpy as np
from scipy.integrate import trapezoid

from .peak_functions import (
    gaussian_peak, voigt_peak, bigaussian_peak, 
    exponentially_modified_gaussian
)


class ResultAnalyzer:
    """
    Result analysis tools for fitted peaks.
    
    Provides static methods for:
    - Peak area calculation (analytical and numerical)
    - Comprehensive peak statistics (FWHM, area %, height %)
    - Origin-style result reporting
    
    All methods are static and can be called directly on the class.
    
    Methods
    -------
    calculate_peak_areas(x, parameters, peak_type)
        Calculate peak areas for all peaks in fit
    calculate_peak_statistics(x, y, fitted_curve, parameters, peak_type)
        Generate comprehensive statistics for all peaks
        
    Examples
    --------
    Calculate peak areas:
    
    >>> from imspartacus.fitting import ResultAnalyzer
    >>> parameters = [100, 5.0, 0.5, 80, 6.0, 0.6]  # 2 Gaussian peaks
    >>> areas = ResultAnalyzer.calculate_peak_areas(x_data, parameters, "Gaussian")
    
    Get full peak statistics:
    
    >>> stats = ResultAnalyzer.calculate_peak_statistics(x_data, y_data, y_fit, 
    ...                                                   parameters, "Gaussian")
    >>> for peak in stats:
    ...     print(f"Peak {peak['peak_number']}: Area = {peak['area']:.2f}, "
    ...           f"FWHM = {peak['fwhm']:.3f}")
    """
    
    @staticmethod
    def calculate_peak_areas(x, parameters, peak_type):
        """
        Calculate peak areas (Origin-style integration).
        
        Parameters
        ----------
        x : array_like
            X-axis data range
        parameters : array_like
            Fitted parameters for all peaks
        peak_type : str
            Peak function type ("Gaussian", "Lorentzian", "Voigt", "BiGaussian", "EMG")
            
        Returns
        -------
        list of float
            Peak areas for each peak
            
        Notes
        -----
        - Gaussian and Lorentzian use analytical formulas
        - Complex peak types use numerical integration
        - Areas are calculated from -∞ to +∞ (or practical limits)
        """
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
        """
        Calculate comprehensive peak statistics (Origin-style report).
        
        Parameters
        ----------
        x : array_like
            X-axis data
        y : array_like
            Original y data
        fitted_curve : array_like
            Fitted y data
        parameters : array_like
            Fitted parameters for all peaks
        peak_type : str
            Peak function type
            
        Returns
        -------
        list of dict
            List of statistics dictionaries, one per peak, containing:
            - peak_number : int - Peak index (1-based)
            - amplitude : float - Peak height
            - center : float - Peak center position
            - fwhm : float - Full width at half maximum
            - area : float - Integrated peak area
            - area_percent : float - Percentage of total area
            - height_percent : float - Percentage of maximum intensity
            
        Examples
        --------
        >>> stats = ResultAnalyzer.calculate_peak_statistics(x, y, y_fit, params, "Gaussian")
        >>> for peak in stats:
        ...     print(f"Peak {peak['peak_number']}:")
        ...     print(f"  Center: {peak['center']:.3f}")
        ...     print(f"  FWHM: {peak['fwhm']:.3f}")
        ...     print(f"  Area: {peak['area']:.2f} ({peak['area_percent']:.1f}%)")
        """
        params_per_peak = {
            "Gaussian": 3, "Lorentzian": 3, "Voigt": 4, "BiGaussian": 4, "EMG": 4
        }.get(peak_type, 3)
        
        n_peaks = len(parameters) // params_per_peak
        areas = ResultAnalyzer.calculate_peak_areas(x, parameters, peak_type)
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
