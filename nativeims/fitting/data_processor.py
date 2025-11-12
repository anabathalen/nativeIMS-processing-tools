"""
Data Processor Module
=====================

This module provides data preprocessing tools for peak fitting, including
smoothing and baseline subtraction methods commonly used in Origin software.

Classes
-------
DataProcessor
    Static methods for data smoothing and baseline subtraction
"""

import numpy as np
from scipy.signal import savgol_filter


class DataProcessor:
    """
    Data preprocessing tools for peak fitting.
    
    Provides static methods for common preprocessing operations:
    - Data smoothing (Savitzky-Golay, Moving Average)
    - Baseline subtraction (Linear, Polynomial)
    
    All methods are static and can be called directly on the class.
    
    Methods
    -------
    smooth_data(x, y, method, window_size, poly_order)
        Smooth data using various algorithms
    subtract_baseline(x, y, method, poly_degree, regions)
        Subtract baseline using various methods
        
    Examples
    --------
    Smooth data:
    
    >>> from nativeims.fitting import DataProcessor
    >>> y_smooth = DataProcessor.smooth_data(x, y, method="Savitzky-Golay", 
    ...                                       window_size=11, poly_order=3)
    
    Subtract linear baseline:
    
    >>> y_corrected, baseline = DataProcessor.subtract_baseline(x, y, method="Linear")
    
    Subtract polynomial baseline using specific regions:
    
    >>> regions = [(0, 2), (8, 10)]  # Use edges for baseline
    >>> y_corrected, baseline = DataProcessor.subtract_baseline(x, y, method="Polynomial",
    ...                                                          poly_degree=2, regions=regions)
    """
    
    @staticmethod
    def smooth_data(x, y, method="Savitzky-Golay", window_size=5, poly_order=2):
        """
        Data smoothing (Origin-style options).
        
        Parameters
        ----------
        x : array_like
            X-axis data (not used for current methods but kept for compatibility)
        y : array_like
            Y-axis data to smooth
        method : str, optional
            Smoothing method: "Savitzky-Golay" or "Moving Average" (default: "Savitzky-Golay")
        window_size : int, optional
            Window size for smoothing (default: 5)
        poly_order : int, optional
            Polynomial order for Savitzky-Golay filter (default: 2)
            
        Returns
        -------
        ndarray
            Smoothed y data
            
        Notes
        -----
        For Savitzky-Golay:
        - Window size is automatically adjusted to be odd and smaller than data length
        - Polynomial order is adjusted to be less than window size
        
        For Moving Average:
        - Uses uniform convolution filter
        """
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
        """
        Baseline subtraction (Origin-style methods).
        
        Parameters
        ----------
        x : array_like
            X-axis data
        y : array_like
            Y-axis data
        method : str, optional
            Baseline method: "None", "Linear", or "Polynomial" (default: "Linear")
        poly_degree : int, optional
            Polynomial degree for polynomial baseline (default: 2)
        regions : list of tuple, optional
            List of (start, end) x-ranges to use for baseline fitting.
            If None, uses automatic region selection (default: None)
            
        Returns
        -------
        y_corrected : ndarray
            Baseline-corrected y data
        baseline : ndarray
            The fitted baseline
            
        Notes
        -----
        For Linear method:
        - If regions=None, uses first and last 10% of data
        - Fits a line through specified or automatic baseline regions
        
        For Polynomial method:
        - If regions=None, uses entire dataset
        - Fits polynomial of specified degree
        
        Examples
        --------
        >>> # Linear baseline using default regions
        >>> y_corrected, baseline = DataProcessor.subtract_baseline(x, y)
        
        >>> # Polynomial baseline with custom regions
        >>> regions = [(0, 1), (9, 10)]
        >>> y_corrected, baseline = DataProcessor.subtract_baseline(x, y, method="Polynomial",
        ...                                                          poly_degree=3, regions=regions)
        """
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
