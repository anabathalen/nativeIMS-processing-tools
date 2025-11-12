"""Peak detection module for curve fitting.

Provides Origin-style peak detection with automatic parameter estimation.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter


class PeakDetector:
    """Peak detection using Origin-style parameters."""
    
    @staticmethod
    def find_peaks_origin_style(x, y, min_height_percent=5, min_prominence_percent=2, 
                               min_distance_percent=5, smoothing_points=5):
        """Peak detection using Origin-style parameters.
        
        Args:
            x: X-axis data
            y: Y-axis data
            min_height_percent: Minimum peak height as percentage of data range
            min_prominence_percent: Minimum peak prominence as percentage of data range
            min_distance_percent: Minimum distance between peaks as percentage of data length
            smoothing_points: Number of points for Savitzky-Golay smoothing (0 = no smoothing)
            
        Returns:
            List of dictionaries containing peak information:
                - index: Peak index in array
                - x: Peak x position
                - y: Peak y value
                - prominence: Peak prominence
                - width_half: Peak width at half maximum
                - width_base: Peak width at base
                - area_estimate: Estimated peak area
        """
        # Smooth data if requested
        if smoothing_points > 0:
            window_length = min(smoothing_points * 2 + 1, len(y) - 1)
            if window_length < 3:
                window_length = 3
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=2)
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
                    dx = x[1] - x[0] if len(x) > 1 else 1.0
                    info = {
                        'index': peak_idx,
                        'x': x[peak_idx],
                        'y': y_smooth[peak_idx],
                        'prominence': properties['prominences'][i],
                        'width_half': widths_half[i] * dx if i < len(widths_half) else 0,
                        'width_base': widths_base[i] * dx if i < len(widths_base) else 0,
                        'area_estimate': properties['prominences'][i] * widths_half[i] * dx if i < len(widths_half) else 0
                    }
                    peak_info.append(info)
                
                return peak_info
            except:
                # Fallback if width calculation fails
                peak_info = []
                x_span = x[-1] - x[0] if len(x) > 1 else 1.0
                for i, peak_idx in enumerate(peaks):
                    info = {
                        'index': peak_idx,
                        'x': x[peak_idx],
                        'y': y_smooth[peak_idx],
                        'prominence': properties['prominences'][i],
                        'width_half': x_span / 20,  # Default width
                        'width_base': x_span / 10,
                        'area_estimate': properties['prominences'][i] * x_span / 20
                    }
                    peak_info.append(info)
                return peak_info
        
        return []
