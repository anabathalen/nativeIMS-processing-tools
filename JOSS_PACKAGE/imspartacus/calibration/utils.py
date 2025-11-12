"""
Calibration utilities including drift time adjustments and file generation.

This module handles instrument-specific corrections and file format generation
for calibration workflows.
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class InstrumentParams:
    """
    IMS instrument parameters needed for calibration.
    
    Attributes:
        wave_velocity: Traveling wave velocity in m/s
        wave_height: Wave height (voltage) in V
        pressure: IMS pressure in mbar
        drift_length: Drift cell length in m
        instrument_type: 'cyclic' or 'synapt'
        inject_time: Injection time in ms (only for Cyclic IMS)
    """
    wave_velocity: float
    wave_height: float
    pressure: float
    drift_length: float
    instrument_type: str
    inject_time: float = 0.0


def adjust_drift_time_for_injection(
    drift_time: float,
    inject_time: float,
    instrument_type: str
) -> float:
    """
    Adjust drift time for injection time (Cyclic IMS only).
    
    In Cyclic IMS, the measured arrival time includes both the drift time
    AND the time the ion spent in the injection region. We need to subtract
    the injection time to get the true drift time.
    
    For Synapt instruments, no adjustment is needed.
    
    Args:
        drift_time: Measured arrival time in ms
        inject_time: Injection time in ms
        instrument_type: 'cyclic' or 'synapt' (case-insensitive)
        
    Returns:
        Adjusted drift time in ms
        
    Example:
        >>> # Cyclic IMS measurement
        >>> measured_time = 5.5  # ms
        >>> inject_time = 0.3    # ms
        >>> true_drift = adjust_drift_time_for_injection(
        ...     measured_time, inject_time, 'cyclic'
        ... )
        >>> print(true_drift)
        5.2
        
        >>> # Synapt - no adjustment
        >>> drift = adjust_drift_time_for_injection(5.5, 0.3, 'synapt')
        >>> print(drift)
        5.5
    """
    if instrument_type.lower() == 'cyclic':
        # Subtract injection time for Cyclic IMS
        return drift_time - inject_time
    else:
        # No adjustment needed for Synapt
        return drift_time


def adjust_dataframe_drift_times(
    df: pd.DataFrame,
    instrument_params: InstrumentParams
) -> pd.DataFrame:
    """
    Adjust all drift times in a DataFrame based on instrument type.
    
    Args:
        df: DataFrame with 'drift time' column
        instrument_params: Instrument parameters including inject_time
        
    Returns:
        New DataFrame with adjusted drift times
        
    Example:
        >>> params = InstrumentParams(
        ...     wave_velocity=281.0,
        ...     wave_height=20.0,
        ...     pressure=1.63,
        ...     drift_length=0.98,
        ...     instrument_type='cyclic',
        ...     inject_time=0.3
        ... )
        >>> adjusted_df = adjust_dataframe_drift_times(results_df, params)
    """
    # Make a copy to avoid modifying the original
    adjusted_df = df.copy()
    
    # Apply adjustment to each drift time
    adjusted_df['drift time'] = adjusted_df['drift time'].apply(
        lambda dt: adjust_drift_time_for_injection(
            dt,
            instrument_params.inject_time,
            instrument_params.instrument_type
        )
    )
    
    return adjusted_df
