"""DTIMS calibration and CCS calculation core utilities.

Pure functions extracted from the Streamlit page, suitable for reuse and testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
except Exception:  # optional dependency for consumers
    LinearRegression = None  # type: ignore
    r2_score = None  # type: ignore


# Physical constants
@dataclass
class PhysicalConstants:
    k_B: float = 1.380649e-23      # Boltzmann constant (J/K)
    e: float = 1.602176634e-19     # Elementary charge (C)
    N_A: float = 6.02214076e23     # Avogadro's number
    da_to_kg: float = 1.66054e-27  # Dalton to kg conversion
    pi: float = float(np.pi)


CONSTANTS = PhysicalConstants()


def parse_dtims_csv(content: Union[str, bytes]) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[List[str]]]:
    """Parse the DTIMS CSV file and extract data.

    Accepts a decoded string or raw bytes. Returns (df, raw_files, range_files).
    """
    try:
        if isinstance(content, bytes):
            text = content.decode("utf-8")
        else:
            text = content

        lines = text.strip().split("\n")
        if len(lines) < 3:
            return None, None, None

        range_files = lines[0].split(',')[1:]
        raw_files = lines[1].split(',')[1:]

        data_rows: List[List[float]] = []
        for line in lines[2:]:
            values = line.split(',')
            if len(values) >= 2:
                try:
                    data_rows.append([float(v) if v else 0.0 for v in values])
                except Exception:
                    # skip malformed line
                    continue

        columns = ['Time'] + [f'File_{i+1}' for i in range(len(raw_files))]
        df = pd.DataFrame(data_rows, columns=columns)
        return df, raw_files, range_files
    except Exception:
        return None, None, None


def find_max_drift_time(df: pd.DataFrame, column: str) -> Tuple[Optional[float], Optional[float]]:
    """Find the drift time with maximum intensity for a given column."""
    if column in df.columns and not df[column].empty:
        max_idx = df[column].idxmax()
        return float(df.loc[max_idx, 'Time']), float(df.loc[max_idx, column])
    return None, None


def calculate_reduced_mass(mass_analyte: float, mass_buffer: float) -> float:
    """Calculate reduced mass in kg from analyte and buffer gas masses in Da."""
    mass_analyte_kg = mass_analyte * CONSTANTS.da_to_kg
    mass_buffer_kg = mass_buffer * CONSTANTS.da_to_kg
    return (mass_analyte_kg * mass_buffer_kg) / (mass_analyte_kg + mass_buffer_kg)


def calculate_ccs_mason_schamp(
    drift_time: float,
    voltage: float,
    temperature: float,
    pressure: float,
    mass_analyte: float,
    mass_buffer: float,
    charge: int = 1,
    length: float = 25.05,
    return_mobility: bool = False,
) -> float | Tuple[float, float]:
    """Compute CCS (Å²) via Mason–Schamp. Returns CCS or (CCS, mobility)."""
    if (
        drift_time is None
        or voltage is None
        or drift_time <= 0
        or voltage <= 0
        or temperature <= 0
        or pressure <= 0
        or mass_analyte <= 0
        or mass_buffer <= 0
    ):
        return (np.nan, np.nan) if return_mobility else float("nan")

    t_s = drift_time * 1e-3  # ms -> s
    L_m = length * 1e-2      # cm -> m
    P_Pa = pressure * 100.0  # mbar -> Pa

    mu = calculate_reduced_mass(mass_analyte, mass_buffer)
    N = P_Pa / (CONSTANTS.k_B * temperature)
    K = (L_m * L_m) / (voltage * t_s)

    if not np.isfinite(K) or K <= 0:
        return (np.nan, np.nan) if return_mobility else float("nan")

    prefactor = (3.0 * charge * CONSTANTS.e) / (16.0 * N * K)
    sqrt_term = np.sqrt((2.0 * CONSTANTS.pi) / (mu * CONSTANTS.k_B * temperature))
    ccs_m2 = prefactor * sqrt_term
    ccs_A2 = ccs_m2 * 1e20

    return (ccs_A2, K) if return_mobility else float(ccs_A2)


def calculate_true_voltage(
    helium_cell_dc: float,
    bias: float,
    transfer_dc_entrance: float,
    helium_exit_dc: float,
) -> float:
    """Calculate true voltage from instrumental parameters."""
    return (helium_cell_dc + bias) - (transfer_dc_entrance + helium_exit_dc)


def perform_linear_regression(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
    """Linear regression: y vs x using scikit-learn if available."""
    if LinearRegression is None or r2_score is None:
        raise ImportError("scikit-learn is required for linear regression")

    x = np.asarray(x_data, dtype=float).reshape(-1, 1)
    y = np.asarray(y_data, dtype=float)

    reg = LinearRegression()
    reg.fit(x, y)
    y_pred = reg.predict(x)
    r2 = float(r2_score(y, y_pred))
    return {
        "gradient": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
        "r2": r2,
        "y_pred": y_pred,  # numpy array for plotting usage
    }
