"""
Calculate CCS using the standard low-field limit Mason-Schamp equation
for drift tube IMS (DTIMS).

The correct relationship is:
K₀ = (18π)^(1/2) / (16 × N₀) × (ze / √(μkT)) × (1/Ω)

Solving for Ω:
Ω = (18π)^(1/2) / (16) × (ze / N₀) × (1/√(μkT)) × (1/K₀)

Where K₀ is the reduced mobility at standard conditions (273.15 K, 101325 Pa)
"""
import numpy as np

# Physical constants
k_B = 1.380649e-23
e = 1.602176634e-19
da_to_kg = 1.66053906660e-27

# Standard conditions for reduced mobility
T0 = 273.15         # K
P0 = 101325.0       # Pa (1 atm)

# Test case
drift_time_ms = 5.0
voltage = 500.0
T = 298.0           # K
P_mbar = 3.0
P_Pa = P_mbar * 100.0
L_cm = 25.05
L_m = L_cm * 1e-2
z = 1

mass_ion_da = 1000.0
mass_buffer_da = 4.002602  # Helium

mass_ion = mass_ion_da * da_to_kg
mass_buffer = mass_buffer_da * da_to_kg
mu = (mass_ion * mass_buffer) / (mass_ion + mass_buffer)

print("=" * 70)
print("DTIMS CCS CALCULATION - CORRECT FORMULATION")
print("=" * 70)

# Step 1: Calculate measured mobility K
t_s = drift_time_ms * 1e-3
K = (L_m ** 2) / (voltage * t_s)
print(f"\nMeasured mobility K = {K:.6e} m²/(V·s)")

# Step 2: Convert to reduced mobility K₀ (at standard T and P)
K0 = K * (P_Pa / P0) * (T0 / T)
print(f"Reduced mobility K₀ = {K0:.6e} m²/(V·s)")
print(f"  Conversion: K₀ = K × (P/P₀) × (T₀/T)")
print(f"  Conversion: K₀ = K × ({P_Pa}/{P0}) × ({T0}/{T})")

# Step 3: Calculate number density at standard conditions
N0 = P0 / (k_B * T0)
print(f"\nNumber density at STP: N₀ = {N0:.6e} m⁻³")

# Step 4: Calculate CCS using Mason-Schamp
# Ω = √(18π)/16 × (ze/N₀) × 1/√(μkT₀) × (1/K₀)
prefactor = np.sqrt(18 * np.pi) / 16
charge_term = (z * e) / N0
sqrt_term = 1 / np.sqrt(mu * k_B * T0)
mobility_term = 1 / K0

omega_m2 = prefactor * charge_term * sqrt_term * mobility_term
omega_A2 = omega_m2 * 1e20

print(f"\n" + "=" * 70)
print("CCS CALCULATION COMPONENTS:")
print("=" * 70)
print(f"Prefactor √(18π)/16 = {prefactor:.6f}")
print(f"Charge term (ze/N₀) = {charge_term:.6e} C·m³")
print(f"Sqrt term 1/√(μkT₀) = {sqrt_term:.6e} kg^(-1/2)·J^(-1/2)")
print(f"Mobility term 1/K₀ = {mobility_term:.6e} V·s/m²")

print(f"\n" + "=" * 70)
print("RESULT:")
print("=" * 70)
print(f"CCS = {omega_m2:.6e} m²")
print(f"CCS = {omega_A2:.2f} Å²")

print(f"\n" + "=" * 70)
print("EXPECTED: 200-400 Å² for 1000 Da peptide")
print("=" * 70)
if 150 < omega_A2 < 500:
    print(f"✓ Result ({omega_A2:.2f} Å²) is REASONABLE")
else:
    print(f"✗ Result ({omega_A2:.2f} Å²) is OUT OF RANGE")
