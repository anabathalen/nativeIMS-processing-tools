"""
Compare different forms of the Mason-Schamp equation
to see which gives reasonable values
"""
import numpy as np

# Physical constants
k_B = 1.380649e-23
e = 1.602176634e-19
da_to_kg = 1.66053906660e-27

# Test case
t_s = 0.005          # 5 ms
V = 500.0            # V
T = 298.0            # K
P_Pa = 300.0         # Pa (3 mbar)
L_m = 0.2505         # m (25.05 cm)
z = 1

mass_ion = 1000.0 * da_to_kg
mass_buffer = 4.002602 * da_to_kg
mu = (mass_ion * mass_buffer) / (mass_ion + mass_buffer)

N = P_Pa / (k_B * T)

print("=" * 70)
print("COMPARING MASON-SCHAMP EQUATION FORMS")
print("=" * 70)

# Form 1: Ω = (3ze/16N) × √(2π/μkT) × (Vt/L²)
print("\nForm 1: Ω = (3ze/16N) × √(2π/μkT) × (Vt/L²)")
factor1 = (3 * z * e) / (16 * N)
sqrt1 = np.sqrt((2 * np.pi) / (mu * k_B * T))
term1 = V * t_s / (L_m * L_m)
omega1_m2 = factor1 * sqrt1 * term1
omega1_A2 = omega1_m2 * 1e20
print(f"  Result: {omega1_A2:.2f} Å²")

# Form 2: Ω = (3ze/16N) × √(18π/μkT) × (Vt/L²) 
# This has √18π instead of √2π
print("\nForm 2: Ω = (3ze/16N) × √(18π/μkT) × (Vt/L²)")
factor2 = (3 * z * e) / (16 * N)
sqrt2 = np.sqrt((18 * np.pi) / (mu * k_B * T))
term2 = V * t_s / (L_m * L_m)
omega2_m2 = factor2 * sqrt2 * term2
omega2_A2 = omega2_m2 * 1e20
print(f"  Result: {omega2_A2:.2f} Å²")

# Form 3: Full Mason-Schamp with (18π)^(1/2) prefactor
# Ω = (18π)^(1/2) / 16 × (ze/N) × 1/√(μkT) × (Vt/L²)
print("\nForm 3: Ω = √(18π)/16 × (ze/N) × 1/√(μkT) × (Vt/L²)")
factor3 = np.sqrt(18 * np.pi) / 16 * (z * e / N)
sqrt3 = 1 / np.sqrt(mu * k_B * T)
term3 = V * t_s / (L_m * L_m)
omega3_m2 = factor3 * sqrt3 * term3
omega3_A2 = omega3_m2 * 1e20
print(f"  Result: {omega3_A2:.2f} Å²")

# Check ratio
print(f"\n" + "=" * 70)
print(f"Form 2 / Form 1 = {omega2_A2 / omega1_A2:.4f}")
print(f"√(18π) / √(2π) = {np.sqrt(18 * np.pi / (2 * np.pi)):.4f} = √9 = 3")
print(f"Form 3 / Form 1 = {omega3_A2 / omega1_A2:.4f}")

print(f"\n" + "=" * 70)
print("EXPECTED RANGE: 200-400 Å² for 1000 Da peptide")
print("=" * 70)
print(f"Form 1: {omega1_A2:.2f} Å² - {'TOO HIGH' if omega1_A2 > 500 else ('TOO LOW' if omega1_A2 < 150 else 'REASONABLE')}")
print(f"Form 2: {omega2_A2:.2f} Å² - {'TOO HIGH' if omega2_A2 > 500 else ('TOO LOW' if omega2_A2 < 150 else 'REASONABLE')}")
print(f"Form 3: {omega3_A2:.2f} Å² - {'TOO HIGH' if omega3_A2 > 500 else ('TOO LOW' if omega3_A2 < 150 else 'REASONABLE')}")
