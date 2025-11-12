"""
Test CCS calculation units and verify against known values
"""
import numpy as np

# Physical constants (CODATA 2018)
k_B = 1.380649e-23           # Boltzmann constant (J/K)
e = 1.602176634e-19          # Elementary charge (C)
da_to_kg = 1.66053906660e-27 # Dalton to kg

# Test parameters (typical DTIMS experiment)
drift_time_ms = 5.0          # ms (typical for small proteins)
voltage = 500.0              # V
temperature = 298.0          # K
pressure_mbar = 3.0          # mbar
length_cm = 25.05            # cm (Synapt G2)
charge = 1                   # z

# Analyte: small peptide/protein
mass_analyte_da = 1000.0     # Da
mass_buffer_da = 4.002602    # Da (Helium)

# Convert units
t_s = drift_time_ms * 1e-3   # ms -> s
L_m = length_cm * 1e-2       # cm -> m
P_Pa = pressure_mbar * 100.0 # mbar -> Pa

# Reduced mass
mass_analyte_kg = mass_analyte_da * da_to_kg
mass_buffer_kg = mass_buffer_da * da_to_kg
mu = (mass_analyte_kg * mass_buffer_kg) / (mass_analyte_kg + mass_buffer_kg)

print("=" * 60)
print("TEST CCS CALCULATION")
print("=" * 60)
print(f"\nInput Parameters:")
print(f"  Drift time: {drift_time_ms} ms = {t_s} s")
print(f"  Voltage: {voltage} V")
print(f"  Temperature: {temperature} K")
print(f"  Pressure: {pressure_mbar} mbar = {P_Pa} Pa")
print(f"  Length: {length_cm} cm = {L_m} m")
print(f"  Charge: {charge}")
print(f"  Analyte mass: {mass_analyte_da} Da = {mass_analyte_kg:.6e} kg")
print(f"  Buffer mass: {mass_buffer_da} Da = {mass_buffer_kg:.6e} kg")
print(f"  Reduced mass: {mu:.6e} kg")

# Calculate number density
N = P_Pa / (k_B * temperature)
print(f"\nNumber density N = {N:.6e} m^-3")

# Calculate mobility
K = (L_m * L_m) / (voltage * t_s)
print(f"Mobility K = {K:.6e} m^2/(V·s)")

# Mason-Schamp equation (current implementation)
prefactor = (3.0 * charge * e) / (16.0 * N)
sqrt_term = np.sqrt((2.0 * np.pi) / (mu * k_B * temperature))
inverse_mobility_term = voltage * t_s / (L_m * L_m)

print(f"\nMason-Schamp components:")
print(f"  Prefactor (3ze/16N) = {prefactor:.6e} C·m^3")
print(f"  Sqrt term = {sqrt_term:.6e} kg^-1·m^-1·s")
print(f"  Inverse mobility term (V·td/L^2) = {inverse_mobility_term:.6e} V·s/m^2")

ccs_m2 = prefactor * sqrt_term * inverse_mobility_term
ccs_A2 = ccs_m2 * 1e20

print(f"\nResults:")
print(f"  CCS = {ccs_m2:.6e} m^2")
print(f"  CCS = {ccs_A2:.2f} Å^2")

# Expected range for ~1000 Da peptide: typically 200-400 Å²
print(f"\n" + "=" * 60)
print("EXPECTED VALUES FOR REFERENCE:")
print("=" * 60)
print(f"Typical CCS for 1000 Da peptide/protein: 200-400 Å²")
print(f"Myoglobin (~17 kDa): ~1500-2000 Å²")
print(f"Small peptides (~500 Da): 100-250 Å²")

if ccs_A2 > 1000:
    print(f"\n⚠️  WARNING: Calculated CCS ({ccs_A2:.2f} Å²) seems too high!")
    print(f"   This suggests a problem with the formula or units.")
elif ccs_A2 < 100:
    print(f"\n⚠️  WARNING: Calculated CCS ({ccs_A2:.2f} Å²) seems too low!")
    print(f"   This suggests a problem with the formula or units.")
else:
    print(f"\n✓ CCS value ({ccs_A2:.2f} Å²) is in expected range")
