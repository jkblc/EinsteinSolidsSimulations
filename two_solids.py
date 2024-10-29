import numpy as np
import matplotlib.pyplot as plt
from math import comb, log

# Constants
k_B = 1.0  # Boltzmann constant (set to 1 for simplicity)

# Parameters
N_A = 3  # Number of oscillators in solid A
N_B = 3  # Number of oscillators in solid B
q_total = 6  # Total energy units

# Arrays to store results
E_A_values = np.arange(0, q_total + 1)
Omega_A = np.zeros_like(E_A_values, dtype=int)
Omega_B = np.zeros_like(E_A_values, dtype=int)
Omega_total = np.zeros_like(E_A_values, dtype=int)
S_A = np.zeros_like(E_A_values, dtype=float)
S_B = np.zeros_like(E_A_values, dtype=float)
S_total = np.zeros_like(E_A_values, dtype=float)

# Calculations
for i, q_A in enumerate(E_A_values):
    q_B = q_total - q_A  # Energy units in solid B

    # Multiplicities
    Omega_A[i] = comb(q_A + N_A - 1, q_A)
    Omega_B[i] = comb(q_B + N_B - 1, q_B)
    Omega_total[i] = Omega_A[i] * Omega_B[i]

    # Entropies
    S_A[i] = k_B * log(Omega_A[i])
    S_B[i] = k_B * log(Omega_B[i])
    S_total[i] = S_A[i] + S_B[i]

# Probability distribution
P = Omega_total / np.sum(Omega_total)

# Mean energy of solid A
E_A_mean = np.sum(E_A_values * P)
print(f"Mean Energy of Solid A: {E_A_mean:.2f} units")

# Plotting multiplicities
plt.figure(figsize=(10, 6))
plt.bar(E_A_values, Omega_total, color='skyblue', edgecolor='black')
plt.title('Total Number of Microstates vs. Energy of Solid A')
plt.xlabel('Energy of Solid A (E_A)')
plt.ylabel('Total Number of Microstates (Ω_total)')
plt.xticks(E_A_values)
plt.grid(True)
plt.show()

# Plotting Entropy
plt.figure(figsize=(10, 6))
plt.plot(E_A_values, S_A, 'o-', label='Entropy of Solid A (S_A)')
plt.plot(E_A_values, S_B, 's-', label='Entropy of Solid B (S_B)')
plt.plot(E_A_values, S_total, 'd-', label='Total Entropy (S_total)', color='red')
plt.title('Entropy vs. Energy of Solid A')
plt.xlabel('Energy of Solid A (E_A)')
plt.ylabel('Entropy (S)')
plt.legend()
plt.grid(True)
plt.show()

# Updated Parameters
N_A_large = 300
N_B_large = 200
q_total_large = 100

# Energy range for solid A
E_A_values_large = np.arange(0, q_total_large + 1)
ln_Omega_total_large = np.zeros_like(E_A_values_large, dtype=float)

# Define function for ln_Omega
def ln_Omega(N, q):
    if q == 0:
        return 0
    else:
        return (q + N) * log(q + N) - q * log(q) - N * log(N)

# Calculations
for i, q_A in enumerate(E_A_values_large):
    q_B = q_total_large - q_A  # Energy units in solid B
    
    ln_Omega_A = ln_Omega(N_A_large, q_A)
    ln_Omega_B = ln_Omega(N_B_large, q_B)
    ln_Omega_total_large[i] = ln_Omega_A + ln_Omega_B

# To prevent overflow in exp, subtract the maximum ln_Omega_total_large
ln_Omega_total_large_shifted = ln_Omega_total_large - np.max(ln_Omega_total_large)
Omega_total_large = np.exp(ln_Omega_total_large_shifted)

# Probability distribution
P_large = Omega_total_large / np.sum(Omega_total_large)
E_A_mean_large = np.sum(E_A_values_large * P_large)
sigma_large = np.sqrt(np.sum((E_A_values_large - E_A_mean_large)**2 * P_large))
relative_width_large = sigma_large / E_A_mean_large * 100
print(f"Mean Energy of Solid A: {E_A_mean_large:.2f} units")
print(f"Relative Width of the Distribution: {relative_width_large:.2f}%")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(E_A_values_large, P_large, color='green')
plt.title('Probability Distribution for Large N')
plt.xlabel('Energy of Solid A (E_A)')
plt.ylabel('Probability P(E_A)')
plt.grid(True)
plt.show()

# Constants
k_B = 1.0  # Boltzmann constant
epsilon = 1.0  # Energy unit (set to 1 for simplicity)

# Parameters for part (e)
N = 50  # Number of oscillators
q_max = 100  # Maximum energy units

# Arrays to store results
q_values = np.arange(1, q_max + 1)  # Start from q = 1 to avoid log(0)
S_values = np.zeros_like(q_values, dtype=float)
E_values = q_values * epsilon
T_values = np.zeros_like(q_values, dtype=float)
C_V_values = np.zeros_like(q_values, dtype=float)

# Calculate entropy using Stirling's approximation
for i, q in enumerate(q_values):
    # Entropy S = k_B * ln(Ω)
    ln_Omega = (q + N) * log(q + N) - q * log(q) - N * log(N)
    S_values[i] = k_B * ln_Omega

# Calculate temperature T using 1/T = dS/dE
dS_dE = np.gradient(S_values, epsilon)
T_values = 1 / dS_dE

# Calculate heat capacity C_V = dE/dT
dE_dT = np.gradient(E_values, T_values)
C_V_values = dE_dT

# Plot Heat Capacity vs. Temperature
plt.figure(figsize=(10, 6))
plt.plot(T_values, C_V_values, label=f'N = {N}')
plt.title('Heat Capacity vs. Temperature for Einstein Solid (N=50)')
plt.xlabel('Temperature (T)')
plt.ylabel('Heat Capacity at Constant Volume (C_V)')
plt.grid(True)
plt.legend()
plt.show()
