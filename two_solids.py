import numpy as np
import matplotlib.pyplot as plt
from math import comb, log, factorial

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
Omega_A_large = np.zeros_like(E_A_values_large, dtype=float)
Omega_B_large = np.zeros_like(E_A_values_large, dtype=float)
Omega_total_large = np.zeros_like(E_A_values_large, dtype=float)

# Calculations
for i, q_A in enumerate(E_A_values_large):
    q_B = q_total_large - q_A  # Energy units in solid B

    # Using Stirling's approximation for large N and q
    ln_Omega_A = (q_A + N_A_large) * log(q_A + N_A_large) - q_A * log(q_A) - N_A_large * log(N_A_large)
    ln_Omega_B = (q_B + N_B_large) * log(q_B + N_B_large) - q_B * log(q_B) - N_B_large * log(N_B_large)
    Omega_A_large[i] = np.exp(ln_Omega_A)
    Omega_B_large[i] = np.exp(ln_Omega_B)
    Omega_total_large[i] = Omega_A_large[i] * Omega_B_large[i]

# Normalize to prevent overflow
Omega_total_large /= np.max(Omega_total_large)

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