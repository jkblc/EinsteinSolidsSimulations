import numpy as np
import matplotlib.pyplot as plt
from math import comb, log
from scipy.stats import norm, binom

# Constants
k_B = 1.0  # Boltzmann constant (set to 1 for simplicity)

# Parameters
N_A = 3  # Number of oscillators in solid A
N_B = 3  # Number of oscillators in solid B
q_total = 6  # Total energy units
p = N_A / (N_A + N_B)  # Probability an energy unit is in solid A (p = 0.5)

# Arrays to store results
E_A_values = np.arange(0, q_total + 1)
Omega_A = np.zeros_like(E_A_values, dtype=int)
Omega_B = np.zeros_like(E_A_values, dtype=int)
Omega_total = np.zeros_like(E_A_values, dtype=int)
S_A = np.zeros_like(E_A_values, dtype=float)
S_B = np.zeros_like(E_A_values, dtype=float)
S_total = np.zeros_like(E_A_values, dtype=float)
P_binomial = np.zeros_like(E_A_values, dtype=float)

# Calculations
for i, q_A in enumerate(E_A_values):
    q_B = q_total - q_A  # Energy units in solid B

    # Multiplicities
    Omega_A[i] = comb(q_A + N_A - 1, q_A)
    Omega_B[i] = comb(q_B + N_B - 1, q_B)
    Omega_total[i] = Omega_A[i] * Omega_B[i]

    # Entropies
    if Omega_A[i] > 0:
        S_A[i] = k_B * log(Omega_A[i])
    else:
        S_A[i] = 0
    if Omega_B[i] > 0:
        S_B[i] = k_B * log(Omega_B[i])
    else:
        S_B[i] = 0
    S_total[i] = S_A[i] + S_B[i]

    # Binomial Probability
    P_binomial[i] = comb(q_total, q_A) * (p ** q_A) * ((1 - p) ** (q_total - q_A))

# Probability distribution from microstates
P = Omega_total / np.sum(Omega_total)

# Scale binomial probabilities to match Omega_total scale
scaled_P_binomial = P_binomial * np.sum(Omega_total)

# Mean energy of solid A
E_A_mean = np.sum(E_A_values * P)
print(f"Mean Energy of Solid A: {E_A_mean:.2f} units")

# Plotting multiplicities with binomial distribution line
plt.figure(figsize=(10, 6))
plt.bar(E_A_values, Omega_total, color='skyblue', edgecolor='black', label='Total Microstates (Ω_total)')
plt.title('Total Number of Microstates vs. Energy of Solid A')
plt.xlabel('Energy of Solid A (E_A)')
plt.ylabel('Total Number of Microstates (Ω_total)')
plt.xticks(E_A_values)
plt.grid(True)
plt.legend()
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
N_A_large = 300000
N_B_large = 200000
q_total_large = 100000

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
epsilon = 1.0  # Energy unit

# Parameters for part (e)
N = 50  # Number of oscillators
q_max = 200  # Increase q_max to capture higher temperatures

# Arrays to store results
q_values = np.arange(0, q_max + 1)
S_values = np.zeros_like(q_values, dtype=float)
E_values = q_values * epsilon

# Calculate entropy using Stirling's approximation with handling q=0
for i, q in enumerate(q_values):
    if q == 0:
        ln_Omega = (N - 1) * log(N - 1)
    else:
        ln_Omega = (q + N - 1) * log(q + N - 1) - q * log(q) - (N - 1) * log(N - 1)
    S_values[i] = k_B * ln_Omega

# Calculate temperature T using dS/dE
# Use central differences for better accuracy
dS_dE = np.zeros_like(S_values, dtype=float)
dS_dE[1:-1] = (S_values[2:] - S_values[:-2]) / (E_values[2:] - E_values[:-2])
dS_dE[0] = (S_values[1] - S_values[0]) / (E_values[1] - E_values[0])
dS_dE[-1] = (S_values[-1] - S_values[-2]) / (E_values[-1] - E_values[-2])

T_values = 1 / dS_dE

# Calculate heat capacity C_V using numerical derivative of E(T)
# Since E is a function of T, we can compute dE/dT numerically
C_V_values = np.zeros_like(E_values, dtype=float)
C_V_values[1:-1] = (E_values[2:] - E_values[:-2]) / (T_values[2:] - T_values[:-2])
C_V_values[0] = (E_values[1] - E_values[0]) / (T_values[1] - T_values[0])
C_V_values[-1] = (E_values[-1] - E_values[-2]) / (T_values[-1] - T_values[-2])

# Remove any NaN or infinite values resulting from division by zero
valid_indices = np.isfinite(T_values) & np.isfinite(C_V_values) & (T_values > 0)

# Analytical expression for Heat Capacity
T_analytical = np.linspace(0.1, 5, 1000)  # Avoid T=0 to prevent division by zero
C_V_analytical = N * (1 / T_analytical) ** 2 * np.exp(1 / T_analytical) / (np.exp(1 / T_analytical) - 1) ** 2

# Plotting both numerical and analytical results
plt.figure(figsize=(10, 6))
plt.plot(T_analytical, C_V_analytical, label='N')
plt.title('Heat Capacity vs. Temperature for Einstein Solid (N=50)')
plt.xlabel('Temperature (T)')
plt.ylabel('Heat Capacity at Constant Volume (C_V)')
plt.grid(True)
plt.legend()
plt.show()

# Constants
k_B = 1.0    # Boltzmann constant
epsilon = 1.0  # Energy quantum

# Parameters
N = 50
q_max = 100

q_values = np.arange(0, q_max + 1)
E_values = q_values * epsilon

# Calculate entropy S(N, q)
S_values = np.zeros_like(q_values, dtype=float)

for i, q in enumerate(q_values):
    if q == 0:
        S_values[i] = 0.0
    else:
        ln_Omega = (q + N - 1) * log(q + N - 1) - q * log(q) - (N - 1) * log(N - 1)
        S_values[i] = k_B * ln_Omega

# Calculate temperature T analytically
T_values = np.zeros_like(q_values, dtype=float)
T_values[1:] = 1.0 / (k_B * np.log((q_values[1:] + N) / q_values[1:]))

# Calculate heat capacity C_V analytically
y_values = 1.0 / (k_B * T_values[1:])
exp_y = np.exp(y_values)
dq_dT = (N * exp_y) / ((exp_y - 1) ** 2) * (1 / (k_B * T_values[1:] ** 2))
C_V_values = np.zeros_like(q_values, dtype=float)
C_V_values[1:] = dq_dT

# Plot heat capacity vs. temperature
valid_indices = T_values > 0

plt.figure(figsize=(10, 6))
plt.plot(T_values[valid_indices], C_V_values[valid_indices], label=f'N = {N}')
plt.title('Heat Capacity vs. Temperature for an Einstein Solid (N=50)')
plt.xlabel('Temperature (T)')
plt.ylabel('Heat Capacity at Constant Volume (C_V)')
plt.grid(True)
plt.legend()
plt.show()
