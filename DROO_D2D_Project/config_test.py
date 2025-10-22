# -----------------
# System Parameters
# -----------------
N_WDs = 3  # Number of Wireless Devices (WDs)
N_NODES = N_WDs + 1  # Total number of nodes (AP + WDs)

# --- Task Model ---
CYCLES_PER_BIT = 100 # Number of CPU cycles required to process one bit

# --- Channel Model ---
NOISE_POWER = 1e-10    # Noise power in Watts (-70 dBm)
BANDWIDTH_HZ = 2e6     # Communication bandwidth in Hz (2 MHz)
V_U = 1.1              # Communication overhead factor from the paper

# --- Energy Model ---
ENERGY_HARVEST_EFFICIENCY = 0.8  # Efficiency of energy harvesting
AP_TX_POWER_W = 3.0            # AP's transmit power for energy harvesting in Watts
ENERGY_EFFICIENCY_COEFF = 1e-26#2e-31 # Computation energy efficiency coefficient (kappa)

# --- Hardware Constraints ---
CPU_FREQ_MAX_HZ = 5e7  # Maximum CPU frequency in Hz (2 GHz)
CPU_FREQ_MIN_HZ = 0.5e7 # Minimum CPU frequency in Hz (0.1 GHz)
TX_POWER_MAX_W = 5e-6   # Maximum transmit power of WDs in Watts (27 dBm)
TX_POWER_MIN_W = 0.1e-6 # Minimum transmit power of WDs in Watts (0 dBm)

# --------------------
# Optimization Weights
# --------------------
TIME_WEIGHT = 1.0     # Weight for total time in the objective function
ENERGY_WEIGHT = 5e4   # Weight for total energy in the objective function
