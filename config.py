# -----------------
# System Parameters
# -----------------
N_WDs = 3  # Number of Wireless Devices (WDs)
N_NODES = N_WDs + 1  # Total number of nodes (AP + WDs)

# --- Task Model ---
TASK_SIZE_MIN_KB = 50  # Minimum task size in KiloBytes
TASK_SIZE_MAX_KB = 100 # Maximum task size in KiloBytes
TASK_SIZE_MIN = TASK_SIZE_MIN_KB * 1000  # Minimum task size in bits
TASK_SIZE_MAX = TASK_SIZE_MAX_KB * 1000  # Maximum task size in bits
CYCLES_PER_BIT = 10 # Number of CPU cycles required to process one bit (phi)

# --- Channel Model ---
MIN_DISTANCE_M = 2.5   # Minimum distance between nodes in meters
MAX_DISTANCE_M = 5.2   # Maximum distance between nodes in meters
AP_POSITION = [0, 0]   # Position of the Access Point (AP)
PATH_LOSS_EXPONENT = 2.8  # Path loss exponent
ANTENNA_GAIN = 4.11    # Antenna gain
CARRIER_FREQ_HZ = 915e6 # Carrier frequency in Hz (915 MHz)
LIGHT_SPEED = 3e8      # Speed of light in m/s

NOISE_POWER = 1e-10    # Noise power in Watts (N0)
BANDWIDTH_HZ = 3e6     # Communication bandwidth in Hz (B)
V_U = 1.1              # Communication overhead factor (v_u)

# --- Energy Model ---
ENERGY_HARVEST_EFFICIENCY = 0.8  # Efficiency of energy harvesting (mu)
AP_TX_POWER_W = 6.0            # AP's transmit power for energy harvesting in Watts (P)
ENERGY_EFFICIENCY_COEFF = 2e-28# Computation energy efficiency coefficient (kappa)
ENERGY_HARVEST_SCALE_S = 1.0   # Energy harvesting scaling factor (S) [NEW]

# --- Hardware Constraints ---
CPU_FREQ_MAX_HZ = 5e7  # Maximum CPU frequency in Hz (f_max)
CPU_FREQ_MIN_HZ = 1e7  # Minimum CPU frequency in Hz (f_min)
TX_POWER_MAX_W = 50e-6   # Maximum transmit power of WDs in Watts (P_max)
TX_POWER_MIN_W = 0.1e-6  # Minimum transmit power of WDs in Watts (P_min)

# --------------------
# Optimization Weights
# --------------------
TIME_WEIGHT = 1.0     # Weight for total time in the objective function (w_t)
ENERGY_WEIGHT = 1e5   # Weight for total energy in the objective function (w_E)

# -----------------
# DRL Agent Hyperparameters
# -----------------
# --- Network Architecture ---
NUM_CHANNELS_FLAT = N_NODES * (N_NODES - 1) // 2
INPUT_DIM = NUM_CHANNELS_FLAT + N_WDs
OUTPUT_DIM = N_WDs * N_NODES
NET_TOPOLOGY = [INPUT_DIM, 120, 80, OUTPUT_DIM]

# --- Training Parameters ---
LEARNING_RATE = 0.001
TRAINING_INTERVAL = 10  # Train the network every X time frames
BATCH_SIZE = 128
MEMORY_SIZE = 1024

# DRL Agent Parameters (for DROO)
# --------------------
K_CANDIDATES = 4
EXPLORATION_NOISE = 0.3

# Simulation Parameters
# -----------------
NUM_TIME_FRAMES = 5000  # Total number of time frames to simulate
EPS = 1e-12             # Small epsilon for numerical stability
