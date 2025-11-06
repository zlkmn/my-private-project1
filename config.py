# DROO_D2D_PARTIAL_OFFLOADING/config.py

"""
Centralized configuration file for the DRL-based D2D Offloading project.
All system parameters, constants, and hyperparameters are defined here.
"""

# -----------------
# System Parameters
# -----------------
N_WDs = 3  # Number of Wireless Devices (WDs)
N_NODES = N_WDs + 1  # Total number of nodes (AP + WDs)

# --- Task Model ---
TASK_SIZE_MIN_KB = 2  #100# Minimum task size in KiloBytes
TASK_SIZE_MAX_KB = 10 #500 # Maximum task size in KiloBytes
TASK_SIZE_MIN = TASK_SIZE_MIN_KB * 1024 * 8  # Minimum task size in bits
TASK_SIZE_MAX = TASK_SIZE_MAX_KB * 1024 * 8  # Maximum task size in bits
# 修改前 (Before):
CYCLES_PER_BIT = 500 # Number of CPU cycles required to process one bit

# 修改后 (After):
# CYCLES_PER_BIT = 8000  # [实验修改] 大幅增加每bit所需的计算周期，使任务变为计算密集型

# --- Channel Model ---
# --- Channel Model ---
# [新增参数] 莱斯K因子。K越大，信道波动越小，各信道质量越接近。
# K=0 时为瑞利衰落。建议实验值：3.0 或 5.0
RICIAN_K_FACTOR = 5.0
MIN_DISTANCE_M = 2.5   # Minimum distance between nodes in meters
MAX_DISTANCE_M = 4   # Maximum distance between nodes in meters
AP_POSITION = [0, 0]   # Position of the Access Point (AP)
PATH_LOSS_EXPONENT = 2.3  # Path loss exponent
ANTENNA_GAIN = 4.11    # Antenna gain
CARRIER_FREQ_HZ = 915e6 # Carrier frequency in Hz (915 MHz)
LIGHT_SPEED = 3e8      # Speed of light in m/s
NOISE_POWER = 1e-10    # Noise power in Watts (-70 dBm)
BANDWIDTH_HZ = 1e6     # Communication bandwidth in Hz (2 MHz)
V_U = 1.1              # Communication overhead factor from the paper

# --- Energy Model ---
ENERGY_HARVEST_EFFICIENCY = 0.8  # Efficiency of energy harvesting
# 修改前 (Before):
AP_TX_POWER_W = 3.0e6             # AP's transmit power for energy harvesting in Watts

# 修改后 (After):
# AP_TX_POWER_W = 0.5              # [实验修改] 显著降低充电功率，让能量变得稀缺

# 修改前 (Before):
ENERGY_EFFICIENCY_COEFF = 1e-30#2e-31 # Computation energy efficiency coefficient (kappa)

# 修改后 (After):
# ENERGY_EFFICIENCY_COEFF = 5e-25  # [实验修改] 增大能效系数，让计算消耗更多能量

# --- Hardware Constraints ---
CPU_FREQ_MAX_HZ = 1.5e9  # Maximum CPU frequency in Hz (2 GHz)
CPU_FREQ_MIN_HZ = 0.5e9 # Minimum CPU frequency in Hz (0.1 GHz)
TX_POWER_MAX_W = 0.1   # Maximum transmit power of WDs in Watts (27 dBm)
TX_POWER_MIN_W = 0.001 # Minimum transmit power of WDs in Watts (0 dBm)

# --------------------
# Optimization Weights
# --------------------
TIME_WEIGHT = 1.0     # Weight for total time in the objective function
ENERGY_WEIGHT = 50   # Weight for total energy in the objective function
PENALTY_ENERGY_VIOLATION = 25000.0 # Penalty factor for energy constraint violation

# -----------------
# DRL Agent Hyperparameters
# -----------------
# --- Network Architecture ---
# Input: channel gains + task sizes
# channel gains: N_NODES * (N_NODES - 1) / 2 unique channels
# task sizes: N_WDs
NUM_CHANNELS_FLAT = N_NODES * (N_NODES - 1) // 2
INPUT_DIM = NUM_CHANNELS_FLAT + N_WDs
# Output: action matrix (N_WDs x N_NODES)
OUTPUT_DIM = N_WDs * N_NODES
NET_TOPOLOGY = [INPUT_DIM, 120, 80, OUTPUT_DIM]

# --- Training Parameters ---
LEARNING_RATE = 0.001
TRAINING_INTERVAL = 10  # Train the network every X time frames
BATCH_SIZE = 128
MEMORY_SIZE = 1024

# --- Action Generation ---
# The number of candidate actions to generate from the relaxed DNN output.
# As we are doing partial offloading, we will work directly with the soft
# output first, so K is effectively 1.
K_CANDIDATES = 1
DECODER_MODE = 'SOFT' # 'SOFT' for direct softmax, 'OP' for Order-Preserving candidates

# -----------------
# Simulation Parameters
# -----------------
NUM_TIME_FRAMES = 10000  # Total number of time frames to simulate