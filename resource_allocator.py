# DROO_D2D_PARTIAL_OFFLOADING/resource_allocator.py

import numpy as np
from scipy.optimize import minimize
from config import *  # Import all parameters from config.py
from environment import SystemEnvironment


class ResourceAllocator:
    """
    Solves the resource allocation sub-problem for a given offloading action.

    This class acts as the "critic" in the DRL framework. For a fixed
    offloading decision matrix, it finds the optimal:
    - a: Time fraction for energy harvesting.
    - cpu_freqs: CPU frequencies for each WD.
    - tx_powers: Transmit powers for each WD.

    The optimization goal is to minimize the weighted sum of total system latency
    and total task energy, subject to hardware and energy harvesting constraints.
    """

    def __init__(self):
        """Initializes the solver."""
        # The number of variables to optimize: 1 (a) + N_WDs (freqs) + N_WDs (powers)
        self.num_vars = 1 + 2 * N_WDs

    def solve(self, action_matrix: np.ndarray, channel_matrix: np.ndarray, task_sizes: np.ndarray) -> dict:
        """
        Main method to solve the resource allocation problem.

        Args:
            action_matrix (np.ndarray): A [N_WDs, N_NODES] matrix of offloading decisions.
            channel_matrix (np.ndarray): A [N_NODES, N_NODES] matrix of current channel gains.
            task_sizes (np.ndarray): An array of task sizes for each WD.

        Returns:
            A dictionary containing the optimization results, including the final
            objective value, optimal variables, and detailed performance metrics.
        """
        # Initial guess for the optimizer. A mid-range start is generally robust.
        x0 = np.concatenate([
            np.array([0.5]),  # a
            np.full(N_WDs, (CPU_FREQ_MIN_HZ + CPU_FREQ_MAX_HZ) / 2),  # cpu_freqs
            np.full(N_WDs, (TX_POWER_MIN_W + TX_POWER_MAX_W) / 2)  # tx_powers
        ])

        # # Initial guess for the optimizer. All scaled variables start at 0.5.
        # x0 = np.full(self.num_vars, 0.5)

        # Define bounds for each variable
        bounds = self._get_bounds()

        # Run the numerical optimization
        result = minimize(
            self._objective_function,
            x0,
            args=(action_matrix, channel_matrix, task_sizes),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-9}  # More precise options
        )


        # If solver fails, use the initial guess as a fallback
        solution = result.x if result.success else x0

        # Recalculate all metrics with the final solution
        final_metrics = self._calculate_all_metrics(solution, action_matrix, channel_matrix, task_sizes)

        return final_metrics

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Defines the feasible range for each optimization variable."""
        bounds = [(0.01, 0.99)]  # Bounds for 'a'
        bounds += [(CPU_FREQ_MIN_HZ, CPU_FREQ_MAX_HZ)] * N_WDs  # Bounds for CPU frequencies
        bounds += [(TX_POWER_MIN_W, TX_POWER_MAX_W)] * N_WDs  # Bounds for transmit powers

        # # All variables are now scaled to [0, 1] for optimization stability
        # num_vars = 1 + 2 * N_WDs
        # bounds = [(0.0, 1.0)] * num_vars
        return bounds

    def _objective_function(self, x: np.ndarray, action_matrix: np.ndarray,
                            channel_matrix: np.ndarray, task_sizes: np.ndarray) -> float:
        """
        The objective function to be minimized by scipy.optimize.minimize.
        It returns a single scalar value: the weighted objective.
        """
        metrics = self._calculate_all_metrics(x, action_matrix, channel_matrix, task_sizes)
        return metrics['objective']

    def _calculate_all_metrics(self, x: np.ndarray, action_matrix: np.ndarray,
                               channel_matrix: np.ndarray, task_sizes: np.ndarray) -> dict:
        """
        Calculates all performance metrics based on the robust 3-phase model.
        This is the core implementation of our system model.

        Args:
            x (np.ndarray): The vector of optimization variables [a, freqs, powers].
            action_matrix, channel_matrix, task_sizes: System state.

        Returns:
            A dictionary containing all calculated performance metrics.
        """
        # 1. Unpack optimization variables
        a = x[0]
        cpu_freqs = x[1: 1 + N_WDs]
        tx_powers = x[1 + N_WDs:]

        # # 1. DENORMALIZE optimization variables
        # x_scaled = x  # x is the scaled vector from the optimizer
        #
        # # 1.1 Denormalize 'a' (a_min=0.01, a_max=0.99 from bounds)
        # a_min, a_max = 0.01, 0.99
        # a = x_scaled[0] * (a_max - a_min) + a_min
        #
        # # 1.2 Denormalize CPU Frequencies
        # freq_min, freq_max = CPU_FREQ_MIN_HZ, CPU_FREQ_MAX_HZ
        # cpu_freqs_scaled = x_scaled[1: 1 + N_WDs]
        # cpu_freqs = cpu_freqs_scaled * (freq_max - freq_min) + freq_min
        #
        # # 1.3 Denormalize Transmit Powers
        # power_min, power_max = TX_POWER_MIN_W, TX_POWER_MAX_W
        # tx_powers_scaled = x_scaled[1 + N_WDs:]
        # tx_powers = tx_powers_scaled * (power_max - power_min) + power_min

        # --- PHASE 1: Energy Harvesting ---
        # Energy harvested by each WD from the AP (h_0j)
        channels_to_ap = channel_matrix[0, 1:]
        harvested_energy = ENERGY_HARVEST_EFFICIENCY * AP_TX_POWER_W * channels_to_ap * a

        # --- PHASE 2: Sequential Offloading (Calculate Latency T_offload) ---
        T_offload = 0
        wd_tx_times = np.zeros(N_WDs)  # Store each WD's total transmit time

        for i in range(N_WDs):  # For each task originator WD_i
            # t_i_off_tx = 0
            # for j in range(N_NODES):  # For each destination node_j
            #     # Local computation is not a transmission
            #     if j == i + 1: continue
            #
            #     offload_ratio = action_matrix[i, j]
            #     if offload_ratio > 1e-9:  # If there is data to offload
            #         task_to_offload = task_sizes[i] * offload_ratio
            #         channel_gain = channel_matrix[i + 1, j]
            #
            #         # Shannon-Hartley theorem for data rate
            #         snr = (tx_powers[i] * channel_gain) / NOISE_POWER
            #         data_rate = (BANDWIDTH_HZ / V_U) * np.log2(1 + snr)
            #
            #         if data_rate > 1e-9:
            #             t_i_off_tx += task_to_offload / data_rate
            #         else:
            #             t_i_off_tx += 1e9  # Effectively infinite time (huge penalty)
            #
            # wd_tx_times[i] = t_i_off_tx

            # --- 修改开始 ---
            # t_i_off_tx_parts 存储WD_i到每个目标节点的独立传输时间
            t_i_off_tx_parts = []
            # --- 修改结束 ---

            for j in range(N_NODES):  # 对于每个目标节点 node_j
                if j == i + 1: continue

                offload_ratio = action_matrix[i, j]
                if offload_ratio > 1e-9:
                    task_to_offload = task_sizes[i] * offload_ratio
                    channel_gain = channel_matrix[i + 1, j]

                    snr = (tx_powers[i] * channel_gain) / NOISE_POWER
                    # 假设总带宽被拆分，每个子信道带宽为 B / N_offloads
                    # 为简化，我们假设功率和带宽可以理想分配，这里主要改变时间模型
                    data_rate = (BANDWIDTH_HZ / V_U) * np.log2(1 + snr)

                    if data_rate > 1e-9:
                        # --- 修改开始 ---
                        transmission_time = task_to_offload / data_rate
                        t_i_off_tx_parts.append(transmission_time)
                        # --- 修改结束 ---
                    else:
                        # --- 修改开始 ---
                        t_i_off_tx_parts.append(1e9)  # 巨大惩罚
                        # --- 修改结束 ---

            # --- 修改开始 ---
            # WD_i的总传输时间现在是所有并行传输中最长的时间
            if t_i_off_tx_parts:
                wd_tx_times[i] = np.max(t_i_off_tx_parts)
            else:
                wd_tx_times[i] = 0
            # --- 修改结束 ---

        T_offload = np.sum(wd_tx_times)

        # --- PHASE 3: Parallel Execution (Calculate Latency T_execute) ---
        wd_computation_loads = np.zeros(N_WDs)
        for j in range(N_WDs):  # For each computing node WD_j
            # Load from its own task
            local_ratio = action_matrix[j, j + 1]
            wd_computation_loads[j] += task_sizes[j] * local_ratio
            # Load from other WDs' tasks
            for i in range(N_WDs):
                if i == j: continue
                d2d_ratio = action_matrix[i, j + 1]
                wd_computation_loads[j] += task_sizes[i] * d2d_ratio

        # Time to compute the total load on each WD
        wd_computation_times = (wd_computation_loads * CYCLES_PER_BIT) / cpu_freqs
        T_execute = np.max(wd_computation_times) if wd_computation_times.size > 0 else 0

        # --- Calculate Total Latency ---
        T_total = a + T_offload + T_execute

        # --- Calculate Total Energy and Penalty ---
        total_task_energy = 0
        penalty = 0

        # A. Calculate total energy FOR each task i (E_i^task)
        for i in range(N_WDs):
            # Local computation energy for task i
            local_comp_time = (task_sizes[i] * action_matrix[i, i + 1] * CYCLES_PER_BIT) / cpu_freqs[i]
            e_loc = ENERGY_EFFICIENCY_COEFF * (cpu_freqs[i] ** 3) * local_comp_time

            # Transmission energy for task i
            e_tx = tx_powers[i] * wd_tx_times[i]

            # D2D computation energy for task i (computed by other WDs)
            e_d2d_comp = 0
            for j in range(N_WDs):
                if i == j: continue
                d2d_task_part = task_sizes[i] * action_matrix[i, j + 1]
                comp_time_on_j = (d2d_task_part * CYCLES_PER_BIT) / cpu_freqs[j]
                e_d2d_comp += ENERGY_EFFICIENCY_COEFF * (cpu_freqs[j] ** 3) * comp_time_on_j

            total_task_energy += (e_loc + e_tx + e_d2d_comp)

        # B. Calculate total energy CONSUMED BY each WD j (E_j^self) for constraint checking
        for j in range(N_WDs):
            # Total computation energy consumed by WD j
            e_j_comp = ENERGY_EFFICIENCY_COEFF * (cpu_freqs[j] ** 3) * wd_computation_times[j]
            # Total transmission energy consumed by WD j
            e_j_tx = tx_powers[j] * wd_tx_times[j]

            e_j_self = e_j_comp + e_j_tx

            # Check for energy neutrality constraint violation
            energy_violation = max(0, e_j_self - harvested_energy[j])
            #print(e_j_comp,e_j_tx,e_j_self, harvested_energy[j], energy_violation)
            penalty += PENALTY_ENERGY_VIOLATION * energy_violation

        # --- Final Objective Value ---
        objective = (TIME_WEIGHT * T_total +
                     ENERGY_WEIGHT * total_task_energy +
                     penalty)
        print(ENERGY_WEIGHT * total_task_energy)
        return {
            "objective": objective, "total_time": T_total, "total_energy": total_task_energy,
            "penalty": penalty, "a": a, "cpu_freqs": cpu_freqs, "tx_powers": tx_powers,
            "T_harvest": a, "T_offload": T_offload, "T_execute": T_execute
        }


# Example usage:
if __name__ == '__main__':
    # Create a dummy environment and state for testing
    env = SystemEnvironment()
    # chan_matrix, _, tasks = env.get_new_state()
    # print("Channel Matrix:", chan_matrix, "Task Sizes:", tasks)
    chan_matrix=np.array([[ 0  ,  2e-8 , 3e-8,  5e-8 ] , # AP row
 [ 2e-8 , 0 ,   8e-8 , 7e-8 ],  # WD1 row (good h12, h13)
 [ 3e-8, 8e-8 , 0  ,  9e-8 ] , # WD2 row (poor h21 implied by action, good h23) - NOTE: h21 = h12, this matrix needs care. Let's make h12 moderate instead.
 [ 5e-8 , 7e-8 , 9e-8 , 0    ]])
    tasks=[4e6, 3e6, 2e6]
    # Create a dummy offloading action (e.g., 50% local, 50% to AP)
    dummy_action = np.zeros((N_WDs, N_NODES))
    # for i in range(N_WDs):
    #     dummy_action[i, 0] = 0.5  # 50% to AP
    #     dummy_action[i, i + 1] = 0.5  # 50% local

    dummy_action[0] = [0.2, 0.3, 0.3, 0.2]
    dummy_action[1] = [0.3, 0.0, 0.4, 0.3]
    dummy_action[2] = [0.5, 0.0, 0.0, 0.5]

    print("\n--- Testing Resource Allocator ---")
    print(f"Dummy Action:\n{dummy_action}")

    allocator = ResourceAllocator()
    results = allocator.solve(dummy_action, chan_matrix, tasks)

    print("\n--- Optimization Results ---")
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: \n{value}")
        else:
            print(f"{key}: {value:.4f}")