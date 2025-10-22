# DROO_D2D_PARTIAL_OFFLOADING/resource_allocator.py

import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import time # For potential timing/debugging
from environment import SystemEnvironment
# Import parameters directly (ensure config.py is accessible)
from config_test import *

EPS = 1e-12 # Small epsilon for numerical stability (avoid division by zero)

class ResourceAllocator:
    """
    Solves the resource allocation sub-problem using auxiliary variables
    and hard constraints for a given offloading action.

    Finds optimal (a, f, P) by solving a Non-Linear Program (NLP).
    """

    def __init__(self):
        """Initializes the solver."""
        self.num_wd = N_WDs
        # Total variables: 1 (a) + N (f) + N (P) + N (t_tx aux) + 1 (T_exec aux)
        self.num_vars = 1 + 3 * self.num_wd + 1
        print(f"ResourceAllocator initialized for {self.num_wd} WDs.")

    def _split_vars(self, z: np.ndarray) -> tuple:
        """Helper to unpack the optimization variable vector z."""
        N = self.num_wd
        a = z[0]
        f = z[1 : 1 + N] * 1e6
        P = z[1 + N : 1 + 2 * N] *1e-6
        t_tx = z[1 + 2 * N : 1 + 3 * N]
        T_exec = z[-1]
        return a, f, P, t_tx, T_exec

    def _calculate_loads(self, action_matrix: np.ndarray, task_sizes: np.ndarray) -> np.ndarray:
        """Calculates the total computation load L_j for each WD j."""
        N = self.num_wd
        loads = np.zeros(N)
        for j in range(N):
            # Local part for WD j
            loads[j] += task_sizes[j] * action_matrix[j, j + 1]
            # Parts offloaded from other WDs i to WD j
            for i in range(N):
                if i == j: continue
                loads[j] += task_sizes[i] * action_matrix[i, j + 1]
        return loads

    def _get_offload_destinations(self, action_matrix: np.ndarray) -> tuple[list, np.ndarray]:
        """Determines offload destinations D_i and counts n_i for each WD i."""
        N = self.num_wd
        destinations = []
        counts = np.zeros(N, dtype=int)
        for i in range(N):
            D_i = [j for j in range(N_NODES) if j != i + 1 and action_matrix[i, j] > EPS]
            destinations.append(D_i)
            counts[i] = len(D_i)
        return destinations, counts

    def _objective_func(self, z: np.ndarray, action_matrix: np.ndarray, task_sizes: np.ndarray) -> float:
        """Calculates the objective function value for the optimizer."""
        N = self.num_wd
        a, f, P, t_tx, T_exec = self._split_vars(z)

        # Calculate total task energy E_i^task components
        total_E_task = 0
        for i in range(N):
            E_loc = ENERGY_EFFICIENCY_COEFF * (f[i]**2) * (task_sizes[i] * action_matrix[i, i + 1]) * CYCLES_PER_BIT
            E_tx = P[i] * t_tx[i] # Uses auxiliary variable t_tx[i]
            E_d2d_comp = 0
            for j in range(N):
                if i == j: continue
                E_d2d_comp += ENERGY_EFFICIENCY_COEFF * (f[j]**2) * (task_sizes[i] * action_matrix[i, j + 1]) * CYCLES_PER_BIT
            total_E_task += (E_loc + E_tx + E_d2d_comp)

        # Calculate total time T_total components
        T_total = a + np.sum(t_tx) + T_exec # Uses auxiliary variables

        # Combine weighted objectives
        objective = TIME_WEIGHT * T_total + ENERGY_WEIGHT * total_E_task
        return objective

    def solve(self, action_matrix: np.ndarray, channel_matrix: np.ndarray, task_sizes: np.ndarray) -> dict:
        """
        Main method to solve the resource allocation NLP using hard constraints.

        Args: see previous version.
        Returns: see previous version, including optimal auxiliary vars.
        """
        N = self.num_wd
        # Pre-calculate loads and destinations
        loads = self._calculate_loads(action_matrix, task_sizes)
        destinations, offload_counts = self._get_offload_destinations(action_matrix)

        # --- Define Bounds ---
        lb = [EPS] + [CPU_FREQ_MIN_HZ*10e-6]*N + [TX_POWER_MIN_W*10e6]*N + [0.0]*N + [0.0]
        ub = [1.0 - EPS] + [CPU_FREQ_MAX_HZ*10e-6]*N + [TX_POWER_MAX_W*10e6]*N + [np.inf]*N + [np.inf]
        bounds = Bounds(lb, ub)

        # --- Define Constraints ---
        constraints = []

        # 1. Transmission Time Constraints (t_i^tx >= actual time for each j in D_i)
        def tx_constraints_func(z):
            a, f, P, t_tx, T_exec = self._split_vars(z)
            constraint_values = []
            for i in range(N):
                n_i = max(1, offload_counts[i]) # Use n_i=1 even if no offload, avoids div by zero if D_i is empty later
                bw_share = BANDWIDTH_HZ / (n_i * V_U)
                for j in destinations[i]:
                    h_ij = channel_matrix[i + 1, j]
                    snr = (P[i] * h_ij) / max(EPS, NOISE_POWER)
                    rate = bw_share * np.log2(1.0 + max(0.0, snr))
                    required_time = (task_sizes[i] * action_matrix[i, j]) / max(EPS, rate)
                    constraint_values.append(t_tx[i] - required_time) # Constraint: func >= 0
                # If no destinations, constraint is implicitly t_tx[i] >= 0 (covered by bounds)
            # Pad with zeros if no constraints generated (should not happen if N>0)
            return np.array(constraint_values) if constraint_values else np.array([0.0])

        num_tx_constraints = sum(offload_counts)
        constraints.append(NonlinearConstraint(tx_constraints_func, 0.0, np.inf)) # All >= 0

        # 2. Execution Time Constraints (T_exec >= actual time for each j)
        def exec_constraints_func(z):
            a, f, P, t_tx, T_exec = self._split_vars(z)
            constraint_values = []
            for j in range(N):
                required_time = (loads[j] * CYCLES_PER_BIT) / max(EPS, f[j])
                constraint_values.append(T_exec - required_time) # Constraint: func >= 0
            return np.array(constraint_values)

        constraints.append(NonlinearConstraint(exec_constraints_func, 0.0, np.inf)) # All >= 0

        # 3. Energy Neutrality Constraints (E_harv_j >= E_self_j for each j)
        def energy_constraints_func(z):
            a, f, P, t_tx, T_exec = self._split_vars(z)
            constraint_values = []
            for j in range(N):
                h0j = channel_matrix[0, j + 1]
                E_harv = ENERGY_HARVEST_EFFICIENCY * AP_TX_POWER_W * h0j * a
                E_comp_self = ENERGY_EFFICIENCY_COEFF * (f[j]**2) * loads[j] * CYCLES_PER_BIT
                E_tx_self = P[j] * t_tx[j]
                E_self = E_comp_self + E_tx_self
                constraint_values.append(E_harv - E_self) # Constraint: func >= 0
            return np.array(constraint_values)

        constraints.append(NonlinearConstraint(energy_constraints_func, 0.0, np.inf)) # All >= 0

        # --- Initial Guess ---
        # Start feasible and slightly perturbed
        a0 = 0.5
        f0 = np.random.uniform(CPU_FREQ_MIN_HZ*1e-6, CPU_FREQ_MAX_HZ*1e-6, N) * 0.5
        P0 = np.random.uniform(TX_POWER_MIN_W*1e6, TX_POWER_MAX_W*1e6, N) * 0.5
        # Estimate initial t_tx, T_exec loosely (avoiding zero)
        t_tx0 = np.ones(N) * 0.1 # Placeholder positive value
        T_exec0 = 0.1           # Placeholder positive value
        z0 = np.concatenate([[a0], f0, P0, t_tx0, [T_exec0]])

        # --- Solve the NLP ---
        # Recommended: try 'trust-constr' first for robustness, then 'SLSQP' if needed
        #solver_method = 'trust-constr'
        solver_method = 'SLSQP'
        start_opt_time = time.time()
        result = minimize(
            lambda z: self._objective_func(z, action_matrix, task_sizes),
            z0,
            method=solver_method,
            bounds=bounds,
            constraints=constraints,
            #options={'maxiter': 500, 'verbose': 0} # Set verbose=2 for detailed output
            options = {'maxiter': 1000, 'ftol': 1e-8, 'disp': False}  # disp=True for detailed output
            #options = {'maxiter': 1000, 'disp': False}  # disp=True for detailed output
        )
        opt_duration = time.time() - start_opt_time

        # --- Process Results ---
        if result.success:
            z_opt = result.x
            a_opt, f_opt, P_opt, t_tx_opt, T_exec_opt = self._split_vars(z_opt)
            final_objective = result.fun
            # Recalculate components for reporting (redundant but clean)
            final_T_total = a_opt + np.sum(t_tx_opt) + T_exec_opt
            final_total_E_task = 0 # Recalculate E_task based on optimal values
            for i in range(N):
                 E_loc = ENERGY_EFFICIENCY_COEFF * (f_opt[i]**2) * (task_sizes[i] * action_matrix[i, i + 1]) * CYCLES_PER_BIT
                 E_tx = P_opt[i] * t_tx_opt[i]
                 E_d2d_comp = 0
                 for j in range(N):
                     if i == j: continue
                     E_d2d_comp += ENERGY_EFFICIENCY_COEFF * (f_opt[j]**2) * (task_sizes[i] * action_matrix[i, j + 1]) * CYCLES_PER_BIT
                 final_total_E_task += (E_loc + E_tx + E_d2d_comp)

            # Check feasibility (constraints should be satisfied by solver)
            final_penalty = 0 # Should be near zero due to hard constraints

            metrics = {
                "success": True, "message": result.message,
                "objective": final_objective, "total_time": final_T_total, "total_energy": final_total_E_task,
                "penalty": final_penalty, # Should be negligible
                "a": a_opt, "cpu_freqs": f_opt, "tx_powers": P_opt,
                "aux_t_tx": t_tx_opt, "aux_T_exec": T_exec_opt,
                "T_harvest": a_opt, "T_offload": np.sum(t_tx_opt), "T_execute": T_exec_opt,
                "opt_duration_s": opt_duration
            }
        else:
            print(f"Warning: Resource allocation optimization failed! Message: {result.message}")
            # Fallback: Return results based on initial guess or indicate failure
            metrics = {
                "success": False, "message": result.message,
                "objective": np.inf, # Indicate failure
                # ... include other fields based on z0 if needed ...
                 "a": z0[0], "cpu_freqs": z0[1:1+N], "tx_powers": z0[1+N:1+2*N],
                "opt_duration_s": opt_duration
            }

        return metrics


# Example usage (within the file for direct testing)
if __name__ == '__main__':
    # Add SystemEnvironment if not imported/defined elsewhere
    class SystemEnvironment:
        def __init__(self):
            self.base_gains = np.random.rand(N_NODES, N_NODES) * 1e-5 # Dummy gains
            np.fill_diagonal(self.base_gains, 0)
            self.base_gains = (self.base_gains + self.base_gains.T)/2

        def get_new_state(self):
            # Simplified: just return base gains and random tasks for testing
            channel_matrix = self.base_gains * (0.5 + np.random.rand(N_NODES, N_NODES)) # Add some fading
            np.fill_diagonal(channel_matrix, 0)
            channel_matrix = (channel_matrix + channel_matrix.T)/2
            task_sizes = np.random.uniform(1e3, 5e3, N_WDs) # Example: 1-5 Mbits
            return channel_matrix, None, task_sizes # No flat channel needed here

    # Create dummy environment and state
    env = SystemEnvironment()
    chan_matrix, _, tasks = env.get_new_state()

    # Create the target Dummy Action
    dummy_action = np.array([
        [0.2, 0.3, 0.3, 0.2],  # WD1
        [0.3, 0.0, 0.4, 0.3],  # WD2
        [0.5, 0.0, 0.0, 0.5]   # WD3
    ])

    print("\n--- Testing Resource Allocator (Hard Constraints) ---")
    print(f"Target Dummy Action:\n{dummy_action}")
    print(f"Task sizes (Mbits): {tasks / 1e6}")
    print(f"Channel Matrix (x 1e7):\n{chan_matrix * 1e7}")


    # Instantiate and solve
    allocator = ResourceAllocator()
    results = allocator.solve(dummy_action, chan_matrix, tasks)

    print("\n--- Optimization Results ---")
    if results["success"]:
        print(f"Success: {results['success']} ({results['message']})")
        print(f"Optimization duration: {results['opt_duration_s']:.4f} s")
        print(f"Optimal Objective: {results['objective']:.6f}")
        print(f"  Total Time: {results['total_time']:.6f}")
        print(f"  Total Energy: {results['total_energy']:.6e}")
        print(f"Optimal 'a': {results['a']:.6f}")
        print(f"Optimal 'f' (GHz): {results['cpu_freqs'] / 1e9}")
        print(f"Optimal 'P' (mW): {results['tx_powers'] * 1000}")
        print(f"  T_harvest: {results['T_harvest']:.6f}")
        print(f"  T_offload: {results['T_offload']:.6f}")
        print(f"  T_execute: {results['T_execute']:.6f}")
        print(f"  Aux t_tx: {results['aux_t_tx']}")
        print(f"  Aux T_exec: {results['aux_T_exec']:.6f}")
    else:
        print(f"Optimization Failed: {results['message']}")