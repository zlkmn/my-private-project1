
# resource_allocator_fixed.py
# Drop-in replacement with soft slacks (zeta, xi_E), no min-power lower bound,
# and a feasibility report. Designed to eliminate "infeasible" returns for SCS
# unless there is a pure modeling bug (e.g., shape mismatch).
#
# Usage in your project:
#   from resource_allocator_fixed import ResourceAllocator
#   allocator = ResourceAllocator()
#   results = allocator.solve(X, H, c)

import numpy as np
import cvxpy as cp
import time

# Pull constants from your config; if missing during standalone tests, provide sane defaults.
try:
    from config_test import *
except Exception:
    # -------- Defaults for standalone quick tests (units consistent with your logs) --------
    N_WDs = 3
    N_NODES = 4  # AP (0) + 3 WDs
    BANDWIDTH_HZ = 1e6
    NOISE_POWER = 1e-13
    V_U = 1.0
    ENERGY_HARVEST_EFFICIENCY = 0.8
    AP_TX_POWER_W = 1.0
    ENERGY_HARVEST_SCALE_S = 1.0
    CPU_FREQ_MIN_HZ = 5e8
    CPU_FREQ_MAX_HZ = 2.0e9
    CYCLES_PER_BIT = 1000.0
    ENERGY_EFFICIENCY_COEFF = 1e-28
    TX_POWER_MAX_W = 0.2
    TIME_WEIGHT = 1.0
    ENERGY_WEIGHT = 1.0
    EPS = 1e-12

class ResourceAllocator:
    """
    CVXPY-based convex solver for the per-frame resource allocation subproblem
    given a fixed offloading matrix X.
    """

    def __init__(self):
        self.C_rate = (np.log(2.0) * V_U) / BANDWIDTH_HZ  # factor to move bits to "tau*log(1+...)" domain
        print(f"[RA] CVXPY allocator ready for {N_WDs} WDs. (rel_entr-based rate, slacks ON)")

    @staticmethod
    def _calc_loads(action_matrix: np.ndarray, task_sizes: np.ndarray) -> np.ndarray:
        """
        L_j = sum_i c_i * x_{i,j} for WD j (j=1..N). AP does not compute in this model.
        action_matrix shape: (N, N+1) with columns [AP, WD1, WD2, ... WD_N]
        """
        N = action_matrix.shape[0]
        loads = np.zeros(N)
        for j in range(N):        # target WD j (node index j+1)
            for i in range(N):    # source WD i
                loads[j] += task_sizes[i] * action_matrix[i, j + 1]
        return loads  # bits

    def _print_feasibility_report(self, X, H, c):
        """
        Cheap pre-check: energy lower bounds that do NOT depend on tau.
        E_tx_min(i) = sum_j s_ij * (ln2*V_U/B) / gamma_ij   (from tau*log(1+gamma*e/tau) <= gamma*e)
        E_comp_min(j) at f_min.
        """
        N = X.shape[0]
        N_nodes = X.shape[1]
        s_ij = np.zeros((N, N_nodes))
        for i in range(N):
            for j in range(N_nodes):
                if j == i + 1:
                    continue
                s_ij[i, j] = c[i] * X[i, j]

        h_ij = H[1:, :]  # i in 1..N
        gamma_ij = h_ij / NOISE_POWER
        L = self._calc_loads(X, c)

        Etx_min = np.zeros(N)
        C = self.C_rate  # ln2*V_U/B
        for i in range(N):
            for j in range(N_nodes):
                if j == i + 1:
                    continue
                if s_ij[i, j] <= 0:
                    continue
                gij = gamma_ij[i, j]
                if gij <= 0:
                    Etx_min[i] = np.inf
                else:
                    Etx_min[i] += s_ij[i, j] * C / gij

        # comp lower bound at f_min
        Ecomp_min = CYCLES_PER_BIT * ENERGY_EFFICIENCY_COEFF * L * (CPU_FREQ_MIN_HZ ** 2)

        Eharv_coeff = ENERGY_HARVEST_EFFICIENCY * AP_TX_POWER_W * ENERGY_HARVEST_SCALE_S * H[0, 1:]

        print("\n[RA] Feasibility lower bounds (per WD):")
        for j in range(N):
            need = Etx_min[j] + Ecomp_min[j]
            cap_per_a = Eharv_coeff[j]
            a_req = need / cap_per_a if cap_per_a > 0 else np.inf
            print(f"  WD{j+1}: E_tx_min={Etx_min[j]:.3e} J, E_comp_min={Ecomp_min[j]:.3e} J, "
                  f"E_harv_coeff={cap_per_a:.3e} J/a  ->  required a >= {a_req:.3f}")

    def solve(self, action_matrix: np.ndarray, channel_matrix: np.ndarray, task_sizes: np.ndarray):
        N = N_WDs
        N_nodes = N_NODES

        t0 = time.time()
        # Optional: print quick feasibility hints
        try:
            self._print_feasibility_report(action_matrix, channel_matrix, task_sizes)
        except Exception as _:
            pass

        # Precompute
        L_j = self._calc_loads(action_matrix, task_sizes)  # length N (WDs only)
        s_ij = np.zeros((N, N_nodes))
        for i in range(N):
            for j in range(N_nodes):
                if j == i + 1:
                    continue
                s_ij[i, j] = task_sizes[i] * action_matrix[i, j]
        h_ij = channel_matrix[1:, :]
        gamma_ij = h_ij / NOISE_POWER
        E_harv_coeff_j = (ENERGY_HARVEST_EFFICIENCY * AP_TX_POWER_W *
                          ENERGY_HARVEST_SCALE_S * channel_matrix[0, 1:])  # length N

        # Variables
        a = cp.Variable(nonneg=True, name="a")
        T_ex = cp.Variable(nonneg=True, name="T_ex")
        f = cp.Variable(N, nonneg=True, name="f")
        tau = cp.Variable((N, N_nodes), nonneg=True, name="tau")
        e = cp.Variable((N, N_nodes), nonneg=True, name="e")
        # soft slacks
        zeta = cp.Variable((N, N_nodes), nonneg=True, name="zeta")  # rate slack
        xi_E = cp.Variable(N, nonneg=True, name="xi_E")             # energy slack

        # Objective
        T_offload = cp.sum(tau)
        T_total = a + T_offload + T_ex
        E_tx = cp.sum(e)
        E_comp = cp.sum(CYCLES_PER_BIT * ENERGY_EFFICIENCY_COEFF * cp.multiply(L_j, cp.power(f, 2)))
        R_RATE = 1e4
        R_ENER = 1e4
        obj = cp.Minimize(TIME_WEIGHT * T_total +
                          ENERGY_WEIGHT * (E_tx + E_comp) +
                          R_RATE * cp.sum(zeta) + R_ENER * cp.sum(xi_E))

        # Constraints
        cons = []
        TAU_FLOOR = 1e-9
        cons += [a >= TAU_FLOOR, a <= 1.0]
        cons += [f >= CPU_FREQ_MIN_HZ, f <= CPU_FREQ_MAX_HZ]

        # power box: ONLY upper bound. min-power lower bound is removed to avoid infeasibilities.
        cons += [e <= TX_POWER_MAX_W * tau]

        # execute time constraints (vectorized): T_ex >= L_j * phi / f_j
        # Note: inv_pos(f) is convex and imposes f > 0 internally; bounds above also ensure positivity.
        cons += [T_ex >= cp.multiply(L_j * CYCLES_PER_BIT, cp.inv_pos(f))]

        # energy neutrality (soft): sum_k e_{j,k} + kappa f_j^2 L_j phi <= coeff_j * a + xi_E_j
        E_tx_j = cp.sum(e, axis=1)
        E_comp_j = CYCLES_PER_BIT * ENERGY_EFFICIENCY_COEFF * cp.multiply(L_j, cp.power(f, 2))
        cons += [E_tx_j + E_comp_j <= E_harv_coeff_j * a + xi_E]

        # rate constraints via perspective/relative-entropy, with soft slack zeta
        for i in range(N):
            for j in range(N_nodes):
                if j == i + 1:
                    # No self-transmission
                    cons += [tau[i, j] == 0, e[i, j] == 0, zeta[i, j] == 0]
                    continue
                if s_ij[i, j] <= 0:
                    cons += [tau[i, j] == 0, e[i, j] == 0, zeta[i, j] == 0]
                    continue
                cons += [tau[i, j] >= TAU_FLOOR]
                LHS = s_ij[i, j] * self.C_rate
                RHS = -cp.rel_entr(tau[i, j], tau[i, j] + gamma_ij[i, j] * e[i, j])  # natural log
                cons += [LHS <= RHS + zeta[i, j]]

        prob = cp.Problem(cp.Minimize(obj), cons)

        try:
            prob.solve(solver=cp.SCS, eps=1e-5, max_iters=20000, warm_start=True, verbose=False)
            status = prob.status
        except Exception as ex:
            return {
                "success": False,
                "message": f"solver exception: {ex}",
                "objective": np.inf,
                "opt_duration_s": time.time() - t0
            }

        print(f"[RA] status={status}, solver=SCS, obj={prob.value if prob.value is not None else None}")
        dt = time.time() - t0

        if status in ("optimal", "optimal_inaccurate"):
            return {
                "success": True,
                "message": status,
                "objective": prob.value,
                "total_time": (a.value + float(np.sum(tau.value)) + T_ex.value),
                "total_energy": (float(np.sum(e.value)) +
                                 float(np.sum(CYCLES_PER_BIT * ENERGY_EFFICIENCY_COEFF * L_j * (f.value ** 2)))),
                "a": float(a.value),
                "cpu_freqs": f.value.copy(),
                "T_harvest": float(a.value),
                "T_offload": float(np.sum(tau.value)),
                "T_execute": float(T_ex.value),
                "sum_rate_slack": float(np.sum(zeta.value)),
                "sum_energy_slack": float(np.sum(xi_E.value)),
                "opt_duration_s": dt
            }
        else:
            # With slacks, infeasible should not happen unless model bugs (e.g., NaNs) appear.
            return {
                "success": False,
                "message": status,
                "objective": np.inf,
                "opt_duration_s": dt
            }


if __name__ == "__main__":
    # Minimal self-test using your numbers (can be removed in your repo).
    H = np.array([[0.0, 6.62546102e-06, 2.31014624e-06, 1.83911263e-06],
                  [6.62546102e-06, 0.0, 4.80614196e-06, 1.01083395e-06],
                  [2.31014624e-06, 4.80614196e-06, 0.0, 3.81168776e-06],
                  [1.83911263e-06, 1.01083395e-06, 3.81168776e-06, 0.0]])
    c = np.array([46170.08926378, 97076.77384453, 12340.47028163])  # bits
    X = np.array([[0.17732602, 0.11025,    0.28628772, 0.42613626],
                  [0.358793,   0.21895458, 0.21797714, 0.20427531],
                  [0.10907146, 0.13038586, 0.6538224,  0.10672026]])

    ra = ResourceAllocator()
    res = ra.solve(X, H, c)
    print("\n--- Optimization Results ---")
    print(res)
