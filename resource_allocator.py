# DROO_D2D_PARTIAL_OFFLOADING/resource_allocator.py
#
# [全新实现]
# 使用 CVXPY 解决 README.md 中定义的凸资源分配子问题。
#

import numpy as np
import cvxpy as cp
import time
from config import * # Import all parameters from config.py

class ResourceAllocator:
    """
    使用 CVXPY 解决凸资源分配子问题。

    根据 README.md 中的新模型，给定一个固定的卸载矩阵 X，
    求解器找到最优的 (a, tau, e, f, T_ex) 以最小化加权和。
    """

    def __init__(self):
        """初始化求解器。"""
        # 预计算一些常量
        self.C_rate = (np.log(2) * V_U) / BANDWIDTH_HZ
        print(f"Convex ResourceAllocator (CVXPY) initialized for {N_WDs} WDs.")

    def _calculate_loads(self, action_matrix: np.ndarray, task_sizes: np.ndarray) -> np.ndarray:
        """
        计算每个 WD j 的总计算负载 L_j。
        L_j = (WD j 的本地任务) + (所有其他 WD i 卸载给 WD j 的任务)
        """
        N = N_WDs
        loads = np.zeros(N)
        for j in range(N):  # 目标 WD j (节点索引 j+1)
            for i in range(N):  # 源 WD i (任务 c_i)
                # action_matrix[i, j+1] 是 x_{i+1, j+1}
                loads[j] += task_sizes[i] * action_matrix[i, j + 1]
        return loads

    def solve(self, action_matrix: np.ndarray, channel_matrix: np.ndarray, task_sizes: np.ndarray) -> dict:
        """
        使用 CVXPY 求解凸子问题。

        Args:
            action_matrix (np.ndarray): 卸载决策 X, shape (N_WDs, N_NODES)
            channel_matrix (np.ndarray): 信道增益 H, shape (N_NODES, N_NODES)
            task_sizes (np.ndarray): 任务大小 c, shape (N_WDs,)

        Returns:
            dict: 包含求解结果
        """
        N = N_WDs
        N_nodes = N_NODES
        start_opt_time = time.time()

        try:
            # --- 1. 预计算常量 ---
            # L_j: 每个 WD (j=0..N-1, 对应节点 1..N) 的总计算负载 (bits)
            L_j = self._calculate_loads(action_matrix, task_sizes)

            # s_ij: 从 WD i (节点 i+1) 发送到节点 j 的比特数
            s_ij = np.zeros((N, N_nodes))
            for i in range(N):
                for j in range(N_nodes):
                    if i + 1 == j:  # 跳过到自己的"传输"
                        continue
                    s_ij[i, j] = task_sizes[i] * action_matrix[i, j]

            # h_ij: 从 WD i (节点 i+1) 到节点 j 的信道增益
            h_ij = channel_matrix[1:, :]

            # gamma_ij = h_ij / N0
            gamma_ij = h_ij / NOISE_POWER

            # E_harv_j (的系数): WD j (节点 j+1) 从 AP (节点 0) 收集的能量
            # E_harv_j = (系数) * a
            E_harv_coeff_j = (ENERGY_HARVEST_EFFICIENCY * AP_TX_POWER_W *
                              ENERGY_HARVEST_SCALE_S * channel_matrix[0, 1:])

            # --- 2. 定义 CVXPY 变量 ---
            # 标量
            a = cp.Variable(pos=True, name="a")
            T_ex = cp.Variable(pos=True, name="T_ex")
            # 向量 (N_WDs)
            f = cp.Variable(N, pos=True, name="f")
            # 矩阵 (N_WDs, N_NODES)
            tau = cp.Variable((N, N_nodes), pos=True, name="tau")
            e = cp.Variable((N, N_nodes), pos=True, name="e")

            # --- 3. 定义目标函数 ---
            # T_total = a + T_offload + T_execute
            T_offload = cp.sum(tau)
            T_total = a + T_offload + T_ex

            # E_total = E_tx + E_comp
            E_tx = cp.sum(e)
            E_comp = cp.sum(CYCLES_PER_BIT * ENERGY_EFFICIENCY_COEFF * cp.multiply(L_j, cp.power(f, 2)))
            E_total = E_tx + E_comp

            objective = cp.Minimize(TIME_WEIGHT * T_total + ENERGY_WEIGHT * E_total)

            # --- 4. 定义约束 ---
            constraints = []

            # 变量界限
            constraints += [a <= 1.0]
            constraints += [f >= CPU_FREQ_MIN_HZ, f <= CPU_FREQ_MAX_HZ]

            # 功率限制 (P_min <= e/tau <= P_max)
            constraints += [e >= TX_POWER_MIN_W * tau]
            constraints += [e <= TX_POWER_MAX_W * tau]

            # 执行时间约束 (T_ex >= L_j * phi / f_j)
            constraints += [T_ex >= cp.multiply(L_j * CYCLES_PER_BIT, cp.inv_pos(f))]

            # 能量中性约束 (E_self_j <= E_harv_j)
            E_tx_j = cp.sum(e, axis=1)  # WD j (i=0..N-1) 的总传输能耗
            E_comp_j = CYCLES_PER_BIT * ENERGY_EFFICIENCY_COEFF * cp.multiply(L_j, cp.power(f, 2))
            E_self_j = E_tx_j + E_comp_j
            E_harv_j = E_harv_coeff_j * a
            constraints += [E_self_j <= E_harv_j]

            # 速率约束 (s_ij <= C * tau * log2(1 + gamma * e / tau))
            # 使用 -rel_entr(tau, tau + gamma*e) 来表示 tau * log( (tau + gamma*e) / tau )
            for i in range(N):
                for j in range(N_nodes):
                    if i + 1 == j:
                        # 约束自传输为 0
                        constraints += [tau[i, j] == 0, e[i, j] == 0]
                        continue
                    
                    if s_ij[i, j] < EPS:
                        # 如果没有数据要发送，则不分配时间和能量
                        constraints += [tau[i, j] == 0, e[i, j] == 0]
                        continue

                    if gamma_ij[i, j] < EPS:
                        # 如果信道增益为0但仍有数据要发送，则问题不可行
                        # 将 tau 设置为非常大，以使其成为一个糟糕的选择
                        constraints += [tau[i, j] >= 1e9] 
                        continue

                    # LHS = s_ij * log(2) * v_u / B
                    LHS = s_ij[i, j] * self.C_rate
                    # RHS = tau * log(1 + gamma * e / tau)
                    RHS = -cp.rel_entr(tau[i, j], tau[i, j] + gamma_ij[i, j] * e[i, j])
                    constraints += [LHS <= RHS]

            # --- 5. 求解问题 ---
            problem = cp.Problem(objective, constraints)
            # 使用 ECOS 或 SCS，它们是为这类问题设计的
            problem.solve(solver=cp.ECOS, verbose=True) 

            opt_duration = time.time() - start_opt_time

            if problem.status in ["optimal", "optimal_inaccurate"]:
                obj_val = problem.value
                t_total_val = (a.value + np.sum(tau.value) + T_ex.value)
                e_total_val = (np.sum(e.value) + np.sum(CYCLES_PER_BIT * ENERGY_EFFICIENCY_COEFF * L_j * (f.value**2)))

                metrics = {
                    "success": True,
                    "message": problem.status,
                    "objective": obj_val,
                    "total_time": t_total_val,
                    "total_energy": e_total_val,
                    "a": a.value,
                    "cpu_freqs": f.value,
                    "T_harvest": a.value,
                    "T_offload": np.sum(tau.value),
                    "T_execute": T_ex.value,
                    "opt_duration_s": opt_duration
                }
            else:
                # 求解失败
                metrics = {
                    "success": False,
                    "message": problem.status,
                    "objective": np.inf,
                    "opt_duration_s": opt_duration
                }
        
        except Exception as e:
            # CVXPY 求解器可能因为数值问题彻底失败
            print(f"Warning: CVXPY solver failed with exception: {e}")
            metrics = {
                "success": False,
                "message": str(e),
                "objective": np.inf,
                "opt_duration_s": time.time() - start_opt_time
            }

        return metrics

# Example usage (within the file for direct testing)
if __name__ == '__main__':
    # 使用来自 Resource_allocator_test.py 的相同测试用例
    chan_matrix = np.array([[0.00000000e+00, 6.62546102e-06, 2.31014624e-06, 1.83911263e-06],
                            [6.62546102e-06, 0.00000000e+00, 4.80614196e-06, 1.01083395e-06],
                            [2.31014624e-06, 4.80614196e-06, 0.00000000e+00, 3.81168776e-06],
                            [1.83911263e-06, 1.01083395e-06, 3.81168776e-06, 0.00000000e+00]])
    tasks = np.array([461705.08926378, 970766.77384453, 123409.47028163])
    dummy_action = np.array([[0.17732602, 0.11025, 0.28628772, 0.42613626],
                             [0.358793, 0.21895458, 0.21797714, 0.20427531],
                             [0.10907146, 0.13038586, 0.6538224, 0.10672026]])

    print("\n--- Testing Resource Allocator (CVXPY Convex Solver) ---")
    print(f"Target Dummy Action:\n{dummy_action}")
    print(f"Task sizes (Mbits): {tasks / 1e6}")

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
        print(f"Optimal 'f' (MHz): {results['cpu_freqs'] / 1e6}")
        print(f"  T_harvest: {results['T_harvest']:.6f}")
        print(f"  T_offload: {results['T_offload']:.6f}")
        print(f"  T_execute: {results['T_execute']:.6f}")
    else:
        print(f"Optimization Failed: {results['message']}")

