# DROO_D2D_PARTIAL_OFFLOADING/environment.py

import numpy as np
from config_test import *  # Import all parameters from config.py


class SystemEnvironment:
    """
    Manages the state of the wireless environment.

    This class is responsible for generating the dynamic aspects of the system
    at each time frame, specifically:
    1. The wireless channel gains between all nodes (AP and WDs).
    2. The computation task sizes for each WD.
    """

    def __init__(self):
        """
        Initializes the environment by pre-calculating distance-based path loss.
        """
        print("Initializing System Environment...")
        self._initialize_positions()
        self._calculate_base_path_loss()
        print("Environment initialized.")

    def _initialize_positions(self):
        """
        Randomly places WDs in a 2D space around the AP at the origin.
        """
        # Note: In the final model, we use random distances, but generating
        # positions is a good practice for potential future visualization.
        ap_pos = np.array([AP_POSITION])
        wd_positions = []
        while len(wd_positions) < N_WDs:
            # Generate a new position and ensure it's not too close to the AP
            new_pos = np.random.uniform(-3, 3, 2)
            if np.linalg.norm(new_pos - ap_pos[0]) >= 0.5:  # Min distance from AP
                wd_positions.append(new_pos)

        self.positions = np.vstack([ap_pos, np.array(wd_positions)])

    def _calculate_base_path_loss(self):
        """
        Calculates the large-scale, distance-dependent path loss between all nodes.
        This is calculated once and combined with small-scale fading later.
        """
        self.base_gains = np.zeros((N_NODES, N_NODES))
        wavelength = LIGHT_SPEED / CARRIER_FREQ_HZ
        #固定距离矩阵
        distance_matrix = np.random.uniform(MIN_DISTANCE_M, MAX_DISTANCE_M,(N_NODES, N_NODES))
        for i in range(N_NODES):
            for j in range(i + 1, N_NODES):
                # We use random distances as in the user's original code for consistency
                #distance = np.random.uniform(MIN_DISTANCE_M, MAX_DISTANCE_M)

                # Friis free-space path loss formula
                path_loss = (wavelength / (4 * np.pi * distance_matrix[i,j])) ** PATH_LOSS_EXPONENT
                gain = ANTENNA_GAIN * path_loss

                self.base_gains[i, j] = gain
                self.base_gains[j, i] = gain  # Symmetric channel

    # def get_new_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Generates and returns a new state for the current time frame.
    #     [已修改] 实现莱斯衰落以降低信道差异。
    #     """
    #     # --- 修改开始 ---
    #     # 1. 生成小尺度衰落 (莱斯衰落)
    #     # 莱斯衰落 = 视线(LoS)分量 + 非视线(NLoS)散射分量 (瑞利)
    #
    #     # NLoS (瑞利) 分量
    #     real_part = np.random.normal(0, 1 / np.sqrt(2), (N_NODES, N_NODES))
    #     imag_part = np.random.normal(0, 1 / np.sqrt(2), (N_NODES, N_NODES))
    #
    #     # LoS 分量的功率由K因子决定，散射分量的总功率归一化为1
    #     # LoS 分量的幅值
    #     los_amplitude = np.sqrt(RICIAN_K_FACTOR / (RICIAN_K_FACTOR + 1))
    #     # NLoS 分量的缩放因子
    #     nlos_scale = np.sqrt(1 / (RICIAN_K_FACTOR + 1))
    #
    #     # 合成复数衰落
    #     fading_complex = (los_amplitude + nlos_scale * real_part) + 1j * (nlos_scale * imag_part)
    #     fading_matrix = np.abs(fading_complex)
    #     # --- 修改结束 ---
    #
    #     # 确保衰落是对称的
    #     fading_symmetric = (fading_matrix + fading_matrix.T) / 2
    #     np.fill_diagonal(fading_symmetric, 0)  # No self-fading
    #
    #     # 结合基础路径损耗得到当前信道条件
    #     channel_matrix = self.base_gains * fading_symmetric
    #
    #     # 2. 生成任务大小 (不变)
    #     task_sizes = np.random.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX, N_WDs)
    #
    #     # 3. 获取展平的上三角矩阵 (不变)
    #     upper_indices = np.triu_indices(N_NODES, k=1)
    #     channel_flat = channel_matrix[upper_indices]
    #
    #     return channel_matrix, channel_flat, task_sizes

    def get_new_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates and returns a new state for the current time frame.

        A state consists of:
        1. Channel gains: Small-scale Rayleigh fading applied to base path loss.
        2. Task sizes: Randomly generated task size for each WD.

        Returns:
            A tuple containing:
            - channel_matrix (np.ndarray): A symmetric [N_NODES, N_NODES] matrix of channel gains.
            - channel_flat (np.ndarray): A flattened array of the upper triangle of the channel matrix.
            - task_sizes (np.ndarray): An array of task sizes for each WD.
        """
        # 1. Generate small-scale Rayleigh fading
        # real_part^2 + imag_part^2 follows an exponential distribution,
        # its square root is a Rayleigh distribution.
        real_part = np.random.normal(0, 1 / np.sqrt(2), (N_NODES, N_NODES))
        imag_part = np.random.normal(0, 1 / np.sqrt(2), (N_NODES, N_NODES))
        fading_matrix = np.sqrt(real_part ** 2 + imag_part ** 2)

        # Ensure fading is symmetric
        fading_symmetric = (fading_matrix + fading_matrix.T) / 2
        np.fill_diagonal(fading_symmetric, 0)  # No self-fading

        # Combine with base path loss to get current channel conditions
        channel_matrix = self.base_gains * fading_symmetric

        # 2. Generate task sizes for each WD
        task_sizes = np.random.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX, N_WDs)

        # 3. Get the flattened upper triangle for the DNN input
        upper_indices = np.triu_indices(N_NODES, k=1)
        channel_flat = channel_matrix[upper_indices]

        return channel_matrix, channel_flat, task_sizes


# Example usage:
if __name__ == '__main__':
    env = SystemEnvironment()
    chan_matrix, chan_flat, tasks = env.get_new_state()

    print("\n--- Example State ---")
    print(f"Number of WDs: {N_WDs}")
    print(f"Task sizes (Mbits): \n{tasks / 1e6}")
    print(f"\nFull Channel Matrix (h_ij): \n{chan_matrix}")
    print(f"\nFlattened Channels for DNN (shape: {chan_flat.shape}): \n{chan_flat}")