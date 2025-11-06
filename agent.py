# DROO_D2D_PARTIAL_OFFLOADING/agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import * # [修改] 导入 config.py


class GroupedSoftmax(nn.Module):
    """
    Custom activation function to apply softmax independently to each WD's action vector.
    This ensures that for each WD, the sum of its offloading ratios equals 1.
    """

    def __init__(self):
        super(GroupedSoftmax, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The output from the last linear layer, with shape
                              (batch_size, N_WDs * N_NODES).

        Returns:
            torch.Tensor: The activated output, with the same shape.
        """
        # Reshape to (batch_size, N_WDs, N_NODES) to group actions by WD
        x_reshaped = x.view(-1, N_WDs, N_NODES)

        # Apply softmax along the last dimension (the nodes dimension)
        softmax_output = torch.softmax(x_reshaped, dim=-1)

        # Reshape back to the original flat shape (batch_size, N_WDs * N_NODES)
        return softmax_output.view(-1, OUTPUT_DIM)


class DRLAgent:
    """
    The Deep Reinforcement Learning Agent (Actor).

    This class encapsulates the DNN, the experience replay memory, and the
    learning logic. It learns to map a system state to an effective
    partial offloading decision.
    """

    def __init__(self):
        """Initializes the agent's components."""
        print("Initializing DRL Agent...")

        # Build the neural network
        self._build_net()

        # Setup the optimizer
        self.optimizer = optim.Adam(self.model_base.parameters(), lr=LEARNING_RATE)

        # Initialize the experience replay memory
        # Memory stores [state, action], so width is INPUT_DIM + OUTPUT_DIM
        self.memory = np.zeros((MEMORY_SIZE, INPUT_DIM + OUTPUT_DIM))
        self.memory_counter = 0

        # For tracking training performance
        self.training_loss_history = []
        print("Agent initialized.")

    def _build_net(self):
        """
        [修改] 将模型拆分为 "base" (logits) 和 "activation" (softmax)
        """
        layers = [
            nn.Linear(NET_TOPOLOGY[0], NET_TOPOLOGY[1]),
            nn.ReLU(),
            nn.Linear(NET_TOPOLOGY[1], NET_TOPOLOGY[2]),
            nn.ReLU(),
            nn.Linear(NET_TOPOLOGY[2], NET_TOPOLOGY[3])  # <-- 停止在 logits 层
        ]
        self.model_base = nn.Sequential(*layers)
        self.activation = GroupedSoftmax()

    def choose_action(self, state: np.ndarray) -> np.ndarray:
        """
        [修改] 此函数现在用于 "exploitation" (利用)
        它返回一个确定性的最佳动作 (无噪声)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # [修改] 使用新的模型结构
        self.model_base.eval()
        with torch.no_grad():
            logits = self.model_base(state_tensor)  # 1. 获取 logits
            action_probs_flat = self.activation(logits)  # 2. 应用激活

        action_matrix = action_probs_flat.squeeze(0).numpy().reshape(N_WDs, N_NODES)
        return action_matrix

    def generate_candidate_actions(self, state: np.ndarray) -> list:
        """
        [新增] DROO K-Candidate 生成器
        为 "exploration" (探索) 生成 K 个带噪声的候选动作
        """
        self.model_base.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 1. 获取基础 logits
        with torch.no_grad():
            base_logits = self.model_base(state_tensor)  # Shape: [1, OUTPUT_DIM]

        candidate_actions = []
        for _ in range(K_CANDIDATES):
            # 2. 添加高斯噪声
            noise = torch.randn_like(base_logits) * EXPLORATION_NOISE
            noisy_logits = base_logits + noise

            # 3. 通过激活函数
            with torch.no_grad():
                action_probs_flat = self.activation(noisy_logits)

            # 4. 转换并存储
            action_matrix = action_probs_flat.squeeze(0).numpy().reshape(N_WDs, N_NODES)
            candidate_actions.append(action_matrix)

        return candidate_actions

    def store_experience(self, state: np.ndarray, action: np.ndarray):
        """
        Stores a (state, action) pair in the experience replay memory.
        """
        # Flatten the action matrix for storage
        action_flat = action.flatten()

        transition = np.hstack((state, action_flat))

        # Use a circular buffer index
        index = self.memory_counter % MEMORY_SIZE
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        """
        Trains the neural network using a batch of experiences from the memory.
        """
        if self.memory_counter < BATCH_SIZE:
            return

        # [修改] 使用新的模型结构
        self.model_base.train()

        # 采样逻辑保持不变
        if self.memory_counter > MEMORY_SIZE:
            sample_indices = np.random.choice(MEMORY_SIZE, size=BATCH_SIZE)
        else:
            sample_indices = np.random.choice(self.memory_counter, size=BATCH_SIZE)
        batch_memory = self.memory[sample_indices, :]
        batch_states = torch.FloatTensor(batch_memory[:, :INPUT_DIM])
        batch_target_actions = torch.FloatTensor(batch_memory[:, INPUT_DIM:])

        # --- [修改] The Training Step ---
        # 1. 获取 logits
        batch_logits = self.model_base(batch_states)
        # 2. 获取预测的动作
        batch_predicted_actions = self.activation(batch_logits)

        # 3. 计算损失 (行为克隆: 最小化与"最佳动作"的交叉熵)
        loss = -torch.mean(torch.sum(batch_target_actions * torch.log(batch_predicted_actions + 1e-10), dim=1))

        # 4. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_loss_history.append(loss.item())

# Example usage:
if __name__ == '__main__':
    print("\n--- Testing DRL Agent (Enhanced Test) ---")

    # 辅助函数
    def create_dnn_state(channel_flat: np.ndarray, task_sizes: np.ndarray) -> np.ndarray:
        normalized_tasks = task_sizes / TASK_SIZE_MAX
        scaled_channels = channel_flat * 1e6
        return np.concatenate([scaled_channels, normalized_tasks])

    # 1. 实例化 Agent
    agent = DRLAgent()

    # 2. 生成一个真实的模拟状态
    print(f"\nGenerating realistic state for {N_WDs} WDs...")
    raw_channel_flat = np.random.rand(NUM_CHANNELS_FLAT) * 5e-7
    raw_task_sizes = np.random.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX, N_WDs)
    dnn_state = create_dnn_state(raw_channel_flat, raw_task_sizes)
    print(f"  Normalized DNN State (shape: {dnn_state.shape}): {np.round(dnn_state, 2)}")

    # 3. 测试 'choose_action' (无噪声的“利用”动作)
    print("\n--- 1. Testing choose_action (Exploitation) ---")
    exploit_action = agent.choose_action(dnn_state)
    print(f"  Deterministic Action (shape: {exploit_action.shape}):\n{np.round(exploit_action, 2)}")
    print(f"  Sum of ratios (should be 1.0): {np.sum(exploit_action, axis=1)}")

    # 4. 测试 'generate_candidate_actions' (带噪声的“探索”动作)
    print(f"\n--- 2. Testing generate_candidate_actions (Exploration, K={K_CANDIDATES}) ---")
    candidate_actions = agent.generate_candidate_actions(dnn_state)

    print(f"  Generated {len(candidate_actions)} candidate actions.")
    difference = np.sum(np.abs(candidate_actions[0] - candidate_actions[1]))
    print(f"  Sum of difference between Action 0 and 1: {difference:.4f}")
    if difference < 1e-3:
        print("  WARNING: Actions are too similar! Check EXPLORATION_NOISE in config.py.")
    else:
        print("  SUCCESS: Actions are different (noise was applied correctly).")
