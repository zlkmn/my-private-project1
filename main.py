# DROO_D2D_PARTIAL_OFFLOADING/main.py

import time
import numpy as np
from tqdm import tqdm

# Import our custom modules
from config_test import *
from environment import SystemEnvironment
from resource_allocator_fixed_fi import ResourceAllocator
from agent import DRLAgent
import utils


def create_dnn_state(channel_flat: np.ndarray, task_sizes: np.ndarray) -> np.ndarray:
    """
    Normalizes and combines channel and task information into a state vector for the DNN.
    Normalization is crucial for stable DNN training.

    Args:
        channel_flat (np.ndarray): Flattened channel gains.
        task_sizes (np.ndarray): Task sizes in bits.

    Returns:
        np.ndarray: The normalized state vector.
    """
    # Normalize task sizes to be roughly in the [0, 1] range
    normalized_tasks = task_sizes / TASK_SIZE_MAX

    # Scale channel gains to a more suitable range for the DNN
    # (e.g., multiply by 1e7 or 1e8 to avoid very small numbers)
    scaled_channels = channel_flat * 1e5

    return np.concatenate([scaled_channels, normalized_tasks])


if __name__ == "__main__":
    # --- 1. Initialization ---
    print("--- Starting D2D Partial Offloading Simulation ---")
    start_time = time.time()

    # Initialize the main components
    env = SystemEnvironment()
    allocator = ResourceAllocator()
    agent = DRLAgent()

    # Initialize lists to store simulation results
    objective_history = []
    actions_history = []

    # --- 2. Main Simulation Loop ---
    print(f"\n--- Running Simulation for {NUM_TIME_FRAMES} Time Frames ---")
    for t in tqdm(range(NUM_TIME_FRAMES)):
        # a. Get the new state from the environment
        channel_matrix, channel_flat, task_sizes = env.get_new_state()
        # b. Create the normalized state for the DNN
        dnn_state = create_dnn_state(channel_flat, task_sizes)

        # c. The DRL agent chooses an offloading action
        # This action is a "soft" decision, representing probabilities or ratios
        # b. [修改] Agent 生成 K 个候选动作
        candidate_actions = agent.generate_candidate_actions(dnn_state)
        # d. The resource allocator (critic) evaluates this action
        # It solves the sub-problem to find the best resources and the resulting cost
        # c. [修改] Critic (Allocator) 评估所有 K 个动作
        results_list = []
        for action_k in candidate_actions:
            eval_k = allocator.solve(action_k, channel_matrix, task_sizes)
            results_list.append((eval_k, action_k))

        # d. [修改] 找出最好的动作
        best_objective = np.inf
        best_action = None
        for eval_results, action_k in results_list:
            if eval_results['success'] and eval_results['objective'] < best_objective:
                best_objective = eval_results['objective']
                best_action = action_k

        # e. The agent stores the experience (state, action) in its memory
        # The agent learns from the best actions found. In this simple case,
        # we store the action it took. In a more complex setup (like original DROO),
        # one would generate K candidates and store the best one.
        # e. [修改] 存储最好的 (state, action) 对
        if best_action is not None:
            # 我们找到了一个有效的最佳动作
            agent.store_experience(dnn_state, best_action)
            objective_history.append(best_objective)  # 记录最好的成本
            actions_history.append(best_action)
        else:
            # 如果求解器全部失败 (罕见), 存入第一个作为惩罚
            print(f"Frame {t}: All {K_CANDIDATES} candidates failed to solve. Skipping experience store.")
            #agent.store_experience(dnn_state, candidate_actions[0])
            objective_history.append(results_list[0][0].get('objective', np.inf))  # 记录失败
            actions_history.append(candidate_actions[0])

        # f. Periodically, the agent learns from its memory
        if (t + 1) % TRAINING_INTERVAL == 0:
            agent.learn()


    # --- 3. Post-Simulation Analysis ---
    end_time = time.time()
    total_duration = end_time - start_time
    print("\n--- Simulation Finished ---")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average time per frame: {total_duration / NUM_TIME_FRAMES:.4f} seconds")

    # Calculate and print the average objective of the last 20% of the simulation
    # This gives an idea of the converged performance
    converged_performance = np.mean(objective_history[-int(NUM_TIME_FRAMES * 0.2):])
    print(f"Average objective value (last 20%): {converged_performance:.4f}")

    # --- 4. Visualization and Saving ---
    # Plot the objective function history
    utils.plot_objective_history(objective_history)

    # Plot the agent's training loss
    utils.plot_training_loss(agent.training_loss_history)

    # Save key results to files
    results_to_save = {
        "objective_history": objective_history,
        "actions_history": actions_history,
        "training_loss": agent.training_loss_history
    }
    utils.save_results(results_to_save)