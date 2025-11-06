# DROO_D2D_PARTIAL_OFFLOADING/utils.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import TRAINING_INTERVAL


def plot_objective_history(objective_his: list, rolling_window: int = 100):
    """
    Plots the history of the objective function value with a rolling average.

    Args:
        objective_his (list): A list of objective values from the simulation.
        rolling_window (int): The window size for the rolling average.
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 8))

    df = pd.DataFrame(objective_his)
    rolling_mean = df.rolling(window=rolling_window, min_periods=1).mean()

    ax.plot(df.index, df.values, 'c-', alpha=0.3, label='Raw Objective')
    ax.plot(rolling_mean.index, rolling_mean.values, 'b-', label=f'Rolling Mean (window={rolling_window})')

    ax.set_title('Objective Function Value Over Time', fontsize=16)
    ax.set_xlabel('Time Frame', fontsize=12)
    ax.set_ylabel('Objective Value (Weighted Cost)', fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_training_loss(loss_his: list):
    """
    Plots the training loss of the DRL agent.

    Args:
        loss_his (list): A list of loss values from agent training.
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 8))

    # The x-axis should correspond to the time frames when training occurred
    training_steps = np.arange(len(loss_his)) * TRAINING_INTERVAL

    ax.plot(training_steps, loss_his, 'r-')

    ax.set_title('DRL Agent Training Loss', fontsize=16)
    ax.set_xlabel('Time Frame', fontsize=12)
    ax.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
    ax.grid(True)
    plt.show()


def save_results(data_dict: dict, folder: str = "results"):
    """
    Saves the simulation results to text files in a specified folder.

    Args:
        data_dict (dict): A dictionary where keys are filenames and values are the data to save.
        folder (str): The directory to save the files in.
    """
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

    for filename, data in data_dict.items():
        filepath = os.path.join(folder, f"{filename}.txt")
        # Handle different data structures (e.g., nested numpy arrays)
        with open(filepath, 'w') as f:
            for item in data:
                if isinstance(item, np.ndarray):
                    # Flatten arrays for saving
                    f.write(' '.join(map(str, item.flatten())) + '\n')
                else:
                    f.write(f"{item}\n")
    print(f"Results saved to '{folder}/' directory.")