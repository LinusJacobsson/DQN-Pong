import matplotlib.pyplot as plt
import seaborn as sns

def parse_rewards_from_log(log_file: str) -> list[float]:
    """
    Parse total rewards for each episode from the log file.

    Args:
        log_file: Path to the log file.

    Returns:
        A list of total rewards for each training episode.
    """
    rewards = []
    with open(log_file, 'r') as file:
        for line in file:
            if "Episode" in line and "Total reward" in line:
                # Extract the reward value
                reward = float(line.split("Total reward:")[-1].strip())
                rewards.append(reward)
    return rewards

def parse_evaluation_rewards(log_file: str) -> list[float]:
    """
    Parse average evaluation rewards from the log file.

    Args:
        log_file: Path to the log file.

    Returns:
        A list of average rewards during evaluation.
    """
    eval_rewards = []
    with open(log_file, 'r') as file:
        for line in file:
            if "Evaluation after" in line and "Average reward:" in line:
                # Extract the average reward value
                reward = float(line.split("Average reward:")[-1].strip())
                eval_rewards.append(reward)
    return eval_rewards

def plot_combined_rewards(
    training_rewards: list[float],
    eval_rewards: list[float],
    evaluation_freq: int,
    save_path: str = None
) -> None:
    """
    Plot both training rewards and average evaluation rewards in the same figure.

    Args:
        training_rewards: List of total rewards for each training episode.
        eval_rewards: List of average evaluation rewards.
        evaluation_freq: Frequency of evaluations during training.
        save_path: Path to save the plot. If None, the plot is displayed.
    """
    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
    plt.figure(figsize=(12, 6))

    # Plot training rewards
    sns.lineplot(x=range(len(training_rewards)), y=training_rewards, label="Training Reward", color="lightblue", linewidth=1.5)

    # Plot evaluation rewards
    eval_episodes = [i * evaluation_freq for i in range(len(eval_rewards))]
    sns.lineplot(x=eval_episodes, y=eval_rewards, label="Evaluation Reward", color="orange", linewidth=2)

    # Set labels and title
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.title("Training and Evaluation Rewards vs. Episode", fontsize=16)
    plt.legend()
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Path to the log file
    log_file_path = "training.log"
    evaluation_frequency = 50  # Replace with the evaluation frequency used in your training loop

    # Parse rewards from the log file
    total_rewards = parse_rewards_from_log(log_file_path)
    eval_rewards = parse_evaluation_rewards(log_file_path)

    # Plot combined rewards
    plot_combined_rewards(total_rewards, eval_rewards, evaluation_frequency, save_path="combined_rewards_plot.png")
