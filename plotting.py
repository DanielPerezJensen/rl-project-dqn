import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def average_csvs(task_folder):

    path = os.path.join("log", task_folder)

    q_vals_dict = {}
    test_reward_avgs_dict = {}
    test_reward_stds_dict = {}

    for algo_folder in os.listdir(path):
        algo_path = os.path.join(path, algo_folder)
        n = len(os.listdir(algo_path))

        q_vals = []
        test_reward_avgs = []
        test_reward_stds = []

        for run_folder in os.listdir(algo_path):
            run_path = os.path.join(algo_path, run_folder, "csv")

            q_vals.append(pd.read_csv(os.path.join(run_path, "q_vals.csv"), index_col=0)["mean"][0])

            test_reward_avgs.append(
                pd.read_csv(os.path.join(run_path, "test_reward.csv"), index_col=0)["value"]
            )

            test_reward_stds.append(
                pd.read_csv(os.path.join(run_path, "test_reward_std.csv"), index_col=0)["value"]
            )

        q_vals_dict[algo_folder] = q_vals
        test_reward_avgs_dict[algo_folder] = (sum(test_reward_avgs) / n).values
        test_reward_stds_dict[algo_folder] = (sum(test_reward_stds) / n).values

    steps = pd.read_csv(os.path.join(run_path, "test_reward.csv"))["step"].values

    return q_vals_dict, test_reward_avgs_dict, test_reward_stds_dict, steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_folder", type=str, default="CartPole-v0",
        help="Name of gym environment"
    )

    args = parser.parse_known_args()[0]

    q_vals_dict, test_reward_avgs_dict, test_reward_stds_dict, steps = average_csvs(args.task_folder)

    os.makedirs("results", exist_ok=True)

    # Training reward over time
    plt.figure()
    for algo_type in test_reward_avgs_dict:
        means = test_reward_avgs_dict[algo_type]
        stds = test_reward_stds_dict[algo_type]
        plt.errorbar(steps, means, stds, capsize=3,
                     linestyle=":", fmt="o", label=algo_type)

    plt.title(args.task_folder)
    plt.xlabel("Step")
    plt.ylabel("(Avg.) Reward")
    plt.legend()
    plt.savefig(f"results/training_{args.task_folder}")

    # Plot for spread of Q-values
    fig = plt.figure()
    ax = plt.subplot(111)

    # Only integers in the tickers
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    qs = q_vals_dict["dqn"]
    qs_double = q_vals_dict["double_dqn"]
    plt.boxplot([qs, qs_double])
    plt.ylabel("Mean Q-Values")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["DQN", "Double DQN"])
    plt.title(args.task_folder)
    plt.savefig(f"results/violin_{args.task_folder}")
