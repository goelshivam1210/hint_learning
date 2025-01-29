import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml


def load_tensorboard_logs(log_dirs, scalar_name):
    """
    Load and interpolate scalar data from TensorBoard logs across multiple runs.

    Args:
        log_dirs (list of str): List of directories containing TensorBoard logs.
        scalar_name (str): Name of the scalar to extract (e.g., "Test/Success_Rate").

    Returns:
        pd.DataFrame: DataFrame with averaged and interpolated scalar data.
    """
    data = []
    common_steps = None

    for log_dir in log_dirs:
        # Find the TensorBoard log file
        logs_path = os.path.join(log_dir, "logs")
        log_files = [
            os.path.join(logs_path, f)
            for f in os.listdir(logs_path)
            if f.startswith("events.out.tfevents")
        ]
        if not log_files:
            print(f"No TensorBoard logs found in {logs_path}")
            continue

        # Parse the first log file in the directory
        event_accumulator = EventAccumulator(log_files[0])
        event_accumulator.Reload()

        # Check if the scalar exists in the logs
        if scalar_name not in event_accumulator.Tags()['scalars']:
            print(f"Scalar {scalar_name} not found in {log_files[0]}")
            continue

        # Extract scalar values
        events = event_accumulator.Scalars(scalar_name)
        steps = np.array([event.step for event in events])
        values = np.array([event.value for event in events])

        if common_steps is None:
            common_steps = steps
        else:
            # Update the common steps (union of all steps for interpolation)
            common_steps = np.union1d(common_steps, steps)

        data.append({"steps": steps, "values": values})

    # Interpolate all runs to align with common steps
    interpolated_data = []
    for run in data:
        interpolated_values = np.interp(common_steps, run["steps"], run["values"])
        interpolated_data.append(interpolated_values)

    # Compute mean and standard deviation
    interpolated_data = np.array(interpolated_data)
    mean_values = np.mean(interpolated_data, axis=0)
    std_values = np.std(interpolated_data, axis=0)

    return pd.DataFrame({"steps": common_steps, "mean": mean_values, "std": std_values})

def align_timesteps(results_dfs):
    """
    Align the timesteps across all DataFrames by trimming to the shortest timesteps range.

    Args:
        results_dfs (dict): Dictionary of DataFrames keyed by condition (e.g., "Hints=True").

    Returns:
        dict: Updated dictionary with aligned DataFrames.
    """
    # Ensure all DataFrames have the correct column name
    for condition, df in results_dfs.items():
        if "steps" in df.columns:
            df.rename(columns={"steps": "timesteps"}, inplace=True)

    # Find the shortest timesteps range
    min_timesteps = min([df["timesteps"].max() for df in results_dfs.values()])

    # Trim each DataFrame to the shortest range
    aligned_results = {}
    for condition, df in results_dfs.items():
        aligned_df = df[df["timesteps"] <= min_timesteps]
        aligned_results[condition] = aligned_df

    return aligned_results


def plot_results(results_df, scalar_name, title, output_file):
    """
    Plot results with success rate vs timesteps.

    Args:
        results_df (pd.DataFrame): DataFrame with 'timesteps', 'mean', and 'std' columns.
        scalar_name (str): Name of the scalar being plotted.
        title (str): Title of the plot.
        output_file (str): Filepath to save the plot.
    """
    sns.set_theme(style="whitegrid", palette="muted")

    # Ensure column names are correct
    if "steps" in results_df.columns:
        results_df.rename(columns={"steps": "timesteps"}, inplace=True)

    # Debugging: Check if DataFrame has expected columns
    print("Columns in results_df:", results_df.columns)
    print("Head of results_df:", results_df.head())

    # Prepare the data for error shading
    results_df["upper_bound"] = results_df["mean"] + results_df["std"]
    results_df["lower_bound"] = results_df["mean"] - results_df["std"]

    # Plot mean line
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_df,
        x="timesteps",
        y="mean",
        linewidth=2,
        label="Mean",
    )

    # Plot shaded region for standard deviation
    plt.fill_between(
        results_df["timesteps"],
        results_df["lower_bound"],
        results_df["upper_bound"],
        alpha=0.3,
        label="Standard Deviation",
    )

    # Configure plot labels and title
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel(scalar_name, fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    # plt.show()

def plot_combined_results(results_dfs, scalar_name, title, output_file):
    sns.set_theme(style="whitegrid", palette="muted")

    # Align timesteps
    results_dfs = align_timesteps(results_dfs)

    # Combine all DataFrames into a single one with a 'Condition' column
    combined_df = pd.concat(
        [df.assign(Condition=condition) for condition, df in results_dfs.items()]
    )

    # Rename columns if necessary
    if "steps" in combined_df.columns:
        combined_df.rename(columns={"steps": "timesteps"}, inplace=True)

    # Debugging: Check combined DataFrame
    print("Columns in combined_df:", combined_df.columns)
    print("Head of combined_df:", combined_df.head())

    # Prepare the data for error shading
    combined_df["upper_bound"] = combined_df["mean"] + combined_df["std"]
    combined_df["lower_bound"] = combined_df["mean"] - combined_df["std"]

    # Plot mean lines with hue for different conditions
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=combined_df,
        x="timesteps",
        y="mean",
        hue="Condition",
        linewidth=2,
    )

    # Add shaded regions for standard deviation
    for condition, group_df in combined_df.groupby("Condition"):
        plt.fill_between(
            group_df["timesteps"],
            group_df["lower_bound"],
            group_df["upper_bound"],
            alpha=0.2,
            label=f"{condition} Std Dev",
        )

    # Configure plot labels and title
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel(scalar_name, fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(output_file)
    print(f"Combined plot saved to {output_file}")
    # plt.show()

def filter_runs_by_name(log_dirs, param_name, param_value):
    """
    Filter run directories based on a specific parameter in the directory name.

    Args:
        log_dirs (list of str): List of run directories.
        param_name (str): Parameter to filter by (e.g., 'hints').
        param_value (str or bool): Value of the parameter to filter by.

    Returns:
        list of str: Filtered list of directories.
    """
    filtered_dirs = []
    param_value_str = str(param_value)  # Convert the value to string for matching
    for log_dir in log_dirs:
        if f"{param_name}_{param_value_str}" in log_dir:
            filtered_dirs.append(log_dir)
    return filtered_dirs


def main():
    base_log_dir = "log"
    scalar_names = ["Test/Success_Rate", "Test/Reward"]
    hints_values = [True, False]
    output_files = {
        "Success_Rate": {
            True: "success_rate_hints_true.png",
            False: "success_rate_hints_false.png",
            "combined": "success_rate_combined.png",
        },
        "Reward": {
            True: "reward_hints_true.png",
            False: "reward_hints_false.png",
            "combined": "reward_combined.png",
        },
    }

    run_dirs = [
        os.path.join(base_log_dir, d)
        for d in os.listdir(base_log_dir)
        if os.path.isdir(os.path.join(base_log_dir, d)) and "ppo_instance" in d
    ]

    print(f"Found {len(run_dirs)} runs in {base_log_dir}")

    for scalar_name in scalar_names:
        print(f"\nProcessing scalar: {scalar_name}")
        results_by_hints = {}

        for hints in hints_values:
            print(f"  Processing runs for hints={hints}...")
            filtered_dirs = filter_runs_by_name(run_dirs, "hints", hints)
            if not filtered_dirs:
                print(f"  No runs found for hints={hints}.")
                continue

            results_df = load_tensorboard_logs(filtered_dirs, scalar_name)
            if results_df.empty:
                print(f"  No valid data for scalar {scalar_name} (hints={hints}).")
                continue

            results_by_hints[f"Hints={hints}"] = results_df

            output_file = os.path.join(base_log_dir, output_files[scalar_name.split("/")[-1]][hints])
            plot_results(
                results_df,
                scalar_name,
                title=f"{scalar_name} vs Timesteps (Hints={hints})",
                output_file=output_file,
            )

        if results_by_hints:
            combined_output_file = os.path.join(
                base_log_dir, output_files[scalar_name.split("/")[-1]]["combined"]
            )
            aligned_results = align_timesteps(results_by_hints)
            combined_df = pd.concat(
                [df.assign(Condition=condition) for condition, df in aligned_results.items()]
            )

        sns.set_theme(style="whitegrid", palette="muted")
        plt.figure(figsize=(10, 6))

        # Draw the mean lines with hue for different conditions
        lineplot = sns.lineplot(
            data=combined_df,
            x="timesteps",
            y="mean",
            hue="Condition",
            linewidth=2,
        )

        # Use the same colors for the fill_between
        for condition, group_df in combined_df.groupby("Condition"):
            color = lineplot.get_lines()[list(combined_df["Condition"].unique()).index(condition)].get_color()
            plt.fill_between(
                group_df["timesteps"],
                group_df["mean"] - group_df["std"],
                group_df["mean"] + group_df["std"],
                alpha=0.2,
                color=color,
                label=f"{condition} Std Dev",
            )

        plt.title(f"{scalar_name} vs Timesteps (Combined)", fontsize=14, weight="bold")
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel(scalar_name, fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(combined_output_file)
        print(f"Combined plot saved to {combined_output_file}")

if __name__ == "__main__":
    main()