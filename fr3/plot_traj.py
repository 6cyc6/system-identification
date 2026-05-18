import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TRAJECTORY_DIR = Path("/home/galois/Downloads/trajectory_2")


def load_array(traj_dir, name):
    path = traj_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required trajectory file: {path}")
    return np.load(path)


def relative_time(times, t0):
    return np.asarray(times, dtype=float) - t0


def derive_joint_velocity(positions, dt):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError(f"Expected joint positions to be 2D, got {positions.shape}")
    return np.gradient(positions, float(dt), axis=0)


def median_sample_period(times, name):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {times.shape}")
    if len(times) < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return float(np.median(np.diff(times)))


def load_target_joint_velocity(traj_dir, target_positions, target_dt):
    candidates = (
        "target_joint_velocities.npy",
        "target_joint_velocity.npy",
        "target_joint_vel.npy",
    )
    for name in candidates:
        path = traj_dir / name
        if path.exists():
            return np.load(path), f"loaded from {path.name}"
    return (
        derive_joint_velocity(target_positions, target_dt),
        f"derived from target positions with uniform dt={target_dt:.9f}s",
    )


def validate_joint_data(actual, target, actual_name, target_name):
    if actual.ndim != 2:
        raise ValueError(f"{actual_name} must be a 2D array, got {actual.shape}")
    if target.ndim != 2:
        raise ValueError(f"{target_name} must be a 2D array, got {target.shape}")
    if actual.shape[1] != target.shape[1]:
        raise ValueError(
            f"{actual_name} and {target_name} must have the same joint count: "
            f"{actual.shape[1]} != {target.shape[1]}"
        )


def plot_joint_comparison(
    actual_time,
    actual_values,
    target_time,
    target_values,
    *,
    title,
    ylabel,
):
    validate_joint_data(actual_values, target_values, "actual_values", "target_values")

    njoints = actual_values.shape[1]
    fig, axes = plt.subplots(njoints, 1, sharex=True, figsize=(12, 1.8 * njoints))
    axes = np.atleast_1d(axes)

    for joint_idx, ax in enumerate(axes):
        ax.plot(
            actual_time,
            actual_values[:, joint_idx],
            label="actual",
            linewidth=1.2,
        )
        ax.plot(
            target_time,
            target_values[:, joint_idx],
            label="target",
            linewidth=1.0,
            linestyle="--",
        )
        ax.set_ylabel(f"J{joint_idx + 1}\n{ylabel}")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def plot_trajectory(traj_dir):
    joint_positions = load_array(traj_dir, "joint_positions.npy")
    target_joint_positions = load_array(traj_dir, "target_joint_positions.npy")
    joint_velocities = load_array(traj_dir, "joint_velocities.npy")
    sample_times = load_array(traj_dir, "sample_times_s.npy")
    target_sample_times = load_array(traj_dir, "target_sample_times_s.npy")
    target_dt = median_sample_period(target_sample_times, "target_sample_times")
    target_joint_velocities, target_velocity_source = load_target_joint_velocity(
        traj_dir,
        target_joint_positions,
        target_dt,
    )

    validate_joint_data(
        joint_positions,
        target_joint_positions,
        "joint_positions",
        "target_joint_positions",
    )
    validate_joint_data(
        joint_velocities,
        target_joint_velocities,
        "joint_velocities",
        "target_joint_velocities",
    )

    t0 = min(float(sample_times[0]), float(target_sample_times[0]))
    actual_time = relative_time(sample_times, t0)
    target_time = relative_time(target_sample_times, t0)

    print(f"Trajectory directory: {traj_dir}")
    print(f"joint_positions: {joint_positions.shape}")
    print(f"target_joint_positions: {target_joint_positions.shape}")
    print(f"joint_velocities: {joint_velocities.shape}")
    print(f"target_joint_velocities: {target_joint_velocities.shape} ({target_velocity_source})")

    plot_joint_comparison(
        actual_time,
        joint_positions,
        target_time,
        target_joint_positions,
        title="Joint Position vs Target Joint Position",
        ylabel="rad",
    )
    plot_joint_comparison(
        actual_time,
        joint_velocities,
        target_time,
        target_joint_velocities,
        title="Joint Velocity vs Target Joint Velocity",
        ylabel="rad/s",
    )

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot actual and target FR3 trajectory data."
    )
    parser.add_argument(
        "trajectory_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
        help=f"Directory containing trajectory .npy files. Default: {DEFAULT_TRAJECTORY_DIR}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_trajectory(args.trajectory_dir.expanduser().resolve())
