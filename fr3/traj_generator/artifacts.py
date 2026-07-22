"""Trajectory selection and output artifact helpers."""

from pathlib import Path

import numpy as np

from loguru import logger

from ..utils.fourier_utils import unflatten_fourier_params


def make_trajectory_record(source, flat_params, eval_result, metrics):
    """Bundle coefficients, quality metrics, and sampled motion for selection."""
    eval_summary = {
        key: value
        for key, value in eval_result.items()
        if key not in {"t", "q", "dq", "ddq"}
    }
    return {
        "source": source,
        "flat_params": np.asarray(flat_params, dtype=float),
        "metrics": dict(metrics),
        "condition_number": float(metrics.get("condition_number", np.inf)),
        "valid": bool(eval_result["valid"]),
        "eval": eval_summary,
        "t": eval_result["t"],
        "q": eval_result["q"],
        "dq": eval_result["dq"],
        "ddq": eval_result["ddq"],
    }


def select_best_valid_record(records):
    """Choose the valid record with the lowest finite condition number."""
    valid_records = [
        record for record in records if record is not None and record["valid"]
    ]
    if not valid_records:
        return None
    return min(
        valid_records,
        key=lambda record: (
            not np.isfinite(record["condition_number"]),
            record["condition_number"],
        ),
    )


def save_traj_csv(save_dir, name, t, q, dq, ddq):
    """Save time, position, velocity, and acceleration samples in one CSV."""
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{name}.csv"
    data = np.column_stack([t, q, dq, ddq])
    njoints = q.shape[1]
    header = (
        ["t"]
        + [f"q_{i}" for i in range(njoints)]
        + [f"dq_{i}" for i in range(njoints)]
        + [f"ddq_{i}" for i in range(njoints)]
    )
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")
    return csv_path


def _condition_number_filename_suffix(condition_number):
    condition_number = float(condition_number)
    if not np.isfinite(condition_number):
        return "nonfinite"
    text = f"{condition_number:.6g}"
    return text.replace("-", "neg").replace("+", "").replace(".", "p")


def _unique_csv_stem(save_dir, stem):
    save_dir = Path(save_dir)
    candidate = stem
    counter = 2
    while (save_dir / f"{candidate}.csv").exists():
        candidate = f"{stem}_{counter}"
        counter += 1
    return candidate


def save_robot_payload_fourier_format(
    save_dir,
    flat_params,
    fourier_config,
    robot_config,
):
    """Export coefficients in the convention consumed by the robot payload."""
    params = unflatten_fourier_params(flat_params, fourier_config, robot_config)
    a_external, b_external = params[0], params[1]
    omega = 2.0 * np.pi / float(fourier_config["duration"])
    harmonic_ids = np.arange(1, int(fourier_config["order"]) + 1, dtype=float)

    a_values = (a_external / (omega * harmonic_ids[:, None])).T
    b_values = (-b_external / (omega * harmonic_ids[:, None])).T
    q0_values = np.asarray(robot_config["init_pos"], dtype=float)

    np.save(save_dir / "a_value.npy", a_values)
    np.save(save_dir / "b_value.npy", b_values)
    np.save(save_dir / "q0_value.npy", q0_values)
    np.save(save_dir / "omega.npy", np.array([omega]))
    np.save(save_dir / "external_flat_fourier_params.npy", flat_params)


def yaml_safe(value):
    """Recursively convert Path and NumPy values into YAML-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: yaml_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe(val) for val in value]
    return value


class BestCandidateRecorder:
    """Validate and archive improving intermediate IPOPT candidates."""

    def __init__(
        self,
        config,
        experiment_dir,
        fourier_config,
        robot_config,
        collision_checker,
        trajectory_validator=None,
    ):
        if trajectory_validator is None:
            from .validation import validate_fr3_trajectory

            trajectory_validator = validate_fr3_trajectory

        self.config = config
        self.experiment_dir = experiment_dir
        self.fourier_config = fourier_config
        self.robot_config = robot_config
        self.collision_checker = collision_checker
        self.trajectory_validator = trajectory_validator
        self.record = None

    def __call__(self, flat_params, metrics, iteration):
        eval_result = self.trajectory_validator(
            f"Iteration {iteration} best-condition candidate",
            flat_params,
            self.fourier_config,
            self.robot_config,
            self.collision_checker,
            self.config,
        )
        if not eval_result["valid"]:
            logger.info(
                f"Iteration {iteration} improved condition number "
                f"{metrics['condition_number']}, but failed validation."
            )
            return False

        self.record = make_trajectory_record(
            "best_condition",
            flat_params,
            eval_result,
            metrics,
        )
        if not self.config.no_save:
            condition_suffix = _condition_number_filename_suffix(
                metrics["condition_number"]
            )
            archive_name = _unique_csv_stem(
                self.experiment_dir,
                f"best_condition_{condition_suffix}",
            )
            save_traj_csv(
                self.experiment_dir,
                archive_name,
                eval_result["t"],
                eval_result["q"],
                eval_result["dq"],
                eval_result["ddq"],
            )
            save_traj_csv(
                self.experiment_dir,
                "best_condition",
                eval_result["t"],
                eval_result["q"],
                eval_result["dq"],
                eval_result["ddq"],
            )
            logger.info(f"Saved valid best-condition archive: {archive_name}.csv")
        logger.info(
            f"Accepted valid best candidate at IPOPT iteration {iteration}: "
            f"condition_number={metrics['condition_number']}"
        )
        return True
