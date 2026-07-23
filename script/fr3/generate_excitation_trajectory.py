"""Generate a safe, informative Fourier excitation trajectory for the FR3.

The entry point coordinates a feasible random seed, Drake/IPOPT optimization,
independent validation, candidate selection, and controller-ready output.

The high-level pipeline is:
    1. Load and validate the YAML/JSON configuration.
    2. Build Pinocchio dynamics data and the Drake collision checker.
    3. Randomly search for a feasible Fourier trajectory.
    4. Improve its identification quality with IPOPT.
    5. Validate and select the safest valid candidate.
    6. Export sampled trajectories and robot-controller Fourier coefficients.

Implementation details live in the corresponding ``fr3`` package modules.
"""

import argparse
import time

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from loguru import logger

from sysid.traj_generator.artifacts import (
    BestCandidateRecorder,
    make_trajectory_record,
    save_robot_payload_fourier_format,
    save_traj_csv,
    select_best_valid_record,
    yaml_safe,
)
from sysid.traj_generator.config import Fr3ExcitationConfig
from sysid.utils.path_utils import BASE_DIR, get_config_path, resolve_repo_path
from sysid.utils.robot_config import retrieve_robot_config


# BASE_DIR comes from env.sh, so the default does not depend on the current
# working directory or the location of this script.
DEFAULT_CONFIG_PATH = get_config_path("fr3", "traj_gen.yaml")


################################################################################
# Load configuration
################################################################################
def parse_args():
    """Load trajectory settings from a YAML or JSON configuration file."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Franka FR3 excitation trajectory with Drake/IPOPT, "
            "workspace wall constraints, and camera-box obstacles."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "YAML or JSON configuration file. Defaults to "
            f"{DEFAULT_CONFIG_PATH.relative_to(BASE_DIR)}."
        ),
    )
    cli_args = parser.parse_args()

    # Report file, schema, and type errors through argparse so CLI failures have
    # one consistent format and exit status.
    try:
        config = Fr3ExcitationConfig.from_file(cli_args.config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    # Downstream Drake code receives an absolute path even when YAML contains a
    # repository-relative robot description.
    config.robot_urdf_path = resolve_repo_path(config.robot_urdf_path)
    return config


def main():
    """Run generation, optimization, validation, selection, and export."""

    ############################################################################
    # Prepare the experiment
    ############################################################################
    config = parse_args()

    # Create one directory per run. An explicit run_name makes repeated runs
    # reproducible; otherwise the timestamp prevents accidental overwrites.
    save_root = config.save_dir.expanduser()
    if not save_root.is_absolute():
        save_root = (Path.cwd() / save_root).resolve()
    run_name = config.run_name or datetime.now().strftime(
        "fr3_excitation_%Y%m%d_%H%M%S"
    )
    experiment_dir = save_root / run_name

    ############################################################################
    # Load the robotics and optimization stack
    ############################################################################
    # These imports happen only after parsing so --help and configuration errors
    # do not need Drake, Pinocchio, IPOPT, or Matplotlib.
    from sysid.model.collision_model import DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS
    from sysid.model.inertia_model import InertiaModel
    from sysid.model.workspace_constraints import build_collision_checker
    from sysid.solver.solver_utils import evaluate_params_metrics
    from sysid.solver.wall_bound_solver import (
        WallBoundDrakeMathematicalProgramExcitationSolver,
    )
    from sysid.traj_generator.initialization import generate_valid_initial_trajectory
    from sysid.traj_generator.validation import validate_fr3_trajectory

    if config.plot:
        # Matplotlib is optional for headless trajectory generation.
        from sysid.utils.visualization import vis_compare_seqs

    if not config.no_save:
        # Save the exact input settings before generating data so every result
        # directory remains self-describing.
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(experiment_dir / "args.yaml", "w", encoding="utf-8") as stream:
            yaml.safe_dump(yaml_safe(config.to_dict()), stream, sort_keys=True)

    # The seed controls the random feasible-trajectory search.
    np.random.seed(config.seed)

    logger.info(f"Loaded trajectory configuration: {config.config_path}")
    logger.info(f"Using local FR3 helpers and models under {BASE_DIR}")
    logger.info(f"Using Drake collision URDF: {config.robot_urdf_path}")
    logger.info(
        "FR3 workspace constraints: "
        f"x > {config.link_x_lower}, "
        f"{config.link_y_lower} < y < {config.link_y_upper}, "
        f"z > {config.link_z_lower}; "
        f"camera min distance {config.drake_min_distance}"
    )
    logger.info(
        "FR3 camera boxes: x/y scale %s, z scale %s",
        config.camera_box_xy_scale,
        config.camera_box_z_scale,
    )
    logger.info(
        "IPOPT Fourier constraint scales: velocity=%s, position=%s; "
        "validation remains at 0.95x",
        config.fourier_velocity_limit_scale,
        config.fourier_position_limit_scale,
    )
    logger.info(
        "Best-condition validation margin thresholds: velocity>=%s, position>=%s",
        config.best_condition_fourier_velocity_margin_min,
        config.best_condition_fourier_position_margin_min,
    )
    if not config.disable_self_collision_constraints:
        logger.info(
            "FR3 self-collision constraints: %d configured body pairs, "
            "clearance >= %s",
            len(DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS),
            config.self_collision_clearance,
        )
    logger.info(f"Output directory: {experiment_dir}")

    ############################################################################
    # Initialize the robot and constraints
    ############################################################################
    # Pinocchio supplies controlled-joint counts and limits from the dynamics
    # URDF selected by robot name. Drake separately uses robot_urdf_path for
    # collision geometry and workspace constraints.
    robot_config = retrieve_robot_config(config.robot)
    fourier_config = {
        "order": int(config.fourier_order),
        "duration": float(config.fourier_duration),
    }
    collision_checker = build_collision_checker(config)

    ############################################################################
    # Initialize a feasible Fourier trajectory
    ############################################################################
    # IPOPT needs a feasible starting point. Random candidates are rejected
    # until joint, collision, self-collision, and workspace constraints pass.
    init_params, initial_eval = generate_valid_initial_trajectory(
        config,
        fourier_config,
        robot_config,
        collision_checker,
    )
    if not config.no_save:
        save_traj_csv(
            experiment_dir,
            "initial",
            initial_eval["t"],
            initial_eval["q"],
            initial_eval["dq"],
            initial_eval["ddq"],
        )

    if config.plot:
        vis_compare_seqs(
            [initial_eval["t"], initial_eval["t"], initial_eval["t"]],
            [initial_eval["q"], initial_eval["dq"], initial_eval["ddq"]],
            ["q", "dq", "ddq"],
            ["time"],
        )

    if config.initial_only:
        ########################################################################
        # Save the initial trajectory without optimization
        ########################################################################
        # This mode is useful for debugging feasibility independently of IPOPT.
        # The feasible seed becomes the selected controller trajectory.
        if not config.no_save:
            save_traj_csv(
                experiment_dir,
                "selected",
                initial_eval["t"],
                initial_eval["q"],
                initial_eval["dq"],
                initial_eval["ddq"],
            )
            save_robot_payload_fourier_format(
                experiment_dir,
                init_params,
                fourier_config,
                robot_config,
            )
            summary = {
                "selected_source": "initial",
                "selected_eval": initial_eval,
                "ipopt_stats": None,
            }
            with open(
                experiment_dir / "summary.yaml", "w", encoding="utf-8"
            ) as stream:
                yaml.safe_dump(yaml_safe(summary), stream, sort_keys=True)
        logger.info(
            "Stopping after valid initial trajectory because initial_only is enabled."
        )
        return

    ############################################################################
    # Configure trajectory optimization
    ############################################################################
    # IPOPT may visit a valid trajectory with a better condition number before
    # its final iterate. The callback validates and archives those improvements.
    best_candidate_recorder = BestCandidateRecorder(
        config,
        experiment_dir,
        fourier_config,
        robot_config,
        collision_checker,
        validate_fr3_trajectory,
    )
    # The inertia regressor measures how informative a trajectory is for system
    # identification. Friction columns are included only for condFriction.
    inertia_model = InertiaModel(config.robot)
    friction_model = (
        config.friction_model if config.excite_type == "condFriction" else None
    )
    solver = WallBoundDrakeMathematicalProgramExcitationSolver(
        # Trajectory parameterization and identification objective.
        fourier_config=fourier_config,
        robot_config=robot_config,
        inertia_model=inertia_model,
        robot_name=config.robot,
        friction_model=friction_model,
        eig_eps=config.eig_eps,
        min_eig_weight=config.objective_lambda,
        # IPOPT and collision-sampling controls.
        camera_collision_stride=config.camera_collision_stride,
        ipopt_max_iter=config.ipopt_max_iter,
        ipopt_print_level=config.ipopt_print_level,
        ipopt_hessian_approximation=config.ipopt_hessian_approximation,
        log_condition_every=config.log_condition_every,
        early_stop_constraint_tol=config.early_stop_constraint_tol,
        use_identifiable_columns=not config.no_identifiable_column_reduction,
        # Workspace walls and robot self-collision constraints.
        link_x_lower=config.link_x_lower,
        link_y_lower=config.link_y_lower,
        link_y_upper=config.link_y_upper,
        link_z_lower=config.link_z_lower,
        use_link_y_bounds=not config.disable_link_y_bounds,
        use_self_collision_constraints=(
            not config.disable_self_collision_constraints
        ),
        self_collision_body_pairs=DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS,
        self_collision_clearance=config.self_collision_clearance,
        # Criteria used by the intermediate best-candidate callback.
        best_condition_initial=config.initial_best_condition_number,
        best_candidate_check_every=config.best_condition_check_every,
        best_candidate_fourier_velocity_margin_tolerance=(
            config.best_condition_fourier_velocity_margin_tolerance
        ),
        best_candidate_fourier_position_margin_tolerance=(
            config.best_condition_fourier_position_margin_tolerance
        ),
        best_candidate_fourier_velocity_margin_min=(
            config.best_condition_fourier_velocity_margin_min
        ),
        best_candidate_fourier_position_margin_min=(
            config.best_condition_fourier_position_margin_min
        ),
        # Safety margins applied to the analytical Fourier bounds.
        fourier_velocity_limit_scale=config.fourier_velocity_limit_scale,
        fourier_position_limit_scale=config.fourier_position_limit_scale,
        best_candidate_callback=best_candidate_recorder,
        collision_checker=collision_checker,
    )

    ############################################################################
    # Optimize the trajectory
    ############################################################################
    # The solver returns its final iterate plus any better intermediate iterate
    # accepted by the callback.
    start = time.perf_counter()
    result = solver.solve(init_params)
    elapsed = time.perf_counter() - start
    stats = result["stats"]
    logger.info(f"Drake MathematicalProgram IPOPT time cost: {elapsed}")
    logger.info(f"Drake MathematicalProgram IPOPT status: {stats['return_status']}")
    logger.info(f"Drake MathematicalProgram IPOPT success: {stats['is_success']}")
    logger.info(f"Recomputed objective after optimization: {result['f']}")
    logger.info(f"Solver-reported objective: {result['solver_f']}")

    ############################################################################
    # Evaluate optimized trajectory candidates
    ############################################################################
    # Reuse the solver's identifiable regressor columns when evaluating every
    # candidate; otherwise condition numbers would not be directly comparable.
    identifiable_columns = result["identifiable_columns"]
    initial_metrics = evaluate_params_metrics(
        init_params,
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=friction_model,
        identifiable_columns=identifiable_columns,
        eig_eps=config.eig_eps,
        min_eig_weight=config.objective_lambda,
    )
    initial_record = make_trajectory_record(
        "initial",
        init_params,
        initial_eval,
        initial_metrics,
    )

    final_metrics = evaluate_params_metrics(
        result["x"],
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=friction_model,
        identifiable_columns=identifiable_columns,
        eig_eps=config.eig_eps,
        min_eig_weight=config.objective_lambda,
    )
    # Re-evaluate the optimized trajectory independently from the constraints
    # reported internally by IPOPT.
    final_eval = validate_fr3_trajectory(
        "Final",
        result["x"],
        fourier_config,
        robot_config,
        collision_checker,
        config,
    )
    final_record = make_trajectory_record(
        "final",
        result["x"],
        final_eval,
        final_metrics,
    )

    # Prefer the solver-returned best iterate when available, while retaining a
    # callback record for solver implementations that only invoke the callback.
    best_record = best_candidate_recorder.record
    if result.get("best_x") is not None:
        best_eval = validate_fr3_trajectory(
            "Best-condition",
            result["best_x"],
            fourier_config,
            robot_config,
            collision_checker,
            config,
        )
        best_record = make_trajectory_record(
            "best_condition",
            result["best_x"],
            best_eval,
            result["best_metrics"],
        )

    ############################################################################
    # Select the best valid trajectory
    ############################################################################
    # Choose the lowest-condition valid result. Including the feasible seed
    # guarantees a safe fallback when optimization produces an invalid iterate.
    selected_record = select_best_valid_record(
        [best_record, final_record, initial_record]
    )
    if selected_record is None:
        raise RuntimeError("No valid FR3 excitation trajectory was produced.")

    ############################################################################
    # Save the selected trajectory
    ############################################################################
    if not config.no_save:
        # Keep the final iterate for diagnosis, but export the selected valid
        # candidate in both sampled CSV and controller Fourier formats.
        save_traj_csv(
            experiment_dir,
            "final",
            final_record["t"],
            final_record["q"],
            final_record["dq"],
            final_record["ddq"],
        )
        selected_csv = save_traj_csv(
            experiment_dir,
            "selected",
            selected_record["t"],
            selected_record["q"],
            selected_record["dq"],
            selected_record["ddq"],
        )
        save_robot_payload_fourier_format(
            experiment_dir,
            selected_record["flat_params"],
            fourier_config,
            robot_config,
        )
        summary = {
            "selected_source": selected_record["source"],
            "selected_condition_number": selected_record["condition_number"],
            "selected_eval": selected_record["eval"],
            "selected_metrics": selected_record["metrics"],
            "ipopt_stats": stats,
            "elapsed_s": elapsed,
            "csv": str(selected_csv),
        }
        with open(
            experiment_dir / "summary.yaml", "w", encoding="utf-8"
        ) as stream:
            yaml.safe_dump(yaml_safe(summary), stream, sort_keys=True)

    ############################################################################
    # Plot and report the selected trajectory
    ############################################################################
    if config.plot:
        vis_compare_seqs(
            [selected_record["t"], selected_record["t"], selected_record["t"]],
            [selected_record["q"], selected_record["dq"], selected_record["ddq"]],
            ["q", "dq", "ddq"],
            ["time"],
        )

    logger.info(
        f"Selected {selected_record['source']} trajectory with condition number "
        f"{selected_record['condition_number']}"
    )
    if not config.no_save:
        logger.info(
            f"Saved selected trajectory and Fourier parameters to {experiment_dir}"
        )


if __name__ == "__main__":
    main()
