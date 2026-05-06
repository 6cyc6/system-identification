import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from generate_excitation_ipopt_drake import save_traj_csv, validate_trajectory
from system_identification.drake.camera_collision import DrakeCameraCollisionChecker
from system_identification.drake.mathematical_program_solver import (
    DrakeMathematicalProgramExcitationSolver,
    fourier_constraint_bounds,
    fourier_constraint_values,
)
from system_identification.excitation_generator_new import obtain_valid_traj_param
from system_identification.fourier_utils import flatten_fourier_params
from system_identification.inertia_model import InertiaModel
from system_identification.ipopt_solver import evaluate_params_metrics
from system_identification.my_utils.path_utils import safe_mkdir
from system_identification.utils import retrieve_robot_config, vis_compare_seqs


def _max_constraint_violation(values, lower, upper):
    values = np.asarray(values, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    lower_violation = np.maximum(lower - values, 0.0)
    upper_violation = np.maximum(values - upper, 0.0)
    return float(max(np.max(lower_violation), np.max(upper_violation)))


def _min_or_inf(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.inf
    return float(np.min(values))


def initial_solver_constraint_report(
    flat_params,
    q,
    fourier_config,
    robot_config,
    collision_checker,
    args,
):
    fourier_lbg, fourier_ubg = fourier_constraint_bounds(
        fourier_config,
        robot_config,
    )
    fourier_values = fourier_constraint_values(
        flat_params,
        fourier_config,
        robot_config,
    )
    fourier_violation = _max_constraint_violation(
        fourier_values,
        fourier_lbg,
        fourier_ubg,
    )

    q_sampled = q[:: max(1, int(args.camera_collision_stride))]
    path_values = [collision_checker.minimum_distance_constraint_values(q_sampled)]
    path_lbg = [np.zeros_like(path_values[0], dtype=float)]
    path_ubg = [np.ones_like(path_values[0], dtype=float)]
    if not args.disable_link_y_bounds:
        link_y_values = collision_checker.robot_link_y_margins(
            q_sampled,
            lower=args.link_y_lower,
            upper=args.link_y_upper,
        ).reshape(-1)
        path_values.append(link_y_values)
        path_lbg.append(np.zeros_like(link_y_values, dtype=float))
        path_ubg.append(np.full_like(link_y_values, np.inf, dtype=float))

    path_values = np.concatenate(path_values)
    path_lbg = np.concatenate(path_lbg)
    path_ubg = np.concatenate(path_ubg)
    path_violation = _max_constraint_violation(path_values, path_lbg, path_ubg)

    njoints = int(robot_config["njoints"])
    fourier_values_by_joint = fourier_values.reshape(njoints, 5)
    fourier_ubg_by_joint = fourier_ubg.reshape(njoints, 5)
    fourier_equality_residual = float(np.max(np.abs(fourier_values_by_joint[:, :3])))
    fourier_velocity_margin = _min_or_inf(
        fourier_ubg_by_joint[:, 3] - fourier_values_by_joint[:, 3]
    )
    fourier_position_margin = _min_or_inf(
        fourier_ubg_by_joint[:, 4] - fourier_values_by_joint[:, 4]
    )
    n_collision = len(q_sampled)
    drake_collision_margin = _min_or_inf(
        np.minimum(
            path_values[:n_collision],
            path_ubg[:n_collision] - path_values[:n_collision],
        )
    )
    link_y_margin = _min_or_inf(path_values[n_collision:])
    max_violation = max(fourier_violation, path_violation)

    return {
        "valid": max_violation <= max(0.0, float(args.early_stop_constraint_tol)),
        "max_violation": max_violation,
        "fourier_equality_residual": fourier_equality_residual,
        "fourier_velocity_margin": fourier_velocity_margin,
        "fourier_position_margin": fourier_position_margin,
        "drake_collision_margin": drake_collision_margin,
        "link_y_margin": link_y_margin,
    }


def make_collision_checker(args):
    xy_prism_height = (
        None if args.drake_physical_camera_height else args.drake_xy_prism_height
    )
    return DrakeCameraCollisionChecker(
        robot_name=args.robot,
        min_distance=args.drake_min_distance,
        robot_sphere_radius=args.drake_robot_sphere_radius,
        robot_link_samples=args.drake_robot_link_samples,
        camera_chamfer_radius=args.drake_camera_chamfer_radius,
        xy_prism_height=xy_prism_height,
    )


def set_worker_seed(args, worker_id):
    if args.seed is not None:
        seed = int(args.seed) + int(worker_id or 0)
    else:
        seed = (
            os.getpid() * 1000003
            + int(worker_id or 0) * 9176
            + int(datetime.now().timestamp() * 1e6)
        ) % (2**32 - 1)
    np.random.seed(seed)
    return seed


def make_trajectory_record(source, flat_params, eval_result, metrics):
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
    valid_records = [record for record in records if record is not None and record["valid"]]
    if not valid_records:
        return None
    return min(
        valid_records,
        key=lambda record: (
            not np.isfinite(record["condition_number"]),
            record["condition_number"],
        ),
    )


def evaluate_trajectory_metrics(
    flat_params,
    fourier_config,
    robot_config,
    inertia_model,
    args,
    identifiable_columns,
):
    return evaluate_params_metrics(
        flat_params,
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=args.friction_model if args.excite_type == "condFriction" else None,
        identifiable_columns=identifiable_columns,
        eig_eps=args.eig_eps,
        min_eig_weight=args.objective_lambda,
    )


def generate_valid_initial_trajectory(
    args,
    fourier_config,
    robot_config,
    collision_checker,
    worker_label,
):
    initial_attempts = 0
    while True:
        initial_attempts += 1
        t, qs, qds, qdds, init_params = obtain_valid_traj_param(
            fourier_config,
            robot_config,
        )
        init_params_flat = flatten_fourier_params(init_params)
        solver_constraint_report = initial_solver_constraint_report(
            init_params_flat,
            qs,
            fourier_config,
            robot_config,
            collision_checker,
            args,
        )
        if not solver_constraint_report["valid"]:
            logger.info(
                f"{worker_label}Discarding initial trajectory that violates solver "
                f"constraints: attempt={initial_attempts}; "
                f"max_violation={solver_constraint_report['max_violation']}; "
                f"fourier_equality_residual="
                f"{solver_constraint_report['fourier_equality_residual']}; "
                f"fourier_velocity_margin="
                f"{solver_constraint_report['fourier_velocity_margin']}; "
                f"fourier_position_margin="
                f"{solver_constraint_report['fourier_position_margin']}; "
                f"drake_collision_margin="
                f"{solver_constraint_report['drake_collision_margin']}; "
                f"link_y_margin={solver_constraint_report['link_y_margin']}"
            )
            continue

        initial_eval = validate_trajectory(
            f"{worker_label}Initial",
            init_params_flat,
            fourier_config,
            robot_config,
            collision_checker,
            args,
        )
        if initial_eval["valid"]:
            logger.info(
                f"{worker_label}Accepted initial trajectory satisfying solver "
                f"constraints and validation after {initial_attempts} attempt(s)."
            )
            return t, qs, qds, qdds, init_params_flat, initial_eval
        logger.info(
            f"{worker_label}Discarding initial trajectory that satisfies solver "
            "constraints but fails full validation."
        )


def run_single_start(
    args_dict,
    worker_id=None,
    save_outputs=False,
    experiment_dir_str=None,
):
    os.chdir(Path(__file__).resolve().parent)
    args = argparse.Namespace(**args_dict)
    seed = set_worker_seed(args, worker_id)
    worker_label = "" if worker_id is None else f"Worker {worker_id}: "
    logger.info(f"{worker_label}Using random seed {seed}")

    experiment_dir = Path(experiment_dir_str) if save_outputs and experiment_dir_str else None
    robot_config = retrieve_robot_config(args.robot)
    fourier_config = {"order": args.fourier_order, "duration": args.fourier_duration}
    collision_checker = make_collision_checker(args)

    t, qs, qds, qdds, init_params_flat, initial_eval = generate_valid_initial_trajectory(
        args,
        fourier_config,
        robot_config,
        collision_checker,
        worker_label,
    )

    if experiment_dir is not None and initial_eval["valid"]:
        save_traj_csv(
            experiment_dir,
            "initial",
            initial_eval["t"],
            initial_eval["q"],
            initial_eval["dq"],
            initial_eval["ddq"],
        )

    if save_outputs and not args.no_plot:
        vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
        plt.show()

    best_callback_record = {"record": None}

    def save_best_valid_candidate(flat_params, metrics, iteration):
        condition_number = float(metrics["condition_number"])
        eval_result = validate_trajectory(
            f"{worker_label}Iteration {iteration} best-condition candidate",
            flat_params,
            fourier_config,
            robot_config,
            collision_checker,
            args,
        )
        if not eval_result["valid"]:
            logger.info(
                f"{worker_label}Iteration {iteration} condition number "
                f"{condition_number} beats the current best threshold, but the "
                "trajectory is invalid; not saving."
            )
            return False
        best_callback_record["record"] = make_trajectory_record(
            "best_condition",
            flat_params,
            eval_result,
            metrics,
        )
        if experiment_dir is not None:
            save_traj_csv(
                experiment_dir,
                "best_condition",
                eval_result["t"],
                eval_result["q"],
                eval_result["dq"],
                eval_result["ddq"],
            )
            logger.info(
                f"{worker_label}Saved new valid best trajectory at Drake IPOPT "
                f"iteration {iteration}: condition_number={condition_number}"
            )
        else:
            logger.info(
                f"{worker_label}Accepted new valid best trajectory at Drake IPOPT "
                f"iteration {iteration}: condition_number={condition_number}"
            )
        return True

    inertia_model = InertiaModel(args.robot)
    solver = DrakeMathematicalProgramExcitationSolver(
        fourier_config=fourier_config,
        robot_config=robot_config,
        inertia_model=inertia_model,
        robot_name=args.robot,
        friction_model=args.friction_model if args.excite_type == "condFriction" else None,
        eig_eps=args.eig_eps,
        min_eig_weight=args.objective_lambda,
        camera_collision_stride=args.camera_collision_stride,
        ipopt_max_iter=args.ipopt_max_iter,
        ipopt_print_level=args.ipopt_print_level,
        ipopt_hessian_approximation=args.ipopt_hessian_approximation,
        log_condition_every=args.log_condition_every,
        early_stop_constraint_tol=args.early_stop_constraint_tol,
        use_identifiable_columns=not args.no_identifiable_column_reduction,
        link_y_lower=args.link_y_lower,
        link_y_upper=args.link_y_upper,
        use_link_y_bounds=not args.disable_link_y_bounds,
        best_condition_initial=args.initial_best_condition_number,
        best_candidate_check_every=args.best_condition_check_every,
        best_candidate_callback=save_best_valid_candidate,
        collision_checker=collision_checker,
    )

    time1 = perf_counter()
    result = solver.solve(init_params_flat)
    time2 = perf_counter()
    stats = result["stats"]
    logger.info(f"{worker_label}Drake MathematicalProgram IPOPT time cost: {time2 - time1}")
    logger.info(
        f"{worker_label}Drake MathematicalProgram IPOPT recomputed objective after "
        f"optimization: {result['f']}"
    )
    logger.info(
        f"{worker_label}Drake MathematicalProgram IPOPT reported cost: "
        f"{result['solver_f']}"
    )
    logger.info(
        f"{worker_label}Drake MathematicalProgram IPOPT return status: "
        f"{stats['return_status']}"
    )
    logger.info(f"{worker_label}Drake MathematicalProgram IPOPT success: {stats['is_success']}")

    identifiable_columns = result["identifiable_columns"]
    initial_metrics = evaluate_trajectory_metrics(
        init_params_flat,
        fourier_config,
        robot_config,
        inertia_model,
        args,
        identifiable_columns,
    )
    initial_record = make_trajectory_record(
        "initial",
        init_params_flat,
        initial_eval,
        initial_metrics,
    )

    best_metrics = result.get("best_metrics")
    if best_metrics is None:
        logger.info(
            f"{worker_label}No checked valid trajectory improved below the initial "
            f"best condition number {args.initial_best_condition_number}."
        )
    else:
        logger.info(
            f"{worker_label}Lowest valid checked candidate source: "
            f"{result.get('best_source')}; condition_number: "
            f"{best_metrics.get('condition_number')}; objective: "
            f"{best_metrics.get('objective')}"
        )

    final_metrics = evaluate_trajectory_metrics(
        result["x"],
        fourier_config,
        robot_config,
        inertia_model,
        args,
        identifiable_columns,
    )
    final_eval = validate_trajectory(
        f"{worker_label}Final",
        result["x"],
        fourier_config,
        robot_config,
        collision_checker,
        args,
    )
    final_record = make_trajectory_record("final", result["x"], final_eval, final_metrics)

    best_record = None
    if result.get("best_x") is not None:
        best_eval = validate_trajectory(
            f"{worker_label}Best-condition",
            result["best_x"],
            fourier_config,
            robot_config,
            collision_checker,
            args,
        )
        best_record = make_trajectory_record(
            "best_condition",
            result["best_x"],
            best_eval,
            result["best_metrics"],
        )
    elif best_callback_record["record"] is not None:
        best_record = best_callback_record["record"]

    if save_outputs and not args.no_plot:
        vis_compare_seqs(
            [final_eval["t"], final_eval["t"], final_eval["t"]],
            [final_eval["q"], final_eval["dq"], final_eval["ddq"]],
            ["q", "qd", "qdd"],
            ["time"],
        )
        plt.show()

    if experiment_dir is not None and stats["is_success"] and final_eval["valid"]:
        save_traj_csv(
            experiment_dir,
            "final",
            final_eval["t"],
            final_eval["q"],
            final_eval["dq"],
            final_eval["ddq"],
        )
    elif experiment_dir is not None:
        logger.warning(
            f"{worker_label}Final Drake MathematicalProgram IPOPT point was not saved."
        )

    selected_record = select_best_valid_record([best_record, final_record, initial_record])
    return {
        "worker_id": worker_id,
        "seed": seed,
        "time_cost": time2 - time1,
        "stats": stats,
        "result_f": result["f"],
        "solver_f": result["solver_f"],
        "best_source": result.get("best_source"),
        "initial": initial_record,
        "best": best_record,
        "final": final_record,
        "selected": selected_record,
    }


def run_single_start_worker(args_dict, worker_id):
    try:
        return run_single_start(
            args_dict,
            worker_id=worker_id,
            save_outputs=False,
            experiment_dir_str=None,
        )
    except Exception:
        return {
            "worker_id": worker_id,
            "error": traceback.format_exc(),
        }


def run_parallel_starts(args, experiment_dir):
    worker_count = max(1, int(args.num_workers))
    logger.info(f"Starting {worker_count} parallel Drake IPOPT multi-start workers.")
    worker_args = vars(args).copy()
    worker_args["no_plot"] = True
    worker_args["no_save"] = True

    results = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(run_single_start_worker, worker_args, worker_id): worker_id
            for worker_id in range(1, worker_count + 1)
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            result = future.result()
            if result.get("error"):
                logger.error(f"Worker {worker_id} failed:\n{result['error']}")
                continue
            selected = result.get("selected")
            if selected is None:
                logger.warning(f"Worker {worker_id} returned no valid trajectory.")
                continue
            logger.info(
                f"Worker {worker_id} selected {selected['source']}: "
                f"condition_number={selected['condition_number']}; "
                f"objective={selected['metrics'].get('objective')}; "
                f"seed={result['seed']}; time_cost={result['time_cost']}"
            )
            results.append(result)

    if not results:
        raise RuntimeError("All parallel Drake IPOPT workers failed or returned no valid trajectory.")

    best_result = min(
        results,
        key=lambda result: (
            not np.isfinite(result["selected"]["condition_number"]),
            result["selected"]["condition_number"],
        ),
    )
    best_record = best_result["selected"]
    logger.info(
        f"Best parallel Drake IPOPT result came from worker "
        f"{best_result['worker_id']} ({best_record['source']}): "
        f"condition_number={best_record['condition_number']}; "
        f"objective={best_record['metrics'].get('objective')}"
    )

    if experiment_dir is not None:
        save_traj_csv(
            experiment_dir,
            "best_condition",
            best_record["t"],
            best_record["q"],
            best_record["dq"],
            best_record["ddq"],
        )

    if not args.no_plot:
        vis_compare_seqs(
            [best_record["t"], best_record["t"], best_record["t"]],
            [best_record["q"], best_record["dq"], best_record["ddq"]],
            ["q", "qd", "qdd"],
            ["time"],
        )
        plt.show()


def main():
    os.chdir(Path(__file__).resolve().parent)

    parser = argparse.ArgumentParser(
        description=(
            "Excitation trajectory optimizer using Drake MathematicalProgram "
            "and Drake IpoptSolver"
        )
    )
    parser.add_argument("--robot", type=str, default="fr3")
    parser.add_argument("--fourier_order", type=int, default=5)
    parser.add_argument("--fourier_duration", type=int, default=10)
    parser.add_argument(
        "--excite_type",
        type=str,
        choices=("cond", "condFriction"),
        default="cond",
    )
    parser.add_argument(
        "--friction_model",
        type=str,
        choices=("symmetric", "asymmetric"),
        default="symmetric",
    )
    parser.add_argument("--objective_lambda", type=float, default=1e-6)
    parser.add_argument("--eig_eps", type=float, default=1e-9)
    parser.add_argument("--ipopt_max_iter", type=int, default=500)
    parser.add_argument("--ipopt_print_level", type=int, default=5)
    parser.add_argument(
        "--ipopt_hessian_approximation",
        type=str,
        choices=("limited-memory", "exact"),
        default="limited-memory",
    )
    parser.add_argument("--log_condition_every", type=int, default=5)
    parser.add_argument("--best_condition_check_every", type=int, default=5)
    parser.add_argument("--initial_best_condition_number", type=float, default=10000000.0)
    parser.add_argument("--early_stop_constraint_tol", type=float, default=1e-6)
    parser.add_argument("--camera_collision_stride", type=int, default=2)
    parser.add_argument("--validation_stride", type=int, default=1)
    parser.add_argument("--drake_min_distance", type=float, default=0.02)
    parser.add_argument("--drake_robot_sphere_radius", type=float, default=0.05)
    parser.add_argument("--drake_robot_link_samples", type=int, default=5)
    parser.add_argument("--drake_camera_chamfer_radius", type=float, default=0.02)
    parser.add_argument("--link_y_lower", type=float, default=-0.5)
    parser.add_argument("--link_y_upper", type=float, default=0.4)
    parser.add_argument("--disable_link_y_bounds", action="store_true")
    parser.add_argument(
        "--drake_xy_prism_height",
        type=float,
        default=None,
        help=(
            "Optional full z height for XY-prism camera obstacles. Defaults to "
            "physical camera-box height."
        ),
    )
    parser.add_argument("--drake_physical_camera_height", action="store_true")
    parser.add_argument("--no_identifiable_column_reduction", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./saves")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Number of independent IPOPT multi-start workers. Values greater than 1 "
            "run separate optimizations in parallel and keep the best valid result."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base random seed. Worker i uses seed + i.",
    )
    args = parser.parse_args()

    experiment_dir = None
    if not args.no_save:
        timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
        experiment_dir = (
            Path(args.save_dir)
            / f"{args.excite_type}_ipopt_drake_solver_{args.robot}_{timestamp}"
        )
        safe_mkdir(experiment_dir)
        logger.info(f"Saving this experiment under {experiment_dir}")

    if args.num_workers > 1:
        run_parallel_starts(args, experiment_dir)
        return

    run_single_start(
        vars(args),
        worker_id=None,
        save_outputs=True,
        experiment_dir_str=str(experiment_dir) if experiment_dir is not None else None,
    )


if __name__ == "__main__":
    main()
