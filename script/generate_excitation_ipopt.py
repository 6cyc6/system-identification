import argparse
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from loguru import logger

from system_identification.excitation_generator_new import (
    camera_box_clearance,
    is_traj_valid,
    obtain_valid_traj_param,
)
from system_identification.fourier_utils import flat_params_to_traj, flatten_fourier_params
from system_identification.inertia_model import InertiaModel
from system_identification.ipopt_solver import (
    IpoptExcitationSolver,
    ipopt_status_is_acceptable,
)
from system_identification.my_utils.path_utils import safe_mkdir
from system_identification.utils import retrieve_robot_config, vis_compare_seqs


def save_traj_csv(save_dir, robot, excite_type, t, q, dq, ddq):
    safe_mkdir(save_dir)
    timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
    csv_path = f"{save_dir}/{excite_type}_{robot}_{timestamp}.csv"
    data = np.column_stack([t, q, dq, ddq])
    njoints = q.shape[1]
    header = (
        ["t"]
        + [f"q_{i}" for i in range(njoints)]
        + [f"dq_{i}" for i in range(njoints)]
        + [f"ddq_{i}" for i in range(njoints)]
    )
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")
    logger.info(f"Saved trajectory CSV to {csv_path}")
    return csv_path


def main():
    os.chdir(Path(__file__).resolve().parent)

    parser = argparse.ArgumentParser(description="IPOPT excitation trajectory optimizer")
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
    parser.add_argument("--ipopt_max_iter", type=int, default=300)
    parser.add_argument("--ipopt_print_level", type=int, default=5)
    parser.add_argument(
        "--ipopt_hessian_approximation",
        type=str,
        choices=("limited-memory", "exact"),
        default="limited-memory",
    )
    parser.add_argument("--log_condition_every", type=int, default=5)
    parser.add_argument("--early_stop_objective", type=float, default=None)
    parser.add_argument("--early_stop_constraint_tol", type=float, default=1e-6)
    parser.add_argument("--camera_collision_stride", type=int, default=5)
    parser.add_argument("--no_identifiable_column_reduction", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./saves")
    args = parser.parse_args()

    robot_config = retrieve_robot_config(args.robot)
    fourier_config = {"order": args.fourier_order, "duration": args.fourier_duration}

    while True:
        t, qs, qds, qdds, init_params = obtain_valid_traj_param(
            fourier_config,
            robot_config,
        )
        init_camera_clearance = camera_box_clearance(qs)
        if init_camera_clearance >= 0.0:
            break
        logger.info(
            f"Discarding initial trajectory with camera box clearance {init_camera_clearance}"
        )

    if not args.no_plot:
        vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
        plt.show()

    inertia_model = InertiaModel(args.robot)
    solver = IpoptExcitationSolver(
        fourier_config=fourier_config,
        robot_config=robot_config,
        inertia_model=inertia_model,
        friction_model=args.friction_model if args.excite_type == "condFriction" else None,
        eig_eps=args.eig_eps,
        min_eig_weight=args.objective_lambda,
        camera_collision_stride=args.camera_collision_stride,
        ipopt_max_iter=args.ipopt_max_iter,
        ipopt_print_level=args.ipopt_print_level,
        ipopt_hessian_approximation=args.ipopt_hessian_approximation,
        log_condition_every=args.log_condition_every,
        early_stop_objective=args.early_stop_objective,
        early_stop_constraint_tol=args.early_stop_constraint_tol,
        use_identifiable_columns=not args.no_identifiable_column_reduction,
    )

    init_params_flat = flatten_fourier_params(init_params)
    time1 = perf_counter()
    result = solver.solve(init_params_flat)
    time2 = perf_counter()
    ipopt_status = result["stats"].get("return_status")
    logger.info(f"IPOPT time cost: {time2 - time1}")
    logger.info(f"IPOPT recomputed objective after optimization: {result['f']}")
    logger.info(f"IPOPT solver-reported objective: {result['solver_f']}")
    logger.info(f"IPOPT return status: {ipopt_status}")
    logger.info(f"IPOPT stopped by early-stop criterion: {result['stopped_early']}")

    t, qs, qds, qdds = flat_params_to_traj(
        result["x"],
        fourier_config,
        robot_config,
    )
    valid_joint_limits = is_traj_valid(qs, qds, qdds, robot_config)
    camera_clearance = camera_box_clearance(qs)
    valid = valid_joint_limits and camera_clearance >= 0.0
    ipopt_status_ok = ipopt_status_is_acceptable(ipopt_status) or (
        ipopt_status == "User_Requested_Stop" and result["stopped_early"]
    )
    acceptable_result = (
        ipopt_status_ok
        and np.isfinite(result["f"])
        and valid
    )
    logger.info(
        f"Trajectories validity with camera box checks: {valid}; "
        f"camera clearance: {camera_clearance}"
    )
    if not acceptable_result:
        logger.warning("Skipping save because IPOPT did not return an acceptable result.")

    if not args.no_plot:
        vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
        plt.show()

    if acceptable_result and not args.no_save:
        save_traj_csv(
            args.save_dir,
            args.robot,
            f"{args.excite_type}_ipopt",
            t,
            qs,
            qds,
            qdds,
        )


if __name__ == "__main__":
    main()
