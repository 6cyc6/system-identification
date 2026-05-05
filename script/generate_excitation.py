import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from datetime import datetime
from time import perf_counter
from scipy.optimize import minimize, NonlinearConstraint
from system_identification.excitation_generator_new import (
    camera_box_clearance,
    obtain_valid_traj_param,
    obtain_fourier_traj,
    is_traj_valid,
)
from system_identification.excitation_optimization import params2cond, constraints, params2coverage, params2condFriction
from system_identification.inertia_model import InertiaModel
from system_identification.my_utils.path_utils import safe_mkdir
from system_identification.utils import retrieve_robot_config, vis_compare_seqs

CAMERA_BOX_MIN_CLEARANCE = 0.0


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


def flat_params_to_traj(flat_params, fourier_config, robot_config):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]
    params = flat_params.reshape(2, njoints, order)
    params = np.transpose(params, (0, 2, 1))
    return obtain_fourier_traj(params, fourier_config, robot_config)


def make_camera_box_constraint(fourier_config, robot_config, optimizer, stride):
    def clearance_from_params(flat_params):
        _, q, _, _ = flat_params_to_traj(flat_params, fourier_config, robot_config)
        return camera_box_clearance(q, stride=stride)

    if optimizer == "SLSQP":
        return {"type": "ineq", "fun": clearance_from_params}

    return NonlinearConstraint(
        clearance_from_params,
        CAMERA_BOX_MIN_CLEARANCE,
        np.inf,
        keep_feasible=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulator")
    # parser.add_argument("--excite_type", type=str, default="condFriction")
    parser.add_argument(
        "--excite_type",
        type=str,
        choices=("cond", "coverage", "condFriction"),
        default="cond",
    )
    # parser.add_argument("--robot", type=str, default="iiwas")
    parser.add_argument("--robot", type=str, default="fr3")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("SLSQP", "trust-constr"),
        default="trust-constr",
    )
    parser.add_argument("--fourier_order", type=int, default=5)
    parser.add_argument("--fourier_duration", type=int, default=10)
    parser.add_argument(
        "--friction_model",
        type=str,
        choices=("symmetric", "asymmetric"),
        default="symmetric",
    )
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--camera_collision_stride", type=int, default=5)
    args = parser.parse_args()

    # setup gym env
    robot_config = retrieve_robot_config(args.robot)
    njoints = robot_config["njoints"]
    friction_model = args.friction_model

    # ===================================================================================
    # Generate exciting trajectories
    # ===================================================================================
    # parameterize trajectories as fourier series and initialize the parameters
    fourier_config = {"order": args.fourier_order, "duration": args.fourier_duration}
    while True:
        t, qs, qds, qdds, init_params = obtain_valid_traj_param(
            fourier_config, robot_config
        )
        init_camera_clearance = camera_box_clearance(qs)
        if init_camera_clearance >= CAMERA_BOX_MIN_CLEARANCE:
            break
        logger.info(
            f"Discarding initial trajectory with camera box clearance {init_camera_clearance}"
        )
    vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
    plt.show()

    # ===================================================================================
    # Optimize exciting trajectories
    # ===================================================================================
    default_maxiter = 300 if args.excite_type == "coverage" else 30
    maxiter = args.maxiter if args.maxiter is not None else default_maxiter
    inertia_model = None
    if args.excite_type == "cond":
        loss = params2cond
        inertia_model = InertiaModel(args.robot)
        optimizer_args = {
            "fourier_config": fourier_config,
            "robot_config": robot_config,
            "sysID": inertia_model,
        }
        optimizer_input_args = (fourier_config, robot_config, inertia_model)
        optimize_options = {"maxiter": maxiter, "disp": True}
    elif args.excite_type == "coverage":
        loss = params2coverage
        optimizer_args = {"fourier_config": fourier_config, "robot_config": robot_config}
        optimizer_input_args = (fourier_config, robot_config)
        optimize_options = {"maxiter": maxiter, "disp": True}
    elif args.excite_type == "condFriction":
        loss = params2condFriction
        inertia_model = InertiaModel(args.robot)
        optimizer_args = {
            "fourier_config": fourier_config,
            "robot_config": robot_config,
            "sysID": inertia_model,
            "friction_model": friction_model,
        }
        optimizer_input_args = (
            fourier_config,
            robot_config,
            inertia_model,
            friction_model,
        )
        optimize_options = {"maxiter": maxiter, "disp": True}
    else:
        raise ValueError(f"Invalid excitation type: {args.excite_type}")

    init_params = np.transpose(init_params, (0, 2, 1)).ravel()
    cons = constraints(init_params, fourier_config, robot_config, args.optimizer)
    camera_box_constraint = make_camera_box_constraint(
        fourier_config,
        robot_config,
        args.optimizer,
        args.camera_collision_stride,
    )
    cons = (*cons, camera_box_constraint)
    loss_init = loss(init_params, **optimizer_args)
    logger.info(
        f"Loss before optimization: {loss_init}"
    )
    time1 = perf_counter()
    res = minimize(
        loss,
        init_params,
        args=optimizer_input_args,
        method=args.optimizer,
        constraints=cons,
        options=optimize_options,
        tol=1e-9,
    )
    time2 = perf_counter()
    logger.info(f"Time cost: {time2 - time1}")
    logger.info(
        f"Loss before optimization {loss_init}; Loss after optimization: {res.fun}"
    )
    # generate joint trajectories parametrized by fourier basis. Since the
    # params are flatten into (2 x order x njoints) to constuct the constraints,
    # here we need to reshape and transpose it to recover the original shape
    t, qs, qds, qdds = flat_params_to_traj(res.x, fourier_config, robot_config)

    valid_joint_limits = is_traj_valid(qs, qds, qdds, robot_config)
    camera_clearance = camera_box_clearance(qs)
    valid_with_camera_boxes = (
        valid_joint_limits and camera_clearance >= CAMERA_BOX_MIN_CLEARANCE
    )
    logger.info(
        f"Trajectories validity with camera box checks: {valid_with_camera_boxes}; "
        f"camera clearance: {camera_clearance}"
    )
    if not res.success:
        if valid_with_camera_boxes:
            logger.warning(
                f"Optimization stopped before convergence ({res.message}), "
                "but the final trajectory passed joint and camera-box validation."
            )
        else:
            logger.warning(f"Optimization did not converge: {res.message}")
    vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
    plt.show()

    if args.excite_type == "cond":
        logger.info(f"condition number: {res.fun}")
    elif args.excite_type == "condFriction":
        logger.info(f"condition number: {res.fun}")

    # save trajectories
    if valid_with_camera_boxes:
        save_traj_csv("./saves", args.robot, args.excite_type, t, qs, qds, qdds)
