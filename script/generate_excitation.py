import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from datetime import datetime
from scipy.optimize import minimize
from system_identification.excitation_generator_new import obtain_valid_traj_param, obtain_fourier_traj, is_traj_valid
from system_identification.excitation_optimization import params2cond, constraints, params2coverage, params2condFriction, generateSymFrictionReg, generateAsymFrictionReg
from system_identification.inertia_model import InertiaModel
from system_identification.my_utils.path_utils import safe_mkdir
from system_identification.utils import retrieve_robot_config, vis_compare_seqs, QR_dim_reduction


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulator")
    # parser.add_argument("--excite_type", type=str, default="condFriction")
    parser.add_argument("--excite_type", type=str, default="cond")
    # parser.add_argument("--robot", type=str, default="iiwas")
    parser.add_argument("--robot", type=str, default="fr3")
    parser.add_argument("--optimizer", type=str, default="trust-constr") # SLSQP, trust-constr
    parser.add_argument("--fourier_order", type=int, default=5)
    parser.add_argument("--fourier_duration", type=int, default=10)
    parser.add_argument("--friction_model", type=str, default="symmetric") # symmetric, asymmetric
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
    t, qs, qds, qdds, init_params = obtain_valid_traj_param(
        fourier_config, robot_config
    )
    save_traj_csv("./saves", args.robot, args.excite_type, t, qs, qds, qdds)
    vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
    plt.show()
    
    
