import numpy as np
from system_identification.utils import *
from system_identification.qp_solver import *
from system_identification.overparam_solver import *
from system_identification.inertia_model import InertiaModel
from system_identification.friction_model import *
from scipy.signal import filtfilt, butter
from system_identification.filtering_data import cal_d

def build_solver(models_config, solver_config):
    # Setup solver
    solver = eval(solver_config["solver_name"])(
        models_config, solver_config
    )
    return solver

def bandpass_filter(signal):
    b_v, a_v = butter(3, 20, btype="lowpass", fs=1000)
    signal_filtered = filtfilt(b_v, a_v, signal, axis=0)
    return signal_filtered

def visualize_result(viz_data, train=True):
    t, q_f, dq_f, ddq_f, tau_f, q_m, dq_m, ddq_m, tau_m, des_pos, des_vel, des_acc, tau_cmd, fd_dq = viz_data

    # prediction and visualize the results
    tau = dynamics_prediction(models, q_f, dq_f, ddq_f).reshape(-1, inertia_model.njoints)
    if train:
        mode = "train"
    else:
        mode = "evaluate"
    vis_compare_seqs([t, t], [tau_m, tau], [f"measured", "RBD + friction"],
                     ["time: s"])
    vis_compare_seqs([t], [tau_m - tau], [f"measured - RBD + friction"],
                     ["time: s"])
    plt.title(f"Data from: {mode}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="System Identification")
    parser.add_argument("--alg", type=str, default="lmi")
    parser.add_argument("--robot_name", type=str, default="iiwas14_cad") # iiwas14_cad or threelink_robot
    parser.add_argument("--load_model", type=str, default="")
    args = parser.parse_args()

    # setup gym env
    robot_config = retrieve_robot_config(args.robot_name)
    njoints = robot_config["njoints"]

    # ===================================================================================
    # Identifying inertia parameters
    # ===================================================================================
    # define training set and validate set
    dset_path = f"iiwas14_cad_10trajs.pkl"
    root_dir = "./pkls/"

    # build dataset from saved pkl file
    dset = Dataset(dset_path, root_dir)
    logger.info(f"Number of datapoints in the dataset: {dset.num_datapoints}")
    t, q_f, dq_f, ddq_f, tau_f, q_m, dq_m, ddq_m, tau_m, des_pos, des_vel, des_acc, tau_cmd = dset.get_train_data()
    fd_dq = cal_d(q_m, t)

    # generate visualize data
    viz_data = [t, q_f, dq_f, ddq_f, tau_f, q_m, dq_m, ddq_m, tau_m, des_pos, des_vel, des_acc, fd_dq]

    # init manual crba solver
    urdf_filename = f"{args.robot_name}.urdf"
    urdf_filename = find_path(urdf_filename, "../robot_description")
    logger.info(f"Number of trajectories for training: {dset.num_traj_training}, for testing: {dset.num_traj_testing}")
    logger.info(f"Number of samples for training: {dset.num_traindata}, for testing: {dset.num_testdata}")
    taus = np.hstack(tau_f)

    # setup configurations for models, models include both inertia parameters and residual
    if args.alg == "lmi":
        models_config = {
            "inertia": "InertiaModel",
            "residual": [] #["AsymMatlabFModel"],
        }
        solver_config = {
            "robot_name": args.robot_name,
            "solver_name": "LmiQP",
            "loss": "loss",
            "regularizer": ["bregman_inertia_param_regularizer"],
            "constraint": "pseudoinertia_totalmass", # "densityrealize",
            "omega_pi": 1e-1,
            "num_solver": "MOSEK",
        }
    elif args.alg == "logcholesky":
        models_config = {
            "inertia": "InertiaModel",
            "residual": ["CoulombViscousModel"],
        }
        solver_config = {
            "robot_name": args.robot_name,
            "solver_name": "LogCholesky",
            "loss": "loss",
            "regularizer": ["bregd_inertia_param_regularizer"],
            "constraint": "",
            "omega_pi": 1e-1,
            "num_solver": "trust-ncg",
            "using_cad_inertia": False,
        }
    else:
        raise NotImplementedError

    solver = build_solver(models_config, solver_config)
    dyn_param = solver.fit(q_f, dq_f, ddq_f, taus)
    solver.log_info()
    solver.save_param_dict(f"./traj_data/identified_{args.alg}_residuals.yml")

    # save parameters as numpy array
    residual_models = models_config["residual"]

    if len(residual_models) == 1:
        model_name = f"./models/tau_{args.alg}_{residual_models[0]}.npy"
    elif len(residual_models) == 2:
        model_name = f"./models/tau_{args.alg}_{residual_models[0]}_{residual_models[1]}.npy"
    else:
        model_name = f"./models/tau_{args.alg}.npy"
    solver.save_param(f"{model_name}")

    # load the pretrained model
    lmi_model = np.load(f"{model_name}", allow_pickle=True).item()
    model_param = lmi_model["model_param"]
    robot_name = "iiwas14_cad"
    inertia_model = InertiaModel(robot_name)
    
    # if there is residual model, load the residual model
    if len(models_config["residual"]) >= 1:
        friction_model_config = {
        "F_brk": 25,
        "F_c": 20,
        "v_brk": 0.1,
        "damping": inertia_model.damping,
        "coulomb": inertia_model.coulomb,
        "njoints": inertia_model.njoints,
        "bias": 0.0
        }
        friction_model = eval(models_config["residual"][0])(friction_model_config)    
        load_model_params(model_param, inertia_model, friction_model)
        models = [inertia_model, friction_model]
    else:
        load_model_params(model_param, inertia_model)
        models = [inertia_model]

    if dset.num_traj_training == 1:
        visualize_result(viz_data, True)

    # ===================================================================================
    # Evaluate the model
    # ===================================================================================
    test_set_id = dset.test_set_id
    t, q_f, dq_f, ddq_f, tau_f, q_m, dq_m, ddq_m, tau_m, des_pos, des_vel, des_acc, tau_cmd = dset.retrieve_dset_data_byindex(test_set_id[0])
    fd_dq = cal_d(q_m, t)

    # data for visualization
    viz_data = [t, q_f, dq_f, ddq_f, tau_f, q_m, dq_m, ddq_m, tau_m, des_pos, des_vel, des_acc, tau_cmd, fd_dq]
    visualize_result(viz_data, False)

    plt.show()