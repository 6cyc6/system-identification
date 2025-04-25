from system_identification.excitation_generator import *
from scipy.optimize import minimize
from system_identification.excitation_optimization import params2cond, constraints, params2coverage, params2condFriction, generateSymFrictionReg, generateAsymFrictionReg
from system_identification.utils import *
from system_identification.inertia_model import *
from datetime import datetime

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulator")
    parser.add_argument("--excite_type", type=str, default="condFriction")
    parser.add_argument("--robot", type=str, default="iiwas")
    parser.add_argument("--optimizer", type=str, default="trust-constr") # SLSQP, trust-constr
    parser.add_argument("--fourier_order", type=int, default=5)
    parser.add_argument("--fourier_duration", type=int, default=80)
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
    vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
    plt.show()
    
    # ===================================================================================
    # load the trajectories from the file
    # ===================================================================================
    # file_path = f"./traj_data/init_params_{args.fourier_duration}.npy"
    # file_ = np.load(file_path, allow_pickle=True).item()
    # t, qs, qds, qdds, init_params = file_["t"], file_["q"], file_["dq"], file_["ddq"], file_["init_params"]
    # vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
    # plt.show()

    # TODO: select loss and setting up parameters
    if args.excite_type == "cond":
        loss = params2cond
        inertia_model = InertiaModel(args.robot)
        optimizer_args = {"fourier_config": fourier_config, "robot_config": robot_config, "sysID": inertia_model}
        optimizer_input_args = (fourier_config, robot_config, inertia_model)
        optimize_options = {"maxiter": 30, "disp": True}
    elif args.excite_type == "coverage":
        loss = params2coverage
        optimize_options = {"maxiter": 100, "disp": True}
        optimizer_args = {"fourier_config": fourier_config, "robot_config": robot_config}
        optimizer_input_args = (fourier_config, robot_config)
    elif args.excite_type == "condFriction":
        loss = params2condFriction
        inertia_model = InertiaModel(args.robot)
        optimizer_args = {"fourier_config": fourier_config, "robot_config": robot_config, "sysID": inertia_model, "friction_model": friction_model}
        optimizer_input_args = (fourier_config, robot_config, inertia_model, friction_model)
        optimize_options = {"maxiter": 30, "disp": True}

    # TODO: starting optimization
    params_shape = init_params.shape
    init_params = np.transpose(init_params, (0, 2, 1))
    init_params = init_params.flatten()
    cons = constraints(init_params, fourier_config, robot_config, args.optimizer)
    loss_init = loss(init_params, **optimizer_args)
    logger.info(
        f"Loss before optimization: {loss_init}"
    )
    import time
    time1 = time.time()
    res = minimize(
        loss,
        init_params,
        args=optimizer_input_args,
        method=args.optimizer,
        constraints=cons,
        options=optimize_options,
        tol=1e-9
    )
    time2 = time.time()
    logger.info(f"Time cost: {time2 - time1}")
    logger.info(
        f"Loss before optimization {loss_init}; Loss after optimization: {loss(res.x, **optimizer_args)}"
    )
    # generate joint trajectories parametrized by fourier basis. Since the
    # params are flatten into (2 x order x njoints) to constuct the constraints,
    # here we need to reshape and transpose it to recover the original shape
    params = res.x.reshape(2, njoints, fourier_config["order"])
    params = np.transpose(params, (0, 2, 1))
    t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config)
    
    save = is_traj_valid(qs, qds, qdds, robot_config)
    logger.info(
        f"Trajectories vidality: {save}"
    )
    vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
    plt.show()

    if args.excite_type == "cond":
        # compute regressor and do dimensionality reduction
        regressor = inertia_model.regressor(qs, qds, qdds)
        reduced_R, cond_num = QR_dim_reduction(regressor)
        logger.info(f"condition number: {cond_num}")
    elif args.excite_type == "condFriction":
        regressor = inertia_model.regressor(qs, qds, qdds)
        if friction_model == "symmetric":
            regressorFriction = generateSymFrictionReg(qds)
        elif friction_model == "asymmetric":
            regressorFriction = generateAsymFrictionReg(qds)
        regressor = np.hstack((regressor, regressorFriction))
        reduced_R, cond_num = QR_dim_reduction(regressor)
        logger.info(f"condition number: {cond_num}")

    # save trajectories
    if save:
        _time = datetime.now().strftime("%d%m%Y%H%M%S")
        _dir = f"./traj_data"
        np.save(
            f"{_dir}/{args.excite_type}_{args.robot}_{_time}",
            {"t": t, "q": qs, "dq": qds, "ddq": qdds},
            allow_pickle=True
        )
