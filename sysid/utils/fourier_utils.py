import numpy as np


def flatten_fourier_params(params):
    return np.transpose(params, (0, 2, 1)).ravel()


def unflatten_fourier_params(flat_params, fourier_config, robot_config):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]
    params = np.asarray(flat_params).reshape(2, njoints, order)
    return np.transpose(params, (0, 2, 1))


def flat_params_to_traj(flat_params, fourier_config, robot_config):
    from ..traj_generator.excitation_generator import obtain_fourier_traj

    params = unflatten_fourier_params(flat_params, fourier_config, robot_config)
    return obtain_fourier_traj(params, fourier_config, robot_config)
