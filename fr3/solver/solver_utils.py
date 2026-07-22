import numpy as np
import scipy.linalg

from ..utils.fourier_utils import flat_params_to_traj
from ..utils.utils import feature2regressor


def generateSymFrictionReg(dq, vbrk=0.001):
    nq = dq.shape[1]
    ndata = dq.shape[0]
    vst = vbrk * np.sqrt(2)
    vcoul = vbrk * 2
    feature_Fbrk_mins_Fc = np.sqrt(2 * np.e) * np.exp(-((dq / vst) ** 2)) * dq / vst
    feature_Fc = np.tanh(dq / vcoul)
    feature_viscous = dq
    features = [feature_Fbrk_mins_Fc, feature_Fc, feature_viscous]
    Y_reg = feature2regressor(features, ndata, nq)
    return Y_reg


def generateAsymFrictionReg(dq, vbrk=0.001):
    nq = dq.shape[1]
    ndata = dq.shape[0]
    vst = vbrk * np.sqrt(2)
    vcoul = vbrk * 2
    feature_Fbrk_mins_Fc_pos = (
        np.sqrt(2 * np.e) * np.exp(-((dq / vst) ** 2)) * dq / vst * (dq > 0)
    )
    feature_Fc_pos = np.tanh(dq / vcoul) * (dq > 0)
    feature_viscous_pos = dq * (dq > 0)
    feature_Fbrk_mins_Fc_neg = (
        np.sqrt(2 * np.e) * np.exp(-((dq / vst) ** 2)) * dq / vst * (dq < 0)
    )
    feature_Fc_neg = np.tanh(dq / vcoul) * (dq < 0)
    feature_viscous_neg = dq * (dq < 0)
    features = [
        feature_Fbrk_mins_Fc_pos,
        feature_Fc_pos,
        feature_viscous_pos,
        feature_Fbrk_mins_Fc_neg,
        feature_Fc_neg,
        feature_viscous_neg,
    ]
    Y_reg = feature2regressor(features, ndata, nq)
    return Y_reg


def select_identifiable_columns(regressor, tol=None):
    _, r, pivots = scipy.linalg.qr(regressor, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        raise ValueError("Regressor has no columns.")

    if tol is None:
        tol = diag.max() * max(regressor.shape) * np.finfo(regressor.dtype).eps

    rank = int(np.sum(diag > tol))
    if rank == 0:
        raise ValueError("Regressor has no identifiable columns.")

    return np.sort(pivots[:rank])


def build_regressor(q, dq, ddq, inertia_model, friction_model=None):
    regressor = inertia_model.regressor(q, dq, ddq)
    if friction_model is None:
        return regressor
    if friction_model == "symmetric":
        return np.hstack((regressor, generateSymFrictionReg(dq)))
    if friction_model == "asymmetric":
        return np.hstack((regressor, generateAsymFrictionReg(dq)))
    raise ValueError(f"Invalid friction model: {friction_model}")


def regressor_gramian_extreme_eigs(regressor):
    gramian = regressor.T @ regressor
    gramian = 0.5 * (gramian + gramian.T)
    eigs = np.linalg.eigvalsh(gramian)
    return max(float(eigs[0]), 0.0), max(float(eigs[-1]), 0.0)


def regressor_condition_metrics(regressor, eig_eps=1e-9, min_eig_weight=1e-6):
    min_eig, max_eig = regressor_gramian_extreme_eigs(regressor)
    if min_eig <= 0.0:
        gramian_condition_number = np.inf
        regressor_condition_number = np.inf
    else:
        gramian_condition_number = max_eig / min_eig
        regressor_condition_number = np.sqrt(gramian_condition_number)

    objective = (
        np.log(max_eig + eig_eps) - np.log(min_eig + eig_eps) - min_eig_weight * min_eig
    )
    return {
        "condition_number": regressor_condition_number,
        "gramian_condition_number": gramian_condition_number,
        "min_eig": min_eig,
        "max_eig": max_eig,
        "objective": objective,
    }


def build_params_regressor(
    flat_params,
    fourier_config,
    robot_config,
    inertia_model,
    friction_model=None,
    identifiable_columns=None,
):
    _, q, dq, ddq = flat_params_to_traj(flat_params, fourier_config, robot_config)
    regressor = build_regressor(q, dq, ddq, inertia_model, friction_model)
    if identifiable_columns is not None:
        regressor = regressor[:, identifiable_columns]
    return regressor


def evaluate_params_metrics(
    flat_params,
    fourier_config,
    robot_config,
    inertia_model,
    friction_model=None,
    identifiable_columns=None,
    eig_eps=1e-9,
    min_eig_weight=1e-6,
):
    regressor = build_params_regressor(
        flat_params,
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=friction_model,
        identifiable_columns=identifiable_columns,
    )
    return regressor_condition_metrics(
        regressor,
        eig_eps=eig_eps,
        min_eig_weight=min_eig_weight,
    )
