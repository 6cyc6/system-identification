"""Conversions, constraints, and regularizers for rigid-body inertia parameters."""

import numpy as np
import scipy.linalg

from ..utils.io_utils import merge_dict
from ..utils.robot_config import _load_pin_model_for_robot


def retrieve_geo_fromCAD(robot_name="fr3"):
    """Return link masses, center-of-mass levers, and inertias from the URDF."""
    model = _load_pin_model_for_robot(robot_name)
    njoints = model.njoints - 1
    system_inertia = model.inertias.tolist()[1 : 1 + njoints]
    masses = [inertia.mass for inertia in system_inertia]
    levers = [inertia.lever for inertia in system_inertia]
    inertias = [inertia.inertia for inertia in system_inertia]
    return masses, levers, inertias


def skew_symmetric(vector):
    vector = np.asarray(vector)
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )


def I(vector):
    """Convert the six unique inertia entries to a symmetric 3x3 matrix."""
    ixx, ixy, iyy, ixz, iyz, izz = vector
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inv_sigmoid(x):
    return np.log(x / (1 - x))


def pinertiaToVec(pseudo_inertia):
    """Convert a 4x4 pseudo-inertia matrix to ten dynamic parameters."""
    mass = pseudo_inertia[3, 3]
    first_moment = pseudo_inertia[3, :3]
    sigma = pseudo_inertia[:3, :3]
    inertia = np.trace(sigma) * np.eye(3) - sigma
    ixx, iyy, izz, iyz, ixz, ixy = (
        inertia[0, 0],
        inertia[1, 1],
        inertia[2, 2],
        inertia[2, 1],
        inertia[2, 0],
        inertia[1, 0],
    )
    return np.array(
        [
            mass,
            first_moment[0],
            first_moment[1],
            first_moment[2],
            ixx,
            ixy,
            iyy,
            ixz,
            iyz,
            izz,
        ]
    )


def _body_inertia_and_center(dynamic_parameters):
    inertia = np.array(
        [
            [dynamic_parameters[4], dynamic_parameters[5], dynamic_parameters[7]],
            [dynamic_parameters[5], dynamic_parameters[6], dynamic_parameters[8]],
            [dynamic_parameters[7], dynamic_parameters[8], dynamic_parameters[9]],
        ]
    )
    mass = dynamic_parameters[0]
    first_moment = np.asarray(dynamic_parameters[1:4])
    center = first_moment / mass
    return inertia, mass, first_moment, center


def inertiaVecToinertiasCoM(dynamic_parameters):
    """Return mass, center of mass, and inertia expressed at the CoM."""
    inertia, mass, _, center = _body_inertia_and_center(dynamic_parameters)
    skew_center = skew_symmetric(center)
    inertia_com = inertia - mass * skew_center @ skew_center.T
    return mass, center, inertia_com


def inertiaVecToPinertia(dynamic_parameters):
    """Convert ten dynamic parameters to a 4x4 pseudo-inertia matrix."""
    inertia, mass, first_moment, _ = _body_inertia_and_center(dynamic_parameters)
    sigma = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    upper = np.hstack([sigma, first_moment.reshape(3, 1)])
    lower = np.hstack([first_moment, mass])
    return np.vstack([upper, lower])


def inertiaVecToIcQs(dynamic_parameters):
    """Return the CoM inertia and normalized covariance matrix Qs."""
    mass, _, inertia_com = inertiaVecToinertiasCoM(dynamic_parameters)
    sigma_com = 0.5 * np.trace(inertia_com) * np.eye(3) - inertia_com
    return inertia_com, sigma_com / mass


def inertiaVecToXsQs(dynamic_parameters):
    """Return the center of mass Xs and normalized covariance matrix Qs."""
    mass, center, inertia_com = inertiaVecToinertiasCoM(dynamic_parameters)
    sigma_com = 0.5 * np.trace(inertia_com) * np.eye(3) - inertia_com
    return center, sigma_com / mass


def inertiaVecToQ(dynamic_parameters):
    first_moment = np.asarray(dynamic_parameters[1:4])
    mass = dynamic_parameters[0]
    center = (first_moment / mass).reshape(-1, 1)
    _, qs = inertiaVecToIcQs(dynamic_parameters)
    qs_inverse = np.linalg.inv(qs)
    qs_inverse_transpose_center = qs_inverse.T @ center
    center_term = center.T @ qs_inverse @ center
    first_columns = np.hstack([-qs_inverse, qs_inverse_transpose_center])
    last_column = np.hstack(
        [qs_inverse_transpose_center.T, 1 - center_term]
    )
    return np.vstack([first_columns, last_column])


def param2dict(inertia_param, friction_param, njoints):
    """Convert flat inertia/friction parameters to the per-link YAML structure."""
    inertia_param_dict = {}
    friction_param_dict = {}
    viscous_params = friction_param[:njoints].tolist()
    coulomb_params = friction_param[njoints:].tolist()

    for joint_index in range(njoints):
        link = f"link_{joint_index + 1}"
        friction_param_dict[link] = {
            "damping": viscous_params[joint_index],
            "friction": coulomb_params[joint_index],
        }

    for joint_index in range(njoints):
        start, end = joint_index * 10, (joint_index + 1) * 10
        parameters = inertia_param[start:end]
        mass = float(parameters[0])
        center = [float(value) / mass for value in parameters[1:4]]
        body_inertia = I([float(value) for value in parameters[4:10]])
        skew_center = skew_symmetric(center)
        inertia_com = body_inertia - mass * (skew_center @ skew_center.T)
        inertia_param_dict[f"link_{joint_index + 1}"] = {
            "mass": mass,
            "xyz": center,
            "ixx": float(inertia_com[0, 0]),
            "ixy": float(inertia_com[0, 1]),
            "iyy": float(inertia_com[1, 1]),
            "ixz": float(inertia_com[0, 2]),
            "iyz": float(inertia_com[1, 2]),
            "izz": float(inertia_com[2, 2]),
        }

    return merge_dict([inertia_param_dict, friction_param_dict])


def pseudoinertia_i(dynamic_parameters):
    """Build one link's pseudo-inertia matrix from dynamic parameters."""
    off_diagonal = np.array(
        [
            [0.0, dynamic_parameters[5], dynamic_parameters[7]],
            [0.0, 0.0, dynamic_parameters[8]],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    inertia = (
        np.diag(
            np.array(
                [dynamic_parameters[4], dynamic_parameters[6], dynamic_parameters[9]]
            )
        )
        + off_diagonal
        + off_diagonal.T
    )
    sigma = np.trace(inertia) / 2.0 * np.eye(3) - inertia
    pseudo_inertia = np.vstack(
        [sigma, np.reshape(dynamic_parameters[1:4], (1, 3))]
    )
    return np.hstack(
        [
            pseudo_inertia,
            np.reshape(
                np.array(
                    [
                        dynamic_parameters[1],
                        dynamic_parameters[2],
                        dynamic_parameters[3],
                        dynamic_parameters[0],
                    ]
                ),
                (4, 1),
            ),
        ]
    )


def logCholeskyVecToParams(dynamic_theta):
    """Map ten log-Cholesky values to physical dynamic parameters."""
    d1, d2, d3, s12, s13, s23, t1, t2, t3, alpha = dynamic_theta
    upper = np.exp(alpha) * np.array(
        [
            [np.exp(d1), s12, s13, t1],
            [0, np.exp(d2), s23, t2],
            [0, 0, np.exp(d3), t3],
            [0, 0, 0, 1],
        ]
    )
    return pinertiaToVec(upper.T @ upper)


def overparam2pi(dynamic_theta):
    """Map log-Cholesky parameters for all links to dynamic parameters."""
    parameters = []
    for link_index in range(len(dynamic_theta) // 10):
        link_theta = dynamic_theta[link_index * 10 : (link_index + 1) * 10]
        parameters.extend(logCholeskyVecToParams(link_theta))
    return parameters


def pi2overparam(dynamic_parameters):
    """Map physical dynamic parameters to the log-Cholesky representation."""
    dynamic_theta = []
    for link_index in range(len(dynamic_parameters) // 10):
        link_parameters = dynamic_parameters[
            link_index * 10 : (link_index + 1) * 10
        ]
        upper = scipy.linalg.cholesky(inertiaVecToPinertia(link_parameters))
        alpha = np.log(upper[3, 3])
        normalized = upper / upper[3, 3]
        d1, d2, d3 = (
            np.log(normalized[0, 0]),
            np.log(normalized[1, 1]),
            np.log(normalized[2, 2]),
        )
        dynamic_theta.extend(
            [
                d1,
                d2,
                d3,
                normalized[0, 1],
                normalized[0, 2],
                normalized[1, 2],
                normalized[0, 3],
                normalized[1, 3],
                normalized[2, 3],
                alpha,
            ]
        )
    return np.array(dynamic_theta)


def bregman_regularizer(dynamic_parameters, prior_parameters):
    """Sum the log-det Bregman divergence over all robot links."""
    regularizer = 0.0
    for link_index in range(len(dynamic_parameters) // 10):
        start, end = link_index * 10, (link_index + 1) * 10
        pseudo_inertia = inertiaVecToPinertia(dynamic_parameters[start:end])
        prior_pseudo_inertia = inertiaVecToPinertia(prior_parameters[start:end])
        regularizer += bregman_i(pseudo_inertia, prior_pseudo_inertia)
    return regularizer


def bregman_i(pseudo_inertia, prior_pseudo_inertia):
    prior_inverse_times_inertia = np.linalg.inv(prior_pseudo_inertia) @ pseudo_inertia
    return (
        -np.log(np.linalg.det(pseudo_inertia))
        + np.log(np.linalg.det(prior_pseudo_inertia))
        + np.trace(prior_inverse_times_inertia)
        - len(pseudo_inertia)
    )


def retrieve_pi_prior_inertia():
    """Load the FR3 URDF prior as a flat dynamic-parameter vector."""
    from ..model.inertia_model import InertiaModel

    return InertiaModel("fr3").ref_param


def check_CoM_bounded(parameters, prior_parameters):
    """Check positive definiteness of each link's CoM bounding matrix."""
    for link_index in range(len(parameters) // 10):
        start, end = link_index * 10, (link_index + 1) * 10
        link_parameters = parameters[start:end]
        prior_link_parameters = prior_parameters[start:end]
        mass = link_parameters[0]
        first_moment = link_parameters[1:4]
        center, qs = inertiaVecToXsQs(prior_link_parameters)
        offset = first_moment - mass * center
        bounding_matrix = np.vstack(
            [
                np.hstack([mass, offset]),
                np.hstack([offset.reshape(3, 1), mass * qs]),
            ]
        )
        if not np.all(np.linalg.eigvals(bounding_matrix) > 0):
            return False
    return True
