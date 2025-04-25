import numpy as np
import torch

from system_identification.abstract_solver import AbstractSolver
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from loguru import logger
import jax.numpy as jnp
import jax
from jax import jacfwd, jacrev
from jax.config import config
config.update("jax_enable_x64", True)  # enable data type of double precision

class BaseOverparam(AbstractSolver):
    def __init__(
        self,
        models_config, solver_config,
        pi_overparam_inertia_init=None,
        overparam_per_joint=10,
    ):
        super(BaseOverparam, self).__init__(
            models_config, solver_config
        )
        self.njoints = self.inertia_model.njoints
        self.omega_pi = solver_config["omega_pi"]
        self.Y = None
        self.weights = None
        self.taus = None
        self.pi = None
        self.pi_overparam_inertia_init = pi_overparam_inertia_init
        self.n_points = 0
        self.overparam_per_joint = overparam_per_joint
        self.virparam = None
        self.using_cad_inertia = solver_config["using_cad_inertia"]
        self.init_variables()
        self.init_pi_indices()
        self.init_overparam()

    def init_variables(self):
        self.num_dyn_param = 0
        ref_dyn_param = []
        for i in range(len(self.models)):
            self.num_dyn_param += self.models[i].num_param_total
            ref_dyn_param.append(self.models[i].ref_param)
        self.pi_prior = np.hstack(ref_dyn_param)
        self.J_prior = self.inertia_model.J_prior
        self.iters = 0

    def load_param(self, file_name):
        load_dict = np.load(f"{file_name}", allow_pickle=True).item()
        self.pi = load_dict["model_param"]
        self.dyn_param = load_dict["model_param"][: self.njoints * 10]
        self.sync_param()

    def save_param(self, file_name):
        model_param = self.pi
        save_dict = {"model_param": model_param, "solver_config": self.solver_config,
                     "models_config": self.models_config}
        np.save(file_name, save_dict)

    def init_overparam(self):
        raise NotImplementedError

    def init_pi_indices(self):
        self.pi_idxes = []
        start_idx = 0
        for i in range(len(self.models)):
            model = self.models[i]
            num_model_param = model.num_param_total
            end_idx = start_idx + num_model_param
            self.pi_idxes.append([start_idx, end_idx])
            start_idx = end_idx

    def generate_regressor(self, q, dq, ddq):
        regressors = []
        for model in self.models:
            regressors.append(model.regressor(q, dq, ddq))
        regressor = np.hstack(regressors)
        return regressor

    def overparam2pi(self, pi_overparam):
        raise NotImplementedError

    def sync_data(self, q, dq, ddq, taus):
        """
        Input:
            q, dq, ddq: (n_points, 7)
            taus: (n_points, )
        """
        self.Y = self.generate_regressor(q, dq, ddq)
        self.taus = taus
        self.n_points = self.Y.shape[0]
        self.n_samples = q.shape[0]
        self.weights = np.ones(self.Y.shape[0])
        self.q = q
        self.dq = dq
        self.ddq = ddq

    def sync_param(self):
        for i in range(len(self.models)):
            model = self.models[i]
            param_idx_start, param_idx_end = self.pi_idxes[i][0], self.pi_idxes[i][1]
            model.sync_param(self.pi[param_idx_start:param_idx_end])

    def inertia_param_regularizer(self):
        return (
            np.linalg.norm(
                self.pi[: self.njoints * self.inertia_model.inertia_param_per_joint]
                - self.pi_prior[
                    : self.njoints * self.inertia_model.inertia_param_per_joint
                ]
            )
            ** 2
        )

    def all_param_regularizer(self):
        return np.linalg.norm(self.pi - self.pi_prior) ** 2

    def loss(self, pi_overparam):
        pi = self.overparam2pi(pi_overparam)
        e = self.Y @ pi - self.taus
        _loss = 1 / 2 * e.T @ e / len(e)
        return _loss

    def avg_loss(self, pi_overparam):
        loss = self.loss(pi_overparam)
        return loss / self.n_samples

    def weighted_loss(self, pi_overparam):
        pi = self.overparam2pi(pi_overparam)
        residual = self.Y @ pi - self.taus
        weighted_residual_square = residual * residual * self.weights
        weighted_loss = np.sum(weighted_residual_square)
        return weighted_loss

    def avg_weighted_loss(self, pi_overparam):
        weighted_loss = self.weighted_loss(pi_overparam)
        return weighted_loss / self.n_samples

    def update_weight(self):
        residual = self.Y @ self.pi - self.taus
        E = residual.reshape((-1, self.njoints)).T
        m = E.shape[1]
        p = self.Y.shape[1]
        Lambda = np.square(np.linalg.norm(E, axis=1, keepdims=True)) / (m - p)
        return np.repeat(1 / Lambda, m, axis=1).flatten(order="F")

    def objective(self, pi_overparam):
        raise NotImplementedError

    def fit(self, q, dq, ddq, taus, n_iter=200, method="L-BFGS-B"):
        solver_config = self.solver_config
        self.sync_data(q, dq, ddq, taus)
        self.method = method

        optimize_options = {"maxiter": n_iter, "disp": True}
        res = minimize(
            self.objective,
            self.pi_overparam_init,
            args=(solver_config),
            method=method,
            options=optimize_options,
        )

        self.pi = self.overparam2pi(res.x)
        self.sync_param()
        return self.pi

    def predict(self, q, dq, ddq):
        Y_eval = self.generate_regressor(q, dq, ddq)
        dyn_param = self.pi.reshape(-1, 1)
        return Y_eval @ dyn_param

class LogCholesky(BaseOverparam):
    def __init__(
        self,
        models_config, solver_config,
        pi_overparam_inertia_init=np.zeros(70),
        overparam_per_joint=10,
    ):
        super(LogCholesky, self).__init__(
            models_config, solver_config,
            pi_overparam_inertia_init,
            overparam_per_joint,
        )

    def init_overparam(self):
        self.num_overparam = self.njoints * self.overparam_per_joint
        for i in range(len(self.residual_models)):
            self.num_overparam += self.residual_models[i].num_param_total
        self.pi_overparam_init = jnp.zeros(self.num_overparam, dtype=np.float64)
        self.pi = self.overparam2pi(self.pi_overparam_init)

    def pinertiaToVec(self, J):
        m = J[3, 3]
        h = J[3, :3]
        c = h / m
        Sigma = J[:3, :3]
        eye3 = jnp.eye(3)
        Ibar = jnp.trace(Sigma) * eye3 - Sigma
        Ixx, Iyy, Izz, Iyz, Ixz, Ixy = (
            Ibar[0][0],
            Ibar[1][1],
            Ibar[2][2],
            Ibar[2][1],
            Ibar[2][0],
            Ibar[1][0],
        )
        return jnp.array([m, h[0], h[1], h[2], Ixx, Ixy, Iyy, Ixz, Iyz, Izz])

    def inertiaVecToPinertia(self, pi):
        Ibar = jnp.array(
            [[pi[4], pi[5], pi[7]], [pi[5], pi[6], pi[8]], [pi[7], pi[8], pi[9]]]
        )
        h = jnp.array([pi[1], pi[2], pi[3]])
        m = pi[0]
        eye3 = jnp.eye(3)
        Sigma = 1 / 2 * jnp.trace(Ibar) * eye3 - Ibar
        tmp1 = jnp.hstack([Sigma, jnp.reshape(h, (3, 1))])
        tmp2 = jnp.hstack([h, m])
        Pinertia = jnp.vstack([tmp1, tmp2])
        return Pinertia

    def pseudoinertia_i(self, pi_dyn_i):
        off_diag = jnp.array(
            [[0.0, pi_dyn_i[5], pi_dyn_i[7]], [0.0, 0.0, pi_dyn_i[8]], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        I = (
            jnp.diag(jnp.array([pi_dyn_i[4], pi_dyn_i[6], pi_dyn_i[9]]))
            + off_diag
            + off_diag.T
        )
        eye3 = jnp.eye(3)
        Sigma = jnp.trace(I) / 2.0 * eye3 - I
        J = jnp.vstack([Sigma, jnp.reshape(pi_dyn_i[1:4], (1, 3))])
        J = jnp.hstack(
            [
                J,
                jnp.reshape(
                    jnp.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3], pi_dyn_i[0]]),
                    (4, 1),
                ),
            ]
        )
        return J

    def logCholeskyVecToParams(self, theta, pi_prior_i):
        d1, d2, d3, s12, s13, s23, t1, t2, t3, alpha = theta
        U = jnp.exp(alpha) * jnp.array(
            [
                [jnp.exp(d1), s12, s13, t1],
                [0, jnp.exp(d2), s23, t2],
                [0, 0, jnp.exp(d3), t3],
                [0, 0, 0, 1],
            ]
        )
        J0 = self.inertiaVecToPinertia(pi_prior_i)
        U_J0 = jnp.dot(U, J0)
        J = jnp.dot(U_J0, U.T)
        pi = self.pinertiaToVec(J)
        return pi

    def overparam2pi(self, pi_overparam):
        num_inertia_param = self.njoints * self.overparam_per_joint
        if len(self.residual_models) > 0:
            num_friction_param = (
                self.njoints * self.residual_models[0].num_coefficient_per_joint
            )

        pi_inertia = jnp.zeros(self.njoints * 10, dtype=np.float64)
        for i in range(self.njoints):
            pi_inertia_overparam_i = pi_overparam[
                i * self.overparam_per_joint : (i + 1) * self.overparam_per_joint
            ]
            pi_prior_i = self.pi_prior[i * 10 : (i + 1) * 10]
            pi_inertia_i = self.logCholeskyVecToParams(pi_inertia_overparam_i, pi_prior_i)
            pi_inertia = pi_inertia.at[
                i * 10 : (i + 1) * 10
            ].set(pi_inertia_i)

        if self.using_cad_inertia: # stop gradient of pi inertia, only learnt residual models
            pi_inertia = jax.lax.stop_gradient(pi_inertia)

        # TODO: refactor, nasty code to deal with residual models
        if len(self.residual_models) > 0:
            pi_friction_overparam = pi_overparam[
                num_inertia_param : num_inertia_param + num_friction_param
            ]
            pi_friction = self.overparam2friction(pi_friction_overparam)

            if len(self.residual_models) == 2:
                if self.models_config["residual"][-1] == "BaseRotor":
                    pi_rotor_overparam = pi_overparam[
                    num_inertia_param + num_friction_param:
                ]
                    pi_rotor = self.overparam2rotor(pi_rotor_overparam)
                    self.residual_models[-1].iden_param = pi_rotor
                    pi = jnp.hstack([pi_inertia, pi_friction, pi_rotor])
                else:
                    raise NotImplementedError
            else:
                pi = jnp.hstack([pi_inertia, pi_friction])
        else:
            pi = pi_inertia
        return pi

    def overparam2friction(self, pi_friction_overparam):
        pi_friction = jnp.zeros(self.njoints * self.residual_models[0].num_coefficient_per_joint)
        for i in range(len(pi_friction_overparam)):
            pi_friction_i = jnp.exp(pi_friction_overparam[i])
            pi_friction = pi_friction.at[i].set(pi_friction_i)
        return pi_friction

    def overparam2rotor(self, pi_rotor_overparam):
        pi_rotor = jnp.zeros(self.residual_models[1].num_param_total)
        for i in range(len(pi_rotor_overparam)):
            pi_rotor_i = jnp.exp(pi_rotor_overparam[i])
            pi_rotor = pi_rotor.at[i].set(pi_rotor_i)
        return pi_rotor

    def bregman_regularizer(self, pi_overparam):
        reg = jnp.float64(0.0)
        pi = self.overparam2pi(pi_overparam)
        for i in range(self.njoints):
            pi_dyn_i = pi[i * 10 : (i + 1) * 10]
            J = self.pseudoinertia_i(pi_dyn_i)
            J_prior = self.J_prior[i]
            bregman_i = self.bregman_i(J, J_prior)
            reg += bregman_i
        return reg

    def bregman_i(self, J, J_prior):
        J1 = J
        J2 = J_prior
        J2inv = jnp.linalg.inv(J2)
        J2inv_J1 = jnp.dot(J2inv, J1)
        breg_d = -jnp.log(jax.scipy.linalg.det(J1)) + jnp.log(jax.scipy.linalg.det(J2)) + jnp.trace(J2inv_J1) - len(J1)
        return breg_d

    def objective(self, pi_overparam):
        loss = self.loss(pi_overparam) + self.omega_pi * self.bregman_regularizer(
            pi_overparam
        )
        return loss

    def jacobian(self, x):
        J = jacrev(self.objective)(x)
        return J

    def hessian(self, x):
        H = jacfwd(jacrev(self.objective))(x)
        return H

    def callback(self, pi_overparam):
        logger.info(f"Iteration: {self.iters}")
        logger.info(f"Overall Loss: {self.objective(pi_overparam)}")
        logger.info(f"L2 sum of sq2 Loss: {self.loss(pi_overparam)}")
        logger.info(f"Gravity: {self.inertia_model.model.gravity.linear}")
        logger.info(f"Bregman Loss: {self.bregman_regularizer(pi_overparam)}\n")
        self.update_weight()
        self.iters += 1

    def fit(self, q, dq, ddq, taus, n_iter=100, method="trust-ncg"):
        self.sync_data(q, dq, ddq, taus)
        self.method = method
        self.callback(self.pi_overparam_init)

        optimize_options = {"maxiter": n_iter, "disp": True}
        res = minimize(
            self.objective,
            self.pi_overparam_init,
            method=method,
            jac=self.jacobian,
            hess=self.hessian,
            options=optimize_options,
            tol=1e-6,
            callback=self.callback,
        )

        self.virparam = res.x
        self.pi = self.overparam2pi(res.x)
        self.sync_param()
        return self.pi

    def get_virparam(self):
        return self.virparam

class LogCholeskyOnlyInertia(LogCholesky):
    def __init__(
        self,
        models_config, solver_config,
        pi_overparam_inertia_init=np.zeros(42),
        overparam_per_joint=6,
    ):
        super(LogCholeskyOnlyInertia, self).__init__(
            models_config, solver_config,
            pi_overparam_inertia_init,
            overparam_per_joint,
        )
        if self.solver_config["remove_gravity"]:
            self.inertia_model._disable_gravity()

    def logCholeskyVecToParams(self, theta, pi_prior_i):
        d1, d2, d3, s12, s13, s23 = theta
        Ubar = jnp.array(
            [
                [jnp.exp(d1), s12, s13],
                [0, jnp.exp(d2), s23],
                [0, 0, jnp.exp(d3)],
            ]
        )
        h_prior_i = jnp.array([pi_prior_i[1], pi_prior_i[2], pi_prior_i[3]])
        m_prior_i = pi_prior_i[0]
        c_prior_i = h_prior_i / m_prior_i
        c_prior_i = jnp.reshape(c_prior_i, (3, 1))

        # Here we need to remove the effect of CoM and mass from the CAD model
        J0 = self.inertiaVecToPinertia(pi_prior_i)
        Sigma0 = J0[:3, :3]
        Sigma0_bar = Sigma0 / m_prior_i
        Sigma0_bar_CoM = Sigma0_bar - jnp.dot(c_prior_i, c_prior_i.T)

        Ubar_Sigma0bar = jnp.dot(Ubar, Sigma0_bar_CoM)
        Sigma_bar_CoM = jnp.dot(Ubar_Sigma0bar, Ubar.T) # mass normalized sigma
        # Sigma_bar_CoM = jnp.dot(Ubar, Ubar.T) # mass normalized sigma

        Sigma_bar = Sigma_bar_CoM + jnp.dot(c_prior_i, c_prior_i.T)

        Sigma = m_prior_i * Sigma_bar
        tmp1 = jnp.hstack([Sigma, jnp.reshape(h_prior_i, (3, 1))])
        tmp2 = jnp.hstack([h_prior_i, m_prior_i])
        J = jnp.vstack([tmp1, tmp2])
        pi = self.pinertiaToVec(J)
        return pi

class LogCholeskyOnlyMCoM(LogCholesky):
    def __init__(
        self,
        models_config, solver_config,
        pi_overparam_inertia_init=np.zeros(28),
        overparam_per_joint=4,
    ):
        super(LogCholeskyOnlyMCoM, self).__init__(
            models_config, solver_config,
            pi_overparam_inertia_init,
            overparam_per_joint,
        )

    def logCholeskyVecToParams(self, theta, pi_prior_i):
        alpha_i, c_shift_i  = theta[0], theta[1:]
        m_scale_i = jnp.exp(alpha_i)

        h_prior_i = jnp.array([pi_prior_i[1], pi_prior_i[2], pi_prior_i[3]])
        m_prior_i = pi_prior_i[0]
        c_prior_i = h_prior_i / m_prior_i

        # scaling mass and shifting CoM with CAD model
        m_i = m_scale_i * m_prior_i
        c_i = c_prior_i + c_shift_i
        h_i = m_i * c_i
        c_i = jnp.reshape(c_i, (3, 1))
        c_prior_i = jnp.reshape(c_prior_i, (3, 1)) # return to the CoM frame

        # Here we need to remove the effect of CoM and mass from the CAD model
        J0 = self.inertiaVecToPinertia(pi_prior_i)
        Sigma0 = J0[:3, :3]
        Sigma0_bar = Sigma0 / m_prior_i
        Sigma0_bar_CoM = Sigma0_bar - jnp.dot(c_prior_i, c_prior_i.T)
        Sigma_bar_CoM = Sigma0_bar_CoM

        Sigma_bar = Sigma_bar_CoM + jnp.dot(c_i, c_i.T)
        Sigma = m_i * Sigma_bar

        tmp1 = jnp.hstack([Sigma, jnp.reshape(h_i, (3, 1))])
        tmp2 = jnp.hstack([h_i, m_i])
        J = jnp.vstack([tmp1, tmp2])
        pi = self.pinertiaToVec(J)
        return pi