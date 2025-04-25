import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap, Wrap
from cvxpy.atoms.affine.upper_tri import vec_to_upper_tri
from system_identification.abstract_solver import AbstractSolver


class BaseQP(AbstractSolver):
    def __init__(
        self, models_config={}, solver_config={}
    ):
        super(BaseQP, self).__init__(models_config, solver_config)
        self.njoints = self.inertia_model.njoints
        self.omega_pi = solver_config["omega_pi"]
        self.Y = None
        self.weights = None
        self.taus = None
        self.n_points = 0
        self.init_variables()
        self.init_pi_indices()

    def load_param(self, file_name):
        load_dict = np.load(f"{file_name}", allow_pickle=True).item()
        self.pi.value = load_dict["model_param"]
        self.dyn_param = load_dict["model_param"][: self.njoints * 10]
        self.sync_param()

    def save_param(self, file_name):
        model_param = self.pi.value
        save_dict = {"model_param": model_param, "solver_config": self.solver_config,
                     "models_config": self.models_config}
        np.save(file_name, save_dict)

    def init_variables(self):
        self.num_dyn_param = self.inertia_model.num_param_total
        self.pi = cp.Variable(self.num_dyn_param)
        self.pi_ref = cp.Constant(self.inertia_model.dyn_param)

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
        return self.inertia_model.regressor(q, dq, ddq)

    def sync_data(self, q, dq, ddq, taus):
        """
        Input:
            q, dq, ddq: (n_points, n_joints)
            taus: (n_points, )
        """
        self.Y = self.generate_regressor(q, dq, ddq)
        self.taus = taus
        self.n_points = self.Y.shape[0]
        self.n_samples = q.shape[0]
        self.weights = np.ones(self.Y.shape[0])

    def loss(self):
        P = self.Y.T @ self.Y
        P_cp = cp.Parameter(P.shape)
        P_cp.value = P
        P_cp = psd_wrap(P_cp)
        q = self.taus @ self.Y
        return cp.quad_form(self.pi, P_cp) / 2 - q.T @ self.pi
        # e_train = self.Y @ self.pi - self.taus
        # e_loss = 1 / 2 * cp.sum_squares(e_train)
        # return e_loss

    def avg_loss(self):
        P = self.Y.T @ self.Y
        P_cp = cp.Parameter(P.shape)
        P_cp.value = P
        P_cp = psd_wrap(P_cp)
        q = self.taus @ self.Y
        return (cp.quad_form(self.pi, P_cp) / 2 - q.T @ self.pi) / self.n_samples

    def weighted_loss(self):
        P = (self.Y.T * self.weights) @ self.Y
        P_cp = cp.Parameter(P.shape)
        P_cp.value = P
        P_cp = psd_wrap(P_cp)
        q = (self.taus * self.weights) @ self.Y
        return cp.quad_form(self.pi, P_cp) / 2 - q.T @ self.pi

    def avg_weighted_loss(self):
        P = (self.Y.T * self.weights) @ self.Y
        P_cp = cp.Parameter(P.shape)
        P_cp.value = P
        P_cp = psd_wrap(P_cp)
        q = (self.taus * self.weights) @ self.Y
        return (cp.quad_form(self.pi, P_cp) / 2 - q.T @ self.pi) / self.n_samples

    def evaluate_loss(self):
        pi = self.pi.value
        return cp.norm2(self.Y @ pi - self.taus)

    def evaluate_avg_loss(self):
        pi = self.pi.value
        return cp.norm2(self.Y @ pi - self.taus) / self.n_samples

    def evaluate_weighted_loss(self):
        pi = self.pi.value
        residual = self.Y @ pi - self.taus
        return cp.sum(residual.T * self.weights * residual)

    def evaluate_avg_weighted_loss(self):
        pi = self.pi.value
        residual = self.Y @ pi - self.taus
        return cp.sum(residual.T * self.weights * residual) / self.n_samples

    def update_weight(self):
        residual = self.Y @ self.pi.value - self.taus
        E = residual.reshape((-1, self.njoints)).T
        m = E.shape[1]
        p = self.Y.shape[1]
        Lambda = np.square(np.linalg.norm(E, axis=1, keepdims=True)) / (m - p)
        return np.repeat(1 / Lambda, m, axis=1).flatten(order="F")

    def objective(self):
        raise NotImplementedError

    def inertia_param_regularizer(self):
        return (
            cp.norm2(
                self.pi[: self.njoints * self.inertia_model.inertia_param_per_joint]
                - self.pi_ref[
                    : self.njoints * self.inertia_model.inertia_param_per_joint
                ]
            )
            ** 2
        )

    def evaluate_inertia_param_regularizer(self):
        pi = self.pi.value
        pi_inertia = pi[: self.njoints * self.inertia_model.inertia_param_per_joint]
        pi_ref_inertia = self.pi_ref[
            : self.njoints * self.inertia_model.inertia_param_per_joint
        ]
        residual = pi_inertia - pi_ref_inertia
        return cp.norm2(residual) ** 2

    def all_param_regularizer(self):
        return cp.norm2(self.pi - self.pi_ref) ** 2

    def objective(self):
        solver_config = self.solver_config
        loss = getattr(self, solver_config["loss"])()
        regularizers = []
        for i in range(len(solver_config["regularizer"])):
            regularizer = getattr(self, solver_config["regularizer"][i])()
            regularizers.append(regularizer)
        regularizer = cp.sum(regularizers)
        return cp.Minimize(loss / self.n_points + self.omega_pi * regularizer)

    def fit(self, q, dq, ddq, taus, n_iter=20):
        self.sync_data(q, dq, ddq, taus)

        for i in range(n_iter):
            obj = self.objective()
            problem = cp.Problem(obj)
            problem.solve(verbose=False, warm_start=True, max_iters=5000)

            self.weights = self.update_weight()
            # print("Iteration: {}/{}, optimal value: {}".format(i + 1, n_iter, problem.value))

        self.sync_param()
        return self.pi.value

    def sync_param(self):
        for i in range(len(self.models)):
            model = self.models[i]
            param_idx_start, param_idx_end = self.pi_idxes[i][0], self.pi_idxes[i][1]
            model.sync_param(self.pi.value[param_idx_start:param_idx_end])

    def predict(self, q, dq, ddq):
        Y_eval = self.generate_regressor(q, dq, ddq)
        dyn_param = self.pi.value.reshape(-1, 1)
        return Y_eval @ dyn_param

class LmiQP(BaseQP):
    def __init__(
        self, models_config, solver_config
    ):
        super(LmiQP, self).__init__(models_config, solver_config)
        self.init_matrix_for_constraints()

    def init_variables(self):
        self.num_dyn_param = 0
        ref_dyn_param = []
        for i in range(len(self.models)):
            self.num_dyn_param += self.models[i].num_param_total
            ref_dyn_param.append(self.models[i].ref_param)
        pi_ref = np.hstack(ref_dyn_param)
        self.pi = cp.Variable(self.num_dyn_param)
        self.pi_ref = cp.Constant(pi_ref)
        self.Js = None
        self.J_prior = self.inertia_model.J_prior
        self.SigmaC = self.inertia_model.Qs
        self.Q = self.inertia_model.Q
        assert (
            self.pi.shape == self.pi_ref.shape
        ), "The shape of reference dynamic parameters should be the same as the regressored ones"

    def init_matrix_for_constraints(self):
        mats = []
        for idx in range(len(self.models)):
            model = self.models[idx]
            num_model_param = model.num_param_total
            mat = np.zeros((num_model_param, self.num_dyn_param))
            start, end = self.pi_idxes[idx][0], self.pi_idxes[idx][1]
            mat[:, start:end] = np.identity(num_model_param)
            mat = cp.Constant(mat)
            mats.append(mat)
        self.mats = mats

    def generate_regressor(self, q, dq, ddq):
        regressors = []
        for model in self.models:
            regressors.append(model.regressor(q, dq, ddq))
        regressor = np.hstack(regressors)
        return regressor

    def pi_dyn_joint_i(self, i):
        if "CoM" in self.inertia_model.name:
            return self.pi[i * 10 : i * 10 + 10]
        elif "GoM" in self.inertia_model.name:
            pi_CoM = self.pi[: 7 * 4]
            pi_I = self.inertia_model.G_block_diag @ pi_CoM
            pi_CoM_i = pi_CoM[i * 4 : i * 4 + 4]
            pi_I_i = pi_I[i * 6 : i * 6 + 6]
            pi_i = cp.hstack([pi_CoM_i, pi_I_i])
            return pi_i

    def inertiaVecToPinertiaCVX(self, pi_dyn_i):
        off_diag = vec_to_upper_tri(pi_dyn_i[[5, 7, 8]], strict=True)
        Ibar = cp.diag(pi_dyn_i[[4, 6, 9]]) + off_diag + off_diag.T
        eye3 = cp.Constant(np.eye(3))
        Sigma = cp.trace(Ibar) / 2.0 * eye3 - Ibar
        J = cp.vstack([Sigma, cp.reshape(pi_dyn_i[1:4], (1, 3))])
        J = cp.hstack([J, cp.reshape(pi_dyn_i[[1, 2, 3, 0]], (4, 1))])
        return J

    def pseudoinertia_constraint(self):
        constraint = []
        self.Js = []
        for i in range(self.njoints):
            pi_dyn_i = self.pi_dyn_joint_i(i)
            J = self.inertiaVecToPinertiaCVX(pi_dyn_i)
            self.Js.append(J)
            constraint.append(J - 1e-4 * np.eye(4) >> 0)
        return constraint

    def CoMbound_constraint(self):
        if self.Js is None:
            self.pseudoinertia_constraint()
        constraint = []
        masses_CAD, GoMs_lever, inertias_CAD = (
            self.inertia_model.masses_CAD,
            self.inertia_model.GoMs_lever,
            self.inertia_model.inertias_CAD,
        )
        self.masses_CAD = masses_CAD
        self.GoMs_lever = GoMs_lever
        for i in range(self.njoints):
            pi_i = self.pi_dyn_joint_i(i)
            pi_cad_xyz = GoMs_lever[i]
            Qs = self.SigmaC[i]
            constraint.append(
                self.CoMbound_i(pi_i, pi_cad_xyz, Qs) - 1e-4 * np.eye(4) >> 0
            )
        return constraint

    def CoMbound_i(self, pi_i, pi_cad_xyz, Qs):
        m = pi_i[[0]]
        h_mx = pi_i[1:4] - m * pi_cad_xyz
        mQs = m * Qs
        C_col1 = cp.reshape(cp.hstack([m, h_mx]), (1, 4))
        C_col2 = cp.hstack([cp.reshape(h_mx, (3, 1)), mQs])
        C = cp.vstack([C_col1, C_col2])
        return C

    def densityrealize_constraint(self):
        if self.Js is None:
            self.pseudoinertia_constraint()
        constraint = []
        for i in range(self.njoints):
            Q = self.Q[i]
            J = self.Js[i]
            constraint.append(self.densityrealize_i(J, Q) >= 1e-3)
        return constraint

    def densityrealize_i(self, J, Q):
        TrJQ = cp.trace(J @ Q)
        return TrJQ

    def bregman_inertia_param_regularizer(self):
        if self.Js is None:
            self.pseudoinertia_constraint()
        regs = []
        for i in range(self.njoints):
            J = self.Js[i]
            J_prior = self.J_prior[i]
            dkl_i = self.bregman_inertia_param_i(J, J_prior)
            regs.append(dkl_i)
        return cp.sum(regs)

    def bregman_inertia_param_i(self, J, J_prior):
        J1 = J
        J2 = J_prior
        J2inv = np.linalg.inv(J2)
        J2inv_J1 = J2inv @ J1
        return -cp.log_det(J1) + cp.log_det(J2) + cp.trace(J2inv_J1) - 4

    def fit(self, q, dq, ddq, taus, n_iter=1):
        solver_config = self.solver_config
        self.sync_data(q, dq, ddq, taus)

        if solver_config["constraint"] != "":
            constraint_types = solver_config["constraint"].split("_")
            constraint_types = [
                constraint_type + "_constraint" for constraint_type in constraint_types
            ]
            constraint = getattr(self, constraint_types[0])()
            for i in range(1, len(constraint_types)):
                constraint.extend(getattr(self, constraint_types[i])())

        reg_type = solver_config["regularizer"]
        regs = ""
        for _reg in reg_type:
            regs += _reg

        for i in range(n_iter):
            obj = self.objective()
            if solver_config["constraint"] != "":
                problem = cp.Problem(obj, constraint)
            else:
                problem = cp.Problem(obj)
            if solver_config["num_solver"] == "MOSEK":
                problem.solve(solver=cp.MOSEK, verbose=True, warm_start=True)
            elif solver_config["num_solver"] == "SCS":
                problem.solve(
                    solver=cp.SCS, verbose=True, warm_start=True, max_iters=5000
                )
            # self.weights = self.update_weight()
        self.sync_param()
        return self.pi.value

    def totalmass_constraint(self):
        total_mass = self.inertia_model.total_mass
        inertia_mat = self.mats[0]
        inertia_param = inertia_mat @ self.pi
        mass_param = []
        for i in range(self.njoints):
            mass_idx = i * self.inertia_model.inertia_param_per_joint
            mass_param.append(inertia_param[mass_idx])
        return [cp.sum(mass_param) == total_mass]

    def friction_constraint(self):
        friction_mat = self.mats[1]
        num_friction_param = self.residual_models[0].num_param_total
        return [friction_mat @ self.pi >= np.ones(num_friction_param) * 1e-4]

    def rotor_constraint(self):
        rotor_mat = self.mats[2]
        num_rotor_param = self.residual_models[1].num_param_total
        return [rotor_mat @ self.pi >= np.ones(num_rotor_param) * 1e-3]
