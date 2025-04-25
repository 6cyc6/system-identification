import numpy as np
from system_identification.utils import feature2regressor


class BaseRotor(object):
    def __init__(self, param, num_coefficient_per_joint=1):
        self.njoints = param["njoints"]
        self.num_coefficient_per_joint = num_coefficient_per_joint
        self.num_param_total = num_coefficient_per_joint * self.njoints
        self.iden_param = None
        self._init_ref_param()

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(0.1)
        self.ref_param = np.array(ref_param)

    def regressor(self, q, dq, ddq):
        # ddq shape (N_points, njoints)
        njoints, n_datapoints = ddq.shape[1], ddq.shape[0]

        feature0 = ddq * (100 ** 2) # times (gear ratio) ** 2 as transmission ratio
        features = [feature0]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg

    def sync_param(self, rotor_param):
        self.iden_param = rotor_param

    def predict(self, q, dq, ddq):
        regressor = self.regressor(q, dq, ddq)
        param = self.iden_param.reshape(-1, 1)
        return regressor @ param

    def param2dict(self):
        rotor_param_dict = {}
        rotor_params = self.iden_param[:self.njoints].tolist()
        for i in range(self.njoints):
            link = f"link_{i+1}"
            rotor_p = rotor_params[i]
            rotor_coeff = {"rotor_param": rotor_p}
            rotor_param_dict[link] = rotor_coeff
        return rotor_param_dict

class NegRotor(object):
    def __init__(self, param, num_coefficient_per_joint=1):
        self.njoints = param["njoints"]
        self.num_coefficient_per_joint = num_coefficient_per_joint
        self.num_param_total = num_coefficient_per_joint * self.njoints
        self.iden_param = None
        self._init_ref_param()

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(0.1)
        self.ref_param = np.array(ref_param)

    def regressor(self, q, dq, ddq):
        # ddq shape (N_points, njoints)
        njoints, n_datapoints = ddq.shape[1], ddq.shape[0]
        reg = np.array([[] for i in range(n_datapoints*njoints)]) # empty regressor
        return reg

    def sync_param(self, rotor_param):
        self.iden_param = rotor_param

    def predict(self, q, dq, ddq):
        regressor = self.regressor(q, dq, ddq)
        param = self.iden_param.reshape(-1, 1)
        return regressor @ param

    def param2dict(self):
        rotor_param_dict = {}
        rotor_params = self.iden_param[:self.njoints].tolist()
        for i in range(self.njoints):
            link = f"link_{i+1}"
            rotor_p = rotor_params[i]
            rotor_coeff = {"rotor_param": rotor_p}
            rotor_param_dict[link] = rotor_coeff
        return rotor_param_dict

class RotorAndGear(object):
    def __init__(self, param, num_coefficient_per_joint=2):
        self.njoints = param["njoints"]
        self.num_coefficient_per_joint = num_coefficient_per_joint
        self.num_param_total = num_coefficient_per_joint * self.njoints
        self.iden_param = None
        self._init_ref_param()

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(0.1)
        self.ref_param = np.array(ref_param)

    def regressor(self, q, dq, ddq):
        # ddq shape (N_points, njoints)
        njoints, n_datapoints = ddq.shape[1], ddq.shape[0]
        reg = np.array([[] for i in range(n_datapoints*njoints)]) # empty regressor
        return reg

    def sync_param(self, rotor_gear_param):
        self.iden_param = rotor_gear_param

    # def predict(self, q, dq, ddq):
    #     rotor_param = self.iden_param.tolist()[: 7]
    #     gear_ratio = self.iden_param.tolist()[7: ]
    #
    #     return regressor @ param

    def param2dict(self):
        rotor_gear_param_dict = {}
        rotor_gear_params = self.iden_param[:self.njoints].tolist()
        rotor_params = rotor_gear_params[: 7]
        gear_params = rotor_gear_params[7: ]
        for i in range(self.njoints):
            link = f"link_{i+1}"
            rotor_p = rotor_params[i]
            rotor_coeff = {"rotor_param": rotor_p}
            rotor_gear_param_dict[link] = rotor_coeff
            gear_p = gear_params[i]
            gear_coeff = {"gear_param": gear_p}
            rotor_gear_param_dict[link] = gear_coeff
        return rotor_gear_param_dict


