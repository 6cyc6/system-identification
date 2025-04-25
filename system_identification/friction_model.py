import numpy as np
from system_identification.utils import feature2regressor

class FrictionModel(object):
    def __init__(self, param, num_coefficient_per_joint=1):
        self.viscous_coeff = param[
            "damping"
        ]  # for clearer visualization the damping term should be 10 times
        # larger while the visualization range should from 0 to 2
        self.coulomb_coeff = param["coulomb"]
        self.njoints = param["njoints"]
        self.F_brk = [param["F_brk"] for i in range(self.njoints)]
        self.v_brk = [param["v_brk"] for i in range(self.njoints)]
        self.F_c = [param["F_c"] for i in range(self.njoints)]
        self.bias = [param["bias"] for i in range(self.njoints)]
        self.num_coefficient_per_joint = num_coefficient_per_joint
        self.num_param_total = self.num_coefficient_per_joint * self.njoints
        self.iden_param = None
        self._init_ref_param()

    def _init_ref_param(self):
        raise NotImplementedError

    def sync_param(self, iden_friction_param):
        self.iden_param = iden_friction_param

    def regressor(self, q, qd, qdd):
        raise NotImplementedError

    def predict(self, q, dq, ddq):
        regressor = self.regressor(q, dq, ddq)
        param = self.iden_param.reshape(-1, 1)
        return regressor @ param


class CoulombModel(FrictionModel):
    def __init__(self, param, num_coefficient_per_joint=1):
        super(CoulombModel, self).__init__(param, num_coefficient_per_joint)

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(self.coulomb_coeff[i])
        self.ref_param = np.array(ref_param)

    def regressor(self, q, qd, qdd):
        # qd shape: (N, njoints)
        njoints, n_datapoints = qd.shape[1], qd.shape[0]
        feature0 = np.sign(qd)
        features = [feature0]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg


class ViscousModel(FrictionModel):
    def __init__(self, param, num_coefficient_per_joint=1):
        super(ViscousModel, self).__init__(param, num_coefficient_per_joint)

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(self.viscous_coeff[i])
        self.ref_param = np.array(ref_param)

    def regressor(self, q, qd, qdd):
        # qd shape: (N, njoints)
        njoints, n_datapoints = qd.shape[1], qd.shape[0]
        feature0 = qd
        features = [feature0]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg


class OffsetModel(FrictionModel):
    def __init__(self, param, num_coefficient_per_joint=1):
        super(OffsetModel, self).__init__(param, num_coefficient_per_joint)

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(0.0)
        self.ref_param = np.array(ref_param)

    def regressor(self, q, qd, qdd):
        njoints, n_datapoints = qd.shape[1], qd.shape[0]
        feature0 = np.ones_like(qd)
        features = [feature0]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg

class MatlabFModel(FrictionModel):
    def __init__(self, param, num_coefficient_per_joint=2):
        super(MatlabFModel, self).__init__(param, num_coefficient_per_joint)
        self.vbrk = 0.35
        self.vst = self.vbrk * np.sqrt(2)
        self.vcoul = self.vbrk / 10

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(self.viscous_coeff[i])
        for i in range(self.njoints):
            ref_param.append(self.coulomb_coeff[i])
        self.ref_param = np.array(ref_param)

    def regressor(self, q, dq, ddq):
        # qd shape: (N, njoints)
        njoints, n_datapoints = dq.shape[1], dq.shape[0]
        feature0 = np.sqrt(2*np.e)*np.exp(-(dq/self.vst)**2)*(dq/self.vst) # note that here we take elementwise product
        feature1 = np.tanh(dq/self.vcoul)
        features = [feature1, feature0]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg

    def param2dict(self):
        friction_param_dict = {}
        Fc = self.iden_param[:self.njoints].tolist()
        FbrkMinusFc = self.iden_param[self.njoints:].tolist()
        for i in range(self.njoints):
            link = f"link_{i + 1}"
            Fc_ = Fc[i]
            FbrkMinusFc_ = FbrkMinusFc[i]
            friction_coeff = {"Fc": Fc_, "FbrkMinusFc": FbrkMinusFc_}
            friction_param_dict[link] = friction_coeff
        return friction_param_dict

class AsymMatlabFModel(FrictionModel):
    def __init__(self, param, num_coefficient_per_joint=4):
        super(AsymMatlabFModel, self).__init__(param, num_coefficient_per_joint)
        self.vbrk = 0.2
        self.vst = self.vbrk * np.sqrt(2)
        self.vcoul = self.vbrk / 10

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(self.viscous_coeff[i])
        for i in range(self.njoints):
            ref_param.append(self.coulomb_coeff[i])
        for i in range(self.njoints):
            ref_param.append(self.viscous_coeff[i])
        for i in range(self.njoints):
            ref_param.append(self.coulomb_coeff[i])
        self.ref_param = np.array(ref_param)

    def regressor(self, q, dq, ddq):
        # qd shape: (N, njoints)
        njoints, n_datapoints = dq.shape[1], dq.shape[0]
        dq_pos = dq*np.float64(np.sign(dq) > 0)
        dq_neg = dq*np.float64(np.sign(dq) <= 0)

        feature0 = np.sqrt(2*np.e)*np.exp(-(dq_pos/self.vst)**2)*(dq_pos/self.vst) # note that here we take elementwise product
        feature1 = np.tanh(dq_pos/self.vcoul)
        feature2 = np.sqrt(2*np.e)*np.exp(-(dq_neg/self.vst)**2)*(dq_neg/self.vst) # note that here we take elementwise product
        feature3 = np.tanh(dq_neg/self.vcoul)
        features = [feature1, feature0, feature3, feature2]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg

    def param2dict(self):
        friction_param_dict = {}
        Fc_pos = self.iden_param[:self.njoints].tolist()
        FbrkMinusFc_pos = self.iden_param[self.njoints: 2*self.njoints].tolist()
        Fc_neg = self.iden_param[2*self.njoints: 3*self.njoints].tolist()
        FbrkMinusFc_neg= self.iden_param[3*self.njoints: ].tolist()

        for i in range(self.njoints):
            link = f"link_{i + 1}"
            Fc_pos_i = Fc_pos[i]
            FbrkMinusFc_pos_i = FbrkMinusFc_pos[i]
            Fbrk_pos_i = Fc_pos_i + FbrkMinusFc_pos_i
            Fc_neg_i = Fc_neg[i]
            FbrkMinusFc_neg_i = FbrkMinusFc_neg[i]
            Fbrk_neg_i = Fc_neg_i + FbrkMinusFc_neg_i
            friction_coeff = {"Fc_pos": Fc_pos_i, "FbrkMinusFc_pos": FbrkMinusFc_pos_i, "Fc_neg": Fc_neg_i, "FbrkMinusFc_neg": FbrkMinusFc_neg_i,
                              "Fbrk_pos_i": Fbrk_pos_i, "Fbrk_neg_i": Fbrk_neg_i, "vbrk_pos": self.vbrk, "vbrk_neg": self.vbrk}
            friction_param_dict[link] = friction_coeff
        return friction_param_dict
    
class AsymModel(FrictionModel):
    def __init__(self, param, num_coefficient_per_joint=4):
        super(AsymModel, self).__init__(param, num_coefficient_per_joint)
        self.vbrk = 0.01
        self.vst = self.vbrk * np.sqrt(2)
        self.vcoul = self.vbrk / 10

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(self.F_brk[i]-self.F_c[i])
        for i in range(self.njoints):
            ref_param.append(self.F_c[i])
        for i in range(self.njoints):
            ref_param.append(self.viscous_coeff[i])
        for i in range(self.njoints):
            ref_param.append(self.bias[i])
        self.ref_param = np.array(ref_param)

    def regressor(self, q, dq, ddq):
        # qd shape: (N, njoints)
        njoints, n_datapoints = dq.shape[1], dq.shape[0]
        feature_Fbrk_mins_Fc = np.sqrt(2*np.e) * np.exp(-(dq / self.vst)**2) * dq / self.vst
        feature_Fc = np.tanh(dq/self.vcoul)
        sign_product = np.sign(dq) * np.sign(ddq)
        sign_product_cond = np.where(sign_product >= 0, 1, 0)
        feature_Fbrk_mins_Fc_pos = np.sqrt(2*np.e) * np.exp(-(dq / self.vst)**2) * dq / self.vst
        feature_Fbrk_mins_Fc_pos *= sign_product_cond
        feature_viscous = dq
        feature_bias = np.ones_like(dq)
        features = [feature_Fbrk_mins_Fc_pos, feature_Fc, feature_viscous, feature_bias]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg

    def param2dict(self):
        friction_param_dict = {}
        FbrkMinusFc = self.iden_param[:self.njoints].tolist()
        Fc = self.iden_param[self.njoints: 2*self.njoints].tolist()
        Fviscous = self.iden_param[2*self.njoints: 3*self.njoints].tolist()
        Fbias = self.iden_param[3*self.njoints: ].tolist()

        for i in range(self.njoints):
            link = f"link_{i + 1}"
            FbrkMinusFc_i = FbrkMinusFc[i]
            Fc_i = Fc[i]
            Fbrk_i = Fc_i + FbrkMinusFc_i
            Fviscous_i = Fviscous[i]
            Fbias_i = Fbias[i]
            friction_coeff = {"FbrkMinusFc": FbrkMinusFc_i, "Fbrk": Fbrk_i, "Fc": Fc_i, "Fviscous": Fviscous_i, "Fbias": Fbias_i,
                              "vbrk_pos": self.vbrk, "vbrk_neg": self.vbrk}
            friction_param_dict[link] = friction_coeff
        return friction_param_dict

class CoulombViscousModel(FrictionModel):
    def __init__(self, param, num_coefficient_per_joint=2):
        super(CoulombViscousModel, self).__init__(param, num_coefficient_per_joint)

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(self.viscous_coeff[i])
        for i in range(self.njoints):
            ref_param.append(self.coulomb_coeff[i])
        self.ref_param = np.array(ref_param)

    def regressor(self, q, dq, ddq):
        # qd shape: (N, njoints)
        njoints, n_datapoints = dq.shape[1], dq.shape[0]

        feature0 = np.sign(dq)
        feature1 = dq
        features = [feature1, feature0]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg

    def param2dict(self):
        friction_param_dict = {}
        viscous_params = self.iden_param[:self.njoints].tolist()
        coulomb_params = self.iden_param[self.njoints:].tolist()
        for i in range(self.njoints):
            link = f"link_{i+1}"
            v = viscous_params[i]
            c = coulomb_params[i]
            friction_coeff = {"damping": v, "friction": c}
            friction_param_dict[link] = friction_coeff
        return friction_param_dict

class AsymCoulombViscousModel(FrictionModel):
    """
    Asymmetric version of coulomb and viscous friction model
    """
    def __init__(self, param, num_coefficient_per_joint=3):
        super(AsymCoulombViscousModel, self).__init__(param, num_coefficient_per_joint)

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(self.viscous_coeff[i])
        for i in range(self.njoints):
            ref_param.append(self.coulomb_coeff[i])
        for i in range(self.njoints):
            ref_param.append(self.coulomb_coeff[i])
        self.ref_param = np.array(ref_param)

    def regressor(self, q, dq, ddq):
        # qd shape: (N, njoints)
        njoints, n_datapoints = dq.shape[1], dq.shape[0]
        feature0_positive = np.float64(np.sign(dq) > 0)
        feature0_negative = np.float64(np.sign(dq) < 0) * -1
        feature1 = dq
        features = [feature1, feature0_positive, feature0_negative]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg

    def param2dict(self):
        friction_param_dict = {}
        viscous_params = self.iden_param[:self.njoints].tolist()
        coulomb_positive_params = self.iden_param[self.njoints: self.njoints+self.njoints].tolist()
        coulomb_negative_params = self.iden_param[self.njoints+self.njoints: ].tolist()

        for i in range(self.njoints):
            link = f"link_{i+1}"
            v = viscous_params[i]
            c_pos = coulomb_positive_params[i]
            c_neg = coulomb_negative_params[i]
            friction_coeff = {"damping": v, "friction_positive": c_pos, "friction_negative": c_neg}
            friction_param_dict[link] = friction_coeff
        return friction_param_dict

class AsymCoulombModel(FrictionModel):
    """
    Asymmetric version of coulomb and viscous friction model
    """
    def __init__(self, param, num_coefficient_per_joint=2):
        super(AsymCoulombModel, self).__init__(param, num_coefficient_per_joint)

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(self.coulomb_coeff[i])
        for i in range(self.njoints):
            ref_param.append(self.coulomb_coeff[i])
        self.ref_param = np.array(ref_param)

    def regressor(self, q, dq, ddq):
        # qd shape: (N, njoints)
        njoints, n_datapoints = dq.shape[1], dq.shape[0]
        feature0_positive = np.float64(np.sign(dq) > 0)
        feature0_negative = np.float64(np.sign(dq) < 0) * -1
        features = [feature0_positive, feature0_negative]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg

    def param2dict(self):
        friction_param_dict = {}
        coulomb_positive_params = self.iden_param[:self.njoints].tolist()
        coulomb_negative_params = self.iden_param[self.njoints: ].tolist()
        for i in range(self.njoints):
            link = f"link_{i+1}"
            c_pos = coulomb_positive_params[i]
            c_neg = coulomb_negative_params[i]
            friction_coeff = {"friction_positive": c_pos, "friction_negative": c_neg}
            friction_param_dict[link] = friction_coeff
        return friction_param_dict

class StribeckModel(FrictionModel):
    """
    Stribeck model motivated by matlab:
    https://de.mathworks.com/help/physmod/simscape/ref/translationalfriction.html#responsive_offcanvas
    """

    def __init__(self, param, num_coefficient_per_joint=3):
        super(StribeckModel, self).__init__(param, num_coefficient_per_joint)
        # self.v_brk = 0.1
        self.v_st = self.v_brk * np.sqrt(2)
        self.v_coul = self.v_brk / 10.0

    def _init_ref_param(self):
        ref_param = []
        for i in range(self.njoints):
            ref_param.append(self.F_brk - self.F_c)
        for i in range(self.njoints):
            ref_param.append(self.F_c)
        for i in range(self.njoints):
            ref_param.append(self.viscous_coeff[i])
        self.ref_param = np.array(ref_param)

    def regressor(self, q, qd, qdd):
        """
        The regressor for friction model

            regressor: (n_datapoints * njoints, num_coefficients)
            friction_parameters: ((F_brk-F_C)*njoints, F_C*njoints, f*njoints)
        """

        # qd shape: (N, njoints)
        njoints, n_datapoints = qd.shape[1], qd.shape[0]

        # desired const shape: (N*njoints, njoints)
        # current const shape: (N, njoints)
        # feature0 = np.sqrt(2)*np.e*np.exp(-qd**2/2)*(qd/np.sqrt(2))
        # feature1 = np.tanh(qd*10)
        # feature2 = qd
        feature0 = (
            np.sqrt(2)
            * np.e
            * np.exp(-(qd**2) / (np.sqrt(2) * self.v_st) ** 2)
            * (qd / (np.sqrt(2) * self.v_st))
        )
        feature1 = np.tanh(qd / (self.v_brk / 10))
        feature2 = qd
        features = [feature0, feature1, feature2]
        reg = feature2regressor(features, n_datapoints, njoints)
        return reg

    def predict_viz(self, qd):
        damping_coeff_viz = self.viscous_coeff * 100
        F = (
            np.sqrt(2 * np.e)
            * (self.F_brk - self.F_c)
            * np.exp(-((qd / self.v_st) ** 2))
            * qd
            / self.v_st
            + self.F_c * np.tanh(qd / self.v_coul)
            + qd * damping_coeff_viz
        )
        return F

    def param2dict(self):
        friction_param_dict = {}
        fbrk_fc_params = self.iden_param[:self.njoints].tolist()
        fc_params = self.iden_param[self.njoints: self.njoints+self.njoints].tolist()
        viscous_params = self.iden_param[self.njoints+self.njoints: ].tolist()
        for i in range(self.njoints):
            link = f"link_{i+1}"
            v = viscous_params[i]
            fbrk_fc = fbrk_fc_params[i]
            fc = fc_params[i]
            friction_coeff = {"damping": v, "fbrk-fc": fbrk_fc, "fc": fc}
            friction_param_dict[link] = friction_coeff
        return friction_param_dict
