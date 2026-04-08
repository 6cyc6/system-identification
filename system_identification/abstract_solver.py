from copy import copy
from system_identification.friction_model import *
from system_identification.rotor_model import *
from system_identification.inertia_model import *
from loguru import logger
from system_identification.my_utils import merge_dict, save_yaml

class AbstractSolver(object):
    def __init__(self, models_config={}, solver_config={}):
        self.solver_config = solver_config
        self.models_config = models_config
        self.robot_name = self.solver_config["robot_name"]
        self._build_inertia_model()
        self._build_residual_models()
        self.models = copy(self.residual_models)
        self.models.insert(0, self.inertia_model)

    def _build_inertia_model(self):
        self.inertia_model = eval(self.models_config["inertia"])(f"{self.robot_name}")

    def _build_residual_models(self):
        damping = self.inertia_model.damping
        coulomb = self.inertia_model.coulomb
        # Define residual model
        residual_model_config = {
            "F_brk": 25,
            "F_c": 20,
            "v_brk": 0.1,
            "damping": damping,
            "coulomb": coulomb,
            "njoints": self.inertia_model.njoints,
            "bias": 0.0
        }
        self.residual_models = [
            eval(res_model)(residual_model_config) for res_model in self.models_config["residual"]
        ]

    def generate_regressor(self):
        raise NotImplementedError

    def objective(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict_inertia_CAD(self, q, dq, ddq):
        inertia_model_CAD = InertiaModel("iiwas14_cad")
        inertia_model_CAD.iden_param = inertia_model_CAD.dyn_param
        predict_tau_CAD = inertia_model_CAD.predict(q, dq, ddq).reshape(
            -1, self.njoints
        )
        predict_tau = self.inertia_model.predict(q, dq, ddq).reshape(-1, self.njoints)
        return predict_tau, predict_tau_CAD

    def log_info(self):
        # show some logs
        inertia_param_dict = self.inertia_model.param2dict()
        logger.info(f"mass: {self.inertia_model.iden_masses}")
        logger.info(f"CoM: {[inertia_param_dict[f'link_{i}']['xyz'] for i in range(1, self.njoints+1)]}")
        
        if len(self.residual_models) > 0:
            friction_param_dict = self.residual_models[0].param2dict()
            logger.info(f"friction: {friction_param_dict}")
        
        logger.info(f"number of all params: {self.pi.shape[0]}")
        logger.info(f"number of inertia params: {len(self.inertia_model.iden_param)}")
        logger.info(
            f"total mass: {sum(self.inertia_model.iden_masses)}, ref total mass: {self.inertia_model.total_mass}"
        )
        
        # print out the regressor infos
        logger.info(f"regressor shape: {self.Y.shape}")
        logger.info(f"regressor rank: {np.linalg.matrix_rank(self.Y)}")


    def save_param_dict(self, yaml_dir):
        inertia_param_dict = self.inertia_model.param2dict()
        param_dict = inertia_param_dict
        if len(self.residual_models) > 0:
            friction_param_dict = self.residual_models[0].param2dict()
            param_dict = merge_dict([inertia_param_dict, friction_param_dict])
        if len(self.models_config["residual"]) > 1:
            rotor_param_dict = self.residual_models[1].param2dict()
            param_dict = merge_dict([param_dict, rotor_param_dict])
        save_yaml(yaml_dir, param_dict)
        # save virtual parameters
        yaml_vir_dir = yaml_dir.split(".yml")[0] + "vir.yml"
        if self.solver_config["solver_name"] == "LogCholesky":
            vir_param = self.get_virparam().tolist()
            save_yaml(yaml_vir_dir, vir_param)

    def save_param(self, save_filename):
        raise NotImplementedError

    def load_param(self, load_filename):
        raise NotImplementedError