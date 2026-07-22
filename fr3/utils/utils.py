"""Backward-compatible facade for the focused utility modules.

New code should import from the module that owns the functionality.  These
re-exports keep older scripts working while callers migrate their imports.
"""

from .config_utils import DataClass
from .dataset import Dataset
from .dynamics import dynamics_prediction, load_model_params
from .inertia import (
    I,
    bregman_i,
    bregman_regularizer,
    check_CoM_bounded,
    inertiaVecToIcQs,
    inertiaVecToPinertia,
    inertiaVecToQ,
    inertiaVecToXsQs,
    inertiaVecToinertiasCoM,
    inv_sigmoid,
    logCholeskyVecToParams,
    overparam2pi,
    param2dict,
    pi2overparam,
    pinertiaToVec,
    pseudoinertia_i,
    retrieve_geo_fromCAD,
    retrieve_pi_prior_inertia,
    sigmoid,
    skew_symmetric,
)
from .io_utils import load_obj, load_yaml, merge_dict, save_obj, save_yaml, yml2urdf
from .logging_utils import WandbLogger
from .path_utils import (
    BASE_DIR,
    _search_roots,
    find_path,
    get_config_path,
    resolve_repo_path,
)
from .regression import (
    QR_dim_reduction,
    QR_dim_reduction_backup,
    SVD_dim_reduction,
    check_features,
    feature2regressor,
    tikhonov_regularization,
)
from .robot_config import (
    RobotConfig,
    _find_robot_urdf,
    _load_pin_model_for_robot,
    pin_joint_config,
    retrieve_robot_config,
    validate_robot_config,
)
from .signal_processing import check_jumps, perturbed_array, savitzy_filter
from .string_utils import concanate_strings, concatenate_strings
from .visualization import draw_spectrum, vis_compare_seqs

# Historical name retained for callers that used it directly.
FR3_ROOT = BASE_DIR
