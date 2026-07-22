"""Serialization and configuration file helpers."""

from copy import deepcopy

import yaml
from six.moves import cPickle as pickle


def save_obj(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def load_obj(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def save_yaml(filename, data):
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def load_yaml(filename):
    with open(filename, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def yml2urdf(yaml_file, urdf_template, urdf_file):
    """Apply inertial parameters from YAML to a URDF template."""
    import urdfpy

    data = load_yaml(yaml_file)
    robot = urdfpy.URDF.load(urdf_template)
    for index in range(2, len(robot.links) - 1):
        link_id = index - 1
        link_data = data[f"link_{link_id}"]
        link = robot.links[index]
        ixx, iyy, izz, ixy, ixz, iyz = (
            link_data["ixx"],
            link_data["iyy"],
            link_data["izz"],
            link_data["ixy"],
            link_data["ixz"],
            link_data["iyz"],
        )
        link.inertial.inertia[0][0] = ixx
        link.inertial.inertia[0][1] = ixy
        link.inertial.inertia[1][0] = ixy
        link.inertial.inertia[0][2] = ixz
        link.inertial.inertia[2][0] = ixz
        link.inertial.inertia[1][1] = iyy
        link.inertial.inertia[2][2] = izz
        link.inertial.inertia[1][2] = iyz
        link.inertial.inertia[2][1] = iyz
        link.inertial.mass = link_data["mass"]
        (
            link.inertial.origin[0][3],
            link.inertial.origin[1][3],
            link.inertial.origin[2][3],
        ) = link_data["xyz"]
    robot.save(urdf_file)


def merge_dict(dicts):
    """Merge dictionaries that share the same top-level keys."""
    merged = deepcopy(dicts[0])
    for data in dicts[1:]:
        for key in dicts[0]:
            merged[key].update(data[key])
    return merged
