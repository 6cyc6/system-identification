"""Loading and train/test splitting for recorded robot trajectories."""

import os
from copy import copy

import numpy as np

from .io_utils import load_obj


class Dataset:
    def __init__(self, dset_path, root_dir):
        dset_path = os.path.join(os.path.abspath(root_dir), dset_path)
        self.dset = load_obj(dset_path)
        self.num_traj = len(self.dset["dset"])
        total_dset = self.dset["dset"]
        self.num_datapoints = np.sum([len(dset["t"]) for dset in total_dset])
        self.trainset_data, self.testset_data = self.get_trainset_testset()
        self.num_traindata = len(self.trainset_data[0])
        self.num_testdata = len(self.testset_data[0])
        self.dset_index = np.arange(self.num_traj)

    def get_train_data(self):
        return tuple(self.trainset_data)

    def get_test_data(self):
        return tuple(self.testset_data)

    def retrieve_dset_data_byindex(self, index):
        if index not in self.dset_index:
            raise IndexError(
                f"Index is out of bounds; the dataset has {self.num_traj} trajectories"
            )
        return self.retrieve_dset_data(self.dset["dset"][index])

    @staticmethod
    def retrieve_dset_data(dset):
        t, q_f, dq_f, ddq_f, tau_f = (
            dset["t"],
            dset["q_f"],
            dset["dq_f"],
            dset["ddq_f"],
            dset["tau_f"],
        )
        q_m, dq_m, ddq_m, tau_m = (
            dset["q_m"],
            dset["dq_m"],
            dset["ddq_m"],
            dset["tau_m"],
        )
        reference = dset["ref"][0]
        return (
            t,
            q_f,
            dq_f,
            ddq_f,
            tau_f,
            q_m,
            dq_m,
            ddq_m,
            tau_m,
            reference["desired_position"],
            reference["desired_vel"],
            reference["desired_acc"],
            reference["tau_cmd"],
        )

    def merge_dsets(self, dsets):
        columns = [[] for _ in range(13)]
        for dset in dsets:
            for column, values in zip(columns, self.retrieve_dset_data(dset)):
                column.append(values)
        return [np.vstack(column) for column in columns]

    def get_trainset_testset(self):
        train_set, test_set = self.split_train_test(self.dset["dset"])
        self.num_traj_training = len(train_set)
        self.num_traj_testing = len(test_set)
        trainset_data = self.merge_dsets(train_set)
        print("Generated training dataset")
        testset_data = self.merge_dsets(test_set)
        print("Generated testing dataset")
        return trainset_data, testset_data

    def split_train_test(self, dset):
        if len(dset) == 1:
            train_set = copy(dset)
            test_set = copy(dset)
            self.train_set_id = [0]
            self.test_set_id = [0]
        else:
            train_set_num = len(dset) // 3 * 2
            train_set = dset[:train_set_num]
            test_set = dset[train_set_num:]
            self.train_set_id = range(train_set_num)
            self.test_set_id = range(train_set_num, len(dset))
        return train_set, test_set
