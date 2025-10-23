""" Create experimental data. """

import random

import numpy as np
import torch

from partial_label_learning.data import (get_mnist_dataset, get_vision_dataset,
                                         random_class_imbalance)

for ds_name, hardness in [
    ("mnist", (0.4, 1.0)),
    ("kmnist", (0.15, 1.0)),
    ("fmnist", (0.15, 1.0)),
]:
    # Fix all seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create dataset
    dss = get_mnist_dataset(ds_name)
    class_imb = random_class_imbalance(dss.y_train.shape[1])
    dss = dss.augment_targets_instance_dependent(hardness, class_imb)
    dss = dss.make_imbalanced(class_imb)
    print(ds_name, dss.y_train.shape[0])
    dss.store_to_file(f"./experiments/{ds_name}.npz")

for ds_name, hardness in [
    ("cifar10", (0.6, 1.0)),
    ("cifar100", (0.15, 0.001)),
]:
    # Fix all seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create dataset
    dss = get_vision_dataset(ds_name)
    class_imb = random_class_imbalance(dss.y_train.shape[1])
    dss = dss.augment_targets_instance_dependent(hardness, class_imb)
    dss = dss.make_imbalanced(class_imb)
    print(ds_name, dss.y_train.shape[0])
    dss.store_to_file(f"./experiments/{ds_name}.npz")
