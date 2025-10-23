""" Run all experiments """

import os
import random
import sys
from typing import Dict, Type

import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef  # type: ignore

import partial_label_learning.methods as rw
from models.model_util import create_model, get_device, get_model_arch
from partial_label_learning.data import Datasplit
from partial_label_learning.pll_classifier_base import PllBaseClassifier


def run_experiment(algo_name: str, dataset_name: str) -> None:
    """ Run single experiment """

    # Check if experiment already completed
    if os.path.exists(f"./results/{dataset_name}_{algo_name}.csv"):
        return

    # Get algorithm to use
    all_algos: Dict[str, Type[PllBaseClassifier]] = {
        # ML methods
        "pl-knn-2005": rw.PlKnn,
        "pl-ecoc-2017": rw.PlEcoc,
        # Deep Learning methods
        "proden-2020": rw.Proden,
        "cavl-2021": rw.Cavl,
        "valen-2021": rw.Valen,
        "pico-2022": rw.PiCO,
        "pop-2023": rw.Pop,
        "crosel-2024": rw.CroSel,
        "cel-2025": rw.Cel,
        # Own methods
        "vi-ablation": rw.ViAblation,
        "vi-pll": rw.ViPll,
    }
    algo_type = all_algos[algo_name]

    # Params
    params = {
        "mnist": (200.0, 0.0),
        "kmnist": (250.0, 0.0),
        "fmnist": (200.0, 0.0),
        "cifar10": (200.0, 0.0),
        "cifar100": (200.0, 0.0),
    }

    # Load dataset
    lmbd, prior = params[dataset_name]
    datasplit = Datasplit.restore_from_file(
        f"./experiments/{dataset_name}.npz")
    x_train = datasplit.x_train
    y_train = datasplit.y_train
    y_train_true = datasplit.y_true_train
    x_test = datasplit.x_test
    y_test_true = datasplit.y_true_test

    x_min = np.min(x_train, axis=0)
    x_max = np.max(x_train, axis=0)
    diff = np.clip(x_max - x_min, a_min=1e-10, a_max=None)
    x_train = (x_train - x_min) / diff
    x_test = (x_test - x_min) / diff

    # Get device
    device = get_device()

    # Fix all seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Run 5-fold cross validation
    with open(
        f"./results/{dataset_name}_{algo_name}.csv", "w", encoding="utf-8",
    ) as res_file:
        for i in range(5):
            # Fix all seeds
            random.seed(i)
            np.random.seed(i)
            torch.manual_seed(i)

            rng = np.random.Generator(np.random.PCG64(i))
            model_type = get_model_arch(algo_name)
            model = create_model(
                model_type, y_train.shape[1], x_train.shape[1:])
            model.to(device)
            model.compile()
            clf = algo_type(rng, False, model, device, lmbd=lmbd, prior=prior)
            train_res = clf.fit(x_train, y_train).pred
            test_res = clf.predict(x_test).pred

            train_acc = accuracy_score(y_train_true, train_res)
            train_mcc = matthews_corrcoef(y_train_true, train_res)
            test_acc = accuracy_score(y_test_true, test_res)
            test_mcc = matthews_corrcoef(y_test_true, test_res)

            res_file.write(
                f"{dataset_name},{algo_name},{i},{train_acc:.6f},"
                f"{test_acc:.6f},{train_mcc:.6f},{test_mcc:.6f}\n"
            )


def main() -> None:
    """ Main """

    if len(sys.argv) != 3:
        print("Expected input: python vi_run_sup.py <algo> <dataset>")
        return

    algo_name_str = str(sys.argv[1]).strip()
    dataset_name_str = str(sys.argv[2]).strip()
    run_experiment(algo_name_str, dataset_name_str)


if __name__ == "__main__":
    main()
