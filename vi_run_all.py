""" Run all experiments in separate processes. """

import subprocess

from joblib import Parallel, delayed  # type: ignore
from tqdm import tqdm  # type: ignore


def run_experiment(algo_name: str, dataset_name: str) -> None:
    """ Runs a single experiment in a separate process. """

    # Run RL experiment
    if dataset_name in (
        "bird-song", "lost",
        "mir-flickr", "msrc-v2",
        "yahoo-news",
    ):
        subprocess.run([
            "./.venv/bin/python", "vi_run_rl.py",
            algo_name, dataset_name,
        ], check=False)

    # Run supervised experiment with added noise
    elif dataset_name in (
        "mnist", "kmnist", "fmnist",
        "cifar10", "cifar100",
    ):
        subprocess.run([
            "./.venv/bin/python", "vi_run_sup.py",
            algo_name, dataset_name,
        ], check=False)

    else:
        raise RuntimeError("Unknown dataset.")


def main() -> None:
    """ Main """

    # Run all experiments
    Parallel(n_jobs=12, backend="threading")(
        delayed(run_experiment)(algo_name_str, dataset_name_str)
        for dataset_name_str in tqdm([
            # RL datasets
            "bird-song", "lost",
            "mir-flickr", "msrc-v2",
            "yahoo-news",
            # Vision datasets
            "mnist", "kmnist", "fmnist",
            "cifar10", "cifar100",
        ])
        for algo_name_str in [
            # ML methods
            "pl-knn-2005",
            "pl-ecoc-2017",
            # Deep Learning methods
            "proden-2020",
            "cavl-2021",
            "valen-2021",
            "pico-2022",
            "pop-2023",
            "crosel-2024",
            "cel-2025",
            # Own methods
            "vi-ablation",
            "vi-pll",
        ]
    )


if __name__ == "__main__":
    main()
