# Variational Inference for Partial-Label Learning: A Probabilistic Approach to Label Disambiguation

This repository contains the code and data of the paper

> Tobias Fuchs, Nadja Klein; "Variational Inference for Partial-Label Learning: A Probabilistic Approach to Label Disambiguation".

This document provides (1) an outline of the repository structure and (2) steps to reproduce the experiments including setting up a virtual environment.

## Repository Structure

* The folder `blip2_feat` contains the extracted features of the `cifar10` and `cifar100` datasets using the vision model `blip2`.
* The folder `data` contains all datasets used within our work.
  * The subfolder `realworld-datasets` contains commonly used real-world datasets for partial-label learning, which were initially provided by [Min-Ling Zhang](https://palm.seu.edu.cn/zhangml/Resources.htm).
  * Pytorch will additionally download datasets within this folder during the first run of the experiments.
* The folder `experiments` contains all supervised datasets with added instance-dependent noise.
* The folder `models` contains code for supervised reference models such as the MLP architecture.
* The folder `partial_label_learning` contains the code for the experiments.
  * The subfolder `methods` contains all implementations of related-work algorithms and our method.
* The folder `results` contains the results of all experiments. Run `uv run vi_run_all.py` to run all experiments.
* Additionally, there are the following files in the root directory:
  * `.gitignore`
  * `.python-version` is the targeted Python version.
  * `LICENSE` describes the repository's licensing.
  * `pyproject.toml` is the `uv` project description file.
  * `README.md` is this document.
  * `uv.lock` is the list of all installed packages in this project.
  * `vi_create_exp.py` is a Python script to create all experimental configurations.
  * `vi_results.py` is a Python script to aggregate all results in the folder `results` into the files `results_rl.txt` and `results_sup.txt`.
  * `vi_run_all.py` runs all experimental configurations in the `experiments` folder on all algorithms.
  * `vi_run_rl.py` runs a single algorithm on a single real-world dataset.
  * `vi_run_sup.py` runs a single algorithm on a single supervised dataset with added candidates.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/).

1. Install `uv`.
2. Create and sync the environment:

```bash
uv sync
```

3. Activate the environment or use the local interpreter directly:

```bash
source .venv/bin/activate
```

The project currently targets Python `>=3.12`; see [`pyproject.toml`](pyproject.toml).

## Reproducing the Experiments

The script `vi_create_exp.py` creates all experimental data including the artificially added candidates.
Note that the folder `experiments` already contains the generated files, so you do not need to run it again.
The script `vi_run_all.py` runs all the experiments.
Running all experiments takes roughly two to three days on a system with 48 cores and one `NVIDIA GeForce RTX 4090`.
Running `uv run vi_run_all.py` creates `.csv` files in `results` containing the results of all experiments.
Note that `uv run vi_run_all.py` only runs experiments that have no results in the folder `results` yet.
To re-evaluate all results, delete the contents of the folder `results` first.

## Reproducing the Tables

To obtain tables from the data, use the Python script `vi_results.py`, i.e., `uv run vi_results.py`.
Generating all of them takes about a few seconds on a single core.
