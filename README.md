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
* The folder `results` contains the results of all experiments. Run `python vi_run_all.py` to reevaluate all experiments.
* Additionally, there are the following files in the root directory:
  * `.gitignore`
  * `LICENSE` describes the repository's licensing.
  * `README.md` is this document.
  * `requirements.txt` is a list of all required `pip` packages for reproducibility.
  * `vi_create_exp.py` is a Python script to create all experimental configurations.
  * `vi_results.py` is a Python script to aggregate all results in the folder `results` into the files `results_rl.txt` and `results_sup.txt`.
  * `vi_run_all.py` runs all experimental configurations in the `experiments` folder on all algorithms.
  * `vi_run_rl.py` runs a single algorithm on a single real-world dataset.
  * `vi_run_sup.py` runs a single algorithm on a single supervised dataset with added candidates.

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all the necessary dependencies.
Our code is implemented in Python (version 3.13.5; other versions, including lower ones, might also work).

First, you need to install the correct Python version yourself.
Then, we used `venv` to create an environment for our experiments, which is pre-installed with Python versions greater than 3.3.
To create a virtual environment for this project, you have to clone this repository first.
Thereafter, change the working directory to this repository's root folder.
Run the following commands to create the virtual environment and install all necessary dependencies:

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
<td>

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
</tr>
</table>

## Reproducing the Experiments

Make sure that you created the virtual environment as stated above.
The script `vi_create_exp.py` creates all experimental data including the artificially added candidates.
Note that the folder `experiments` already contains the generated files, so you do not need to run it again.
The script `vi_run_all.py` runs all the experiments.
Running all experiments takes roughly two to three days on a system with 48 cores and one `NVIDIA GeForce RTX 4090`.

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
source venv/bin/activate
python vi_create_exp.py
python vi_run_all.py
```

</td>
<td>

``` powershell
.\venv\Scripts\Activate.ps1
python vi_create_exp.py
python vi_run_all.py
```

</td>
</tr>
</table>

This creates `.csv` files in `results` containing the results of all experiments.
Note that `python vi_run_all.py` only runs experiments that have no results in the folder `results` yet.
To re-evaluate all results, delete the contents of the folder `results` first.

## Reproducing the Tables

To obtain tables from the data, use the Python script `vi_results.py`.
Use the following snippets to generate all tables.
Generating all of them takes about a few seconds on a single core.

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
source venv/bin/activate
python vi_results.py
```

</td>
<td>

``` powershell
.\venv\Scripts\Activate.ps1
python vi_results.py
```

</td>
</tr>
</table>
