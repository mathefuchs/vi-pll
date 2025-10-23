""" Aggregate results """

import sys
from glob import glob

import numpy as np
import pandas as pd  # type: ignore
from scipy.stats import ttest_ind  # type: ignore


def main(output_latex: bool) -> None:
    """ Main """

    # Datasets and algorithms
    datasets_rl = [
        "bird-song", "lost",
        "mir-flickr", "msrc-v2",
        "yahoo-news",
    ]
    datasets_sup = [
        # Vision datasets
        "mnist", "kmnist", "fmnist",
        "cifar10", "cifar100",
    ]
    algorithms = [
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

    # Extract results
    res_matrix = {}
    for fname in sorted(glob("./results/*.csv")):
        test_accs = []
        ds, algo = "error", "error"
        with open(fname, "r", encoding="utf-8") as file:
            for line in file:
                if line:
                    parts = line.split(",")
                    assert len(parts) == 7
                    ds, algo, _, _, test_acc, _, _ = parts
                    test_accs.append(float(test_acc))
        res_matrix[ds, algo] = test_accs

    # Build table
    def build_table(dss):
        rows = []
        for algo in algorithms:
            row = []
            for ds in dss:
                # Check significant differences
                best_algo_idx = int(np.argmax([
                    np.mean(res_matrix[ds, algo1])
                    for algo1 in algorithms
                ]))
                if algo == algorithms[best_algo_idx]:
                    sig_str = " **"
                else:
                    test = ttest_ind(
                        res_matrix[ds, algo],
                        res_matrix[ds, algorithms[best_algo_idx]],
                    )
                    if test.pvalue < 0.05:
                        sig_str = ""
                    else:
                        sig_str = " *"

                test_accs = res_matrix[ds, algo]
                mean, std = float(np.mean(test_accs)), float(np.std(test_accs))
                row.append(f"{mean:.4f} (+/- {std:.4f})" + sig_str)
            rows.append(row)
        return rows

    tab1 = build_table(datasets_rl)
    tab2 = build_table(datasets_sup)

    if output_latex:
        pd.DataFrame(
            tab1, index=algorithms, columns=datasets_rl,
        ).to_latex("results_rl.tex")
        pd.DataFrame(
            tab2, index=algorithms, columns=datasets_sup,
        ).to_latex("results_sup.tex")
    else:
        pd.DataFrame(
            tab1, index=algorithms, columns=datasets_rl,
        ).to_markdown("results_rl.txt")
        pd.DataFrame(
            tab2, index=algorithms, columns=datasets_sup,
        ).to_markdown("results_sup.txt")

    # Compute significant wins/ties/losses
    sig_wtl = {
        algo: np.array([0, 0, 0], dtype=int)
        for algo in algorithms
    }
    for algo1 in algorithms:
        for algo2 in algorithms:
            if algo1 <= algo2:
                continue

            for ds in datasets_rl + datasets_sup:
                test = ttest_ind(
                    res_matrix[ds, algo1],
                    res_matrix[ds, algo2],
                )
                if test.pvalue < 0.05:
                    if np.mean(res_matrix[ds, algo1]) < np.mean(res_matrix[ds, algo2]):
                        sig_wtl[algo1] += np.array([0, 0, 1])  # Loss
                        sig_wtl[algo2] += np.array([1, 0, 0])  # Win
                    else:
                        sig_wtl[algo1] += np.array([1, 0, 0])  # Win
                        sig_wtl[algo2] += np.array([0, 0, 1])  # Loss
                else:
                    sig_wtl[algo1] += np.array([0, 1, 0])  # Tie
                    sig_wtl[algo2] += np.array([0, 1, 0])  # Tie
    print("Method & Wins & Ties & Losses \\\\")
    print("\\midrule")
    for algo in algorithms:
        w, t, l = list(sig_wtl[algo])
        print(f"{algo} & {w} & {t} & {l} \\\\")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        use_latex = int(sys.argv[1]) == 1
        main(use_latex)
    else:
        main(False)
