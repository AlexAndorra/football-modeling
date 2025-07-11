"""
Exploratory Data Analysis (EDA) for the synthetic soccer substitution
simulation dataset.  This script pairs with `soccer_substitution_simulation.py`
from the same project.

Goals
-----
1. Load or generate the simulated dataset.
2. Produce summary statistics and distribution plots for key features
   and outcomes.
3. Quantify naïve vs. adjusted estimates of substitution impact on shots
   to motivate the causal analysis step.
4. Save figures and simple CSV summaries for easy inclusion in reports
   or dashboards.

Usage Examples
--------------
$ python soccer_substitution_eda.py               # generate data and run EDA
$ python soccer_substitution_eda.py --n 20000     # simulate 20k matches
$ python soccer_substitution_eda.py --csv data.csv  # load existing CSV

Dependencies
------------
- pandas
- numpy
- matplotlib
- seaborn (optional, improves plotting aesthetics)

The script follows Data Science Best Practices:
- Clear, modular structure with functions for each logical step (DS BP #1)
- Type hints for maintainability (Python BP #2)
- Main‐guard for CLI execution (Python BP #5)
- Parameterization via argparse for reproducibility (DS BP #1)

"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Local import – assumes both scripts live in the same directory or PYTHONPATH
try:
    from soccer_substitution_simulation import simulate_matches
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Could not import `simulate_matches`. Ensure that "
        "`soccer_substitution_simulation.py` is on your PYTHONPATH."
    ) from e


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="EDA for soccer substitution simulation"
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=None,
        help="Path to an existing CSV to load instead of simulating.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10000,
        help="Number of matches to simulate if --csv is not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility when simulating.",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("eda_outputs"),
        help="Directory to save figures and summaries.",
    )
    return parser.parse_args()


def load_or_simulate(
    n: int, seed: int, csv_path: Optional[pathlib.Path] = None
) -> pd.DataFrame:
    """Load a dataset from CSV or simulate a fresh one."""
    if csv_path and csv_path.exists():
        print(f"Loading data from {csv_path}")
        return pd.read_csv(csv_path)
    print(f"Simulating dataset with n={n}, seed={seed}")
    return simulate_matches(n_matches=n, seed=seed)


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with basic descriptive statistics."""
    summary = df.describe().T
    # Add count of substitutions for context
    summary.loc["T", "count"] = df["T"].sum()
    return summary


def plot_distributions(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    """Plot key distribution histograms and save figures."""
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "Shots",
        "Goals",
        "Passes",
        "Tackles",
        "Clearances",
        "Pressures",
    ]
    for metric in metrics:
        plt.figure()
        sns.histplot(df[metric], kde=True)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.savefig(outdir / f"{metric.lower()}_hist.png", dpi=300)
        plt.close()


def plot_naive_vs_adjusted(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    """Compare naïve vs. score‐diff adjusted effect of substitutions on Shots."""
    outdir.mkdir(parents=True, exist_ok=True)
    # Naïve difference
    mean_shots_sub = df.loc[df["T"] == 1, "Shots"].mean()
    mean_shots_no_sub = df.loc[df["T"] == 0, "Shots"].mean()
    naive_diff = mean_shots_sub - mean_shots_no_sub

    # Adjusted for score_diff using simple group means
    grouped = df.groupby(["T", "score_diff"])["Shots"].mean().unstack(0)
    adjusted_diff = (grouped[1] - grouped[0]).mean()

    # Bar plot
    plt.figure()
    sns.barplot(x=["Naïve", "Adjusted"], y=[naive_diff, adjusted_diff])
    plt.ylabel("Δ Shots (Sub − No Sub)")
    plt.title("Naïve vs. Score‐Adjusted Impact of Substitution on Shots")
    for idx, diff in enumerate([naive_diff, adjusted_diff]):
        plt.text(idx, diff + 0.02, f"{diff:.2f}", ha="center")
    plt.savefig(outdir / "naive_vs_adjusted_shots.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    df = load_or_simulate(n=args.n, seed=args.seed, csv_path=args.csv)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Summary statistics
    summary_df = basic_summary(df)
    summary_path = outdir / "summary_stats.csv"
    summary_df.to_csv(summary_path)
    print(f"Saved summary stats to {summary_path}")

    # 2. Distributions
    plot_distributions(df, outdir)
    print("Saved distribution plots.")

    # 3. Naïve vs. adjusted substitution effect on Shots
    plot_naive_vs_adjusted(df, outdir)
    print("Saved naïve vs. adjusted effect plot.")


if __name__ == "__main__":
    main()
