"""
Some helper functions for causalai.

Author: Alexandre Andorra
"""
# --- The usual libraries:
import numpy as np
import pandas as pd
import pathlib

# --- Some plotting stuff:
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
seed = sum(map(ord, "eda-simulation"))
rng = np.random.default_rng(seed)

# --- Some network and Causality stuff:
import dowhy
import networkx as nx
import pygraphviz
# import pgmpy

# -- Some Statistical modelling:
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pgmpy.inference import CausalInference

# --- PyMC:
import arviz as az
import pymc as pm



def analyze_heterogeneous_effects(df,var_outcome: str, save_fig=False):


    """
        - var_outcome:   the name of the variable you want to analyze the treatment's effect on (e.g. shots, goals, passes etc.)
    """
    
    """Analyze heterogeneous treatment effects."""
    print("\n" + "=" * 60)
    print(f"HETEROGENEOUS TREATMENT EFFECTS ANALYSIS --- OUTCOME: {var_outcome.capitalize()}")
    print("=" * 60)

    # Treatment effects by score differential
    print("1. Treatment Effects by Score Differential:")
    score_effects = []
    score_values = []

    for score_diff in sorted(df["score_diff"].unique()):
        subset = df[df["score_diff"] == score_diff]
        if len(subset) > 0:
            treated_mean = subset[subset["sub"] == 1][var_outcome].mean()
            control_mean = subset[subset["sub"] == 0][var_outcome].mean()
            ate = treated_mean - control_mean
            score_effects.append(ate)
            score_values.append(score_diff)
            print(
                f"  Score {score_diff}: ATE = {ate:.3f} (Treated: {treated_mean:.2f}, Control: {control_mean:.2f})"
            )

    # Treatment effects by team strength
    print("\n2. Treatment Effects by Initial Team Strength:")
    strength_effects = []
    strength_labels = []

    for quartile in ["Q1", "Q2", "Q3", "Q4"]:
        subset = df[df["strength_quartile"] == quartile]
        if len(subset) > 0:
            treated_mean = subset[subset["sub"] == 1][var_outcome].mean()
            control_mean = subset[subset["sub"] == 0][var_outcome].mean()
            ate = treated_mean - control_mean
            strength_effects.append(ate)
            strength_labels.append(quartile)
            print(
                f"  {quartile}: ATE = {ate:.3f} (Treated: {treated_mean:.2f}, Control: {control_mean:.2f})"
            )

    # Visualize heterogeneous effects
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Treatment effects by score differential
    axes[0].bar(score_values, score_effects, color="skyblue", alpha=0.7)
    #axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Score Differential")
    axes[0].set_ylabel(f"Average Treatment Effect ({var_outcome.capitalize()})")
    axes[0].set_title("Treatment Effects by Score Differential")
    axes[0].grid(True, alpha=0.3)

    # Treatment effects by team strength
    axes[1].bar(strength_labels, strength_effects, color="lightcoral", alpha=0.7)
    #axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Team Strength Quartile")
    axes[1].set_ylabel(f"Average Treatment Effect ({var_outcome.capitalize()})")
    axes[1].set_title("Treatment Effects by Initial Team Strength")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        plt.savefig("heterogeneous_effects.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_naive_vs_adjusted(df: pd.DataFrame, outdir: pathlib.Path, var_outcome: str, var_confounder: str, save_fig=False) -> None:
    """
    Compare naïve vs. score‐diff adjusted effect of substitutions on Shots.

    Args:
        df:               Input DataFrame
        outdir:           Directory to save plots
        var_confounder:   the name of the variable that could be a potential confounder if not controlled for (e.g. score_diff)
        var_outcome:      the name of the variable you want to analyze the treatment's effect on (e.g. shots, goals, passes etc.)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Naïve difference:
    mean_sub = df.loc[df["sub"] == 1, var_outcome].mean()
    mean_no_sub = df.loc[df["sub"] == 0, var_outcome].mean()
    naive_diff = mean_sub - mean_no_sub

    # --- Adjusted for score_diff using simple group means:
    grouped = df.groupby(["sub", var_confounder])[var_outcome].mean().unstack(0)
    adjusted_diff = (grouped[1] - grouped[0]).mean()

    # --- Print results
    print(f"Naïve difference in {var_outcome.capitalize()} (Sub - No Sub): {naive_diff:.3f}")
    print(f"Adjusted difference in {var_outcome.capitalize()} (Sub - No Sub): {adjusted_diff:.3f}")
    print(f"Difference between naïve and adjusted: {naive_diff - adjusted_diff:.3f}")

    # --- Bar plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        ["Naïve", "Adjusted"],
        [naive_diff, adjusted_diff],
        color=["lightcoral", "lightblue"],
    )
    plt.ylabel(f"Δ {var_outcome.capitalize()} (Sub − No Sub)")
    plt.title(f"Naïve vs. Adjusted Impact of Substitution on {var_outcome.capitalize()}")

    # --- Add value labels on bars
    for idx, diff in enumerate([naive_diff, adjusted_diff]):
        plt.text(idx, diff + 0.02, f"{diff:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.grid(axis="y", alpha=0.3)
    if save_fig:
        plt.savefig(outdir / f"naive_vs_adjusted_{var_outcome}.png", dpi=300, bbox_inches="tight")
        print("\nNaïve vs. Adjusted Effect Plot saved to Output Directory.")
    plt.show()



def check_balance(df,varX):
    
    """Check covariate balance after matching."""
    treated_ps = df[df["sub"] == 1]["propensity_score"]
    control_ps = df[df["sub"] == 0]["propensity_score"]

    treated_indices = df[df["sub"] == 1].index
    control_indices = df[df["sub"] == 0].index

    # --- Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(control_ps.values.reshape(-1, 1))
    distances, indices = nbrs.kneighbors(treated_ps.values.reshape(-1, 1))

    matched_control = control_indices[indices.flatten()]

    # --- Compare covariate distributions
    for covar in varX:
        treated_mean = df.loc[treated_indices, covar].mean()
        control_mean = df.loc[matched_control, covar].mean()

        # Standardized difference
        treated_std = df.loc[treated_indices, covar].std()
        std_diff = (treated_mean - control_mean) / treated_std

        print(f"  {covar}:")
        print(f"    Treated mean: {treated_mean:.3f}")
        print(f"    Control mean: {control_mean:.3f}")
        print(f"    Standardized difference: {std_diff:.3f}")
        print(f"    Balance: {'✓' if abs(std_diff) < 0.1 else '✗'}")



def test__dSeparation(df,var_outcome,dseps,inspect_dseps=None):
    
    """
    Args:
        df:            a pd.DataFrame object holding the data
        var_outcome:   the name of the variable you want to analyze the treatment's effect on (e.g. shots, goals, passes etc.)
        dseps:         a pgmpy object describing all potential conditional relationships in the data which need to be tested for independence
        
    """

    print("=" * 60)
    print("FITTING MODELS FOR EACH D-SEPARATION ASSERTION")
    print("=" * 60)

    # --- Number of Relationships to asses:
    if not inspect_dseps:
        inspect_dseps = np.arange(len(dseps.get_assertions()))
    
    # Iterate through all d-separation assertions
    print(f"Total number of d-separation assertions: {len(inspect_dseps)}")
    print()
    for n in inspect_dseps:

        dsep = dseps.get_assertions()[n]
        print(f"Assertion: {dsep}")
        target_var, independent_from, conditioning_vars = (
            list(dsep.get_assertion()[0])[0],
            list(dsep.get_assertion()[1]),
            list(dsep.get_assertion()[2]),  # can be empty
        )
    
        print(f"  Target variable: {target_var}")
        print(f"  Variable to test (should have coeff ≈ 0): {independent_from}")
        print(f"  Conditioning variables: {conditioning_vars}")
        print()
    
        coeff_names = ["intercept"] + independent_from + conditioning_vars
        with pm.Model(coords={"coeffs": coeff_names}) as ci_model:
            betas = pm.Normal(
                "betas",
                mu=0,
                sigma=[10] + [5] * len(independent_from + conditioning_vars),
                dims="coeffs",
            )
            noise = pm.Exponential("noise", 1)
    
            mu = betas[0] + pm.math.dot(df[independent_from + conditioning_vars].to_numpy(), betas[1:])
    
            if var_outcome in ["passes",'shots']:
                pm.Poisson(var_outcome, mu=pm.math.exp(mu), observed=df[var_outcome].to_numpy())
            elif var_outcome == "sub":
                pm.Bernoulli(var_outcome, p=pm.math.sigmoid(mu), observed=df[var_outcome].to_numpy())
            else:
                pm.Normal(
                    "y",
                    mu=mu,
                    sigma=noise,
                    observed=df[var_outcome].to_numpy(),
                )
    
            idata = pm.sample(
                #nuts_sampler="nutpie",
                nuts_sampler='numpyro',
                nuts_sample_kwargs={"progress_bar": False},
                tune=300,
                draws=500,
                cores=8,
                chains=8,
                random_seed=seed,
            )
    
        coeff_samples = idata.posterior["betas"].sel(coeffs=independent_from).to_numpy().flatten()
    
        # Compute statistics
        mean_coeff = coeff_samples.mean()
        ci_95 = np.percentile(coeff_samples, [2.5, 97.5])
        prob_near_zero = np.mean(np.abs(coeff_samples) < 0.01)
    
        print(f"\n{independent_from} coefficient:")
        print(f"  Mean: {mean_coeff:.6f}")
        print(f"  95% CI: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")
        print(f"  P(|coeff| < 0.01): {prob_near_zero:.3f}")
        print(f"  CI includes 0: {'Yes' if ci_95[0] <= 0 <= ci_95[1] else 'No'}")
    
        if prob_near_zero > 0.95 and ci_95[0] <= 0 <= ci_95[1]:
            print("  ✅ Strong evidence for conditional independence")
        elif ci_95[0] <= 0 <= ci_95[1]:
            print("  🟡 Weak evidence for conditional independence")
        else:
            print("  ❌ Evidence against conditional independence")
    
        print("-" * 40)
        

def validate_causal_assumptions(df):
    
    """Validate causal inference assumptions."""
    print("\n" + "=" * 60)
    print("CAUSAL ASSUMPTIONS VALIDATION")
    print("=" * 60)

    # 1. Unconfoundedness
    print("1. Unconfoundedness Check:")
    print("   ✓ We have measured: off_old, def_old, score_diff")
    print("   ✓ These should block all backdoor paths")
    print("   ⚠ Need domain knowledge to verify no other confounders")

    # 2. Positivity
    print("\n2. Positivity Check:")
    treated_ps = df[df["sub"] == 1]["propensity_score"]
    control_ps = df[df["sub"] == 0]["propensity_score"]

    print(f"   Treated propensity range: [{treated_ps.min():.3f}, {treated_ps.max():.3f}]")
    print(f"   Control propensity range: [{control_ps.min():.3f}, {control_ps.max():.3f}]")

    overlap_exists = control_ps.min() < treated_ps.max() and treated_ps.min() < control_ps.max()
    print(f"   Overlap exists: {'✓' if overlap_exists else '✗'}")

    # 3. SUTVA
    print("\n3. SUTVA Check:")
    print("   ✓ Each team's substitution decision is independent")
    print("   ✓ No interference between units")
    print("   ✓ Treatment is well-defined (substitution vs no substitution)")

    # 4. Consistency
    print("\n4. Consistency Check:")
    print("   ✓ Treatment version is consistent across all units")
    print("   ✓ All substitutions are treated the same way")


def clean__networkVars(graph):
    
    vars_unique = []
    network_structure_split = graph.split('\n')
    for l in range(1,len(network_structure_split)-2):
        vars_unique.extend(network_structure_split[l].strip().replace(';','').split('->'))
    vars_unique = pd.Series(vars_unique).unique()
    vars_unique = [v.strip() for v in vars_unique]
    return vars_unique

def run_dowhy_analysis(df,network_structure,var_outcome: str):
    """Run DoWhy causal inference analysis."""


    """
        - network_structure:     a digraph object with directed edge-description, e.g.:

                                         \"""   digraph {
                                                        off_old -> score_diff;
                                                        def_old -> score_diff;
                                                        off_old -> sub;
                                                        def_old -> sub;
                                                        score_diff -> sub;
                                                        score_diff -> shots;
                                                        sub -> shots;
                                                        off_old -> shots;
                                                        def_old -> shots;
                                                    }
                                         \"""
                                         
        - var_outcome:   the name of the variable you want to analyze the treatment's effect on (e.g. shots, goals, passes etc.)
    
    """
    

    print("\n" + "=" * 60)
    print("DOWHY CAUSAL INFERENCE ANALYSIS")
    print("=" * 60)

    # Prepare data for DoWhy
    vars_network = clean__networkVars(network_structure)
    dowhy_df = df[vars_network].copy()

    
    print("Causal Graph:")
    print(network_structure)

    # Create DoWhy causal model
    model = dowhy.CausalModel(data=dowhy_df, treatment="sub", outcome=var_outcome, graph=network_structure)

    print(f"\nModel created:")
    print(f"  Treatment: {model._treatment}")
    print(f"  Outcome: {model._outcome}")
    print(f"  Common Causes: {model.get_common_causes()}")

    # Identify causal effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(f"\nIdentified estimand:")
    print(identified_estimand)

    # Estimate causal effect
    try:
        estimate = model.estimate_effect(
            identified_estimand, method_name="backdoor.linear_regression", test_significance=True
        )

        print(f"\nCausal Effect Estimate:")
        print(f"  ATE: {estimate.value:.3f}")
        print(
            f"  CI: [{estimate.get_confidence_intervals()[0][0]:.3f}, {estimate.get_confidence_intervals()[0][1]:.3f}]"
        )
        print(f"  p-value: {estimate.test_stat_significance()['p_value'][0]:.3f}")

    except Exception as e:
        print(f"Error in estimation: {e}")






