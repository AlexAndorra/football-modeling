"""Phase 2 of the Bayesian workflow: criticize the fitted model.

Loads the per-fold InferenceData saved by sfmmo_fit.py and runs two checks: prior sensitivity
(power-scaling, `arviz_stats.psense_summary`, needs the committed log_prior + log_likelihood)
and identifiability (posterior correlations among the intercept mu, the league effects kappa,
and the mean home advantage). W/D/L forecast calibration is covered in sfmmo_eval.py.

Run after sfmmo_fit.py: uv run --group model python 00_code/model_criticism.py
"""

import glob
import os

import arviz as az
import numpy as np
import pandas as pd

try:
    import arviz_stats
except Exception:
    arviz_stats = None

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "10_data", "102_Development")
OUT = os.path.join(HERE, "artifacts")
STEM = "Evaluation__SFMMOwm_DevK__scaleCS__EW__improved_idata_"
# Power-scaling sensitivity above this (Kallioinen et al.) flags a prior- or likelihood-sensitive
# parameter worth examining.
PSENSE_THRESHOLD = 0.05


def fold_paths():
    return sorted(glob.glob(os.path.join(DATA, STEM + "*.nc")))


def fold_name(p):
    return os.path.basename(p).replace(STEM, "").replace(".nc", "")


def prior_sensitivity(idata):
    """Return the psense_summary DataFrame (per-variable power-scaling sensitivity)."""
    return arviz_stats.psense_summary(idata)


def psense_family_summary(ps):
    """Collapse per-element psense rows to per-parameter-family max sensitivity.

    Excludes the per-observation deterministic `eta` — its power-scaling sensitivity is a
    mechanical consequence of being a data-driven transform, not a prior-robustness signal.
    Returns (summary_df, flagged_param_names).
    """
    base = ps.index.to_series().str.replace(r"\[.*\]", "", regex=True)
    sens_cols = [c for c in ps.columns if c in ("prior", "likelihood")]
    rows = []
    for fam, grp in ps.groupby(base):
        if fam == "eta":
            continue
        rec = {"param": fam, "n": len(grp)}
        for c in sens_cols:
            rec[f"max_{c}"] = float(grp[c].max())
        rec["n_flagged"] = int((grp[sens_cols] > PSENSE_THRESHOLD).any(axis=1).sum())
        rows.append(rec)
    summary = (
        pd.DataFrame(rows).sort_values("max_prior", ascending=False) if rows else pd.DataFrame()
    )
    flagged = summary.loc[summary["n_flagged"] > 0, "param"].tolist() if len(summary) else []
    return summary, flagged


def identifiability(idata):
    """Posterior correlations among mu, mean beta_home, and per-league kappa.

    Returns (corr_df, (var_a, var_b, max_off_diagonal_corr))."""
    post = idata.posterior
    cols = {"mu": post["mu"].values.reshape(-1)}
    bh = post["beta_home"]
    extra = [d for d in bh.dims if d not in ("chain", "draw")]
    cols["beta_home(mean)"] = bh.mean(dim=extra).values.reshape(-1)
    for i, lg in enumerate(post.coords["leagues"].values):
        tag = str(lg).split("-")[-1]  # e.g. 'asien', 'europa', 'weltmeisterschaft'
        cols[f"kappa:{tag}"] = post["kappa"].isel(leagues=i).values.reshape(-1)
    df = pd.DataFrame(cols)
    corr = df.corr()
    c = corr.to_numpy().copy()
    np.fill_diagonal(c, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    return corr, (corr.index[i], corr.columns[j], float(c[i, j]))


def main():
    if arviz_stats is None:
        raise SystemExit("arviz_stats not installed — run with `uv run --group model`.")
    os.makedirs(OUT, exist_ok=True)
    paths = fold_paths()
    if not paths:
        raise SystemExit(f"no idata .nc found in {DATA} (run sfmmo_fit.py first)")
    print("folds:", [fold_name(p) for p in paths])

    blocks = []
    for p in paths:
        f = fold_name(p)
        idata = az.from_netcdf(p)

        # --- prior sensitivity (per-parameter-family, eta excluded) ---
        try:
            ps = prior_sensitivity(idata)
            summary, flagged = psense_family_summary(ps)
            ps_txt = summary.round(4).to_string(index=False) if len(summary) else "no parameters"
        except Exception as e:
            ps_txt = f"psense_summary failed: {e}"
            flagged = [ps_txt]

        # --- identifiability ---
        corr, (va, vb, mx) = identifiability(idata)

        print(f"\n[{f}] prior-sensitive (>{PSENSE_THRESHOLD}): {flagged or 'none'}")
        print(f"[{f}] identifiability max|corr| = {mx:+.3f}  ({va} vs {vb})")

        blocks.append(
            f"### Fold {f}\n\n"
            f"**Prior sensitivity** (power-scaling; flagged > {PSENSE_THRESHOLD}): "
            f"{', '.join(flagged) if flagged else 'none — conclusions robust to the prior'}\n\n"
            f"```\n{ps_txt}\n```\n\n"
            f"**Identifiability**: max |posterior correlation| among mu / kappa / mean home "
            f"advantage = **{mx:+.3f}** ({va} vs {vb}) "
            f"({'OK — well separated' if abs(mx) < 0.9 else 'WARNING — near-collinear, not separately identified'}).\n"
        )

    md = (
        "# SFMMO improved model — criticism (prior sensitivity + identifiability)\n\n"
        "*Generated by `model_criticism.py`. Power-scaling sensitivity > "
        f"{PSENSE_THRESHOLD} flags a prior/likelihood-sensitive parameter; "
        "|posterior corr| ~ 1 between components means they are not separately identified.*\n\n"
        + "\n".join(blocks)
    )
    with open(os.path.join(OUT, "model_criticism.md"), "w") as fh:
        fh.write(md)
    print(f"\nwrote {OUT}/model_criticism.md")


if __name__ == "__main__":
    main()
