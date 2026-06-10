"""compare_improved.py — did the re-fit actually change the verdict vs the bookmakers?

Compares three forecasters on the held-out folds, on proper scores with paired
Bayesian-bootstrap uncertainty:
  - ORIGINAL  : Max's committed DevK predictions, post-hoc renormalized + LOFO-calibrated
  - IMPROVED  : the re-fit from sfmmo_fit.py (tightened team priors + k_max=15), LOFO-calibrated
  - BOOK      : de-vigged consensus odds ("Avg")

Answers two questions the post-hoc eval could not: (1) does the *actually re-fit* model
beat / close on the consensus books? (2) did the modeling changes improve out-of-sample
over the original (paired, same fixtures)?

Run AFTER sfmmo_fit.py has produced the __improved.pkl:
  uv run python 00_code/compare_improved.py
"""

import os

import cloudpickle
import numpy as np
import pandas as pd

from foresight_scoring import (
    bayesian_bootstrap,
    log_score,
    randomized_pit,
    ranked_probability_score,
)
from scipy.stats import kstest  # noqa: E402
from sfmmo_eval import (  # reuse the tested primitives
    ALL_FOLDS,
    DATA,
    N_BOOT,
    SEED,
    book_probs,
    fmt_metrics_table,
    fold_arrays,
    lofo_calibrated,
    point_metrics,
    renorm,
)

ORIG_PKL = "Evaluation__SFMMOwm_DevK__scaleCS__EW.pkl"
IMP_PKL = "Evaluation__SFMMOwm_DevK__scaleCS__EW__improved.pkl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def load(stem):
    with open(os.path.join(DATA, stem), "rb") as f:
        return cloudpickle.load(f)["dict_preds"]


def align(ids_a, Pa, ya, ids_b, Pb):
    """Inner-join two forecasters' per-match probs on id_match; return aligned (Pa, Pb, y)."""
    a = pd.DataFrame(Pa, columns=["0", "1", "2"], index=ids_a)
    a["y"] = ya
    b = pd.DataFrame(Pb, columns=["0", "1", "2"], index=ids_b)
    j = a.join(b, how="inner", rsuffix="_b").dropna()
    return (j[["0", "1", "2"]].to_numpy(), j[["0_b", "1_b", "2_b"]].to_numpy(),
            j["y"].to_numpy(dtype=int))


def pooled_diff(P_model, P_other, y, rng, lower_better_metric=True):
    """Pooled paired Bayesian bootstrap of (model - other) on RPS and logLik."""
    d_rps = bayesian_bootstrap(
        ranked_probability_score(P_model, y, reduce="none")
        - ranked_probability_score(P_other, y, reduce="none"), n_draws=N_BOOT, rng=rng)
    d_ll = bayesian_bootstrap(log_score(P_model, y) - log_score(P_other, y), n_draws=N_BOOT, rng=rng)
    return {
        "n": len(y),
        "rps": (d_rps.mean(), np.quantile(d_rps, .025), np.quantile(d_rps, .975), float(np.mean(d_rps < 0))),
        "ll": (d_ll.mean(), np.quantile(d_ll, .025), np.quantile(d_ll, .975), float(np.mean(d_ll > 0))),
    }


def main():
    rng = np.random.default_rng(SEED)
    odds = pd.read_csv(os.path.join(DATA, "odds_byMatch__WM.csv"))
    book_df, _ = book_probs(odds, "Avg")
    orig, imp = load(ORIG_PKL), load(IMP_PKL)
    folds = [f for f in ALL_FOLDS if f in orig and len(orig[f]) and f in imp and len(imp[f])]
    print(f"folds with both original & improved: {folds}")

    # raw arrays
    o_raw = {f: fold_arrays(orig, f) for f in folds}
    i_raw = {f: fold_arrays(imp, f) for f in folds}

    # original: renorm (truncation) then LOFO-calibrate; improved: renorm (already ~1) then calibrate
    o_renorm = {f: (renorm(o_raw[f][0]), o_raw[f][1]) for f in folds}
    i_renorm = {f: (renorm(i_raw[f][0]), i_raw[f][1]) for f in folds}
    o_cal = lofo_calibrated(o_renorm)
    i_cal = lofo_calibrated(i_renorm)

    # per-fold metric tables
    rows_imp_raw = [(f, {**point_metrics(i_raw[f][0], i_raw[f][1]), "n": len(i_raw[f][1])}) for f in folds]
    rows_imp_cal = [(f, {**point_metrics(i_cal[f][0], i_renorm[f][1]), "n": len(i_renorm[f][1])}) for f in folds]

    # head-to-head vs book (pooled over folds with odds), for original+cal and improved+cal
    def vs_book(cal, renorm_dict):
        Pm, Pb, yy = [], [], []
        for f in folds:
            ids = o_raw[f][2] if renorm_dict is o_renorm else i_raw[f][2]
            mdf = pd.DataFrame(cal[f][0], columns=["away", "draw", "home"], index=ids)
            mdf["y"] = renorm_dict[f][1]
            j = mdf.join(book_df, how="inner", rsuffix="_bk").dropna()
            if len(j):
                Pm.append(j[["away", "draw", "home"]].to_numpy())
                Pb.append(j[["away_bk", "draw_bk", "home_bk"]].to_numpy())
                yy.append(j["y"].to_numpy(dtype=int))
        Pm, Pb, yy = np.vstack(Pm), np.vstack(Pb), np.concatenate(yy)
        return pooled_diff(Pm, Pb, yy, rng), Pm, yy

    orig_vs_book, _, _ = vs_book(o_cal, o_renorm)
    imp_vs_book, Pm_imp, y_imp = vs_book(i_cal, i_renorm)

    # improved vs original (paired on id_match, all folds)
    Pi_all, Po_all, y_all = [], [], []
    for f in folds:
        Pi, Po, yy = align(i_raw[f][2], i_cal[f][0], i_renorm[f][1], o_raw[f][2], o_cal[f][0])
        Pi_all.append(Pi)
        Po_all.append(Po)
        y_all.append(yy)
    Pi_all, Po_all, y_all = np.vstack(Pi_all), np.vstack(Po_all), np.concatenate(y_all)
    imp_vs_orig = pooled_diff(Pi_all, Po_all, y_all, rng)

    pit_imp = float(kstest(randomized_pit(Pm_imp, y_imp, rng), "uniform").pvalue)

    # ---- console ----
    def show(tag, d):
        print(f"\n{tag} (n={d['n']}):")
        print(f"  ΔRPS  ={d['rps'][0]:+.4f} [{d['rps'][1]:+.4f},{d['rps'][2]:+.4f}]  P(model better)={d['rps'][3]:.3f}")
        print(f"  ΔlogL ={d['ll'][0]:+.4f} [{d['ll'][1]:+.4f},{d['ll'][2]:+.4f}]  P(model better)={d['ll'][3]:.3f}")

    print("\n=== improved DevK: raw re-fit (per fold) ===\n" + fmt_metrics_table(rows_imp_raw))
    print("\n=== improved DevK: re-fit + calibration ===\n" + fmt_metrics_table(rows_imp_cal))
    show("IMPROVED+cal  vs  consensus book", imp_vs_book)
    show("ORIGINAL+renorm+cal  vs  consensus book", orig_vs_book)
    show("IMPROVED  vs  ORIGINAL (paired, same fixtures)", imp_vs_orig)
    print(f"\nimproved PIT KS p = {pit_imp:.3f}")

    # ---- report ----
    os.makedirs(OUT, exist_ok=True)

    def block(tag, d):
        return (f"**{tag}** (n={d['n']}): "
                f"ΔRPS {d['rps'][0]:+.4f} [{d['rps'][1]:+.4f}, {d['rps'][2]:+.4f}] "
                f"(P model better {d['rps'][3]:.3f}); "
                f"ΔlogLik {d['ll'][0]:+.4f} [{d['ll'][1]:+.4f}, {d['ll'][2]:+.4f}] "
                f"(P model better {d['ll'][3]:.3f}).")

    md = f"""# Re-fit vs bookmakers — does the improvement change the verdict?

*Generated by `compare_improved.py` (seed {SEED}). Negative ΔRPS / positive ΔlogLik ⇒ the
first model is better. Improved = re-fit with ZeroSumNormal(0.30) team priors + k_max=15.*

## Improved DevK — re-fit, per fold (raw)

{fmt_metrics_table(rows_imp_raw)}

## Improved DevK — re-fit + leave-one-fold-out calibration

{fmt_metrics_table(rows_imp_cal)}

## Head-to-head (pooled over folds with odds)

- {block("IMPROVED + calibration vs consensus book", imp_vs_book)}
- {block("ORIGINAL + renorm + calibration vs consensus book", orig_vs_book)}

## Did the re-fit help? (improved vs original, paired on the same fixtures)

- {block("IMPROVED vs ORIGINAL", imp_vs_orig)}

## Calibration

- improved DevK (re-fit + cal) randomized-PIT KS p = {pit_imp:.3f}

Per-fold convergence (max R-hat, divergences) is in
`Evaluation__SFMMOwm_DevK__scaleCS__EW__improved_diagnostics.json`.
"""
    with open(os.path.join(OUT, "improved_comparison.md"), "w") as f:
        f.write(md)
    print(f"\nwrote {OUT}/improved_comparison.md")


if __name__ == "__main__":
    main()
