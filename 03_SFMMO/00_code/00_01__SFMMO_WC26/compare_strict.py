"""Strict-holdout robustness check.

Scores the improved re-fit against the de-vigged consensus books under both forecasting
semantics: sequential (Elo updates on prior held-out matches, matching the closing-odds
information set) and strict-holdout (Elo frozen at the train cut, a pre-tournament forecast).
Both renorm + LOFO-calibrated, paired bootstrap; also splits qualifiers vs World-Cup finals.

Run after sfmmo_fit.py and SFMMO_STRICT_HOLDOUT=1 sfmmo_fit.py:
uv run --group model python 00_code/compare_strict.py
"""

import os

import cloudpickle
import numpy as np
import pandas as pd
from scipy.stats import kstest

from foresight_scoring import (
    bayesian_bootstrap,
    log_score,
    randomized_pit,
    ranked_probability_score,
)
from sfmmo_eval import (
    ALL_FOLDS,
    DATA,
    N_BOOT,
    SEED,
    book_probs,
    competition,
    fold_arrays,
    lofo_calibrated,
    renorm,
)

VARIANTS = {
    "sequential (primary)": "Evaluation__SFMMOwm_DevK__scaleCS__EW__improved.pkl",
    "strict-holdout": "Evaluation__SFMMOwm_DevK__scaleCS__EW__improved_strict.pkl",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def load(stem):
    with open(os.path.join(DATA, stem), "rb") as f:
        return cloudpickle.load(f)["dict_preds"]


def vs_book(dp, book_df, rng, score_folds=None):
    """Pooled improved(renorm+LOFO-cal)-minus-book paired bootstrap on the matched fixtures.

    Calibration is leave-one-fold-out over ALL folds; `score_folds` (default all) selects which
    folds to score, so a qualifiers/finals subset reuses the same calibration.
    """
    all_folds = [f for f in ALL_FOLDS if f in dp and len(dp[f])]
    raw = {f: fold_arrays(dp, f) for f in all_folds}
    renormd = {f: (renorm(raw[f][0]), raw[f][1]) for f in all_folds}
    cal = lofo_calibrated(renormd)
    folds = score_folds if score_folds is not None else all_folds
    Pm, Pb, yy = [], [], []
    for f in folds:
        ids = raw[f][2]
        mdf = pd.DataFrame(cal[f][0], columns=["away", "draw", "home"], index=ids)
        mdf["y"] = renormd[f][1]
        j = mdf.join(book_df, how="inner", rsuffix="_bk").dropna()
        if len(j):
            Pm.append(j[["away", "draw", "home"]].to_numpy())
            Pb.append(j[["away_bk", "draw_bk", "home_bk"]].to_numpy())
            yy.append(j["y"].to_numpy(dtype=int))
    Pm, Pb, yy = np.vstack(Pm), np.vstack(Pb), np.concatenate(yy)
    d_rps = bayesian_bootstrap(
        ranked_probability_score(Pm, yy, reduce="none")
        - ranked_probability_score(Pb, yy, reduce="none"),
        n_draws=N_BOOT,
        rng=rng,
    )
    d_ll = bayesian_bootstrap(log_score(Pm, yy) - log_score(Pb, yy), n_draws=N_BOOT, rng=rng)
    return {
        "n": len(yy),
        "rps": (
            d_rps.mean(),
            np.quantile(d_rps, 0.025),
            np.quantile(d_rps, 0.975),
            float(np.mean(d_rps < 0)),
        ),
        "ll": (
            d_ll.mean(),
            np.quantile(d_ll, 0.025),
            np.quantile(d_ll, 0.975),
            float(np.mean(d_ll > 0)),
        ),
        "pit": float(kstest(randomized_pit(Pm, yy, rng), "uniform").pvalue),
    }


def main():
    rng = np.random.default_rng(SEED)
    odds = pd.read_csv(os.path.join(DATA, "odds_byMatch__WM.csv"))
    book_df, _ = book_probs(odds, "Avg")

    results = {}
    for label, stem in VARIANTS.items():
        if not os.path.exists(os.path.join(DATA, stem)):
            print(f"[skip] {label}: {stem} not found (run the fit for this variant first)")
            continue
        r = vs_book(load(stem), book_df, rng)
        results[label] = r
        print(f"\n{label} (n={r['n']}) vs consensus book:")
        print(
            f"  ΔlogLik = {r['ll'][0]:+.4f} [{r['ll'][1]:+.4f}, {r['ll'][2]:+.4f}]  P(model better)={r['ll'][3]:.3f}"
        )
        print(
            f"  ΔRPS    = {r['rps'][0]:+.4f} [{r['rps'][1]:+.4f}, {r['rps'][2]:+.4f}]  P(model better)={r['rps'][3]:.3f}"
        )
        print(f"  PIT KS p = {r['pit']:.3f}")

    # Per-competition split (qualifiers vs World-Cup finals) for the sequential primary variant.
    comp_results = {}
    seq_stem = VARIANTS["sequential (primary)"]
    if os.path.exists(os.path.join(DATA, seq_stem)):
        dp = load(seq_stem)
        all_folds = [f for f in ALL_FOLDS if f in dp and len(dp[f])]
        for comp in ("qualifiers", "finals"):
            cf = [f for f in all_folds if competition(f) == comp]
            r = vs_book(dp, book_df, rng, score_folds=cf)
            comp_results[comp] = r
            print(f"\n[sequential] {comp} (n={r['n']}) vs consensus book:")
            print(
                f"  ΔlogLik = {r['ll'][0]:+.4f} [{r['ll'][1]:+.4f}, {r['ll'][2]:+.4f}]  P(model better)={r['ll'][3]:.3f}"
            )
            print(
                f"  ΔRPS    = {r['rps'][0]:+.4f} [{r['rps'][1]:+.4f}, {r['rps'][2]:+.4f}]  P(model better)={r['rps'][3]:.3f}"
            )

    if results:
        os.makedirs(OUT, exist_ok=True)
        lines = [
            "# Improved model vs consensus books — robustness\n",
            "*Negative ΔlogLik / positive ΔRPS ⇒ the books are better. Renorm + LOFO-calibrated; "
            "paired Bayesian bootstrap.*\n",
            "## By forecasting semantics (all folds)\n",
            "| variant | n | ΔlogLik (model−book) [95% CI] | ΔRPS [95% CI] | P(model better) | PIT p |",
            "|---|--:|--:|--:|--:|--:|",
        ]
        for label, r in results.items():
            lines.append(
                f"| {label} | {r['n']} | {r['ll'][0]:+.4f} [{r['ll'][1]:+.4f}, {r['ll'][2]:+.4f}] "
                f"| {r['rps'][0]:+.4f} [{r['rps'][1]:+.4f}, {r['rps'][2]:+.4f}] "
                f"| {r['ll'][3]:.3f} / {r['rps'][3]:.3f} | {r['pit']:.3f} |"
            )
        if comp_results:
            lines += [
                "\n## By competition (sequential variant) — the finals are the actual event\n",
                "| competition | n | ΔlogLik (model−book) [95% CI] | ΔRPS [95% CI] | P(model better) |",
                "|---|--:|--:|--:|--:|",
            ]
            for comp, r in comp_results.items():
                lines.append(
                    f"| {comp} | {r['n']} | {r['ll'][0]:+.4f} [{r['ll'][1]:+.4f}, {r['ll'][2]:+.4f}] "
                    f"| {r['rps'][0]:+.4f} [{r['rps'][1]:+.4f}, {r['rps'][2]:+.4f}] "
                    f"| {r['ll'][3]:.3f} / {r['rps'][3]:.3f} |"
                )
        with open(os.path.join(OUT, "strict_robustness.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {OUT}/strict_robustness.md")


if __name__ == "__main__":
    main()
