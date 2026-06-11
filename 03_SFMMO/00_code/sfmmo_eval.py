"""Properly-scored evaluation of the committed SFMMO predictions vs the bookmaker consensus.

Scores log-loss / RPS / Brier against the de-vigged "Avg" odds, with Bayesian-bootstrap
uncertainty and randomized-PIT calibration. Two post-hoc fixes are applied to the predictions:
renormalizing the k_max=5 goal-PMF truncation, and leave-one-fold-out temperature calibration.
Scoring is vendored from foresight (see foresight_scoring/).

Run: uv run python 00_code/sfmmo_eval.py
"""

import os

import cloudpickle
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foresight_scoring import (  # noqa: E402
    bayesian_bootstrap,
    brier_score,
    log_score,
    mean_log_loss,
    randomized_pit,
    ranked_probability_score,
)
from scipy.stats import kstest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "10_data", "102_Development")
OUT = os.path.join(HERE, "artifacts")
SEED = sum(map(ord, "sfmmo"))
N_BOOT = 8000
ALL_FOLDS = ["WMQ2018", "WM2018", "WMQ2022", "WM2022", "WMQ2026"]
# Column / class order in Yhat and match_outcome: 0 = away win, 1 = draw, 2 = home win.
CLASSES = ["away win", "draw", "home win"]


def competition(fold):
    """Classify a fold as 'qualifiers' (WMQ*) or 'finals' (the actual World Cup, WM*).

    The two are different populations: the qualifiers are full of mismatches; the finals are
    the event being forecast. The model-vs-market comparison is reported per competition.
    """
    return "qualifiers" if fold.startswith("WMQ") else "finals"


# ----------------------------------------------------------------------------- model preds
def load_dict_preds(dev):
    path = os.path.join(DATA, f"Evaluation__SFMMOwm_Dev{dev}__scaleCS__EW.pkl")
    with open(path, "rb") as f:
        return cloudpickle.load(f)["dict_preds"]


def fold_arrays(dict_preds, fold):
    """(P (n,3) [away,draw,home], y (n,), id_match (n,)). Yhat & Y__SFM are row-aligned."""
    v = dict_preds[fold]
    P = v["Yhat"][["0", "1", "2"]].to_numpy(dtype=float)
    y = v["Y__SFM"]["match_outcome"].to_numpy(dtype=int)
    ids = v["Y__SFM"]["id_match"].to_numpy()
    return P, y, ids


def renorm(P):
    return P / P.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- calibration
def apply_temperature(P, T, eps=1e-12):
    """p^(1/T) renormalised == softmax(log p / T). T<1 sharpens, T>1 softens."""
    z = np.log(np.clip(P, eps, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(P, y, bounds=(0.3, 3.0)):
    res = minimize_scalar(
        lambda T: mean_log_loss(apply_temperature(P, T), y), bounds=bounds, method="bounded"
    )
    return float(res.x)


def lofo_calibrated(per_fold):
    """Leave-one-fold-out temperature: calibrate each fold with T fit on the OTHER folds.

    `per_fold` maps fold -> (P, y). Returns {fold: (P_calibrated, T_used)} (no leakage).
    """
    out = {}
    for f in per_fold:
        P_tr = np.vstack([per_fold[g][0] for g in per_fold if g != f])
        y_tr = np.concatenate([per_fold[g][1] for g in per_fold if g != f])
        T = fit_temperature(P_tr, y_tr)
        out[f] = (apply_temperature(per_fold[f][0], T), T)
    return out


# ----------------------------------------------------------------------------- bookmaker
def book_probs(odds, bookmaker="Avg"):
    """id_match -> [away, draw, home] de-vigged probabilities, + the raw overround.

    Decimal odds H/D/A -> implied [1/H, 1/D, 1/A] (= home, draw, away), reordered to
    [away, draw, home] and renormalised. The raw overround (sum of implied before
    renormalising) should exceed 1 for a real margin; a best-price line ('Max') often
    sums to < 1 and is NOT a coherent forecast — we flag it rather than launder it.
    """
    cols = [f"{bookmaker}{x}" for x in ["H", "D", "A"]]
    sub = odds.dropna(subset=cols).copy()
    H, D, A = (sub[c].astype(float).to_numpy() for c in cols)
    implied = np.vstack([1 / A, 1 / D, 1 / H]).T  # [away, draw, home]
    overround = implied.sum(axis=1)
    probs = implied / overround[:, None]
    df = pd.DataFrame(probs, columns=["away", "draw", "home"], index=sub["id_match"].to_numpy())
    return df, overround


# ----------------------------------------------------------------------------- metrics
def point_metrics(P, y):
    return {
        "logLik": mean_log_loss(P, y),
        "ACC": float(np.mean(np.argmax(P, axis=1) == y)),
        "RPS": ranked_probability_score(P, y),  # normalised (÷ K-1); notebook RPS = ×2
        "Brier": brier_score(P, y),
    }


def head_to_head(P_model, y, ids_model, book_df, rng):
    """Match model & book on id_match; paired Bayesian bootstrap -> P(model beats book)."""
    mdf = pd.DataFrame(P_model, columns=["away", "draw", "home"], index=ids_model)
    mdf["y"] = y
    j = mdf.join(book_df, how="inner", rsuffix="_bk").dropna()
    if len(j) == 0:
        return None
    yj = j["y"].to_numpy(dtype=int)
    Pm = j[["away", "draw", "home"]].to_numpy()
    Pb = j[["away_bk", "draw_bk", "home_bk"]].to_numpy()
    out = {"n": len(j)}
    # log score: higher is better -> P(model better) = P(diff > 0)
    dm = log_score(Pm, yj) - log_score(Pb, yj)
    bd = bayesian_bootstrap(dm, n_draws=N_BOOT, rng=rng)
    out["logLik"] = {
        "p_model_better": float(np.mean(bd > 0)),
        "delta_mean": float(bd.mean()),
        "ci": (float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))),
    }
    # RPS & Brier: lower is better -> P(model better) = P(diff < 0)
    for name, fn in [("RPS", ranked_probability_score), ("Brier", brier_score)]:
        dd = fn(Pm, yj, reduce="none") - fn(Pb, yj, reduce="none")
        bd = bayesian_bootstrap(dd, n_draws=N_BOOT, rng=rng)
        out[name] = {
            "p_model_better": float(np.mean(bd < 0)),
            "delta_mean": float(bd.mean()),
            "ci": (float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))),
        }
    return out


def pit_ks(P, y, rng):
    return float(kstest(randomized_pit(P, y, rng), "uniform").pvalue)


# ----------------------------------------------------------------------------- report
def df_to_md(df):
    """Minimal DataFrame -> markdown (avoids a tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    head = "| | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join("--:" for _ in cols) + "|"
    body = "\n".join(
        "| " + str(idx) + " | " + " | ".join(f"{df.loc[idx, c]}" for c in df.columns) + " |"
        for idx in df.index
    )
    return head + "\n" + sep + "\n" + body


def fmt_metrics_table(rows):
    head = "| fold | n | logLik↓ | ACC↑ | RPS↓ | Brier↓ |\n|---|--:|--:|--:|--:|--:|"
    body = "\n".join(
        f"| {f} | {m['n']} | {m['logLik']:.4f} | {m['ACC']:.4f} | {m['RPS']:.4f} | {m['Brier']:.4f} |"
        for f, m in rows
    )
    return head + "\n" + body


def main(focal="K"):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    odds = pd.read_csv(os.path.join(DATA, "odds_byMatch__WM.csv"))
    book_df, overround = book_probs(odds, "Avg")
    _, overround_max = book_probs(odds, "Max")

    dp = load_dict_preds(focal)
    folds = [f for f in ALL_FOLDS if f in dp and len(dp[f])]

    # --- raw / renormalized arrays + truncation diagnostics ----------------------------
    raw = {f: fold_arrays(dp, f) for f in folds}
    diag, per_fold_raw, per_fold_renorm = [], {}, {}
    for f in folds:
        P, y, ids = raw[f]
        rs = P.sum(axis=1)
        diag.append((f, len(y), float(np.median(rs)), float(np.mean(rs < 0.9))))
        per_fold_raw[f] = (P, y)
        per_fold_renorm[f] = (renorm(P), y)

    cal = lofo_calibrated(per_fold_renorm)  # renorm -> LOFO temperature

    # --- per-fold metric tables for the focal model: raw / renorm / calibrated ---------
    def rows_for(stage):
        rows = []
        for f in folds:
            if stage == "raw":
                P, y = per_fold_raw[f]
            elif stage == "renorm":
                P, y = per_fold_renorm[f]
            else:
                P, y = cal[f][0], per_fold_renorm[f][1]
            m = point_metrics(P, y)
            m["n"] = len(y)
            rows.append((f, m))
        return rows

    rows_raw, rows_renorm, rows_cal = rows_for("raw"), rows_for("renorm"), rows_for("cal")

    # --- A–K logLik clustering (reproduces the emailed table; sanity check) ------------
    ak = {}
    for dev in list("ABCDEFGHIJK"):
        d = load_dict_preds(dev)
        ak[dev] = {f: mean_log_loss(*fold_arrays(d, f)[:2]) for f in folds}
    ak_df = pd.DataFrame(ak).round(3)

    # --- book metrics on its own matched rows, per overlap fold ------------------------
    book_rows = []
    for f in folds:
        ids = raw[f][2]
        yb = pd.Series(raw[f][1], index=ids)
        j = book_df.join(yb.rename("y"), how="inner").dropna()
        if len(j) == 0:
            continue
        Pb = j[["away", "draw", "home"]].to_numpy()
        yj = j["y"].to_numpy(dtype=int)
        m = point_metrics(Pb, yj)
        m["n"] = len(j)
        book_rows.append((f, m))

    # --- head-to-head: improved model (renorm+calibrated) vs consensus book ------------
    h2h, h2h_raw = {}, {}
    pooled_m, pooled_b, pooled_y = [], [], []
    for f in folds:
        ids = raw[f][2]
        h2h[f] = head_to_head(cal[f][0], per_fold_renorm[f][1], ids, book_df, rng)
        h2h_raw[f] = head_to_head(per_fold_raw[f][0], per_fold_raw[f][1], ids, book_df, rng)
        # pooled overlap set (improved model)
        mdf = pd.DataFrame(cal[f][0], columns=["away", "draw", "home"], index=ids)
        mdf["y"] = per_fold_renorm[f][1]
        j = mdf.join(book_df, how="inner", rsuffix="_bk").dropna()
        if len(j):
            pooled_m.append(j[["away", "draw", "home"]].to_numpy())
            pooled_b.append(j[["away_bk", "draw_bk", "home_bk"]].to_numpy())
            pooled_y.append(j["y"].to_numpy(dtype=int))
    Pm, Pb, yov = np.vstack(pooled_m), np.vstack(pooled_b), np.concatenate(pooled_y)
    d_rps = bayesian_bootstrap(
        ranked_probability_score(Pm, yov, reduce="none")
        - ranked_probability_score(Pb, yov, reduce="none"),
        n_draws=N_BOOT,
        rng=rng,
    )
    d_ll = bayesian_bootstrap(log_score(Pm, yov) - log_score(Pb, yov), n_draws=N_BOOT, rng=rng)

    # --- PIT calibration (pooled overlap) ----------------------------------------------
    pit_model = pit_ks(Pm, yov, rng)
    pit_book = pit_ks(Pb, yov, rng)

    # --- calibration figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for P, lab, c in [(Pm, "model K (renorm+cal)", "C0"), (Pb, "book (Avg)", "C1")]:
        axes[0].hist(
            randomized_pit(P, yov, np.random.default_rng(SEED)),
            bins=20,
            histtype="step",
            density=True,
            label=lab,
            color=c,
            lw=2,
        )
    axes[0].axhline(1.0, ls="--", c="k", alpha=0.4)
    axes[0].set_title("Randomized PIT (flat = calibrated)")
    axes[0].set_xlabel("PIT")
    axes[0].legend()
    edges = np.linspace(0, 1, 11)
    for P, lab, c in [(Pm, "model K", "C0"), (Pb, "book", "C1")]:
        pk, ak_ = P[:, 2], (yov == 2).astype(float)
        cx, cy = [], []
        for i in range(10):
            msk = (pk >= edges[i]) & (pk < edges[i + 1])
            if msk.sum() >= 10:
                cx.append(pk[msk].mean())
                cy.append(ak_[msk].mean())
        axes[1].plot(cx, cy, "o-", color=c, label=lab)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("Reliability — home win")
    axes[1].set_xlabel("predicted P")
    axes[1].set_ylabel("observed")
    axes[1].legend()
    fig.tight_layout()
    figpath = os.path.join(OUT, "honest_calibration.png")
    fig.savefig(figpath, dpi=120)
    plt.close(fig)

    # --- write report.md ---------------------------------------------------------------
    def h2h_line(f):
        r = h2h[f]
        if r is None:
            return f"| {f} | — | no odds | | |"
        return (
            f"| {f} | {r['n']} | {r['RPS']['p_model_better']:.2f} "
            f"| {r['RPS']['delta_mean']:+.4f} [{r['RPS']['ci'][0]:+.4f}, {r['RPS']['ci'][1]:+.4f}] "
            f"| {r['logLik']['p_model_better']:.2f} |"
        )

    md = (
        f"""# SFMMO World-Cup model vs. the bookmakers — honest, properly-scored evaluation

*Generated by `sfmmo_eval.py` (seed {SEED}). Scoring vendored from the foresight package.
Lower is better for logLik / RPS / Brier; higher for ACC. RPS is normalized (÷ K−1);
multiply by 2 to compare with the notebook's un-normalized RPS.*

Focal model: **Dev{focal}**. Benchmark: **de-vigged consensus odds ("Avg")**. The best-price
"Max" line is NOT used as the headline — its raw implied probabilities sum below 1 on
{float(np.mean(overround_max < 1.0)) * 100:.0f}% of matches (median overround {np.median(overround_max):.3f}),
so it is not a coherent forecast; the consensus median overround is {np.median(overround):.3f}.

## 1. Probability-mass truncation (a real artifact in the committed predictions)

The W/D/L probabilities are a Poisson goal-PMF collapse truncated at k_max=5. On
extreme-λ fixtures (blowouts, cold-start qualifier teams) they lose most of their mass:

| fold | n | median row-sum | % rows summing < 0.9 |
|---|--:|--:|--:|
"""
        + "\n".join(f"| {f} | {n} | {md_:.3f} | {bad * 100:.1f}% |" for f, n, md_, bad in diag)
        + f"""

This **inflates the model's log-loss on the qualifiers** (every outcome gets a shrunken
probability). Renormalizing recovers most of it; the proper fix re-runs with a larger
k_max (`fit_sfmmo.py`).

## 2. Focal model Dev{focal} — raw vs renormalized vs calibrated

**Raw (as committed — what the emailed tables used):**

{fmt_metrics_table(rows_raw)}

**Renormalized (truncation fixed):**

{fmt_metrics_table(rows_renorm)}

**Renormalized + leave-one-fold-out temperature calibration:**

{fmt_metrics_table(rows_cal)}

LOFO temperatures: {", ".join(f"{f}={cal[f][1]:.2f}" for f in folds)}.

## 3. Bookmaker (consensus "Avg") on the matched fixtures

{fmt_metrics_table(book_rows)}

## 4. Head-to-head — improved model (renorm+calibrated) vs consensus book

Paired Bayesian bootstrap on the *same* fixtures ({N_BOOT} draws). `P(model better)` and the
RPS difference (model − book; negative ⇒ model better) with 95% CI:

| fold | n | P(model better, RPS) | ΔRPS (model − book) [95% CI] | P(model better, logLik) |
|---|--:|--:|--:|--:|
"""
        + "\n".join(h2h_line(f) for f in folds)
        + f"""

**Pooled across the {len(pooled_y)} overlapping folds (n={len(yov)}):**
- ΔRPS (model − book) = {d_rps.mean():+.4f}  [{np.quantile(d_rps, 0.025):+.4f}, {np.quantile(d_rps, 0.975):+.4f}]  → P(model better) = {np.mean(d_rps < 0):.3f}
- ΔlogLik (model − book) = {d_ll.mean():+.4f}  [{np.quantile(d_ll, 0.025):+.4f}, {np.quantile(d_ll, 0.975):+.4f}]  → P(model better) = {np.mean(d_ll > 0):.3f}

## 5. Calibration (randomized PIT, KS-uniformity p-value; pooled overlap)

- model Dev{focal} (renorm+cal): p = {pit_model:.3f}
- consensus book: p = {pit_book:.3f}

(p > 0.05 ⇒ not distinguishable from calibrated.) See `artifacts/honest_calibration.png`.

## 6. Model family A–K (mean log-loss) — the ladder is a dead end

{df_to_md(ak_df)}

All variants are within ~0.001–0.01 of each other on every fold: the momentum/elo-encoding
choices barely move anything. The gap that matters is model-vs-book, not model-vs-model.
"""
    )
    with open(os.path.join(OUT, "honest_report.md"), "w") as f:
        f.write(md)

    # --- console summary ---------------------------------------------------------------
    print(f"focal=Dev{focal}  folds={folds}")
    print("\ntruncation (median row-sum, %<0.9):")
    for f, n, md_, bad in diag:
        print(f"  {f:>8s} n={n:<4d} median={md_:.3f}  bad={bad * 100:.1f}%")
    print("\nlogLik raw -> renorm -> calibrated (focal):")
    for (f, mr), (_, mn), (_, mc) in zip(rows_raw, rows_renorm, rows_cal):
        print(f"  {f:>8s}: {mr['logLik']:.4f} -> {mn['logLik']:.4f} -> {mc['logLik']:.4f}")
    print(f"\nPOOLED overlap (n={len(yov)}) improved model vs consensus book:")
    print(
        f"  ΔRPS  ={d_rps.mean():+.4f} [{np.quantile(d_rps, 0.025):+.4f},{np.quantile(d_rps, 0.975):+.4f}]  P(model better)={np.mean(d_rps < 0):.3f}"
    )
    print(
        f"  ΔlogL ={d_ll.mean():+.4f} [{np.quantile(d_ll, 0.025):+.4f},{np.quantile(d_ll, 0.975):+.4f}]  P(model better)={np.mean(d_ll > 0):.3f}"
    )
    print(f"  PIT KS p: model={pit_model:.3f} book={pit_book:.3f}")
    print(f"\nwrote {OUT}/honest_report.md + honest_calibration.png")


if __name__ == "__main__":
    main()
