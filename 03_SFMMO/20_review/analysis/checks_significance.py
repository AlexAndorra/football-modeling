"""
A1 — Significance / power of the reported SFMMO-vs-bookmaker ACCURACY gaps.

No MCMC. Quantifies whether the headline accuracy differences are distinguishable
from noise given the sample sizes. Reported by Max:
  (I)  World Cup 2022 finals : SFMMO 0.64 vs books 0.60   (n = 64 matches)
  (II) "Qualifiers 2026"     : SFMMO 0.62 vs books 0.66   (n = computed from data)

Caveat baked into the output: model and bookmaker score the SAME matches, so the
proper test is PAIRED (McNemar), which needs the per-match discordant counts (b, c)
we do not have. The two-proportion z-test below treats the samples as INDEPENDENT,
which is CONSERVATIVE when the two forecasters are positively correlated (they almost
always agree on easy matches) — i.e. it OVER-states the p-value. So "not significant
even under the conservative test" is a safe one-way conclusion.
"""
import numpy as np
import pandas as pd
from scipy import stats

CSV = "03_SFMMO/10_data/data_byPlayer__SFM_II__TM__WM.csv"


def match_counts():
    df = pd.read_csv(CSV, usecols=["id_match", "season"])
    n = df.groupby("season")["id_match"].nunique().sort_index()
    return n


def wald_ci(p, n, z=1.96):
    se = np.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se


def two_prop_z(p1, p2, n):
    """Unpaired two-proportion z for a difference, equal n per group."""
    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return se, z, p


def n_for_power(p1, p2, power=0.80, alpha=0.05):
    """Per-group n to detect |p1-p2| at given power (two-sided, unpaired)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    delta = abs(p1 - p2)
    return (za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / delta ** 2


def report(label, p_model, p_book, n):
    print(f"\n=== {label}  (n = {n} matches) ===")
    print(f"  model acc = {p_model:.3f}   95% CI {wald_ci(p_model, n)[0]:.3f}–{wald_ci(p_model, n)[1]:.3f}")
    print(f"  book  acc = {p_book:.3f}   95% CI {wald_ci(p_book, n)[0]:.3f}–{wald_ci(p_book, n)[1]:.3f}")
    se, z, p = two_prop_z(p_model, p_book, n)
    diff = p_model - p_book
    print(f"  diff = {diff:+.3f}  ({diff * n:+.1f} matches)   SE(diff) = {se:.3f}   z = {z:+.2f}   p = {p:.3f} (unpaired, conservative)")
    need = n_for_power(p_model, p_book)
    print(f"  → to detect a {abs(diff):.2f} gap at 80% power (a=0.05) you'd need ~{need:,.0f} matches/group")
    sig = "NOT significant" if p > 0.05 else "significant"
    print(f"  → verdict: {sig} at the reported n")
    return dict(label=label, n=n, p_model=p_model, p_book=p_book, diff=diff,
                net_matches=diff * n, se=se, z=z, p=p, n_for_80pct_power=need,
                significant=p <= 0.05)


if __name__ == "__main__":
    print("Unique matches per season in the data file:")
    nc = match_counts()
    for s, n in nc.items():
        print(f"  {s:9s} {n:5d}")

    rows = []
    rows.append(report("(I) World Cup 2022 finals", 0.64, 0.60, 64))
    # (II) qualifiers — Max said "2026"; report both candidate seasons for n
    n_q26 = int(nc.get("WMQ2026", np.nan))
    n_q22 = int(nc.get("WMQ2022", np.nan))
    rows.append(report("(II) Qualifiers 2026  [WMQ2026]", 0.62, 0.66, n_q26))
    rows.append(report("(II-alt) Qualifiers 2022 [WMQ2022]", 0.62, 0.66, n_q22))

    out = pd.DataFrame(rows)
    out.to_csv("03_SFMMO/20_review/analysis/significance_results.csv", index=False)
    print("\nsaved -> 03_SFMMO/20_review/analysis/significance_results.csv")
