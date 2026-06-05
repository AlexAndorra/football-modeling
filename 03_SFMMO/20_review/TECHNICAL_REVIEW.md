# SFMMO (World-Cup 2026 edition) -- technical review

**Scope:** `03_SFMMO/` v0 + the reported bookmaker comparison · **Date:** 2026-06-04

This reviews the committed code in `03_SFMMO/00_code/SFMMOwm__dev_EW.ipynb` and the headline result Max reported:

| Validation set | SFMMO acc | Best-of-~10 books | Verdict reported |
|---|---|---|---|
| (I) World Cup 2022 finals | 0.64 | 0.60 | SFMMO "knocked it out of the park" |
| (II) "Qualifiers 2026" | 0.62 | 0.66 | bookmakers ahead |

We re-ran the model locally (devVersion 'A', a 3-cycle window, `nutpie`) to put numbers behind the comments. **Headline: the model itself is in good shape -- several concerns we started with did not survive contact with the data.** The actionable items are mostly about *reproducibility*, *how the comparison is framed statistically*, and *one genuine prior-specification fix*. Everything below is ordered by priority. Artifacts referenced live in `20_review/analysis/`.

> **Scope of our re-run:** devVersion A on a 3-cycle window, probing the **model only** (the bookmaker side isn't committed, so we couldn't validate it). Treat the diagnostic numbers as config-specific -- please cross-check against the exact variant/window behind your numbers.

---

## 0. What's genuinely strong (and now evidence-backed)

- **Clean out-of-sample design.** Expanding window, model **re-fit per window**, OOS features standardized with **training** statistics (`SFMMOwm__dev_EW.ipynb` Grand Loop ~L351, L594). Momentum is a backward-looking within-season EWMA -- **not** a leakage path. No gross leakage.
- **Proper scoring already implemented.** RPS, log-loss, Brier (Cell 7) plus calibration work (PPC, LOO-PIT, rootograms, P(draw) by league, Cells 16–20). This is *not* an accuracy-only setup, which is exactly why the bookmaker comparison should use these too (§2).
- **The model is well-behaved -- we checked the things a reviewer would attack:**
  - **Draws are well-calibrated** (in-sample): observed P(draw) = **0.197** vs model **0.205**. The classic independent-Poisson draw deficit **does not bite here** -- the covariates carry it. *No Dixon-Coles / bivariate Poisson needed.* (This is on the fit window; an OOS draw-calibration check -- which the notebook already supports -- would confirm it.) (`draw_calibration.png`)
  - **Poisson is the right likelihood.** LOO: Poisson vs NegBin **ΔELPD = 0.8 ± 0.5** (Poisson marginally ahead, NegBin weight 0). Marginal overdispersion (var/mean = 1.65) is absorbed by the predictors. *Leaving the NegBin branch off is the correct call.* (`loo_compare.csv`)
  - **Well-identified.** Despite α being `ZeroSumNormal` and δ a free `Normal` with no global intercept, mean(δ) ≈ **−0.008** (it does *not* drift to absorb a baseline) and attack/defense corr = **+0.30** (benign -- good teams are good at both, not a degenerate trade-off). (`identifiability.png`)
  - **0 divergences** at `target_accept=0.9` (the notebook's 0.99 is unnecessarily conservative → slower for no benefit).

---

## 1. Reproducibility & housekeeping  — **highest priority (it blocks the headline claim)**

1. **The bookmaker comparison is not in the repo.** Full-repo search finds no odds data, no de-vigging, no "best quote of ~10 books" logic; the data file has 45 columns and **none are odds**. The committed notebook also has **no executed World-Cup outputs** (the only committed `df_eval` is left over from a *club-season* run, rows `2020/21–2023/24` -- a different dataset, so it says nothing about your WC numbers; it just means the run behind the slides isn't captured). So the 0.64/0.60/0.62/0.66 numbers can't be reproduced or vetted from what's committed.
    **Please commit** (a) the odds source + which books, (b) the de-vig method, (c) the exact run/config + **Dev variant** that produced the numbers.
2. **Label/version mismatch.** The committed code validates on **2022** (`WMQ2022` + `WM2022`). Your set (II) is labelled "**Qualifiers 2026**." `WMQ2026` exists in the data (**778 matches**) but the string `WMQ2026` appears **0 times** in the notebook --> there's an uncommitted run. Worth reconciling so the slide labels match the code.

## 2. How the comparison is framed (the statistics)

**Accuracy can't crown a winner in *either* direction at these sample sizes** (`checks_significance.py`, `significance_results.csv`):

| Set | n | diff (model−book) | z (unpaired) | p | significant? |
|---|---|---|---|---|---|
| (I) WC2022 finals | 64 | **+0.040** (+2.6 matches) | +0.47 | 0.64 | no -- squarely within noise |
| (II) Qualifiers 2026 | 778 | **−0.040** (−31 matches) | −1.65 | 0.10 | no (suggestive; a *paired* test could push lower) |

To detect a 4-point gap at 80% power you'd need **~2,300 matches/group**. So the WC-2022 "win" is ~2–3 matches of luck; the qualifier gap is the more substantial of the two but still not nailed down on accuracy alone. **Recommendations:**
- **Score the bookmaker on RPS / log-loss**, not accuracy -- you already compute these for the model (Cell 7). Accuracy throws away calibration and the ordinal W/D/L structure, and argmax-of-3 almost never picks "draw."
- **Use a paired test** (McNemar or a paired bootstrap on the per-match scores) -- model and books predict the *same* fixtures, so the paired test is both correct and more powerful. (This needs the per-match bookmaker probabilities).
- **De-vig is moot for accuracy** (argmax is scale-invariant) **but required for proper scores.** And "best quote per outcome across ~10 books" is **not a coherent probability vector** (the three best prices are taken from different books and needn't sum to 1). Report a **consensus benchmark** too -- e.g. de-vigged median/average odds -- alongside "best quote", and name the de-vig method (proportional / Shin / log).
- **Break results down by confederation** -- the qualifier pool mixes 6 very different leagues (§5).

---

## 3. Priors & prior predictive — **the one real model-spec fix**

There is **no prior predictive check** in the notebook. We added one, and it's the only place the model misbehaves (`prior_predictive.png`):

- prior-predictive goals: **mean ≈ 8.6**, **P(goals > 10) ≈ 9.6%** (vs ~0.5% observed), **max ≈ 304,000**.
- Cause: ~22 factors at `β ~ Normal(0, 0.3)` **plus** the team effects, with **no global intercept**, make the log-rate prior far too wide; `exp(η)` then has an absurd right tail.

The likelihood (~3,000 obs) rescues the **posterior**, so this doesn't wreck the fitted results — but it's worth fixing because:
- **Regularization vs overfitting.** Tighter priors = more shrinkage. This is directly relevant to the SFM II finding that momentum components **overfit out-of-sample** — wider priors let momentum features chase noise.
- **Cold-start.** For teams unseen in training (lots of them in qualifiers, §5), the prior dominates the posterior, so a pathological prior genuinely degrades those predictions.

Add a prior predictive check to the workflow; tighten the factor priors (try `β ~ Normal(0, 0.1–0.15)`, validated OOS -- we haven't confirmed the tighter prior *improves* OOS, only that the current one is implausible a priori), and consider an **explicit intercept** with a goals-scale prior so the baseline is set deliberately rather than implicitly via team-effect variance.

---

## 4. Convergence & hygiene (minor)

- At 1,000 draws we get **R̂ₘₐₓ = 1.03** (just above the 1.01 rule) and **ESS_bulk ≈ 200**. Bump draws and use **4 chains** at least.
- Drop `target_accept` **0.99 to ~0.9**: we saw **0 divergences** at 0.9, so 0.99 is just slowing sampling.
- Use a **descriptive seed** (`sum(map(ord, "..."))`) instead of `random_seed=42`.
- Minor likelihood-coherence nits: goals are **clipped at 7** for fitting (`np.where(...>7,7,...)`, L670) but predictions use an unclipped Poisson; and the W/D/L collapse uses **`k_max=5`**, dropping 6+ goal mass so the three probabilities don't sum to 1. Raise `k_max` (≥10) and prefer `pm.Censored` over clipping if you keep the cap. Low impact (draws calibrate fine), but easy to tidy.

---

## 5. Open design questions (not bugs) & why qualifiers likely lag

- **Which Dev variant produced the headline numbers?** Let's make sure the reported run is the OOS-robust one, not the in-sample winner.
- **Cold-start prior is ad-hoc.** Unseen teams get a 25th-percentile anchor for α/δ (Grand Loop ~L674). A confederation-level hierarchical prior would be more principled, and matters most for qualifiers, which are full of teams with little/no training history.
- **Why the books pull ahead on qualifiers -- testable hypotheses:**
  1. **Information edge (most likely).** The 2026 qualifiers are recent, so a 10-book ensemble prices current form, injuries, line-ups and dead-rubber motivation that a historical-aggregate + ELO + momentum model structurally can't. This fits "the books suddenly showed vastly improved performance" story.
  2. **Confederation heterogeneity + cold-start.** Six leagues with very different strength spreads; `κ ~ ZeroSumNormal(σ=0.3)` may be too tight to separate them; Oceania blow-outs are still in; many unseen teams hit the anchor + the over-wide prior (§3).
  3. **The WC-2022 "win" may just be 64-match variance** (§2), so the contrast with qualifiers is partly a small-sample artifact, not only a real degradation.

## 6. Suggested next steps (ranked)

1. Commit the bookmaker pipeline + the exact run/Dev variant; reconcile the 2026 label.
2. Re-issue the comparison on **RPS/log-loss with a paired test + per-confederation split**; add a consensus de-vigged benchmark next to "best quote".
3. Add a **prior predictive check** and tighten factor priors (+ optional explicit intercept).
4. Convergence: 4 chains, more draws, `target_accept≈0.9`, descriptive seed.
5. Report the **DevA–DevI OOS selection** and firm up the **cold-start prior**.

*If you want to reproduce our analyses:* `python 03_SFMMO/20_review/analysis/run_sfmmo.py` (env: a PyMC ≥5.28 / arviz ≥0.23 stack) and `python .../checks_significance.py`. Traces (`*.nc`) are gitignored but regenerate from the script.
