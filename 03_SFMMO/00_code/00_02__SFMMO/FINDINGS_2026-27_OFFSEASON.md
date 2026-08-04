# SFMMO — Off-Season Revision 2026/27: Findings

*August 2026. Status: specification selected and frozen pending collaborator review;
holdout (2024/25) deliberately unburned. All numbers below are from the repaired
evaluation harness; RPS is reported /2-normalized (3-class, ∈ [0, 1]) throughout.*

*Reproducibility: the per-fold prediction pickles for all eight variants
(`10_data/102_Development/Evaluation__SFMMO_Dev{A,B,C,E,F,H,K,L}__scaleCS__EW.pkl`,
each holding `dict_preds` with DC and independent probabilities, outcomes and ρ per
fold) and the bookmaker odds (`odds_byMatch.csv`, 1X2 incl. Pinnacle closing) suffice
to reproduce every table in §4–§6 without re-estimation. The source data CSV (~110 MB)
exceeds GitHub's file limit and is available on request.*

---

## 1 · Summary

The World Cup 2026 campaign graded the SFMMO honestly against the betting market and
found it **−25% behind (RPS skill, vs opening averages)** on frozen pre-match forecasts.
A referee-style audit of the league codebase then identified a set of specification and
evaluation defects. After repairing them — and re-running the model selection that some
of those defects had contaminated — the league model stands at:

| benchmark (6,347 matches, 4 seasons) | RPS skill |
|---|---|
| vs. market average (opening odds) | **−0.5%** |
| vs. Pinnacle closing line | **−0.7%** |

The residual gap is diagnosed (§5) as **information, not statistics**: the market knows
*which* matches are drawish; no dependence correction can buy that. The selected
2026/27 specification is **Dev E** (team effects + ELO + points-diff + momentum at three
horizons) with a Dixon–Coles low-score correction.

---

## 2 · Protocol

- **Model.** Bayesian Poisson regression for per-team goals:
  `log λ = μ + α_team − δ_opp + home·β_home,team + Xβ`, with α, δ zero-sum team effects
  (attack / defense), hierarchical per-team home advantage, features cross-sectionally
  standardized per gameday. Estimated with PyMC (NUTS via numpyro).
- **Evaluation.** Expanding-window: train through season *t*, score season *t+1*
  frozen; four folds (validation seasons 2020/21 → 2023/24; 6,844 matches, 5 leagues).
  All model comparisons are **paired tests on identical matches**.
- **Market benchmark.** De-vigged 1X2 odds (proportional normalization), Pinnacle
  closing and cross-book opening averages (~93% coverage).
- **Holdout.** Season 2024/25 is sealed and enforced by asserts in the notebook config.
  It will be scored **once**, for the final committed spec. Selection cannot touch it.
- **Selection rule (pre-registered).** A challenger replaces the incumbent only if it
  wins **both** log-loss and RPS with paired |t| ≥ 2.5; ties resolve to parsimony.

---

## 3 · Specification repairs (and what each was worth)

| # | Defect | Fix | Verified effect |
|---|---|---|---|
| 1 | **Momentum EWMA computed over doubled rows** (both perspective-rows of each match entered a team's points sequence) | single-perspective series; opponent columns mapped via match id | old values ≈ **½ of truth** (e.g. 1.63 vs 3.00); fixed block matches hand-computed ground truth to machine precision |
| 2 | **Cold-start regime silently deleted**: a blanket `dropna()` removed 100% of gamedays 1–2 (3,776 matches) under the momentum feature set — and gameday 1 under *every* set (zero-variance cross-sections) | drop only on target/IDs; undefined features set to league average **in standardized space**; per-gameday audit printed; metrics reported by gameday bucket | the regime that failed at the WC (MD1 log-loss worse than uniform) was structurally invisible in development; league cold start turns out mild (§4) |
| 3 | **No global intercept; asymmetric team effects** (α zero-sum, δ free) — the baseline rate lived in an implicit drift | explicit `μ ~ N(log ȳ, 0.2)`; both α and δ zero-sum, fixed scale 0.30 | identification cleaned; sampler runtime ~halved (`target_accept` 0.99 → 0.9); the WC "μ-bug" class is closed by an **η parity test** (reconstruction vs. graph, asserts < 1e-8; observed 4.4e-16) that gates every export |
| 4 | **Favourites mechanically shaved**: goals capped at 5 under an uncensored Poisson (λ=3.5 ⇒ 14% of mass above cap) and the scoreline grid truncated at k_max=5 (lopsided fixture ⇒ 10.6% of joint mass lost, nearly all favourite-win cells; win prob 0.730 vs 0.835 true) | cap removed entirely (only 0.57% of 72,726 rows exceed 5; max = 10); k_max = 15 | the pre-repair "under-confidence on favourites" was substantially artifact: post-repair, top-pick conviction matches the closing market (0.535 vs 0.530) |
| 5 | Censored-Poisson remedy for #4 **fails under the numpyro/JAX backend** (Poisson log-CDF gradient → 100% divergences) though it passes on the default C backend | no cap (above); finding recorded | *smoke-test on the sampler the notebook actually uses* |
| 6 | Reproducibility: unseeded sampler, seven duplicate eval cells, uniform-RPS baseline misprinted | seeds everywhere; single eval cell with per-bucket reporting; baseline computed from the outcome mix | reported numbers are now reproducible and self-auditing |

---

## 4 · Baseline results (Dev K, post-repair)

Per-season log-loss 0.965–0.985 (uniform: 1.099); accuracy ≈ 0.53. Sampling clean
(0–1 divergences per fold, R-hat and ESS satisfactory).

**Calibration is already correct.** Temperature scaling fitted on folds 1–3 gives
**T\* = 1.02** and changes nothing on the frozen fold-4 test. The T\* ≈ 0.85 measured
pre-repair was the artifact of defect #4 — a cautionary result for post-hoc calibration
fitted over specification errors.

**Cold start is mild in leagues** (gd 1–2 log-loss 0.982 vs 0.977 overall): unlike the
WC pre-tournament board, league matchday 1 is not informationally cold — team effects
and carried ELO know the teams. The planned cold-start prior work is deprioritized.

**Prior-predictive check fails formally** (implied 2.6 goals/team-game vs 1.37
empirical; the β prior width contributes ~70% of prior Var(η)). Queued: β 0.30 → 0.15.
With ~66k training rows the posterior is data-dominated, so no metric movement is
expected — this is workflow hygiene, and T\* ≈ 1 already confirms posterior spread is
right.

---

## 5 · Dixon–Coles low-score dependence

Two-stage: ρ fitted by ML on each fold's *training* matches given fitted λ's
(only 0-0/0-1/1-0/1-1 contribute); τ applied per posterior sample to the out-of-sample
score grids. τ preserves total probability mass analytically.

- **ρ is stable and well-identified: −0.061 … −0.064 in all four folds.**
- Priced draw share: 23.0% → 24.4% (realized 25.5%; closing market 24.9%).
- vs. independent: paired t = −3.4 (RPS), −3.8 (log-loss); accuracy unchanged. Kept.
- But it closes only **6% of the RPS gap** to the closing line. The decomposition
  explains why:

| P(draw) priced on … | independent | Dixon–Coles | closing market |
|---|---|---|---|
| actual draws | 0.238 | 0.252 | 0.261 |
| non-draws | 0.228 | 0.241 | 0.245 |
| **discrimination spread** | **+1.1 pp** | **+1.1 pp** | **+1.5 pp** |

τ lifts the draw *level* almost uniformly; it cannot know *which* matches are drawish.
The market's remaining edge is **draw discrimination — information (lineups,
incentives, style, late money), not statistics**. This bounds what any dependence
modeling can contribute and is the central strategic finding of the exercise.

---

## 6 · Variant re-trial and the 2026/27 specification

The incumbent (K: points-diff + cumulative goals + ELO levels) had been selected while
defects #1–#2 were active — momentum was judged *while broken*. The re-trial on the
repaired harness (8 variants, 6,844 paired matches):

| variant | features | Δ log-loss vs K (t) | Δ RPS vs K (t) | verdict |
|---|---|---|---|---|
| A | all 22 (incl. momentum) | −0.0017 (−3.0) | −0.0006 (−3.2) | qualifies |
| **C** | A + market values | −0.0019 (−3.3) | −0.0007 (−3.7) | qualifies |
| **E** | A − cumulative goals | −0.0017 (−3.1) | −0.0006 (−3.2) | **selected** |
| H | A − momentum-FD terms | −0.0013 (−2.6) | −0.0004 (−2.5) | qualifies (marginal) |
| F | K + ELO interaction | −0.0000 (−0.1) | −0.0000 (−0.5) | ≡ K |
| L | K + market values | −0.0001 (−0.3) | −0.0001 (−2.0) | no effect |
| B | market values only | +0.029 (+13.7) | +0.009 (+13.5) | rejected |

- **Momentum, once correctly computed, earns decisive readmission** — and its value is
  W/D/L *accuracy* (+0.4 pp) and conviction, not draw discrimination.
- **Cumulative goals are fully redundant given momentum** (A vs E: t = −0.2).
- **Transfermarkt values add nothing** in any combination.
- A/C/E/H are pairwise indistinguishable; **E** is the simplest member of the top class:
  *strength (α/δ) + class (ELO) + form (momentum)*.
- Market ladder: E reaches **−0.67%** vs closing (K: −0.96%) — momentum closes ~⅓ of
  K's residual gap, ~5× the Dixon–Coles contribution.
- Caveat, stated for the record: every momentum variant loses to K in the 2020/21 fold
  (+0.0015 log-loss) and wins the three seasons since.

**Committed spec (pending review): Dev E + Dixon–Coles τ, parity-tested.**

---

## 7 · Open items

1. **Holdout run** (2024/25, once, after collaborator review and any final fold-tested
   changes). Mechanism is wired (`RUN_HOLDOUT`), sealed by default.
2. β prior width 0.30 → 0.15 (hygiene; must be fold-re-validated before the holdout,
   not slipped in after selection).
3. Joint (in-likelihood) Dixon–Coles via `pm.Potential`, and diagonal inflation for the
   2-2/3-3 residual — expected effects below noise; "paper polish" tier.
4. Code unification: extract the feature/model pipeline shared by notebook and
   deployment into one module (the η parity test currently guards the seam).
5. Draw *discrimination* is the only identified path to further market convergence and
   requires new information (e.g. lineup/rotation proxies), not new statistics.

---

*Selection closed 2026-08-03 under the pre-registered rule. Any subsequent change to
the specification re-opens fold validation but never the holdout.*
