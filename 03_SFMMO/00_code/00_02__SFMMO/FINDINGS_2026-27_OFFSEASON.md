# SFMMO — Off-Season Revision 2026/27: Findings

*August 2026, **revision 2**. This document supersedes the 2026-08-03 version, whose
headline numbers were measured on a harness later found to leak same-match outcomes
through the ELO features (repair #7, §3) — the earlier market-skill claims were
flattered ~5× and the earlier variant selection did not survive the fix. Everything
below is from the honest harness. Status: specification committed (**Dev K**,
re-confirmed); holdout (2024/25) still unburned. RPS is /2-normalized throughout.*

*Reproducibility: the per-fold prediction pickles for all eight variants
(`10_data/102_Development/Evaluation__SFMMO_Dev{A,B,C,E,F,H,K,L}__scaleCS__EW.pkl`,
honest-harness runs of 2026-08-13; the superseded leak-era runs are preserved under
`102_Development/superseded_leak_era_2026-08-03/`) and the bookmaker odds
(`odds_byMatch.csv`, 1X2 incl. Pinnacle closing) suffice to reproduce every table
below. Note the dataset update of 2026-08 changed `id_match` gameday tokens from float
to int — normalize with `re.sub(r'_GD(\d+)\.0_', r'_GD\1_', id)` when joining odds or
leak-era pickles. The source data CSV (~110 MB) exceeds GitHub's file limit and is
available on request.*

---

## 1 · Summary

The World Cup 2026 campaign graded the SFMMO against the betting market and found it
**−25% behind (RPS skill, vs opening averages)** on frozen pre-match forecasts. A
referee-style audit of the league codebase identified six defect classes; a
collaborator's port review later found a seventh — an ELO implementation that
double-updated every match **and leaked each match's own outcome into the features of
one of its two rows**, present in fold validation but impossible in production. After
all repairs, on the honest harness:

| benchmark (7,224 matched, 4 seasons, 5 leagues) | RPS skill |
|---|---|
| vs. market average (opening odds) | **−3.1%** |
| vs. Pinnacle closing line | **−3.5%** |

An earlier revision of this document reported −0.5%/−0.7%; that was the leak talking
(§3, #7). The honest position: a competent public-data model, far from the WC's −25%,
not at market parity. The residual gap decomposes into a draw-discrimination component
the market resolves with information we don't have (§5), plus a general information
deficit. The committed 2026/27 specification is **Dev K** (team effects + ELO levels +
points-diff + cumulative goals) with a Dixon–Coles low-score correction — the
parsimonious incumbent, re-confirmed after a challenger's apparent victory proved to
be a harness artifact (§6).

---

## 2 · Protocol

- **Model.** Bayesian Poisson regression for per-team goals:
  `log λ = μ + α_team − δ_opp + home·β_home,team + Xβ`, with α, δ zero-sum team effects
  (attack / defense), hierarchical per-team home advantage, features cross-sectionally
  standardized per gameday. Estimated with PyMC (NUTS via numpyro).
- **Evaluation.** Expanding-window: train through season *t*, score season *t+1*
  frozen; four folds (validation seasons 2020/21 → 2023/24; **7,225 matches** after the
  2026-08 data update extended Ligue 1's history to 2015/16, which brings its 2020/21
  season into fold 1). All model comparisons are **paired tests on identical matches**.
- **Market benchmark.** De-vigged 1X2 odds (proportional normalization), Pinnacle
  closing and cross-book opening averages (~100% coverage on the updated id join).
- **Holdout.** Season 2024/25 is sealed and enforced by asserts in the notebook config.
  It has survived two selection reversals untouched — which is the point.
- **Selection rule (pre-registered).** A challenger replaces the incumbent only if it
  wins **both** log-loss and RPS with paired |t| ≥ 2.5; ties resolve to parsimony.

---

## 3 · Specification repairs (and what each was worth)

| # | Defect | Fix | Verified effect |
|---|---|---|---|
| 1 | **Momentum EWMA computed over doubled rows** (both perspective-rows of each match entered a team's points sequence) | single-perspective series; opponent columns mapped via match id | old values ≈ **½ of truth** (e.g. 1.63 vs 3.00); fixed block matches hand-computed ground truth to machine precision |
| 2 | **Cold-start regime silently deleted**: a blanket `dropna()` removed 100% of gamedays 1–2 (3,776 matches) under the momentum feature set — and gameday 1 under *every* set (zero-variance cross-sections) | drop only on target/IDs; undefined features set to league average **in standardized space**; per-gameday audit printed; metrics reported by gameday bucket | the regime that failed at the WC (MD1 log-loss worse than uniform) was structurally invisible in development; league cold start turns out mild (§4) |
| 3 | **No global intercept; asymmetric team effects** (α zero-sum, δ free) — the baseline rate lived in an implicit drift | explicit `μ ~ N(log ȳ, 0.2)`; both α and δ zero-sum, fixed scale 0.30 | identification cleaned; sampler runtime ~halved (`target_accept` 0.99 → 0.9); the WC "μ-bug" class is closed by an **η parity test** (reconstruction vs. graph, asserts < 1e-8; observed 4.4e-16) that gates every export |
| 4 | **Favourites mechanically shaved**: goals capped at 5 under an uncensored Poisson (λ=3.5 ⇒ 14% of mass above cap) and the scoreline grid truncated at k_max=5 (lopsided fixture ⇒ 10.6% of joint mass lost, nearly all favourite-win cells) | cap removed entirely (only 0.57% of rows exceed 5; max = 10); k_max = 15 | the pre-repair "under-confidence on favourites" was substantially artifact; post-repair conviction is calibrated (§4) |
| 5 | Censored-Poisson remedy for #4 **fails under the numpyro/JAX backend** (Poisson log-CDF gradient → 100% divergences) though it passes on the default C backend | no cap (above); finding recorded | *smoke-test on the sampler the notebook actually uses* |
| 6 | Reproducibility: unseeded sampler, seven duplicate eval cells, uniform-RPS baseline misprinted | seeds everywhere; single eval cell with per-bucket reporting; baseline computed from the outcome mix | reported numbers are now reproducible and self-auditing |
| 7 | **ELO double-update + same-match outcome leak** *(found after the audit, by the collaborator's port review — the audit itself missed it)*: the loop iterated perspective rows, so every match updated twice (effective K ≈ 2×20), the +50 home advantage went to both sides across the two calls (washing out), and the second-processed row was assigned **post-match ratings — the match's own result inside a feature** (point-biserial +0.46 with the row's outcome; 50% of rows). Present in training *and* fold validation; **impossible in production** (future fixtures have no outcome to leak) — so validation was systematically optimistic vs. deployment | one update per match, home side correctly attributed, pre-match ratings written to both rows (verified: within-match symmetry exactly 0; volatility 101 → 89 elo; feature corr. with intended ELO 0.976) | **the largest defect of the off-season**: on common matches every ELO-carrying variant scores +0.014–0.016 log-loss worse once honest (t ≈ +11); netting the concurrent data-update benefit (the ELO-free variant *improved*), the leak alone was worth ≈ 0.02 — ~10× the Dixon–Coles effect. It also manufactured a spurious variant-selection result (§6) |

---

## 4 · Baseline results (Dev K, honest harness)

| fold | n | log-loss | RPS | accuracy |
|---|---|---|---|---|
| 2020/21 | 1,826 | 1.0130 | 0.2085 | 0.504 |
| 2021/22 | 1,825 | 0.9894 | 0.2002 | 0.525 |
| 2022/23 | 1,825 | 0.9908 | 0.2043 | 0.528 |
| 2023/24 | 1,749 | 0.9751 | 0.1940 | 0.533 |
| **all** | **7,225** | **0.9922** | **0.2018** | **0.522** |

(Uniform baseline: log-loss 1.0986.) Sampling clean; η parity 4.4e-16.

**Calibration holds on the honest harness**: temperature fitted on folds 1–3 gives
T\* = 1.05 with no gain on the frozen fold-4 test; top-pick conviction 0.525 vs. a
0.522 hit rate (the closing market: 0.529 priced, 0.539 hit — its edge is accuracy,
not our calibration). **Cold start remains mild** (gd 1–2 log-loss 0.996 vs 0.992
overall; n=374) — league matchday 1 is not informationally cold. **Prior-predictive
check**: the β width (σ=0.3) still implies an inflated goals scale and, under
heavy-tailed momentum features, admits absurd rates (the check now reports this as
numbers rather than crashing); β 0.30 → 0.15 remains queued behind fold re-validation.

---

## 5 · Dixon–Coles low-score dependence (survives the honest harness)

Two-stage: ρ fitted by ML on each fold's *training* matches given fitted λ's; τ applied
per posterior sample to the out-of-sample score grids; mass preserved analytically.

- **ρ stable in all four folds: −0.053 … −0.055** (leak-era: −0.061 … −0.064).
- Priced draw share 23.3% → 24.4% (realized 25.5%; closing market 25.0%).
- vs. independent: paired t = **−3.3 (RPS), −3.2 (log-loss)**; accuracy unchanged. Kept.

| P(draw) priced on … | model (K + DC) | closing market |
|---|---|---|
| actual draws | 0.252 | 0.262 |
| non-draws | 0.242 | 0.247 |
| **discrimination spread** | **+1.05 pp** | **+1.53 pp** |

The conclusion of revision 1 stands, strengthened: τ fixes the draw *level*; the
market's remaining draw edge is **discrimination — information (lineups, incentives,
style, late money), not statistics**. No dependence correction can buy it.

---

## 6 · Variant selection: a cautionary tale in three verdicts

**Verdict 1 (June, WC data):** K selected; momentum judged inert — *measured while the
momentum EWMA was broken (repair #1).*

**Verdict 2 (2026-08-03, league folds):** with momentum fixed, the momentum variants
(A/C/E) beat K decisively (paired t ≈ −3.0 … −3.7); E was selected as the simplest
member of the winning class. *Measured while the ELO leaked (repair #7).*

**Verdict 3 (2026-08-13/14, honest harness + updated data, 7,225 paired matches):**

| variant | features | Δ log-loss vs K (t) | Δ RPS vs K (t) | verdict |
|---|---|---|---|---|
| A | all 22 (incl. momentum) | +0.0002 (+0.8) | +0.0001 (+1.0) | ≈ K |
| C | A + market values | +0.0002 (+0.8) | +0.0001 (+0.8) | ≈ K |
| E | A − cumulative goals | +0.0001 (+0.4) | +0.0001 (+0.8) | ≈ K |
| H | A − momentum-FD terms | +0.0001 (+0.5) | +0.0001 (+0.8) | ≈ K |
| F | K + ELO interaction | −0.0000 (−0.3) | −0.0000 (−0.6) | ≡ K |
| L | K + market values | +0.0001 (+0.3) | −0.0000 (−0.1) | ≡ K |
| B | market values only | +0.0112 (+7.9) | +0.0036 (+7.8) | rejected |

**Momentum's advantage vanished entirely with the leak.** No challenger clears the
pre-registered bar; parsimony retains **K**. The apparent vindication of verdict 2 was
a harness artifact — the paired t-statistics were correct arithmetic on contaminated
features. Two lessons, earned twice over:

1. **Every claim inherits the bugs of the harness that measured it.** A pre-registered
   selection rule executed flawlessly and was still wrong, because its inputs were.
2. **The holdout discipline is what kept this recoverable.** 2024/25 was nearly burned
   on Dev E between verdicts 2 and 3; it was postponed (for unrelated reasons) and so
   remains a clean, never-touched estimate for the final spec. Selection evidence is
   replaceable; a burned holdout is not.

**Committed spec (pending collaborator review): Dev K + Dixon–Coles τ, parity-tested,
honest-harness-confirmed.**

---

## 7 · Open items

1. **Holdout run** (2024/25, once, after collaborator review and any final fold-tested
   changes). Mechanism wired (`RUN_HOLDOUT`), sealed by default, still unburned.
2. β prior width 0.30 → 0.15 (hygiene; fold-re-validate before the holdout).
3. Joint (in-likelihood) Dixon–Coles and diagonal inflation for the 2-2/3-3 residual —
   expected effects below noise; "paper polish" tier.
4. Code unification into one shared feature/model module (three of the seven repairs —
   #1, #7, and the WC μ-bug — were divergence or double-row defects the η parity test
   and a single module would have prevented structurally).
5. Draw *discrimination* remains the only identified path to further market
   convergence and requires new information, not new statistics.

---

*Selection re-closed 2026-08-14 under the pre-registered rule (verdict 3). Any
subsequent change to the specification re-opens fold validation but never the holdout.*
