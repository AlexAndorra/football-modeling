# Referee Report — SFMMO (league variant)

**Manuscript under review:** `SFMMO__dev_EW.ipynb` (2026-05-29, pre-World-Cup) with
`SEASON_PLAN_2026-27.md` (the analyst's lessons-learnt) read alongside.
**Unusual and valuable feature of this review:** the model family has a completed,
frozen, honestly-graded out-of-sample experiment — the WC2026 campaign (104 matches,
`SFMMOwm_validation__*`). Wherever possible, each point below is tied to that record
rather than to taste.

**Overall assessment:** The core design is sound and several practices are above the
bar for applied work in this space (leakage-free cross-sectional scaling with stored
train moments; a true expanding-window design; posterior-predictive checking cells;
principled handling of unseen teams). The recommendation is **major revision**: the
evaluation design cannot observe the model's known weakest regime, the calibration
deficit has identifiable *specification* causes that should be fixed before any
post-hoc temperature is applied, and the model now exists in three diverged copies.

---

## Major points

### M1 — The evaluation design is blind to the model's weakest regime (cold start)

Cell 11 drops all rows with any missing engineered feature **before** the
train/validation split (`complete_data[IDvar+[Yvar]+factors].dropna()`, line ~325).
The momentum block's first-difference terms are undefined for each team's first
appearance(s) of a season, so **early-gameday matches are silently excluded from both
training and validation**. Consequence: the dev harness *never scored* the
early-season regime — and the first time the model met a genuine cold start (WC MD1),
it produced log-loss 1.133, *worse than uniform* (the plan's L1). The weakness was
discovered in production precisely because it was structurally invisible in
development. **Remedy:** the E-harness must include gamedays 1–5 in the scored set
(impute/zero the undefined features exactly as the production pipeline would, rather
than dropping the rows), and report metrics *by gameday bucket*. Note this remedy is
free under Model-K-style feature sets (M5), which have no first-difference features.

### M2 — The under-confidence on favourites (L2) is partly *built into the spec*; fix that before fitting a temperature

The plan attributes the RPS deficit to calibration and proposes temperature sharpening
(E2). Three mechanical contributors in this notebook should be repaired first, because
they shave favourite win-probability *deterministically*:

1. **Winsorizing the likelihood target at 5** (line ~322) with an uncensored Poisson
   likelihood. For a strong side (λ = 3.5) the truncated tail is **14.2%** of its goal
   mass; at λ = 4.5, **29.7%**. Fitting an uncensored Poisson to a capped outcome
   biases λ̂ downward for exactly the teams L2 identifies. If capping is desired,
   use a censored likelihood (`pm.Censored`); otherwise raise the cap (the WC variant
   already uses 7) or drop it.
2. **`k_max = 5` in the joint-PMF grid** (cell 9). For a lopsided fixture
   (λₕ = 3.2, λₐ = 0.8) the 5×5 grid loses **10.6%** of joint probability mass, almost
   all of it in favourite-win cells; if W/D/L is read off the truncated grid without
   renormalization the favourite's price is shaved directly. The WC variant was
   patched to `k_max = 15` for this reason; the league notebook was not.
3. **Shrinkage with no explicit intercept** (see M3): all β's are shrunk toward zero
   against a standardized design, so extreme fixtures are pulled toward the field
   mean by construction.

Only after these are fixed does the *residual* miscalibration merit a temperature —
otherwise E2 will estimate a sharpening constant that mostly compensates for
truncation artifacts, and will silently mis-calibrate again when those are later
repaired.

> **RESOLUTION (August 2026) — the cap is removed entirely; do NOT re-introduce
> censoring.** The first remedy above (a right-censored likelihood) was implemented and
> then **reverted after failing on Colab**: `pm.Censored` requires the Poisson log-CDF,
> whose gradient is NaN under the **numpyro/JAX** backend this notebook samples with →
> 100% divergences. It samples cleanly on PyMC's default C backend, which is precisely
> why the original smoke test passed and the failure only appeared in production —
> *always smoke-test on the sampler the notebook actually uses, not the default one.*
> The adopted fix is simpler and strictly better: **no cap at all** (`GOALS_CAP = None`).
> Measured on the real data (72,726 team-match rows, max = 10 goals): only **414 rows
> (0.57%) exceed 5** and **32 rows (0.04%) exceed 7**, so the cap bought almost no
> robustness while introducing a real downward bias on strong attacks. `k_max = 15`
> covers the tail in the scoreline grid, and the W/D/L evaluation already reads the
> uncapped `match_outcome__orig`. Note the diagnosis in M2.1 stands — capping at 5 under
> an uncensored Poisson *was* biasing λ̂ down for favourites; only the remedy changed.

### M3 — No intercept, and an asymmetric α/δ specification

The linear predictor (line ~530) is
`η = α_team − δ_opp + home·β_home,team + Xβ` with **no global intercept**:
`α ~ ZeroSumNormal(σ_α)` but `δ ~ Normal(0, σ_δ)` (free, *not* zero-sum). The
baseline scoring rate is therefore identified only through the drift of E[−δ]
against its own shrinkage prior — an implicit intercept estimated under a prior that
says it is zero. This is not hypothetical: the WC deployment's **7.7% λ-deflation
bug** arose exactly because the intercept's location was ambiguous between notebook
and pipeline. The analyst has *already recommended* an explicit intercept with a
goals-scale prior (and prior-predictive checks); this manuscript still lacks it.
**Remedy:** `μ ~ Normal(log(ȳ), 0.2)`; make **both** α and δ zero-sum (the WC FinalK
already does, with fixed scale 0.30 — adopt it here); report a prior-predictive check
of implied goals per game.

### M4 — Draw underpricing (L3) is structural under independent double-Poisson

The two sides of a match are conditionally independent Poissons given η. It is well
documented (Dixon & Coles 1997; Karlis & Ntzoufras 2003) that this underprices draws
via missing low-score dependence — consistent with the WC knockout evidence (draw
probs ~21% vs market ~33% on 90′-level games). The plan's E3 is the correct response
and this notebook already contains the grading cells for it (cells 16–18, observed vs
posterior-predictive draw rates by league). **Endorsed**, with two additions: (i) fit
the dependence parameter ρ jointly, not as a post-hoc bump; (ii) evaluate under the
**90′ convention** as well as folded (the plan's L4/L7) since draw calibration is
precisely where the two conventions diverge.

### M5 — The feature set: a known-buggy momentum block, known collinearity, and a default the WC evidence already rejected

Three sub-points:

1. **The momentum EWMA is computed over doubled rows.** The block (lines ~104–154)
   selects `(name_team == tt) | (name_opp == tt)` from a table with **two
   perspective-rows per match**, so each match enters the team's points sequence
   twice; `ewm(halflife=1)` over that sequence has an effective halflife of half a
   match and the first-differences alias. The WC port fixed this
   (single-perspective); the league notebook did not inherit the fix.
2. **Collinearity is not academic here.** The WC knockout experiment showed the
   champion board *reorders* depending on which of the correlated form features
   (cum-goals vs points_diff) is included — large offsetting betas on correlated
   regressors. DevA carries 22 standardized factors including 14 momentum terms and
   two unmotivated interactions (`teamMOM__S_L`, `elo_team_opp`).
3. **The tournament's own model selection discarded nearly all of this.** The
   deployed Model K = points_diff + 4 cum-goals + ELO. Momentum contributed nothing
   measurable across variants D–H.

**Remedy:** make the K-style parsimonious set the league *default*; re-admit momentum
only if it beats that default on the frozen harness (M1's version, early gamedays
included) — in which case fix the doubled-row bug first. This also resolves M1's
dropna problem at the root.

### M6 — The model now exists in three diverged copies; the divergences are not innocuous

League notebook vs WC final notebook vs `006_050` currently differ on: δ prior
(free Normal vs zero-sum fixed-scale), intercept (absent vs present), winsorization
cap (5 vs 7), `k_max` (5 vs 15), `target_accept` (0.99 vs 0.9 — the WC showed 0.99
is unnecessary after reparameterization and roughly doubles runtime),
**cross-sectional standardization grouping** (here `groupby('gameday')` pools across
leagues *and seasons*; the WC variant scales per season × league × gameday — these
are materially different estimators of the cross-section, and late gamedays here pool
only the 38-game leagues, changing bucket composition), and gameday parsing
(bespoke string slicing vs regex — the WC id-collision incident shows how this class
of duplication bites). **Remedy:** extract one shared `sfmmo_features.py` +
`sfmmo_model.py` consumed by notebook and pipeline alike; every divergence above
becomes a single documented switch. Until then, every "lesson learnt" must be ported
by hand three times, and the record shows at least two (momentum fix, k_max) were not.

### M7 — Reproducibility and selection effects in the dev loop

1. The production sampler call (line ~561, numpyro branch) has **no `random_seed`** —
   only the unused BART branch sets one. Reported numbers are not exactly
   reproducible.
2. Cells 20–26 are **seven identical copies** of the evaluation cell; version
   comparison is by eye against overwritten state. Which run produced which reported
   table is not recoverable from the artifact.
3. Nine dev variants (A–I, later K) were compared on the **same four validation
   seasons**, and the winner's headline numbers are those same folds — a garden of
   forking paths. The WC served as a genuine test once; 2026/27 should keep one
   season (or the WC vintages) as an untouched holdout that no experiment may tune
   against.

**Remedy:** seed every sampler call; replace the duplicate cells with the planned
harness writing one results ledger (version, seed, data hash, both conventions,
market column); pre-register the E1–E3 success criteria against the holdout —
the plan's own receipts ethos, applied to development.

---

## Minor points

- **Feature-timing audit.** Momentum applies `ewm().mean()` *including* the current
  row; whether `points_team`/`FD_points` are pre- or post-match states is a data
  contract inherited from the upstream builder. If post-match, the current match's
  result leaks into its own in-sample features (OOS rows can't leak — the values
  don't exist — which would *itself* create a train/test feature-definition
  asymmetry). Add one assertion to the harness that recomputes a sample of features
  from prior matches only. Cheap insurance either way.
- **Promoted-team ELO = 1300 flat**; returning teams regress 25% to 1500. Reasonable,
  but E1's informed priors should supersede the constant (second-division
  performance, promoted-team historical hazard). Also note `kick_off` is
  date-normalized, so same-day sequential ELO updates occur in arbitrary row order —
  deterministic but unprincipled; harmless today, worth one sorting tiebreak.
- **New-team anchor** at the 25th percentile of posterior medians (line ~679) is
  sensible but ad hoc — estimate the promoted-team offset from history instead, and
  state it.
- **W/D/L normalization:** if the truncated k_max grid is not renormalized, report
  row sums; with k_max = 15 this is cosmetic, with 5 it is not (see M2.2).
- `idata_kwargs={"log_likelihood": True}` on every fold, with full idata stored in
  `dict_fitEval`, is memory-heavy and unused by any LOO/WAIC comparison — either use
  it (LOO between variants would strengthen model selection) or drop it.
- The per-match `pd.concat`-in-loop evaluation (lines ~765–785) is O(n²); vectorize
  before the harness scales to five leagues × season.
- The market column (L8) is absent from the dev loop entirely — the single most
  informative benchmark the WC produced. The harness should carry de-vigged closing
  odds beside every dev metric from day one.

---

## On the season plan (read as the "response to reviewers")

The plan is unusually honest and mostly correct; the receipts-first premise is the
right strategic conclusion from the ledger. Three amendments:

1. **Priority order.** The plan ranks E2 (temperature) as "the single highest-value
   fix". This review disagrees on sequencing: the M2/M3 specification repairs are
   near-zero-cost, remove *known* deterministic biases, and change what E2 would
   estimate. Do E0 (unify code, M6) → spec fixes (M2/M3) → **then** E2/E3 on the
   harness, with E1 alongside. Fitting a temperature over truncation artifacts bakes
   the artifact into the calibration constant.
2. **L1's root cause is mislabelled.** The cold start is not merely the model's
   weakest product; it was *unmeasurable* under the dev design (M1). The fix is as
   much harness as model.
3. **Add the holdout discipline** (M7.3) to §2 explicitly: no experiment may tune
   against the frozen WC vintages *and* claim them as validation.

---

## Verdict

**Major revision — with enthusiasm.** The infrastructure around this model (frozen
vintages, honest grading, the market benchmark, the convention discipline learned at
cost) is now stronger than the model specification itself. The revision path is
unusually concrete because the WC ledger already priced every weakness: fix what is
mechanical (M2, M3, M5.1), unify what is duplicated (M6), make the harness see what
production sees (M1, M7), and only then calibrate what remains (E2, E3). None of it
requires new data, and most of it requires deleting code rather than writing it.
