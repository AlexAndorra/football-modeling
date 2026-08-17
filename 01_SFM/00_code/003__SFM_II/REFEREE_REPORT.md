# Referee Report — SFM II (player-scoring model)

**Manuscript under review:** `SFM_II__dev.ipynb` (last run: DevM, 2025-08 vintage of the
harness) with `SFM_OG.ipynb` as the frozen baseline and the two evaluation documents
(`SFM_II__EvaluationReport.pdf`, `SFM_vs_SFM-TM.pdf`) read as the analyst's selection
record. The SFMMO referee report and its off-season findings were read alongside as the
sister-model precedent.
**Evidence discipline of this review:** every checkable claim was re-derived against the
current data vintage (`10_data/106_Website/data_byPlayer__SFM_II.csv`; 374,919 rows,
5 leagues, 2000/01–2024/25; checks run 2026-08-04). Claims are tagged **[verified]**
(recomputed from data or by executing the notebook's own function), **[code]** (established
by reading the cell), or **[implied]** (algebraic consequence, posterior not re-sampled —
re-sampling was impossible because the export saves `idata: None`, itself a finding, M7).

**Overall assessment:** The core design is more sophisticated than the average applied
model in this space, and several practices deserve explicit credit: the same-player OOS
path uses `pm.set_data` + `sample_posterior_predictive(predictions=True)` — exactly the
right pattern; new players' cutpoint offsets are drawn hierarchically from the posterior
hyperpriors (principled); the `ZeroSumNormal` + implied-flat-baseline construction is a
nice reparameterization; PSIS-LOO correctly unmasked the P/R/S/U overfitting that
in-sample log-lik had crowned; and — stated for the record — the 3+ winsorization is
**category-safe** under an ordered likelihood: the SFMMO's Poisson-cap bias (its M2) has
**no analog here**. The recommendation is nonetheless **major revision**: the evaluation
layer contains three outright bugs, the new-player out-of-sample path contains two more,
model selection ran 21+ variants against criteria that cannot answer the deployment
question, and the workflow ships with zero convergence diagnostics. Almost every remedy
below is a deletion or a one-liner; the one real project is the harness (M4).

---

## Major points

### M1 — The momentum factor: acquitted on the main charge, convicted on three lesser ones

The stated suspicion was that the momentum calculation is buggy. The central worry — that
the EWMA leaks the current match's goals into its own feature — is **false on the current
data**: `goalsscored_cum_player` is a *pre-match* cumulative (within season/player,
`diff(cum)` equals the *previous* appearance's goals on 99.95% of rows, and equals the
current match's goals on far fewer) **[verified]**, and the first-appearance `fillna(cum)`
resolves to exactly 0 on 100% of the 16,153 (season, player) groups **[verified]**. So
`ewm(halflife=1)` over the lagged series, current row included, uses only past goals. The
construction in cell 11 is correct as a leakage matter, and this report says so plainly.

What survives scrutiny less well:

1. **Inconsistent treatment.** MOM is the only continuous factor that is *never*
   standardized (it sits in `other_factors`/`factors_player`, outside the `factors_CS`
   scaling path) **[code]**. It therefore enters in raw goals units (≈0–2) under the same
   β prior as the standardized factors, and — the sharper consequence — "set to zero"
   means "zero recent goals," not "average form," which silently changes the SAR
   counterfactual's meaning (M5).
2. **Season reset + zero seed.** Momentum restarts from an EWMA seeded at 0 every season
   (the seed's weight decays to ~7% after three appearances **[verified]** — modest, but
   it is a cold-start-shaped artifact, the same regime the SFMMO's WC campaign was
   punished for), and the clock is *appearances*, not calendar time: a player returning
   from three months out carries the same "momentum" as one who played last week.
3. **Where momentum is genuinely broken is downstream, not in the feature:** the
   new-player OOS path redraws its β from the prior (M2), and the W/X "regularized"
   variants replace `Normal(0, 2.5)` with `HalfNormal(0.5)` — that is not a tighter
   prior, it is a **sign constraint** on the very effect whose sign is the research
   question **[code]**. The evaluation report's DevX proposal ("tighter hierarchical
   prior") was the right idea and the implementation quietly became a different one.
   P/R/S/U's per-player β's, meanwhile, are *unpooled* `Normal(0, 2.5)` per player — no
   hyperprior — and LOO already convicted them of overfitting. **Remedy:** if player
   heterogeneity in momentum is to be rescued, the correct object is a hierarchical
   `β_p ~ Normal(μ_mom, τ)` with `τ ~ Exponential`, i.e., actual partial pooling; and
   standardize MOM like every other continuous factor (or document why not, and fix M5's
   zeroing semantics accordingly).

### M2 — The new-player out-of-sample path contains two bugs; every "new player" conclusion is downstream of them

Cell 84 (`SFM_II__newP`) reconstructs the model for the 58 players unseen in training
(1,350 rows, 7.3% of the 18,491-row 2024/25 OOS set **[verified]**):

1. **The baseline is applied at 1/5 of its trained scale.** Training defines the
   effective baseline as `baseline_sigma * Normal("baseline")` with
   `baseline_sigma = sqrt(5² + sd²/2850) ≈ 5.0`. The OOS block contains the correct
   reconstruction *commented out* and uses `baseline = SFM_II__newP.baseline` — the raw
   standard-normal node — instead **[code]**. Every posterior draw therefore shifts every
   new player's η by `(1 − baseline_sigma)·ẑ ≈ −4ẑ`. Back-of-envelope **[implied]**:
   with c₁ = 4 and P(goal) ≈ 0.174, typical η ≈ 2.4, so ẑ ≈ 0.49 and new players lose
   ≈ 2.0 logits — their scoring probabilities collapse to roughly a sixth of what the
   model intends. The two commented lines above the assignment show this is a bug, not a
   choice.
2. **The momentum coefficient is redrawn from the prior.** For every pooled-β version
   (A, F–O, Q, V — including the versions the evaluation report recommends), cell 84
   creates `pm.Normal("beta__player_new", sigma=2.5)`. That name is not in the posterior,
   so `sample_posterior_predictive` draws it **from the prior**: the *learned* momentum
   effect is discarded and each posterior draw injects N(0, 2.5) noise, multiplied by an
   unstandardized MOM of up to ~2 — swings of ±5 logits **[code]**. The fitted global
   `beta__player` sits in `idata.posterior` and should simply be reused, exactly as
   `beta__team` is one line above.

**Remedy:** restore the `baseline_sigma` multiplication; reuse the trained `beta__player`
node for pooled versions (only genuinely player-specific parameters should be redrawn,
as the cutpoint deltas correctly are); then adopt the SFMMO's **η parity test** — an
assert that a NumPy reconstruction of η from posterior draws matches the graph to
< 1e-8 — as a gate on both OOS paths and on any deployment export. The SFMMO's 7.7%
λ-deflation incident and this M2 are the same class of bug: a hand-rebuilt linear
predictor drifting from the fitted graph. The class deserves a permanent test, not two
incident reports. Until then, the `oos_newP` panel of every exported
`Evaluation__SFM_II_Dev*.pkl`, and the combined `oos` panel to ~7% weight, measure the
bugs, not the model.

### M3 — The metrics layer has three defects that contaminate every reported table

1. **Predictive probabilities are the posterior *median* per category** (cells 71 and 89)
   **[code]**. The Bayesian predictive distribution is the posterior *mean*; per-category
   medians need not sum to 1, so every downstream logLik/Brier/RPS/ECE consumes
   unnormalized pseudo-probabilities, with a Jensen bias concentrated exactly where
   posteriors are skewed (rare categories — i.e., goals). One-word fix:
   `.mean(dim='samples')`.
2. **The ECE function is broken at confidence = 1.0.** Its bins are
   `[edge_i, edge_{i+1})`, so a prediction with max-probability exactly 1.0 falls out of
   *every* bin. Executing the notebook's own function on the always-zero benchmark
   returns **0.0000** where the true calibration gap is **0.1738** **[verified]** — the
   "Naive ECE = 0.000" printed in the notebook and in `SFM_vs_SFM-TM.pdf` is the bug on
   display, advertising the maximally miscalibrated predictor as perfectly calibrated.
   Use ArviZ's PIT machinery (`azp.plot_ppc_pit` is *already imported and commented out*
   in cell 74) rather than repairing a hand-rolled binning.
3. **The naive benchmark is degenerate and the metric suite rewards it.** "Always P(0)=1"
   (a) wins ordMAE/xGoalsMAE *by construction* (MAE is minimized by the median, which is
   0 for 82.6%-zero data), and (b) has an epsilon-driven log-likelihood — its logLik is
   `n_nonzero · log(1e-10)`, a number about the `1e-10`, not about football. The honest
   naive is the **marginal-frequency forecast** (0.826/0.151/0.020/0.003), which is
   properly scored by logLik/Brier/RPS and unbeatable by accident. Relatedly, the
   94%-HDI "coverage" statistic is near-vacuous on a 4-category outcome where
   P(y ≤ 1) = 0.977 — 98.15% coverage is a restatement of the marginal distribution, not
   a calibration result.

**Remedy:** mean not median; ArviZ PIT; marginal naive; drop the MAE-family rows or
demote them to descriptive footnotes; and adopt one RPS normalization convention shared
with the SFMMO's ledger so the two model families' numbers can ever sit in one table.

### M4 — Model selection cannot answer the deployment question, and one criterion actively rewards confounding

The evaluation report ranks 21 variants by **in-sample PSIS-LOO/WAIC**, with **SAR
dispersion (σ) as the secondary criterion**, and the notebook separately scores a
**single OOS season (2024/25) that every variant reuses**. Three problems, in
increasing order of importance:

1. **LOO answers the wrong question.** Leave-one-*observation*-out estimates predictive
   accuracy for a hidden appearance amid known seasons — not the deployment task, which
   is *next-season* prediction with player form, promotion/relegation, and transfers
   moved. It is also unaudited: with 2,850 player effects plus 5,700 cutpoint deltas,
   the Pareto-k̂ distribution should be reported before any 15-point LOO gap (C vs F) is
   treated as signal.
2. **The one true OOS season has been consulted by ~24 variants.** That is the garden of
   forking paths the SFMMO report called out at nine variants (its M7.3) — here it is
   worse, and there is no sealed holdout at all.
3. **SAR σ is not a virtue.** The report reads higher σ as "better player
   differentiation," and duly finds the maximum (0.066) at **OG — the variant with the
   fewest controls**. That is not a coincidence; it is the mechanism. Remove position,
   opponent, and form covariates and their variance does not vanish — it is *reassigned
   to the player effects*. Selecting for SAR σ therefore selects for **confounding**,
   i.e., for models whose "skill" estimates absorb the most non-skill variance. The
   correct secondary criteria for a skill product are validity checks: season-to-season
   stability of the player ranking, and whether this year's SAR predicts next year's
   out-of-sample performance.

**Remedy:** port the SFMMO harness wholesale — expanding-window folds (train ≤ t, score
t+1 frozen) over several validation seasons; **paired** comparisons on identical
player-appearances with a pre-registered win rule (both logLoss and RPS, |t| ≥ threshold,
ties → parsimony); one season sealed by asserts and scored once. The SFMMO's own re-trial
is the cautionary tale this manuscript needs: its incumbent had been selected *while its
momentum feature was broken*, and the repaired harness reversed the verdict. Here the
selection ran while M2 and M3 were active; the ranking (C over F over K …) should be
considered unestablished until re-run on the repaired harness.

### M5 — The SAR counterfactual does not implement its own caption

Cell 95 ("*Set all Teams Equal*") zeroes the whole `factor_data__team` container. But
`factors_team` is defined as *everything except MOM* **[code]**, so the counterfactual
also zeroes `home_pitch`, `position_MID`, and `position_FOR` — and since dummies are
unstandardized, zero is not "average": it is *"every match away, every player recoded to
the reference position."* Meanwhile MOM (raw, unstandardized) is *not* zeroed — the
commented-out line shows the ambivalence — so "skill" retains current form but loses
position. The position coding itself is worse than impure — see the **Addendum**: the chained
`groupby(...).bfill().ffill()` runs its second fill *globally*, so the 37,361 rows
(10.0%) whose (season, player) group has no position at all — goalkeeper-like, 0.022
goals/appearance — inherit the *alphabetically previous player's* position (20,386
"Mittelfeld", 16,963 "Sturm") **[verified]**.
Finally, "replacement level = the average player" deviates from the paper's replacement
concept (the notebook says so itself), and SAR/PAR are means of *capped* categories, so
elite scorers' totals are understated at exactly the top of the board being published.

**Remedy:** write the estimand down first — *which* factors define the counterfactual
(team context only, presumably), *at what values* (cross-sectional means, which requires
the standardized representation — motivating M1.1), holding *what* fixed (position,
surely; form, decide and say so). Then: an explicit Unknown/GK category (or a deliberate,
documented exclusion of goalkeepers), and a stated replacement definition reconciled
with the paper. None of this is sampling work; it is one cell and a paragraph.

### M6 — "Maturity" is not identified the way the narrative claims

The markdown motivates per-player non-linear trajectories (injuries, preparation,
fatigue). The implementation is **two global curves**: one `f_within(gameday)` and one
`f_long(career-season index)` shared by all 2,850 players **[code]**. That is a
defensible population-average design — but it is not what the text promises, and its
input has a flaw: `season_nbr` is factorized *within the data window*, so everyone
present in 2000/01 — 18-year-old debutants and 34-year-old veterans alike — enters at
"career season 0," as does every mid-career arrival into the big five leagues. The curve
therefore mixes career age with *data-entry cohort*. And because the model contains **no
calendar-period term**, secular drift has nowhere legitimate to go: per-appearance
scoring fell from 0.2245 (2000/01) to 0.1844 (2022/23) **[verified]** — a ~15–18% decline
containing at least one structural break (the five-substitution era from 2020 changed
appearance composition). This is a classic age–period–cohort identification problem, and
currently period is forced into the age curve and the player effects — i.e., into
cross-era skill comparisons, which is precisely what the SFM sells.

Two honest exonerations from the same audit **[verified]**: pooling the five leagues
without league effects is *mild* (per-appearance rates span only 0.196–0.206, ~5%), so a
league intercept is hygiene, not urgency; and the feared `gameday`-dtype/`Categorical`
mismatch in `SFM_OG.ipynb` is benign (float 4.0 matches integer category 4; and the dev
notebook's `id_match` re-parse agrees with the raw column on all 374,919 rows — the
re-parse is redundant, not wrong). One composition note stands: Bundesliga ends at
gameday 34, so `f_within`'s 35–38 tail and those standardization buckets pool only four
leagues.

**Remedy:** add a calendar-season term (even a coarse era spline or season random
effect) so period stops masquerading as maturity; either left-censor `season_nbr` at
data entry (flag entry-cohort) or accept and *state* the population-average reading; and
rewrite the maturity markdown to describe the model actually fitted.

### M7 — The workflow ships blind: no diagnostics, no prior predictive, no seed, no saved posterior

In both `SFM_OG.ipynb` and `SFM_II__dev.ipynb` **[code]**:

1. **There is not a single convergence check.** No R-hat, no ESS, no divergence count,
   no rank plot — the only diagnostic cells (60–61) are duplicated *and* disabled with
   `if 1==2`. Two chains, GPU-vectorized, `target_accept=0.99` (the SFMMO measured 0.99
   as roughly a 2× runtime tax and unnecessary after reparameterization — this model has
   fixed first cutpoints and ZSN effects, so the same likely holds). Every number in
   both PDFs rests on unexamined chains.
2. **Prior predictive sampling is commented out** — and the dead cell still addresses
   `SFM_I`, a fossil of the copy lineage. Given `β ~ Normal(0, 2.5)` on 5–7 factors,
   `intercept_sigma = 5`, and cutpoint-spacing priors centered at ~2× the empirical
   spacing via an ad-hoc ×4 "probit→logit" conversion (the correct factor is ~1.6–1.8),
   the implied category probabilities have never been looked at. The SFMMO's formally
   failing prior-predictive check (2.6 goals implied vs 1.37 empirical) is the family
   precedent.
3. **The production sampler call has no `random_seed`** while the OOS predictive cells
   use a magic `42` and other cells thread `rng` — three seeding conventions, one of
   them absent. Reported numbers are not reproducible.
4. **The export saves `idata: None`** (cell 103) and the in-sample PPC is sampled
   *twice* (cells 62 and 93), the second silently discarded because
   `InferenceData.extend` keeps the existing group. The posterior behind the published
   evaluation cannot be re-opened — this review had to mark several quantities
   **[implied]** for exactly that reason.

**Remedy:** the standard battery, in order — `sample_prior_predictive` before fitting;
seed every call from one named rng; `arviz_stats.diagnose(idata)` after; PIT/coverage via
ArviZ; `psense_summary` for prior sensitivity (log-likelihood is already being stored —
add log-prior); `idata.to_netcdf()` immediately after sampling; delete the duplicate PPC
cell. This is one afternoon and mostly deletions.

### M8 — The model exists in diverged copies, and the standardization contract differs between train and score time

The lineage is `SFM_OG.ipynb` → `SFM_II__dev.ipynb` (24 variants via if-branches
hand-repeated across ~9 cells) → the deployment path (`003_ModelDeployment` /
`006_website` scripts, not audited here). The SFMMO record shows exactly how this class
of duplication bites (its momentum fix and k_max patch each failed to propagate to one
copy). Two concrete seams already visible **[code]**:

1. **Cross-sectional scaling is a different estimator at train and score time.**
   Training standardizes each factor within `groupby(gameday)` buckets that pool **five
   leagues × 24 seasons**; the OOS block standardizes within the **single 2024/25
   season**. Beyond the estimator mismatch, pooling seasons means a 2005 row is scaled
   with moments that include 2023 data — no outcome leakage, but a train/deploy
   asymmetry the deployed pipeline cannot reproduce. The SFMMO flagged the identical
   issue (its M6) and settled on season × league × gameday. Adopt one definition,
   write it down, use it on both sides.
2. **The variant matrix is stringly-typed.** The `devVersion` membership lists are
   repeated in cells 17, 43, 46–50, 53, 77, 84; this audit found them currently
   consistent, which is luck with a half-life. One `VERSIONS = {...}` dict at the top
   (factors, HSGP config, β_player treatment) consumed everywhere would make drift a
   syntax error instead of a silent fork. The same consolidation should decide, in one
   place, *which variant is canonical* — the notebook's last run is M, the evaluation
   report recommends C/F/K, and the website pickles say `SFM_II`/`SFM_IIa`/`SFM_OG`.

**Remedy:** extract `sfm_features.py` + `sfm_model.py` shared by notebooks and
deployment, with the η parity test (M2) guarding the seam — the same E0-first sequencing
the SFMMO review argued for, for the same reason: every lesson below this line otherwise
has to be learned three times.

---

## Minor points

- **Cutpoint prior arithmetic** (cell 52–53): probit-scale spacings scaled by
  `cutpoint_offset = 4` as a logit conversion (~1.7 is correct); the softplus applied at
  sampling is ignored when centering the prior (benign at these magnitudes); and the
  prior is empirical-Bayes on the training outcome — all data-dominated at n = 356k,
  none checked. One prior-predictive plot of implied category probabilities settles it.
- **`SFM_OG.ipynb` cell 35** computes `empirical_probs` from `value_counts()` *without*
  `sort_index()` — correct only because goal frequencies happen to be monotone in the
  category. The dev notebook fixed this; the fix never travelled back (M8 in miniature).
- **The zero-variance rule zeroes everything**: the CS-scaling lambda returns `x * 0.0`
  for *all* factors in a bucket if *any* factor has zero std (`.std().gt(0).all()`).
  Scale factor-wise, not bucket-wise.
- **`goalsscored_rank_team_wo_player` exists in the data and is unused** — `goal_appeal`
  is built from the *with*-player rank, letting the player's own past scoring
  contaminate an ostensibly contextual factor. The cleaner column is already shipped;
  use it.
- **Collinearity, from the notebook's own output**: corr(points_diff, goal_appeal) =
  0.72 (cell 19). The SFMMO's champion-board reordering under correlated form features
  is the family's documented failure mode; report VIFs alongside any β interpretation,
  or drop one.
- **Positions are backward-filled first** (future informs past within a season) —
  benign for a slow-moving attribute; the real defect in this line is the global
  second fill (see Addendum), and the substantive design fix (Unknown/GK category)
  is in M5.
- **Naive logLik epsilon**: `log(p + 1e-10)` makes any degenerate-forecast comparison a
  statement about the epsilon (M3.3); once the marginal naive is adopted this artifact
  disappears on its own.
- **Housekeeping**: `data_oos` is mutated as a slice (SettingWithCopy exposure); dead
  cell 64 references an undefined `use__HSGP`; the two disabled amplitude-check cells
  are verbatim duplicates; unused `Naive` rows in `df_metrics`. Ten minutes of deletion.

---

## Addendum (2026-08-04, revision phase) — the position fill is a genuine data bug

During the revision, the position-fill line (cell 12) was re-examined:
`groupby(['season','name_player'])['position_player'].bfill().ffill()`. The grouped
`bfill()` returns a plain Series, so the chained `ffill()` ignores group boundaries.
Verified against the data **[verified]**: all 37,361 rows in all-NaN groups (10.0%
of the data; 0.022 goals/appearance, i.e. goalkeeper-like) receive an *invented*
position from the previous player's block — 20,386 as "Mittelfeld", 16,963 as
"Sturm", 12 as "Abwehr". The original review understated this as "coded as
reference class"; in fact a mass of near-zero scorers is labelled midfielder or
striker, biasing β_MID/β_FOR downward in every position-carrying variant (C, F,
K, …). **Fixed in the revision** with a grouped `transform`, which restores the
intended NaN → reference-class coding; the explicit GK/Unknown dummy (M5's design
remedy) enters the re-trial as a variant.

---

## Vintage update (2026-08-10) — which findings the new data changes

The dataset was rebuilt and re-audited (`SFM_data_audit.py`, all checks PASS; 411,007 rows,
3,191 players, 26 seasons). Ligue 1 now reaches back to 2015/16; the Champions League was
trialled and removed (its two-legged `GD10-2` match ids break the notebooks' integer gameday
parse — keep it out). Effects on the findings above:

- **M5 / Addendum — superseded by the data.** Positions are now **100% populated** with all
  three labels, so the global-fill leak has nothing left to leak and the "reference class is
  a defender/GK blend" objection is gone. The unlabelled goalkeeper-like group (9.75% in the
  Nov-2025 vintage, scoring 0.021) has been removed from the universe entirely. The grouped
  `transform` fix stays in place as the correct implementation either way.
- **New consequence, fixed 2026-08-10:** with the unlabelled group gone, `position_MID` and
  `position_FOR` **partition 99.88% of rows** (corr = −0.9975; reference class = the 498
  `Abwehr` rows, 0.12%) — collinear with the intercept, so only β_FOR − β_MID is identified
  and the levels trade off against `baseline`. Note `Abwehr` was always ~0.1% (343 rows in
  Nov-2025): the SFM universe is scorer-oriented, and it was the goalkeeper group that had
  been serving as the reference class. **Remedy applied:** a single `position_FOR` dummy
  (reference = midfield), which spans the same space without the collinearity, in both
  notebooks and all variants. `F_GKu` is thereby void and the harness now raises rather than
  running it as a duplicate of F.
- **M6 — unchanged in substance.** Era drift is −15.3% (0.2247 → 0.1903, 2000/01 → 2023/24)
  on the new vintage; the league spread is 0.195–0.205, so "league intercepts are hygiene,
  not urgency" still holds — though Ligue 1 is now the thinnest league (38,201 rows vs
  83–105k) because its history starts in 2015/16.
- **M1 — unchanged.** The momentum timing contract still verifies clean (pre-match on 99.95%
  of rows; first-appearance seed 0 on 100% of groups).
- **Caught and corrected during the rebuild** (recorded because it is exactly the class of
  defect this report exists to catch): an interim vintage carried a **2025/26 appearance
  definition change** — 18.54 player-rows per match against a median of 10.11, with
  goals *per match* normal — which mechanically depressed goals-per-appearance by 35% in the
  season designated as the pristine holdout. Now resolved (10.36 rows/match, 0.185
  goals/appearance). `SFM_data_audit.py` gained a permanent appearance-density check.

---

## On the two evaluation documents (read as the response-to-reviewers)

Both PDFs predate the current dataset and are contaminated by M3 wherever OOS metric
tables appear (the `SFM_vs_SFM-TM` table prints the ECE bug — Naive 0.000 — verbatim, and
its Naive row "wins" exactly the metrics a median-0 predictor wins by construction). The
LOOCV/WAIC columns are the exception — `az.loo`/`az.waic` consume the stored
log-likelihood directly and bypass the median bug — but they inherit the stale data and
M4.1's task mismatch. The ablation report's *reasoning* is largely sound given its
inputs (the P/R/S/U overfitting diagnosis is correct and well-argued); its
recommendations should nonetheless be considered open until re-run, because selection
operated while M2/M3 were active (and its "ordered-probit" label should read ordered
*logit*). When re-issued: add per-gameday and per-position breakdowns, and consider the
SFM analog of the SFMMO's most informative benchmark — the betting market's
anytime-scorer prices, de-vigged, beside every dev metric. The SFMMO's central strategic
finding (the residual gap to market is *information, not statistics*) is exactly the
question the SFM should ask of itself before more factor engineering.

---

## Verdict

**Major revision — with enthusiasm.** The suspect the manuscript itself flagged — the
momentum calculation — is *acquitted* on the leakage charge by direct verification
against the data, which should be recorded as good news: the feature engineering
instincts here are better than the team feared. The convictions are elsewhere, and they
are cheap: two one-line bugs in the new-player path (M2), a one-word fix and a deleted
hand-rolled function in the metrics layer (M3), an afternoon of workflow hygiene (M7).
The two genuine projects are the harness (M4 — and the SFMMO has already built the
blueprint: expanding windows, paired pre-registered tests, a sealed holdout) and the
estimand work (M5/M6 — deciding what SAR and "maturity" actually mean before the next
season's boards are published). The sister model's off-season demonstrated the payoff
sequence precisely: repair the mechanical defects first, *then* re-run selection —
because at least one of its incumbents had been chosen while a defect was active, and
the verdict flipped. There is no reason to expect this model family to be different.
