# SFM II — Off-Season Revision 2026: Selection Findings

*2026-08-11. Status: selection closed under the pre-registered rule; committed spec pending
collaborator review; holdouts (2024/25, 2025/26) unburned. All numbers from the repaired
expanding-window harness (`SFM_II__dev_EW.ipynb`) on the frozen 2026-08-10 vintage
(411,007 rows; positions fully populated; Ligue 1 from 2015/16; no Champions League).
RPS is /3-normalized (4-class, ∈ [0,1]) throughout.*

---

## 1 · Protocol (as pre-registered)

- **Folds:** train ≤ {2020/21, 2021/22, 2022/23} → score {2021/22, 2022/23, 2023/24}
  frozen; 59,871 paired player-appearances, identical across all variants (verified).
- **Rule (fixed 2026-08-04, before any run):** a challenger replaces the incumbent **OG**
  (the published Andorra & Göbel 2024 spec: points_diff + home, inter+intra HSGPs) only if
  it wins **both** log-loss and RPS with paired |t| ≥ 2.5 pooled across folds; ties resolve
  to parsimony (fewest factors, then fewest HSGP timescales).
- **Sampler (uniform, stamped in every ledger):** centered parameterization, 4 chains,
  `target_accept 0.95`, numpyro/A100. Divergences 0.3–2.7 %/fold documented-accepted for
  selection (replicate experiments showed run-to-run variance dominates these counts and
  ledger aggregates are insensitive at 4-decimal resolution).
- **Gates passed:** identical row sets; identical config stamps; diagnostics in band —
  with **one exception**: F_LEAGUE fold 2022/23 failed (239 divergences, ESS 64,
  R-hat 1.054). Its row is flagged; this does not affect the outcome (§3).

## 2 · Verdict

| challenger vs **OG** | Δ log-loss (t) | Δ RPS (t) | qualifies |
|---|---|---|---|
| **C** | **−0.00846 (−18.1)** | **−0.00089 (−16.5)** | **yes** |
| F | −0.00841 (−17.9) | −0.00089 (−16.4) | yes |
| K | −0.00445 (−13.9) | −0.00045 (−13.2) | yes |
| F_MOMs | −0.00842 (−17.8) | −0.00089 (−16.3) | yes |
| F_LEAGUE | −0.00843 (−17.8) | −0.00089 (−16.2) | yes *(fold-2 diagnostics failed)* |
| F_ERA | −0.00842 (−18.4) | −0.00089 (−17.2) | yes |

Every challenger beats OG in **every fold** (per-fold |t| ≥ 7.6). The post-paper factor
engineering is **vindicated with receipts**: skill vs the marginal-frequency naive rises
from **+2.83 % (OG) to +4.47 % (C)** on log-loss.

**Tie resolution.** {C, F, F_MOMs, F_LEAGUE, F_ERA} are pairwise indistinguishable
(all |t| ≤ 0.73); K is *dominated within the challenger set* (K vs F: +0.00396, t = +13.0),
so — following the SFMMO precedent that parsimony arbitrates only within the pairwise-
indistinguishable top class, a clarification recorded here because the rule text alone
("if several qualify, most parsimonious wins") would absurdly crown the dominated K —
the pick is the top class's most parsimonious member by factor count:

> **Committed spec (pending review): Dev C** — points_diff, cumulative player goals,
> opponent conceded-rank, player share, goal appeal (all CS-standardized, season × gameday)
> + home_pitch + position_FOR; inter- **and** intra-season HSGPs; **no momentum**.

## 3 · What each experiment answered

- **Momentum is dead for the SFM** (F vs C: t = +0.55). Once cumulative goals, share, and
  ranks are in, the appearance-EWMA adds nothing — the mirror image of the SFMMO, where
  correctly-computed momentum earned decisive readmission. The referee report's M1 arc
  closes: acquitted of leakage, rejected on merit. (Standardized vs raw MOM: t = −0.73 —
  moot given C.)
- **No era term needed for prediction** (F_ERA vs F: t = −0.04). The −15 % secular drift is
  real but the expanding window retrains yearly, so the drift is absorbed fold by fold; the
  M6 confound remains an *interpretation* caveat for f_long/player effects, not a
  predictive defect.
- **No league intercepts needed** (F_LEAGUE vs F: t = −0.33). The 0.195–0.205 spread
  verdict ("hygiene, not urgency") confirmed out-of-sample — now with thin-Ligue-1 in the
  data. Incidentally its fold-2 convergence failure moved its ledger by < 0.0001 vs F:
  further evidence the selection metrics sit far above sampler noise.
- **Sparse K is refuted** (K vs F: t = +13.0). The old evaluation report's "best
  SAR-σ/LOO compromise" loses ~half the factor edge; its selling point was SAR dispersion,
  which the referee report (M4.3) already reclassified as confounding, not virtue.
- **OG stands taller than the old in-sample LOO ranking suggested** (17th of 21 there) but
  is genuinely beaten: +1.6 pp of skill separates the paper model from the 2026 spec.

## 4 · Segments (log-loss, C vs OG)

- **New players** (5,824 rows): 0.4802 vs 0.4854 — the M2-repaired path prices unseen
  players *better than* the average row.
- **Cold start** (gd 1–2): 0.5042 vs 0.5062 — mild everywhere; league matchdays are not
  informationally cold (SFMMO league finding replicated).
- **Calibration:** ECE 0.0046 (C) vs 0.0067 (OG) vs 0.0104 (naive).

## 5 · Open items

0. **Reviewer objection (2026-08-11), accepted — the momentum epitaph is provisional.**
   The grid confounded momentum with the HSGP configuration (C = no-MOM/inter+intra,
   F = MOM/inter-only); the clean ±MOM cell at fixed HSGPs was never run. `C_MOMs`
   (= C + season×gameday-standardized momentum, identical HSGPs) is added to the registry:
   it replaces C only if it wins **both** metrics at paired |t| ≥ 2.5; a tie keeps C.
   Until then, §3's finding reads precisely: *momentum is not needed* (two roads tie);
   whether it adds anything on top of C is the open cell. Standardized MOM is used because
   raw momentum's early-season seed ramp is a gameday profile partially absorbable by
   `f_within` — the one real overlap channel between momentum and the GPs.
   **Extended (same review):** two no-GP cells complete the momentum design — `A` (raw MOM,
   `hsgp='none'`, the old DevA never re-run on the repaired harness) and `A_MOMs`
   (standardized). Readout map: C_MOMs vs C isolates momentum *under* full GPs; A vs
   A_MOMs isolates standardization *without* GPs; A-family vs F isolates what the
   long-GP adds in momentum's presence. If the GPs were masking momentum, the A cells
   are where it shows. Same rule for all three: replace C only on a both-metric win at
   |t| ≥ 2.5; A-family fits carry no GP block (dedicated `hsgp='none'` code path,
   alpha = player effect + factors only).

0b. **Reviewer variant `C_ELO` (same review) — the missing 'class' covariate.** ELO — the
   SFMMO's core covariate — was never an SFM factor. Ported into the harness as an
   intent-faithful implementation of the family conventions (K=20, home_adv=50, init 1500,
   season-start 0.75·r + 0.25·1500 for returning teams, 1300 promoted; **one** update per
   match with home side correctly attributed — the SFMMO's in-notebook version processes
   both perspective rows, i.e. double-updates with home_adv on both sides; recorded as a
   finding for the SFMMO's next revision). Validated on the frozen vintage: 39,931 matches,
   exact perspective antisymmetry, sane top-8, no NaNs. `C_ELO` = C + CS-standardized
   elo_diff. **Hypothesis pre-registered before the run:** elo_diff ≈ points_diff mid-season
   (r = 0.80) but carries cross-season information where points_diff is empty
   (gd 1: sd(elo_diff) = 130 vs sd(points_diff) = 0) — so any edge should concentrate in
   the gd 1–2 bucket. Same replacement rule as all reviewer variants.
   **Plus `A_ELO`** (= A + elo_diff, raw momentum, no GPs): A_ELO vs A isolates ELO's
   increment in momentum's presence; A_ELO vs C_ELO asks whether the GPs still matter once
   ELO carries the strength signal. Full reviewer round: C_MOMs, A, A_MOMs, C_ELO, A_ELO —
   15 fits.

1. **Holdout runs** (once, committed spec only, after review): 2024/25 via `RUN_HOLDOUT`;
   2025/26 stays sealed for the next cycle.
2. **β-prior hygiene** (prior-predictive gate: 4.8–9.7 % prior mass on 3+ vs 0.3 %
   empirical): queued experiment, fold-re-validated before the holdout if attempted —
   never after. OG evidence says it is *not* the divergence cause.
3. **Production port**: `SFM_II__dev.ipynb` → devVersion 'C'; note the M5 SAR machinery
   simplifies — C carries no momentum factor, so the historical-average / form-neutral SAR
   pair collapses to a single SAR (the code's `len(factors_player) > 0` guards already
   handle this).
4. **M8 consolidation** after the holdout: one notebook, shared feature/model module,
   retire `SFM_II__dev.ipynb` to `XX__vintage/`.

---

## 6 · Reviewer round (2026-08-13) — VERDICT REVISED: C_ELO

The five reviewer variants of §5.0/§5.0b ran under identical settings (stamps verified;
row sets identical). Results:

| contrast | Δ log-loss (t) | answer |
|---|---|---|
| **C_ELO vs C** | **−0.00047 (−4.11)**, RPS −4.06 | **ELO dethrones C** under the pre-registered rule |
| C_MOMs vs C | −0.00009 (−1.65) | momentum adds nothing on top of C — the clean cell; **epitaph final** |
| A vs A_MOMs | −0.00000 (−0.02) | standardization irrelevant without GPs |
| A_ELO vs A | −0.00048 (−4.10) | ELO's increment is the same without GPs — cleanly **additive** |
| A_ELO vs C_ELO | +0.00042 (+4.00) | the GPs still earn ~0.0004 even with ELO |
| A_ELO vs C | −0.00006 (−0.36) | the dark horse: ELO+MOM with **no GPs** ties the former champion |

**C_ELO beats every one of the 11 other variants pairwise at |t| ≥ 2.5 on both metrics —
the sole member of the top class; no parsimony tiebreak required.** Per-fold vs C: −2.08,
−1.96, −3.08 (direction unanimous). Skill vs naive: **+4.56 %** (C: +4.47 %, OG: +2.83 %).

> **Committed spec (pending review), revised: Dev C_ELO** — C's factor set + CS-standardized
> `elo_diff` (intent-faithful family ELO: K = 20, home_adv = 50, one update/match), inter- and
> intra-season HSGPs, no momentum.

**The pre-registered cold-start hypothesis: direction confirmed, power honest.** The per-row
effect follows exactly the predicted gradient — gd 1–2: −0.00180, gd 3–10: −0.00069,
gd 11+: ≈ −0.0003 — a 5× concentration in the cold start, identical in the no-GP replication
(A_ELO vs A). Per-bucket t at gd 1–2 is −2.11 (n = 3,188 — underpowered at bucket level);
the significant residue at gd 21+ (t = −2.89) says ELO also carries some *within*-season
information beyond points_diff. New players specifically: no extra edge (t = −0.14).

**Structural bonus finding: the A-family sampled with ZERO divergences in all folds**
(GP-carrying variants: 35–108/fold). The divergence source is hereby empirically confirmed
as the GP block (amplitude/lengthscale geometry) — closing the question the target-accept
ladder left open. The GPs' predictive value (+0.0004, t ≈ 4 vs A_ELO) is real but now has a
known sampling cost attached; a `Gamma(2,·)`-style amplitude prior is the queued hygiene
candidate if the divergences ever need to go to zero.

**Production consequences of C_ELO** (for the port): the ELO block must move into the
production feature pipeline (006/003x — it exists only in the harness today), and the SAR
counterfactual's CONTEXT_FACTORS list must include `elo_diff` (it is opponent-relative
context, to be equalized like points_diff).

---

**§6.1 — One follow-up cell before commitment (2026-08-13): `C_ELO_noGP`.** The reviewer
noted A_ELO's practical appeal (98.2 % of the champion's edge with no GP machinery, zero
divergences, a "strength + form" narrative) but momentum's contribution *inside* A_ELO was
untested. `C_ELO_noGP` (= A_ELO − momentum ≡ C_ELO − GPs; single-knob against both) is
added with two pre-registered contrasts: **A_ELO vs C_ELO_noGP** — if |t| < 2.5, momentum
is a passenger in the lean spec too and the "form" story is unmeasured; and
**C_ELO vs C_ELO_noGP** — the pure GP ablation at fixed factors. The commitment decision
(protocol pick C_ELO vs a documented operational deviation to the lean spec) is deferred
until this cell reports. Magnitude context, on the record: the GP block's entire edge is
0.026 pp of geometric-mean assigned probability (61.10 % vs 61.07 %); the full
naive→champion journey is +1.42 pp.

**§6.1 RESOLVED (2026-08-13, C_ELO_noGP ledger; zero divergences, gates clean):**

1. **Momentum is a passenger — final.** A_ELO vs C_ELO_noGP: −0.00010 (t = −1.86), RPS
   t = −1.37. Momentum has now failed every clean cell it was ever offered: under full GPs
   (t −1.65), without GPs alongside ELO (t −1.86), raw vs standardized (t −0.02). Its point
   estimate is consistently a hair negative-of-zero and never significant. Any "strength +
   form" narrative for a lean spec would be unmeasured marketing; the lean candidate is
   therefore C_ELO_noGP itself (ELO + factors, no MOM), not A_ELO.
2. **The pure GP ablation is LARGER than the momentum-confounded estimate:** +0.00052
   (t = +5.69; folds +2.82/+4.37/+2.65, unanimous) vs the +0.00042 measured through A_ELO.
   In magnitude terms: 0.032 pp of geoP. Segment anatomy: late-season (gd 21+: +0.00074,
   t 4.8) and mid-season (gd 11–20: t 2.7); nothing at the cold start (t 1.5), exactly zero
   for new players (t −0.1).

**Commitment inputs, final:** protocol pick **C_ELO** (beats all 12, incl. the lean cell at
t 5.7). Documented-deviation candidate **C_ELO_noGP**: keeps 97.8 % of the edge vs naive,
zero divergences, no GP machinery in production — but with momentum dead, its case rests on
operational simplicity alone, against a GP contribution now measured larger and
fold-unanimous.

---

## 7 · The holdout (2024/25, scored once, 2026-08-14) — reported as-is

| | log-loss | RPS | geoP | skill vs naive |
|---|---|---|---|---|
| **C_ELO** | 0.50742 | 0.05254 | 60.21 % | **+3.56 %** (paired t = 12.0) |
| marginal naive | 0.52616 | — | 59.09 % | — |

Gates clean (21 divergences, R-hat 1.006, ESS 840, stamps correct; 18,996 rows, 1,463
new-player rows). **The honest estimate of next-season performance is +3.56 % skill** —
decisively above naive, and one point below the pooled validation estimate (+4.56 %;
per-fold range +4.30 % to +4.96 %). The shortfall vs the fold range is the expected
combination of out-of-sample selection shrinkage (thirteen variants were compared on those
folds; the winner's fold numbers are optimistically biased in the usual way) and
single-season variation — 2024/25 simply ran harder for everyone, naive included (its own
log-loss is the worst of any season scored). Segment notes: cold start strong (gd 1–2:
0.4891 vs naive 0.5068), mid-season weakest (gd 11–20), new players harder this season
(0.5362 vs 0.5050 for known players; n = 1,463 rows). Per protocol this number is
recorded, not acted on: no re-runs, no re-opening. 2025/26 remains sealed as the pristine
holdout for the next cycle.

---

*Selection first closed 2026-08-11 (C); re-opened at collaborator review by pre-registered
reviewer variants; re-closed 2026-08-13 with **C_ELO** as the sole top-class member;
**COMMITTED 2026-08-13 (user decision, protocol pick): Dev C_ELO** — the lean deviation was
considered against final numbers (§6.1) and declined. **Holdout burned 2026-08-14: +3.56 %
skill (§7).** The selection is closed at every level. 2025/26 remains sealed for the next
cycle.*
