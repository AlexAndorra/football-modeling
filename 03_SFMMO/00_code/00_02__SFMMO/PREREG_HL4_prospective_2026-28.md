# Pre-registration — HL=4 versus K, prospective, 2026/27–2027/28

Fixed 2026-10-07, before any HL=4 production bundle was fitted and before any HL=4 title odds
for 2026/27 existed. This is the follow-up FINDINGS_2026-27_OFFSEASON.md §7 names: experiment M2
found HL=4 better than K on champion log-loss at t = −2.46, short of the pre-registered 2.5.
Requested by Alex; approved by Max.

## Arms

- **K** — the live specification (devVersion K, no decay).
- **HL=4** — K with each training row's log-likelihood weighted `0.5^(age_years/4)`, exactly
  the M2 arm (`DECAY_HALFLIFE_YEARS = 4` in `SFMMO__dev_EW.ipynb`).

Both are fitted once on all data through 2025/26, seed 326, NumPyro NUTS, **4 chains** ×
2,000 draws (Amendment 2), and are never refitted during a season (the expanding-window protocol). K's live
bundle moves from 2 to 4 chains at the same time; same specification, same data, only more
posterior draws.

## Primary metric

Champion log-loss of the **pre-season board**: for each league-season, −log of the probability
the board gave the eventual champion, floored at 1e-4. Paired difference HL=4 − K across
league-seasons. This is M2's metric; it reproduces M2's table from
`SFMMO_decay_experiment_M2__results.csv` exactly (K 1.1775, HL=4 1.0694, t = −2.46).

Sample: 5 leagues × 2026/27 and 2027/28 = **10 league-seasons**, none of them seen by any
experiment.

- 2026/27, both arms: rebuilt by `preseason_board.py` from each arm's 4-chain bundle, with the
  same code and the same inputs, every 2026/27 result removed (see Amendment 1). Honest because
  neither bundle contains 2026/27 data.
- 2027/28: both boards made fresh before the first kick-off, 4 chains.

## Decision rule (title-odds product only; the match model stays K, per experiment M)

Adopt HL=4 for the season-odds board from 2028/29 if **both** hold:

1. the pooled 50 league-seasons (M2's 40 plus these 10) give paired t ≤ −2.5, and
2. the 10 prospective league-seasons alone give t ≤ −1.0.

Otherwise K stays. Why this rule: the prospective 10 alone have 8% power at M2's effect size,
so they cannot decide anything by themselves; pooling them with M2 is legitimate because they
are independent and their inclusion is fixed now. Condition 2 stops the old 40 from carrying
the decision. Bootstrapped from M2's own differences, the rule passes with probability 0.64 if
the effect is as in M2, 0.14 if HL=4 truly adds nothing this time, and 0.01 if HL=4 is worse.

## Secondary — reported, never decisive

- Champion log-loss of every weekly snapshot, averaged over each season (how the title odds
  hold up during the season — Alex's question).
- Full-table rank-RPS, champion hit rate (as in M2).
- Match-level RPS and log-loss of the weekly forecasts (experiment M already settled match
  forecasting for K).

## Fixed in advance

No arm, metric, floor, league or season is added or changed after this date. No early stopping:
interim numbers are descriptive only. Postponed or abandoned seasons are scored on the official
final champion; a league-season without one is dropped from both arms.

## Operation

HL=4 runs every week straight after the live run, with the same code and inputs, writing only to
`10_data/106_Website/_shadow/hl4/`. It never writes the live ledger, the live feed or any published
file.

## Caveats (must appear in any write-up)

1. M2's 40 league-seasons include its 20-season pilot (M2 caveat i); the pooled test inherits
   that overlap. The prospective 10 do not.
2. HL=4's 2026/27 pre-season board is reconstructed after the season began. The model cannot see
   2026/27, but the choice to run it was made with the season under way. Everything about it is
   fixed here, before it is computed.

## Amendment 1 — 2026-10-07, before any HL=4 bundle or number existed

The original text scored K's 2026/27 pre-season board as "the published board of 2026-08-15".
There are two candidates: `SFMMO_preseason_odds__2026-27.csv` (14 Aug) and 006_061's first
snapshot (15 Aug). They differ from each other by up to 0.9 points of title probability, and the
code that produced the 14-Aug file is in no repository. Both arms are therefore rebuilt
by the same script (`preseason_board.py`, which calls 006_061's own simulation) from the same
inputs, so the comparison is symmetric by construction.

Validation of the rebuild, on the live 2-chain K bundle: 0 of 1,752 fixtures pinned (no 2026/27
result leaks in), starting ELO within 0.7 points of the 14-Aug file for all 96 teams, and title
probabilities within 1.3 points of the 14-Aug file and 0.9 of the 15-Aug snapshot, i.e. within
Monte-Carlo noise of 8,000 simulated seasons. The published boards are reported alongside, not
scored.

## Amendment 2 — 2026-10-07, before any bundle was fitted

4 chains × **2,000** draws instead of 4 × 4,000. The fit holds three per-row arrays (eta, the
log-likelihood, the posterior predictive) for every draw before the bundle discards them: about
29 GB at 16,000 draws, more than the Colab runtime or the local 32 GB machine holds. 8,000 draws
is the live bundle's own total (2 × 4,000), so Monte-Carlo precision is unchanged while the
4-chain convergence check is gained. Nothing else changes.
