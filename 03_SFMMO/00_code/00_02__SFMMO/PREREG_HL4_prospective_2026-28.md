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
4,000 draws, and are never refitted during a season (the expanding-window protocol). K's live
bundle moves from 2 to 4 chains at the same time; same specification, same data, only more
posterior draws.

## Primary metric

Champion log-loss of the **pre-season board**: for each league-season, −log of the probability
the board gave the eventual champion, floored at 1e-4. Paired difference HL=4 − K across
league-seasons. This is M2's metric; it reproduces M2's table from
`SFMMO_decay_experiment_M2__results.csv` exactly (K 1.1775, HL=4 1.0694, t = −2.46).

Sample: 5 leagues × 2026/27 and 2027/28 = **10 league-seasons**, none of them seen by any
experiment.

- 2026/27, K: the published pre-season board of 2026-08-15 (2 chains — it is the receipt).
- 2026/27, HL=4: reconstructed from the same inputs with every 2026/27 result removed. Honest
  because the bundle contains no 2026/27 data. The method is validated first by rebuilding K's
  published pre-season board the same way; the agreement is reported with the result.
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
