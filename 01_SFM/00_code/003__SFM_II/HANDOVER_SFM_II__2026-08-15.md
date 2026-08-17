# Handover — SFM II (C_ELO) production artifacts

*2026-08-15. For the website developer. Everything below is generated, verified and on disk.*

---

## 1 · What changed

The site has been served by **SFM I / OG** (`points_diff + home_pitch`). It is now served by
**SFM II · C_ELO**, the spec committed after the 2026 off-season revision: a Bayesian
ordered-logistic over goal categories {0, 1, 2, 3+} with player effects, two Gaussian-process
curves (within-season and career-age), seven contextual factors and — new — an **ELO rating
difference**.

Honest performance, measured once on a sealed holdout season (2024/25) the model never saw:
**+3.56 % skill vs a marginal-frequency baseline** (log-loss 0.5074 vs 0.5262). Validation
folds ran +4.3–5.0 %. The full audit trail is in
`00_code/001_Development/001_00__SFM/FINDINGS_2026_REVISION.md` and `REFEREE_REPORT.md`.

---

## 2 · The two files you consume

Both live in `SFMwebsite__v2/static/data/`.

### `040_ScoringProb__prod.pkl` — 71 MB — scoring probabilities

**The structure is unchanged from the OG file**, so existing readers keep working:

```python
{ 'train' | 'test' | 'oos' : { player_slug : {
      'low'  : DataFrame,   # index '2026/27__1', columns Goals_0..Goals_3
      'mid'  : DataFrame,   # 90 % credible bands; 'mid' is the median
      'up'   : DataFrame,
      'match_stats': DataFrame,  # goals_in_match, name_league, name_team, name_opp, points_*
}}}
```

| dataset | players | content |
|---|---|---|
| `train` | 3,191 | full history through 2025/26 (was 2,850) |
| `test` | 0 | empty — the model trains through the last complete season, so nothing sits between train and oos |
| `oos` | 599 | **the live board**: 2026/27 matchday 1, all five leagues |

`P(scores ≥ 1)` on the live board runs 0.064 → 0.440, median 0.159. Top of the board:
Mbappé 0.440, Lautaro Martínez 0.391, Havertz 0.374, Haaland 0.343, Kane 0.323.

### `041_SARPAR__prod__SFM_II.pkl` — 102 MB — skill boards

⚠️ **New filename and new shape.** The OG artifact (`041_SARPAR__prod.pkl`, still on disk)
was **10.5 GB** because it stored the full observation-level posterior. The aggregation the
site actually needs now happens upstream:

```python
{ 'SAR': xr.DataArray(chain, draw, name_player),   # skill above replacement
  'PAR': xr.DataArray(chain, draw, name_player),   # performance above replacement
  'n_rows_player': Series, 'meta': dict }
```

Use the drop-in accessor at the bottom of `006_041__SAR_PAR__SFM_II.py`:

```python
from importlib.machinery import SourceFileLoader
m = SourceFileLoader('sarpar', '.../006_041__SAR_PAR__SFM_II.py').load_module()
out = m.get__SAR_PAR__SFM_II(cred_region=0.9)
out['SAR']      # DataFrame: low / mid / up per player, sorted
out['draws']    # the raw draws, for violin plots
```

Top-10 by mean SAR: Messi +0.554, Haaland +0.553, Kane +0.488, Ronaldo +0.483,
Lewandowski +0.475, Mbappé +0.475, Suárez +0.388, Henry +0.345, Ibrahimović +0.340,
Neymar +0.327. Units: capped goals per appearance above the average player.

**Two changes worth knowing about before you compare to the old boards:**

1. **The SAR estimand was corrected.** The OG code zeroed *every* factor, which meant "every
   match away, every player recoded to the reference position" — not "all teams equal".
   Now: team **context** is equalized (points-diff, opponent rank, goal appeal, ELO → the
   cross-sectional average; home advantage → the average schedule) while **player
   attributes** (position, cumulative output, goal share) are held at their observed values.
2. **Credible bands are narrower.** SAR/PAR now use the analytic expected goals per posterior
   draw instead of a simulated count per row, which removes simulation noise and keeps
   genuine posterior uncertainty. Set `GOALS_MODE = 'sampled'` in the script for the old,
   wider spread.

`SAR − PAR` is the team-context contribution: Ronaldo, Messi and Mbappé are the players
whose raw output most flatters them relative to their equalized skill (they played in strong
teams); the reverse tail is players in weak sides.

---

## 3 · Rerunning it weekly

Both scripts are **pure NumPy — no PyMC, no GPU** — and run locally in the `sfmII` venv in
minutes:

```
006_Website/006_040__Predictions_ScoringProb__SFM_II.ipynb   # scoring probabilities
006_Website/006_041__SAR_PAR__SFM_II.py                      # skill boards
```

Both read the light bundle `10_data/01_Models/SFM_II_FinalC_ELO_scaleCS__2526__LIGHT.pkl`
(posterior draws only). The model is refitted **once a season**, not weekly.

Each run self-tests before producing anything:

- **golden rows** — 64 stored rows are recomputed and must match the values written at export
  (currently 3.3 × 10⁻¹⁶). Any drift aborts with `DO NOT SERVE`.
- **bundle ↔ CSV contract** — training row count and goal-category mix must still match what
  the model was fitted on, else it prints `REFIT before serving`.
- **factor coverage** — prints the share of OOS rows where each factor is zero. At matchday 1
  `points_diff`, cumulative goals and share are legitimately zero (nothing has been played);
  **from matchday 3 onward they must not be**, or the upstream fixture builder (`006_021`)
  is not filling them.

Weekly inputs: refresh `10_data/106_Website/data_byPlayer__SFM_II.csv` (results) and
`data_byPlayer__OOS.csv` (upcoming fixtures), then run the two scripts. ELO rolls forward
automatically from the bundle over newly played matches and freezes for unplayed fixtures.

---

## 3b · Point-in-time ledger — READ THIS BEFORE BUILDING THE TRACKER

`10_data/106_Website/SFM_predictions__frozen.csv` (599 rows, seeded 2026-08-15).

**This is the only honest record of what the model predicted before a match.** The
scoring-probability pickle is regenerated wholesale every run, and
`migrate_pkl_to_postgres.py` currently does `WeeklyPick.objects.all().delete()` and rebuilds
from it — so anything the site shows as "past predictions" today is a *re-forecast*, not a
track record. Three reasons a re-forecast differs from what was published: the
cross-sectional standardization bucket changes (predicted squads → actual appearances); a
forecast player who did not play disappears while a surprise starter gains a forecast that
was never made; and the annual refit re-scores all history in-sample.

The ledger is maintained by `006_040__…__SFM_II.ipynb` on every run:

| | rule |
|---|---|
| fixture **unplayed** | row refreshes each run (new information, still pre-match) |
| fixture **played** | row is **never rewritten**; the result is attached beside the probabilities standing at kickoff |

Columns: `id_match, name_player, season, gameday, name_league, name_team, name_opp`,
`p0..p3 × {mid, low, up}`, `p_scores_mid` (= 1 − p0, the anytime-scorer headline),
`status` ∈ {upcoming, finished}, `forecast_frozen_at`, `model`, `actual_goals`, `appeared`.

**`appeared` matters and has no SFMMO analogue.** Teams always play; players do not. A
forecast player who was benched has `appeared=False` and a null result — the tracker must
**void** that pick, not score it as a miss, or the published hit-rate is biased downward.
Rows with a null `forecast_frozen_at` are the reverse case: someone who played but was never
forecast, recorded so the coverage gap is visible rather than silent.

Verified by a two-run sandbox on 2026-08-15: results injected for half the fixtures, the
re-forecast deliberately altered — frozen probabilities moved by **0.0e+00**; 41 no-shows
flagged; 12 unforecast appearances recorded; upcoming rows refreshed as intended.

**Required change on the Django side:** replace the `WeeklyPick`/`NaiveWeeklyPick`
delete-and-rebuild with a populate-from-ledger (insert new rows, update outcomes on existing
ones, never overwrite `predicted_prob`). Until that lands, treat the tracker and the
model-validation page as **not publishable** — and never present the `train` split as
forecasts; it is in-sample fit.

## 4 · Known gaps / your call

- **`006_042__SAR_PAR_funcCalc.py` still targets the OG artifact.** Point it at the accessor
  above, or keep both paths while you migrate.
- **The old 10.5 GB `041_SARPAR__prod.pkl` is still on disk.** Safe to delete once the new
  boards are live — it is regenerable and nothing new reads it.
- **Backup:** the previous scoring-probability file is preserved as
  `040_ScoringProb__prod__PRE_SFM_II_backup_20260815.pkl`. Delete when you are satisfied.
- **`test` is empty by construction.** If any view assumes a non-empty `test` split, it needs
  a guard.
- **Django migration scripts** (`SFMwebsite__v2/scripts/migrate_*.py`) have not been re-run
  against these artifacts — that is the next step if the site reads from Postgres rather than
  the pickles.
