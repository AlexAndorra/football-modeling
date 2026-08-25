# Weekly Update — Pipeline Runbook

When Max says **"run the weekly update"**, this is the sequence. Run the stages in
order. Each stage has a **gate**: do not start the next stage until the gate passes.

Everything runs from the shared venv:

```bash
source /Users/maximilian/Dropbox/Max/51_SoccerAnalytics/00_code/sfmII/bin/activate
cd /Users/maximilian/Dropbox/Max/51_SoccerAnalytics/00_code/006_Website
```

Notebooks are executed headlessly with `run_notebook.py`, which applies config
overrides from the command line so nobody has to hand-edit a config cell:

```bash
python run_notebook.py <notebook> --set "KEY=VALUE" --skip 20,21,23
python run_notebook.py <notebook> --list          # show the cell map, run nothing
```

`CURRENT_SEASON` below means the season being played, in `YYYY-YY` form — e.g.
`2026-27`. Its short form (`S2627`) appears in the raw data filenames.

---

## Stage map

| # | Stage | Script | Typical time |
|---|-------|--------|--------------|
| 1 | Scrape played matches | `006_000__WebScrape.ipynb` | 10–45 min |
| 2 | Scrape upcoming fixtures | `006_001__WebScrape_OOS.ipynb` | 10–30 min |
| 3 | Fixture feed (website/app) | `006_007__FixturesKickoff.py` | seconds |
| 4 | Odds | `001_100` → `001_101` → `001_102` | 2–5 min |
| 5 | Training features | `006_020__Features_Train.ipynb` | 2–10 min |
| 6 | Refresh research mirror | `cp` (one command) | seconds |
| 7 | OOS features | `006_021__Features_OOS.ipynb` | 5–20 min |
| 8 | Scoring probabilities | `006_040__Predictions_ScoringProb__SFM_II.ipynb` | — |
| 9 | SAR / PAR | `006_041__SAR_PAR__SFM_II.py` | — |
| 10 | SFMMO match outcome | `006_060`, `006_061` | — |
| 11 | Load to Railway Postgres | `scripts/migrate_*.py` | 2–10 min |
| 12 | Deploy | `git push origin main` | ~3 min (Railway) |

**Stages 4 and 1–2 hit different hosts** (football-data.co.uk vs kicker.de), so
stage 4 can run in parallel with stages 1–2 to save wall-clock time.

Stages 1–7 were executed end-to-end on 2026-08-25 and the gates below are the
actual observed outputs. **Stage 10 was likewise executed and verified on
2026-08-25.** **Stages 8–9 were executed and verified on 2026-08-22** (117 fixtures
scored, 81 ledger rows frozen) and their gates below are observed outputs, rewritten
2026-08-25 against the scripts. Stages 11–12 are transcribed from `ReadMe.txt` and
`CLAUDE.md` and have **not** been re-verified — treat their gates as provisional and
confirm against the scripts before relying on them.

---

## Stage 1 — Scrape played matches

Pulls results, scorers, lineups and player metadata from kicker into
`10_data/100_RawData/{league}/S{ss}_*.csv`.

```bash
python run_notebook.py 006_000__WebScrape.ipynb \
    --set "N_leagues=['premier-league','bundesliga','la-liga','serie-a','ligue-1']" \
    --set "N_seasons=['2026-27']" \
    --set "N_gamedays=[range(1,4)]*5" \
    --set "BACKFILL=True"
```

- `BACKFILL=False` scrapes **only the matchday kicker currently displays**. That is
  the normal mid-season weekly behaviour.
- `BACKFILL=True` sweeps every round listed in `N_gamedays`. Use it when catching
  up more than one round, or in the opening weeks when rounds overlap.
- `N_gamedays` must have **one entry per league**, in the same order as `N_leagues`.
- `SEED_FROM_PRIOR_SEASON=True` (default) fills player metadata from last season's
  files instead of re-fetching it — about **80% fewer requests** at season start.
  Set it to `False` only if you deliberately want fresh metadata for everyone.

**Gate.** Goals must reconcile against scorer rows, in every league:

```bash
python - <<'PY'
import pandas as pd, re
for lg in ['premier-league','bundesliga','la-liga','serie-a','ligue-1']:
    g=pd.read_csv(f'../../10_data/100_RawData/{lg}/S2627_games.csv')
    s=pd.read_csv(f'../../10_data/100_RawData/{lg}/S2627_scorers.csv')
    p=pd.read_csv(f'../../10_data/100_RawData/{lg}/S2627_players.csv')
    if not len(g): print(f'{lg:16} no played matches'); continue
    goals=int(g['score_home_full'].sum()+g['score_away_full'].sum())
    pos=(p['position_player'].notna()&p['position_player'].ne('NA')).sum()
    print(f'{lg:16} games {len(g):3d} | goals {goals:3d} vs scorers {len(s):3d}'
          f' | positions {pos}/{len(p)} | dup {g.duplicated(subset=["match_id","team_home","team_away"]).sum()}')
PY
```

Goals **must equal** scorer rows, positions must be 100%, duplicates must be 0.

> **Do not trust the progress bar.** `fetch_html` returns `None` silently when all
> retries fail, and the loop counts that as a completed iteration — a "3/3" bar can
> mean three failures. The CSV gate above is the only proof a scrape worked.

---

## Stage 2 — Scrape upcoming fixtures

Writes `S{ss}_games__OOS.csv` per league: every fixture not yet played, with
kick-off date **and exact time**.

```bash
python run_notebook.py 006_001__WebScrape_OOS.ipynb \
    --set "N_seasons=['2026-27']"
```

To repair specific rounds instead of the whole season:

```bash
python run_notebook.py 006_001__WebScrape_OOS.ipynb \
    --set "N_leagues=['bundesliga']" \
    --set "N_gamedays=[range(1,35)]" \
    --set "GAMEDAYS_ONLY=[8,24]"
```

`GAMEDAYS_ONLY` takes the **exact list** of rounds, and the export **merges** into
the existing file rather than replacing it.

**Gate.** Full season length per league — 380 for Premier League / La Liga /
Serie A (20 clubs), 306 for Bundesliga / Ligue 1 (18 clubs), counting played +
upcoming together. A short count means rounds are missing; repair them with
`GAMEDAYS_ONLY` before continuing.

---

## Stage 3 — Fixture feed for website and app

```bash
python 006_007__FixturesKickoff.py
```

Writes `10_data/106_Website/fixtures_{season}__kickoff.csv` — the file the web
developer consumes. Carries `kick_off_utc` (use this for countdowns and
notifications; local offsets shift at DST) and `time_confirmed` (`False` means the
league has not published the slot yet, stored as `00:00` local — **not** a midnight
kick-off).

**Gate.** The script refuses to write if the stable key is not unique, so a clean
exit is the gate. Confirm the printed played/upcoming split looks right.

> Join this file on `(season, league, team_home, team_away)` — **never** on
> `match_id`. kicker re-files a postponed fixture onto the matchday it is actually
> played, which changes `match_id`; a `match_id` join drops such fixtures silently.

---

## Stage 4 — Odds

```bash
cd ../001_Development/001_01__SFMMO_WC26
python 001_100__Odds_Scrape_FootballData.py   # download football-data.co.uk CSVs
python 001_101__Odds_TeamName_Mapping.py      # map their names -> our club names
python 001_102__Odds_Join_ByMatch.py          # join -> odds_byMatch.csv
cd ../../006_Website
```

`001_100` always re-downloads the current season, so it is safe to re-run.

**Gate.** `001_102` prints per-season coverage. Current-season coverage is low
early on — football-data.co.uk publishes in batches, typically a few days behind —
so a small current-season count is expected, not a failure. Any **historical**
season dropping below its previous coverage means the name mapping broke: check
`MANUAL_FD` in `001_101` for an unmapped club (this is what promoted clubs break).

---

## Stage 5 — Training features

Builds the played-match feature set for the current season and appends it to the
frozen historical base.

```bash
python run_notebook.py 006_020__Features_Train.ipynb \
    --set "N_seasons=['2026-27']"
```

Reads **and** writes `data_byPlayer.csv` — the production file.

> **Changed 2026-08-25.** This notebook used to write `data_byPlayer__SFM_II.csv`
> while reading `data_byPlayer.csv`, so the two had to be kept in step by a manual
> `cp`. Forgetting it made stage 7 build features against last week's player list
> with no error at all. There is now one production file. `__SFM_II` is a research
> dataset and lives in `10_data/102_Development/102_01__data/`.

> **Re-running this stage requires a rollback first.** The export *appends*, so a
> second run adds a duplicate copy of the season. Before any re-run:
>
> ```bash
> python - <<'PY'
> import pandas as pd
> f='../../10_data/106_Website/data_byPlayer.csv'
> d=pd.read_csv(f,low_memory=False); b=len(d)
> d=d[d['season']!='2026/27']; d.to_csv(f,index=False)
> print(f'{f.split("/")[-1]}: {b:,} -> {len(d):,}')
> PY
> ```
>
> Note the season format differs: config uses `2026-27`, the data column uses
> `2026/27`.

**Gate.** The notebook prints `# duplicated Match-IDs: 0` — anything else means the
rollback was skipped. Also confirm the row count grew by roughly
`matches × scoring players per match`, and that `[note ] ... ladder padded to N
clubs` appears in the opening weeks (it means clubs yet to kick off are correctly
present in the table).

---

## Stage 6 — Refresh the research mirror

```bash
cp ../../10_data/106_Website/data_byPlayer.csv \
   ../../10_data/102_Development/102_01__data/data_byPlayer__SFM_II.csv
```

Research code (`SFM_II__dev`, `SFM_OG`, the `001_0*` Transfermarkt scripts, the
SFMMO dev notebooks) reads the mirror under `102_Development/102_01__data/`, so
production runs can never be disturbed by exploratory work. The mirror only moves
when you run this line.

**Gate.** `cmp` reports the two files identical:

```bash
cmp -s ../../10_data/106_Website/data_byPlayer.csv \
       ../../10_data/102_Development/102_01__data/data_byPlayer__SFM_II.csv \
  && echo "mirror current" || echo "mirror STALE"
```

> Skip this step deliberately if a researcher is mid-experiment and wants their
> base frozen — nothing in production depends on it. But tell them, because a
> stale mirror looks exactly like a current one.

---

## Stage 7 — OOS features

Builds the feature rows for every upcoming fixture — what the model scores.

```bash
python run_notebook.py 006_021__Features_OOS.ipynb --skip 20,21,23
```

> **`--skip 20,21,23` is required.** Those are stale one-line diagnostic cells from
> earlier sessions; cell 20 references a variable that only exists inside a disabled
> `if 1==2:` block, so a plain run-all dies *after* the expensive main loop and
> *before* the export.

Writes `data_byPlayer__OOS.csv` and `data_byPlayer__OOS__currentTEAM.csv`.

**Gate.** Fixture counts per league should match stage 2's upcoming counts, and
duplicates on `(id_match, name_player)` must be 0.

Two things that look wrong but are expected:

- `position_player` is all-NaN in the OOS file. Long-standing behaviour, not a
  regression.
- A fixture between two **promoted** clubs has no rows at all, because neither
  club has a player in `PlayerUniverse__all.csv`. Those matches get no prediction.

**After stage 7 the modeller deliverables are complete.** SFM and SFMMO both consume
exactly these two files:

| File | Contents |
|------|----------|
| `10_data/106_Website/data_byPlayer.csv` | played history, all seasons |
| `10_data/106_Website/data_byPlayer__OOS.csv` | upcoming fixtures |

Hand those over. Stages 8–12 are the publishing legs.

---

## Stage 8 — Scoring probabilities

```bash
python run_notebook.py 006_040__Predictions_ScoringProb__SFM_II.ipynb --set "SMOKE=False"
```

> **`--set "SMOKE=False"` is required.** The notebook ships with `SMOKE = True` so
> that a careless run cannot overwrite production: it restricts to two players and
> the export cell then raises `AssertionError: SMOKE run — a two-player dictionary
> must not overwrite production`. That error **is** the smoke test passing. Run it
> once as shipped if you want the cheap end-to-end check (~1 min), then re-run with
> `SMOKE=False`.

Reads `data_byPlayer.csv` + `data_byPlayer__OOS.csv` and the fitted **light bundle**
`10_data/01_Models/SFM_II_FinalC_ELO_scaleCS__2526__LIGHT.pkl` — posterior draws
only, pure NumPy, no PyMC and no GPU. Writes:

| Output | Path |
|--------|------|
| board the site reads | `01__SFMcom/SFMwebsite__v2/static/data/040_ScoringProb__prod.pkl` |
| training-history board | `10_data/106_Website/040_ScoringProb__{model}__train.pkl` |
| point-in-time ledger | `10_data/106_Website/SFM_predictions__frozen.csv` |

Config worth knowing (`--set` any of them):

- `datasets_to_process` — ships as `['oos']`, which is the weekly job. Add `'train'`
  only when the model is refitted; it takes ~10 min and is merged back automatically
  from its own pickle on later runs.
- `FORECAST_HORIZON_DAYS = 14` — stage 7 emits the **whole season**; only fixtures
  inside the horizon are scored. Date-based, never round-based (see below).
- `SFM_model__NAME` / `train_end` — change only when a new model is committed.

**Gate.** The notebook self-checks and prints each result. All four must be right:

| Printed line | Requirement |
|---|---|
| `[golden rows] ... PASS` | ~1e-16. Anything else = engine drift, **do not publish** |
| `bundle/CSV contract: rows … OK \| category mix OK` | a MISMATCH means the dataset changed under the fitted model → **refit before serving** |
| `OOS factor coverage` table | in the `zero-share: played` column no factor may read ≥0.95 — that is the O-C1 bug (fixture builder keying on round, not history). The `not yet played` column reading 1.00 is **correct** |
| `[ledger] … (+N frozen this run)` | N ≈ the fixtures played since the last run |

Then eyeball the board: `P(≥1 goal)` should top out ~0.3–0.5 for elite strikers.

> **Early-season caveat.** `goalsscored_cum_player` is standardised within
> season × gameday. In the opening rounds ~90 % of players have zero, so one early
> goal produces a very large z-score and the board over-rewards whoever scored first
> — in round 2 of 2026/27 it put Gouiri (0.69) and Aubameyang (0.65) above Mbappé
> (0.48). It self-corrects as the season fills in. Not a bug; decide whether it is
> acceptable publicly before it is on the front page.

> **Fractional gamedays are real.** After the round-1 postponements La Liga ran
> rounds 2, **2.5**, 3, 4 and 6 concurrently. The notebook truncates (`2.5 → 2`,
> the house convention) and prints a note. This is why the horizon is by date.

> **The ledger is the only honest track record.** Unplayed fixtures refresh every
> run; a played fixture's row is **never rewritten** and its result is attached
> beside the probabilities standing at kick-off. `appeared=False` marks a forecast
> player who did not feature — the tracker must **void** those, not score them as
> misses (~35 % of rows in the opening weeks). Rows with an empty
> `forecast_frozen_at` are players who featured but were never forecast.
> Upstream `id_match` is **not stable** — a postponement renumbers a whole league —
> so the ledger re-points itself each run by team pair, choosing the earlier leg.
> Never regenerate it from post-match data.

> The shipped model **is not re-estimated weekly.** Refits happen once a season via
> `001_Development/001_00__SFM/SFM_II__dev_EW.ipynb` (`RUN_PRODUCTION=True`), which
> re-runs selection and writes a new bundle. Feature definitions must stay
> consistent with what it was trained on — never "fix" a feature here to make a
> number look better; the bundle/CSV gate exists to catch exactly that.

---

## Stage 9 — SAR / PAR

```bash
python 006_041__SAR_PAR__SFM_II.py
```

Skill/performance-above-replacement boards. Reads the same light bundle plus
`data_byPlayer.csv`; ~4 min. Writes
`01__SFMcom/SFMwebsite__v2/static/data/041_SARPAR__prod__SFM_II.pkl` (~100 MB).

**Gate.** `golden rows : PASS`, the printed SAR counterfactual lists each factor as
`EQUALIZED`/`held`/`-> mean`, and the top-10 board is recognisable — Messi, Haaland,
Kane, Ronaldo, Lewandowski, Mbappé at ~+0.33 to +0.55 capped goals per appearance
above the average player.

> Computed over the **training window**, so it barely moves week to week — it is a
> season-long skill board, not a form table. Safe to skip on a normal week; re-run
> after a refit.

> **New filename and new shape.** The old `041_SARPAR__prod.pkl` (**10.5 GB**, the
> full observation-level posterior) is dead — nothing reads it, and it can be
> deleted. The new file is pre-aggregated per player per draw; consume it via
> `get__SAR_PAR__SFM_II()` at the bottom of the script. `006_042__SAR_PAR_funcCalc.py`
> still points at the OG artifact and needs redirecting.

---

## Stage 10 — SFMMO match outcome

```bash
python 006_060__Predictions_MatchOutcome__SFMMO.py
python 006_061__SeasonOdds__SFMMO.py
```

**Prerequisite — the season bundle.** `006_060` fits nothing. It loads
`10_data/01_Models/SFMMO_DevK__scaleCS__train202526__PROD.pkl`: posterior draws,
model graph, team index, cross-sectional scaling moments and the Dixon–Coles ρ, all
fitted **once before the season** by `SFMMO__dev_EW.ipynb` with `FIT_PRODUCTION=True`
(GPU, minutes). During the season the parameters never move — only the features roll
forward as results arrive. That is deliberate: it is exactly the protocol the
expanding-window validation measured, so the published intervals mean what they say.
Re-fitting mid-season would silently invalidate them.

**Only the next two weeks are forecast.** `HORIZON_DAYS = 14`. The OOS file carries
the whole season (~1,700 fixtures); the run scores only those kicking off inside the
window — typically 100–150. A May fixture predicted in August carries no form
information, and forecasting it would freeze that near-useless number into the
ledger. The horizon is **date-based, not round-based**, for the same reason stage 3
joins on teams: rounds interleave when matches are postponed (La Liga, August 2026 —
round 2 was played before three round-1 fixtures).

Writes into `10_data/106_Website/`:

| File | Contents |
|------|----------|
| `SFMMO_predictions__matches.csv` | one row per fixture, **played and upcoming** — `status`, results, W/D/L with credible bands, expected goals, most likely score |
| `SFMMO_predictions__scorelines.csv` | score grids, `p_mid` / `p_lo` / `p_up` per cell |
| `SFMMO_predictions__team_goals.csv` | per team-match P(0/1/2/3+ goals) with bands |
| `SFMMO_predictions__prod.pkl` | all three tables plus run metadata |
| `SFMMO_predictions__frozen.csv` | internal ledger — the last **pre-match** forecast per fixture |

Finished fixtures carry the probabilities frozen *before kickoff*, never a
re-forecast: by the next run ELO has absorbed the result, and grading against a
re-derived board flatters the model by roughly 4 percentage points (measured at the
World Cup). `forecast_frozen_at` records when each was fixed.

**Gate.** Three lines from the run, all printed:

1. `[eta parity] PASS` — the NumPy reconstruction of η must equal the model graph's
   to < 1e-8 (observed ~5e-16). This is the check that would have caught the
   World-Cup `mu` bug on day one. **Nothing is exported if it fails.**
2. `sanity (N rows with a forecast): mean row sum 1.000000` — W/D/L must normalise.
3. The `⚠️ played fixture(s) have NO frozen forecast` warning must name **only
   fixtures already known to be affected** (as of 2026-08-25: Rayo–Alavés and
   Betis–Real Sociedad, La Liga round 2, which kicked off before ever entering a
   run). A *new* name means a fixture was played without ever appearing in a run
   while unplayed — that receipt is gone and cannot be reconstructed honestly. If it
   recurs, widen `HORIZON_DAYS` or run more often.

Observed on 2026-08-25: `45 finished + 101 upcoming`, parity `6.7e-16`, row sums
`1.000000`, previous board archived to `_vintages/`.

> **The frozen ledger is keyed on `(season, home_team, away_team)` — never
> `id_match`.** Same lesson as stage 3, and it has already bitten once: when
> Celta–Osasuna was postponed out of La Liga round 1, the survivors were renumbered
> (`G6`→`G5`, `G7`→`G6`) and the then id-keyed ledger handed each match its
> *neighbour's* frozen forecast — Deportivo–Elche was graded with Celta's number.
> Silent, and fatal to the receipts. `006_060` now refuses to run against a ledger on
> the old schema; `rebuild_frozen_ledger.py` reconstructs one from the archived board
> vintages in `_vintages/`, which carry team names.

> **Promoted clubs** — stage 7's note has a consequence here. A fixture where only
> one side has players gets its missing perspective reconstructed by swapping the
> `*_team`/`*_opp` columns (exact, not approximate; the run reports the count). A
> fixture between *two* promoted clubs has no rows at all and gets no forecast.

### `006_061` — season odds (title / top-4 / relegation)

Simulates the **remainder** of the season, once per posterior draw, pinning every
played match to its real score. Writes:

| File | Contents |
|------|----------|
| `SFMMO_season_odds.csv` | current board — `as_of`, `p_title`, `p_top4`, `p_releg`, `exp_pts` with `pts_lo`/`pts_up`, `elo_now`, `new_team` |
| `SFMMO_season_odds__tracker.csv` | one row per (`as_of`, league, team) — the time series behind the website's odds chart; idempotent per day, so re-running does not duplicate |
| `_vintages/SFMMO_season_odds__<date>.csv` | dated snapshot of the outgoing board |

Fixtures are simulated as a full double round-robin rather than from the published
calendar (identical for a complete season, and independent of fixture-list
availability); played matches are matched **by team pair** and pinned. Form features
sit at league average — folding real form in was tested at the World Cup and
rejected; ELO carries recency instead.

**Gate.** `sum p_title 1.000` per league, and the printed `N of M fixtures pinned`
must match the number actually played (0 is correct for a league that has not
started). **A league that played nothing since the last run must report numbers
identical to the last run** — the RNG is seeded per league from
`(SEED, crc32(league))`, so its simulation is independent of iteration order and of
which other leagues ran. Two consecutive runs on unchanged data are bit-identical
(verified 2026-08-25).

> **`elo_now` must come from each team's EARLIEST UNPLAYED fixture**, never its
> latest row. `compute_elo` applies a fake-draw update to unplayed fixtures so that
> upcoming gamedays carry sensible ratings — which means a team's rating on its May
> fixture has ~34 phantom draws in it and has regressed hard toward the mean. Taking
> the last row wrecked the first full-season board: Bayern entered at 1537 instead of
> 1656 (−119 elo) and its title probability fell 76% → 62% having played no matches
> at all. This was invisible while the OOS file held a single matchday and appeared
> the moment it carried the whole season. Fixed 2026-08-25; the guard is the gate
> above — an unchanged league whose numbers move is the symptom.

Worth running weekly once a meaningful number of fixtures is complete. In the opening
weeks it will closely reproduce the pre-season board, which is correct rather than
suspicious: with 2–3% of fixtures pinned there is little to condition on, and
favourites should look much as they did in August.

---

## Stage 11 — Load into Railway Postgres

```bash
cd 01__SFMcom/SFMwebsite__v2
DATABASE_URL="$DATABASE_URL" python scripts/migrate_pkl_to_postgres.py
DATABASE_URL="$DATABASE_URL" python scripts/migrate_sarpar_to_postgres.py
```

Export `DATABASE_URL` from your own environment or `.env`. **Do not paste the
connection string into a file that gets committed** — `ReadMe.txt` contains old
inline credentials and is the example not to follow.

`migrate_sarpar_to_postgres.py` uses `clear_existing=True` because rankings
recompute. `migrate_pkl_to_postgres.py` is safe to re-run incrementally.

> **The prediction board is read from the working tree, not from git.** It is
> gitignored on purpose, so a fresh clone will not have it — the migration fails
> with a message telling you where to get it. That also means stage 11 publishes the
> new numbers whether or not stage 12 has run.

> **`WeeklyPick` is populated from the frozen ledger** (stage 8), not re-derived from
> the current board — picks are detached via `SET_NULL` and re-linked, so receipts
> survive a full rebuild. Do not "simplify" that back into a delete-and-recreate: the
> published track record would silently become a re-forecast, which measured ~4 pp
> flattering on the SFMMO side. Anything shown as a past prediction must trace to
> `SFM_predictions__frozen.csv`; the `train` split is in-sample fit and must never be
> presented as a forecast.

`migrate_users.py` is a **one-time** DB-to-DB copy. Never run it as part of a
weekly update.

---

## Stage 12 — Deploy

```bash
git add -A && git commit -m "weekly update <date>" && git push origin main
```

Railway runs `collectstatic` → `migrate` → `gunicorn` on push. If models changed,
run `makemigrations` locally first and commit the migration — the deploy only runs
`migrate`.

---

## If kicker blocks you (stages 1–2)

1. **Stop.** Do not switch VPN nodes and do not swap HTTP client. Both were tested
   in August 2026; neither helps, and a browser-impersonating client measured
   *worse* than plain `requests` with a session.
2. The historical cause was **self-inflicted**: an un-paced retry loop firing
   thousands of requests. That is fixed, so a fresh block most likely means the
   day's budget is genuinely spent.
3. kicker's allowance is **shared across every script** you run from that IP. Order
   the day accordingly: results first (small, time-critical), bulk re-scrapes later.
4. Blocks have consistently cleared overnight. Resume the next morning — stages 1–2
   are re-runnable, and the player-metadata set-difference means only genuinely new
   players get re-fetched.
