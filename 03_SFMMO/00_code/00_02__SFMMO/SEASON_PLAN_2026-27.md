# SFM/SFMMO — Receipts-First Season Plan 2026/27

*Written July 2026, post-World Cup. For the analyst: what the tournament taught us,
what to fix in the off-season, and the standing rhythm for the league season.
Strategy premise: we will not out-predict the closing market — our asset is the
**public, frozen, honestly-graded track record** ("the receipts") plus the voice
that comes with it. Everything below serves that.*

---

## 1 · What the World Cup taught us (the mistakes to learn from)

Full evidence: `SFMMOwm_validation__bymatch.csv` / `__summary.csv`, the
`_vintages/` boards, and the website Validation page.

**L1 — The cold-start board is our weakest product.**
MD1 log-loss 1.133 — *worse than uniform* (ln 3 ≈ 1.099). With no in-tournament
form, ELO had nothing to chew on. Every forecast after MD1 was fine (0.77–0.92).
The league equivalent is matchdays 1–4 of a new season (promotions, transfers,
stale carryover). → Fix in §2-E1.

**L2 — Systematic under-confidence on favourites.**
The entire RPS deficit vs the market came from favourite-won games (avg top-pick
conviction 56% vs the market's 64%; on our *correct* calls we priced the winner
57% vs their 67%). Ranking was fine — we even edged the market on upsets. This is
a **calibration** problem, not a discrimination problem. Note `T_STAR=None` in
`006_050` — a ~0.85 temperature was validated and left OFF. → §2-E2.

**L3 — Knockout draws are structurally underpriced.**
9 of 32 KO games (28%) were level at 90′; our draw probs on those games averaged
~21% and were *thinner than the market's* on almost every one (e.g. 19% vs 33%
Australien–Ägypten). Elimination football at 90′ is drawier than the same fixture
in a league context — teams play for extra time. The model prices them
identically. → §2-E3. (League relevance: cups — DFB-Pokal/FA Cup — and any
future tournament; also check league draw calibration generally.)

**L4 — Scoring-convention traps are real and expensive.**
We nearly published "3rd on PriorLab's board" by comparing our folded-score
log-loss (0.859) to their 90′-outcome metric; the true number under their rules
was 1.010 — behind their market (0.890) and NEXUS (0.901). **Rules before
claims: read the exact metric, re-score under it, only then compare.** Corollary:
the folded convention had been flattering us on shootout games all along.

**L5 — Hindsight-grading flatters by ~4pp.**
Scoring the *current* board against past results gave 56.2% accuracy; the frozen
pre-match boards gave 52.1%. Live tables absorb results through ELO. Never grade
anything but the frozen vintage (the website's `pub_*` freeze + `006_053`'s
`MATCHDAY_FORECAST` dict enforce this — keep them sacred).

**L6 — Archive before load, always.**
We lost (and only by luck recovered) the pre-MD2 board. Now automated
(`006_050` `archive_existing_outputs()` → `_vintages/`), but the rule is
cultural, not just technical: **a forecast that isn't archived before the next
refresh never existed.**

**L7 — Capture the full result, not just the headline score.**
Folded scores alone cost us the PriorLab comparison until
`SWM2026_games_90min.csv` was produced. Standing order: every result row carries
90′ score + HT + ET/pens flags (that file's schema is now the standard).

**L8 — Odds are part of the dataset, not an afterthought.**
Pinnacle closing (PSC) was empty all tournament; knockout odds arrived late and
gappy. The market column is our most important benchmark — capture odds weekly
*with* each vintage, prefer closing where available, and store them alongside
the frozen board.

---

## 2 · Off-season experiments (August, before the season)

Build one **evaluation harness first**, then run every experiment through it:
score candidate models on the frozen WC vintages + last league season, reporting
accuracy / RPS / log-loss under **both** conventions (folded + 90′) and vs the
de-vigged market. Extend `006_053` — it already does 80% of this. No experiment
"works" unless it beats the current model *on frozen out-of-sample boards*.

- **E1 — Cold-start priors.** Shrink early-season team strength toward an
  informed prior (previous-season posterior + promoted-team prior + optional
  transfer-window adjustment) with a decaying weight over MD1–5. Target: kill
  the L1 cold-start hole without touching mid-season behaviour.
- **E2 — Calibration / sharpening.** Turn on temperature (start at the
  validated ~0.85), fit on one season, test frozen on another. Success = RPS
  gap to market narrows with calibration curves staying honest (no
  over-sharpening into overconfidence). This is the single highest-value fix.
- **E3 — Context-dependent draw model.** Draw inflation (Dixon-Coles-style ρ or
  an explicit draw bump) conditioned on elimination context; check league draw
  calibration while at it. Success metric: 90′-convention log-loss on the WC
  knockout boards drops from 1.010 toward the market's 0.90.
- **E4 — (stretch) Entry-ready pipeline.** A script that converts any frozen
  board into a competition submission under an arbitrary stated metric —
  because of L4, and because §4 makes competitions a channel.

Priority if time is short: **E2 → E3 → E1** (E2 helps every single forecast).

---

## 3 · The standing weekly rhythm (in season)

Per league matchday (the WC drill, generalized — commands in
`SFMwebsite__v2/scripts/README_MIGRATION.md`):

1. **Archive** the outgoing board to `_vintages/` (automatic — verify, don't trust).
2. Refresh data → re-estimate → export pickle (with results incl. 90′ + ET
   flags where applicable, and this week's odds).
3. Load prod (`migrate_sfmmo_to_postgres.py`) → odds
   (`migrate_sfmmo_odds.py`) → **verify** (validation endpoint numbers match
   `006_053`'s summary) → push notification only when warranted.
4. `006_053` re-run: extend `MATCHDAY_FORECAST` with the new frozen board.

The freeze does the honesty automatically now; the human job is the archive
check and the verify step.

---

## 4 · Competitions (the free attention channel)

Placing in public forecasting competitions is how a small shop earns the
credibility that cold emails can't. Standing orders:

- Watch for PriorLab's next contest (they run these seasonally), Kaggle sports
  comps, and academic challenges (e.g. RoboCup-adjacent, Machine Learning for
  Soccer workshops).
- **Read the metric first** (L4). Tailor the submission to the stated scoring
  rule — submitting draw-inflated probs to a 90′-outcome metric is answering
  the question asked, not gaming it.
- Enter with the E2/E3-improved model; even mid-pack placement is content (§5).

---

## 5 · Content calendar (receipts-first)

The publishable asset is the honest arc, not the forecast. Cadence:

- **August — Season preview post.** Publish frozen pre-season forecasts for the
  top-5 leagues (title odds, relegation, top-4). *The stake in the ground —
  this is the receipt everything else gets graded against.*
- **Weekly** (light): matchday forecast + last week's scorecard. App push +
  X/LinkedIn one-liner. Automate from the social kit (`006_052`).
- **Monthly** (substantial): running validation — us vs market vs coin-flip,
  calibration plot, best/worst calls. The WC posts are the template.
- **Event-driven post-mortems** whenever we're publicly wrong — the
  knockout-draw diagnosis is the house style: *here's what we got wrong, here's
  why, here's the fix.* These outperform victory laps.
- **May — Full-season ledger** + what changes next season.

Rule inherited from the WC: **never lead with the metric that flatters us;
always show the market column.**

---

## 6 · Product (app) — in support of the above

- Ship v1.3.0 (5-tab reorg — built, on `main`, unpushed) before the season.
- **Pick'em / "Beat the SFMMO"** for league matchdays: users tap W/D/L, picks
  freeze at kickoff (same integrity rule as `pub_*`), graded on the same
  scorecard, shown as You vs SFMMO vs Market. This is the retention engine and
  the receipts ethos extended to users. Design guard: forecasting game framing,
  never gambling-adjacent.
- League data lights up the `COMPETITIONS` switcher in `MatchesScreen.tsx`.

---

## 7 · What success looks like in May 2027 (honest KPIs)

- **Model:** skill-vs-market gap narrowed from −25% toward −15% RPS (E2+E3);
  early-season cold-start hole gone (E1); at least one competition entry placed
  above its market baseline.
- **Receipts:** every league matchday frozen, graded, published — zero gaps,
  zero overwrites.
- **Audience:** the monthly validation post has a repeat readership; pick'em has
  active weekly users; one piece has been shared by someone we don't know.
- **Not a KPI:** beating the closing market overall. If that ever happens,
  celebrate — but the plan doesn't depend on it.
