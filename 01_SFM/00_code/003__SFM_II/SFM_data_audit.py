# ======================================================================================== #
#                              SFM -- Data Vintage Audit                                    #
# ======================================================================================== #
#
# Referee-report verification battery (2026-08-04), packaged as a runnable script so every
# NEW data vintage gets the same scrutiny the audited one did. Run BEFORE any model sees
# the file:
#
#     python SFM_data_audit.py                                   # audits the canonical CSV
#     python SFM_data_audit.py path/to/new_vintage.csv           # audits a specific file
#     python SFM_data_audit.py new.csv --baseline old.csv        # + coverage diff vs old
#
# Checks (each prints PASS / FAIL / WARN):
#   1. schema           -- required columns present, (name_player, id_match) unique
#   2. gameday          -- integer, agrees with the id_match parse, per-league ranges
#   3. timing contracts -- THE load-bearing ones (referee M1):
#                            cum_player is PRE-match; points are PRE-match;
#                            share == cum_player / cum_team; first-appearance cum == 0
#   4. positions        -- known labels only; NaN share + scoring rate of the NaN group
#   5. sanity           -- share > 1 rows, goals distribution, kick_off parseable
#   6. base rates       -- goals/appearance by league and by season (era drift table)
#   7. holdout guard    -- WARN loudly if a sealed/holdout season enters the vintage
#   8. baseline diff    -- new/lost leagues & seasons, row-count deltas (if --baseline)
#
# ======================================================================================== #

import argparse
import sys

import numpy as np
import pandas as pd

CANONICAL = "/Users/maximilian/Dropbox/Max/51_SoccerAnalytics/10_data/106_Website/data_byPlayer__SFM_II.csv"

# --- seasons no experiment may tune against (keep in sync with SFM_II__dev_EW.ipynb):
SEALED_SEASONS = ["2024/25", "2025/26"]

REQUIRED_COLS = [
    "goal", "goals_in_match", "goalsscored_cum_player", "goalsscored_cum_team",
    "points_team", "points_opp", "points_diff", "home_pitch",
    "goalsscored_rank_team", "goalsconceded_rank_opp", "goalsscored_rank_team_wo_player",
    "goalsscored_share_player_team", "id_match", "name_team", "name_opp",
    "name_league", "season", "gameday", "kick_off", "name_player", "position_player",
]

KNOWN_POSITIONS = {"Sturm", "Mittelfeld", "Abwehr", "Torwart"}   # Torwart tolerated if added

_n_fail = 0
_n_warn = 0


def report(ok, label, detail="", warn=False):
    global _n_fail, _n_warn
    if ok:
        tag = "PASS"
    elif warn:
        tag = "WARN"
        _n_warn += 1
    else:
        tag = "FAIL"
        _n_fail += 1
    print(f"[{tag}] {label}" + (f" -- {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=CANONICAL)
    ap.add_argument("--baseline", default=None, help="previous vintage CSV for a coverage diff")
    args = ap.parse_args()

    print(f"\n=== SFM data vintage audit: {args.csv} ===\n")
    df = pd.read_csv(args.csv)
    print(f"rows: {len(df):,}   players: {df['name_player'].nunique():,}   "
          f"leagues: {df['name_league'].nunique()}   seasons: {df['season'].nunique()}\n")

    # ---------------------------------- 1. schema ---------------------------------- #
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    report(not missing, "schema: required columns", f"missing: {missing}" if missing else "")
    dup = df.duplicated(subset=["name_player", "id_match"]).sum()
    report(dup == 0, "schema: (name_player, id_match) unique", f"{dup} duplicate rows" if dup else "")

    # ---------------------------------- 2. gameday ---------------------------------- #
    try:
        gd_int = df["gameday"].astype(int)
        ok_int = (gd_int == df["gameday"]).all()
    except (ValueError, TypeError):
        ok_int = False
    report(ok_int, "gameday: integer-valued column")
    try:
        # --- split matchdays ('GD10-2') carry a dash; the leg suffix is not a gameday
        parsed = df["id_match"].apply(lambda i: int(float(str(i).split("_")[1][2:].split("-")[0])))
        gd_base = pd.to_numeric(df["gameday"].astype(str).str.split("-").str[0], errors="coerce")
        n_mm = int((parsed != gd_base).sum())
        report(n_mm == 0, "gameday: agrees with id_match parse", f"{n_mm} mismatches" if n_mm else "")
        n_dash = int(df["id_match"].astype(str).str.contains("GD[0-9]+-", regex=True).sum())
        report(n_dash == 0, "gameday: no split-matchday suffixes in id_match",
               f"{n_dash} rows -- the notebooks' int() parse CRASHES on these" if n_dash else "",
               warn=False)
    except Exception as e:  # id format changed -> the notebooks' assert would also fire
        report(False, "gameday: id_match parseable", str(e))
    print("       max gameday by league:",
          df.groupby("name_league")["gameday"].max().astype(int).to_dict())

    # ------------------------- 3. timing contracts (referee M1) ------------------------- #
    d = df.copy()
    d["kick_off"] = pd.to_datetime(d["kick_off"])
    d = d.sort_values(["name_player", "season", "kick_off"])
    g = d.groupby(["season", "name_player"])

    diff_cum = g["goalsscored_cum_player"].diff()
    prev_goals = g["goals_in_match"].shift(1)
    cur_goals = d["goals_in_match"]
    m = diff_cum.notna()
    frac_pre = float((diff_cum[m] == prev_goals[m]).mean())
    frac_post = float((diff_cum[m] == cur_goals[m]).mean())
    report(frac_pre > 0.995 and frac_pre > frac_post,
           "timing: goalsscored_cum_player is PRE-match",
           f"diff==prev {frac_pre:.4f} vs diff==current {frac_post:.4f} "
           f"(momentum leaks the current match if this flips!)")

    first = g.head(1)
    frac0 = float((first["goalsscored_cum_player"] == 0).mean())
    report(frac0 == 1.0, "timing: first-appearance cum_player == 0 (momentum seed)",
           f"{frac0:.4f}")

    tg = d.groupby(["season", "name_team"])
    first_team = d.loc[tg["kick_off"].transform("min") == d["kick_off"]]
    frac_pts0 = float((first_team["points_team"] == 0).mean())
    report(frac_pts0 == 1.0, "timing: points_team == 0 at each team's first match (PRE-match)",
           f"{frac_pts0:.4f}")

    with np.errstate(divide="ignore", invalid="ignore"):
        share_chk = d["goalsscored_cum_player"] / d["goalsscored_cum_team"]
    both = share_chk.notna() & d["goalsscored_share_player_team"].notna()
    frac_share = float(np.isclose(share_chk[both], d.loc[both, "goalsscored_share_player_team"]).mean())
    report(frac_share > 0.999, "timing: share == cum_player / cum_team", f"{frac_share:.4f}")

    # ---------------------------------- 4. positions ---------------------------------- #
    vals = set(df["position_player"].dropna().unique())
    unknown = vals - KNOWN_POSITIONS
    report(not unknown, "positions: known labels only", f"new labels: {unknown}" if unknown else str(sorted(vals)))
    # --- the audited (Apr-2026) vintage carried all three outfield labels on ~90% of rows;
    # --- a missing label means the position scrape regressed, not that football changed.
    for lbl in ("Abwehr", "Mittelfeld", "Sturm"):
        report(lbl in vals, f"positions: label '{lbl}' present",
               "MISSING -- present in the audited vintage" if lbl not in vals else "")
    filled = d.groupby(["season", "name_player"])["position_player"].transform(lambda s: s.bfill().ffill())
    nan_share = float(filled.isna().mean())
    nan_rate = float(d.loc[filled.isna(), "goals_in_match"].mean()) if filled.isna().any() else float("nan")
    # --- audited vintage: 10% NaN scoring 0.022 (goalkeeper-like). A large NaN share, or a
    # --- NaN group scoring like ordinary players, means positions are MISSING rather than
    # --- structurally absent -- position_MID/FOR collapse to ~0 and F_GKu loses its premise.
    report(nan_share <= 0.25, "positions: NaN-after-groupfill share",
           f"{nan_share*100:.1f}% of rows (audited vintage: 10.0%), scoring {nan_rate:.4f} "
           f"goals/app (GK-like ~0.02; ~0.19 means these are ordinary players with no label)")
    print("       position coverage by league:")
    print(df.assign(_p=df["position_player"].notna()).groupby("name_league")["_p"]
          .mean().round(4).to_string().replace("\n", "\n       "))

    # ---------------------------------- 5. sanity ---------------------------------- #
    n_share = int((df["goalsscored_share_player_team"] > 1).sum())
    report(n_share < 50, "sanity: share > 1 rows (dropped by the notebooks)", f"{n_share} rows",
           warn=n_share >= 50)
    dist = df["goals_in_match"].value_counts(normalize=True).sort_index()
    print("       goals/appearance distribution:", {int(k): round(v, 4) for k, v in dist.items()})
    report(dist.index.max() <= 6 or dist.loc[dist.index > 6].sum() < 0.001,
           "sanity: goal counts in plausible range", f"max = {int(dist.index.max())}", warn=True)

    # -------------------- 5b. appearance density (the denominator contract) -------------------- #
    # --- goals/appearance is a RATIO: a changed appearance definition moves it without any
    # --- football changing. 2025/26 arrived at 18.5 player-rows/match vs ~11.1 elsewhere,
    # --- which alone explained a "35% scoring collapse" while goals/match stayed normal.
    dens = df.groupby("season").apply(
        lambda x: pd.Series({"rows_per_match": len(x) / x["id_match"].nunique(),
                             "goals_per_match": x.groupby("id_match")["goals_in_match"].sum().mean()}),
        include_groups=False)
    med = dens["rows_per_match"].median()
    off = dens[(dens["rows_per_match"] - med).abs() / med > 0.25]
    report(len(off) == 0, "appearance density: rows/match stable across seasons",
           f"median {med:.2f}; OUTLIERS -> {dict(off['rows_per_match'].round(2))}" if len(off)
           else f"median {med:.2f} rows/match")
    if len(off):
        print("       (goals/match for the same seasons: "
              f"{dict(dens.loc[off.index, 'goals_per_match'].round(3))} -- if these look normal, "
              "the APPEARANCE definition changed, not the scoring)")

    # ---------------------------------- 6. base rates ---------------------------------- #
    print("\n--- goals/appearance by league (league-effect watch; audited vintage spread was ~5%):")
    print(df.groupby("name_league")["goals_in_match"].mean().round(4).sort_values().to_string())
    print("\n--- goals/appearance by season (era-drift watch; audited vintage fell ~18% 2000->2023):")
    print(df.groupby("season")["goals_in_match"].mean().round(4).to_string())

    # ---------------------------------- 7. holdout guard ---------------------------------- #
    sealed_present = sorted(set(SEALED_SEASONS) & set(df["season"].unique()))
    if sealed_present:
        print("\n" + "!" * 92)
        for s in sealed_present:
            print(f"!!  sealed/holdout season {s} is IN this vintage -- it must be listed in "
                  f"HOLDOUT_SEASONS in SFM_II__dev_EW.ipynb BEFORE any run touches this file.")
        print("!" * 92)
        report(True, "holdout: sealed seasons present", ", ".join(sealed_present), warn=True)
    else:
        report(True, "holdout: no sealed seasons in vintage")

    # ---------------------------------- 8. baseline diff ---------------------------------- #
    if args.baseline:
        old = pd.read_csv(args.baseline, usecols=["name_league", "season", "name_player"])
        print(f"\n--- coverage diff vs baseline ({args.baseline}):")
        new_lg = set(df["name_league"].unique()) - set(old["name_league"].unique())
        lost_lg = set(old["name_league"].unique()) - set(df["name_league"].unique())
        new_ssn = sorted(set(df["season"].unique()) - set(old["season"].unique()))
        lost_ssn = sorted(set(old["season"].unique()) - set(df["season"].unique()))
        print(f"    leagues  +{sorted(new_lg)}  -{sorted(lost_lg)}")
        print(f"    seasons  +{new_ssn}  -{lost_ssn}")
        print(f"    rows     {len(old):,} -> {len(df):,}  ({len(df) - len(old):+,})")
        print(f"    players  {old['name_player'].nunique():,} -> {df['name_player'].nunique():,}")
        if new_lg:
            print("    NOTE: new leagues -> re-check the league base-rate spread above; the "
                  "'league intercepts are hygiene' verdict was measured on the big-5 only.")

    # ---------------------------------- summary ---------------------------------- #
    print(f"\n=== audit complete: {_n_fail} FAIL, {_n_warn} WARN ===")
    if _n_fail:
        print("!!  Do NOT run the harness on this vintage until every FAIL is resolved.")
    return 1 if _n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
