#!/usr/bin/env python
"""One-off: rebuild SFMMO_predictions__frozen.csv on the team-keyed schema.

The original ledger keyed on id_match. When a postponed fixture leaves a round, the source
renumbers the survivors, so id-keyed frozen forecasts get attached to the WRONG matches
(La Liga 2026-08-22: Deportivo-Elche was handed Celta-Osasuna's probability). The archived
board vintages carry team names, so the correct mapping is fully recoverable.

Rebuilds from the OLDEST vintage forward, so that for each fixture the EARLIEST pre-match
forecast is kept where a later run would only have re-frozen the same thing.
"""
import glob, os, re, pandas as pd

D = '/Users/maximilian/Dropbox/Max/51_SoccerAnalytics/10_data/106_Website'
PCOLS = ['p_home_win','p_home_win_lo','p_home_win_up','p_draw','p_draw_lo','p_draw_up',
         'p_away_win','p_away_win_lo','p_away_win_up','exp_goals_home','exp_goals_away',
         'ml_score_home','ml_score_away']
KEY = ['season','home_team','away_team']

vints = sorted(glob.glob(f'{D}/_vintages/SFMMO_predictions__matches__*.csv'))
print(f"rebuilding from {len(vints)} archived vintage(s)")
rows = []
for v in vints:
    d = pd.read_csv(v)
    stamp = re.search(r'__(\d{4}-\d{2}-\d{2})', os.path.basename(v)).group(1)
    if 'status' in d.columns:                      # newer vintages carry played rows too
        d = d[d['status'] == 'upcoming']
    d = d[d['p_home_win'].notna()]
    if not len(d):
        continue
    d = d[KEY + ['id_match'] + PCOLS].copy()
    d['forecast_frozen_at'] = stamp
    rows.append(d)
    print(f"  {os.path.basename(v)}: {len(d)} pre-match forecasts")

led = pd.concat(rows, ignore_index=True)
led = led.sort_values('forecast_frozen_at').drop_duplicates(KEY, keep='last')   # latest pre-match
out = f'{D}/SFMMO_predictions__frozen.csv'
if os.path.exists(out):
    os.replace(out, out.replace('.csv', '__idkeyed_CORRUPT.csv.bak'))
    print("  old id-keyed ledger moved aside -> ...__idkeyed_CORRUPT.csv.bak")
led.to_csv(out, index=False)
print(f"\nwrote {len(led)} fixtures, frozen dates {sorted(led['forecast_frozen_at'].unique())}")
print(led[led['home_team'].isin(['Deportivo A Coruna','Atletico Madrid','Celta Vigo'])]
      [KEY + ['p_home_win','forecast_frozen_at']].to_string(index=False))
