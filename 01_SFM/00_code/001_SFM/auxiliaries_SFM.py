# --- The usual suspects:
import numpy as np
import pandas as pd

# --- For ranking of arrays
from scipy.stats import rankdata


# ================================ Create the Ladder for Each Season ================================ #
# 
# ---- Rows:    Dates
# ---- Columns: Clubs
# ---- Cells:   Points

def func_ladder(seasons,games):

  table_dict = dict.fromkeys(seasons, [])

  for ss in seasons:

    # --- Extract only the season 'ss' games
    games_ss = games[games.season == ss]

    # --- Order according to date
    games_ss = games_ss.sort_values('kickoff_dt').reset_index(drop=True)

    # --- Which are the match-days?
    match_days_ss = np.unique([i[0:10] for i in games_ss['kickoff_dt']])

    # --- Which teams played in season 'ss'?
    teams_ss = np.unique(games_ss.home_team)

    # --- Create the dataframe: Rows = 'match_days_ss', Columns = 'teams_ss'
    table_ss = pd.DataFrame(data=np.concatenate((np.reshape(match_days_ss,(len(match_days_ss),1)),
                                                np.zeros((len(match_days_ss),len(teams_ss)))),axis=1), 
                            columns=['match_day']+list(teams_ss))
    

    # --- Start the loop for all Teams
    for tt in teams_ss:

      # --- Matches of team 'tt'
      games_ss_tt = games_ss[(games_ss.home_team == tt) | (games_ss.away_team == tt)].reset_index(drop=True)

      # --- Start with the first 'match_days'
      match_days_tt = list()
      match_days_tt = match_days_tt + [games_ss_tt.kickoff_dt[0][0:10]]

      # --- --- Score of the match?
      points_tt = list()
      
      # --- --- --- Draw
      if games_ss_tt.outcome[0] == 'd':
        points_tt = points_tt + [1]
      # --- --- --- Home team wins & 'tt' is the home team
      elif (games_ss_tt.outcome[0] == 'h') & (games_ss_tt.home_team[0] == tt):
        points_tt = points_tt + [3]
      # --- --- --- Away team wins & 'tt' is the away team
      elif (games_ss_tt.outcome[0] == 'a') & (games_ss_tt.away_team[0] == tt):
        points_tt = points_tt + [3]
      else:
        points_tt = points_tt + [0]

      # --- --- Assign the points
      idx = table_ss[table_ss.match_day == match_days_tt[0]].index[0]
      table_ss.loc[idx,tt] = pd.to_numeric(table_ss.loc[idx,tt]) + points_tt[0]



      # --- Run loop over remaining 'match_days'
      for dd in range(1,games_ss_tt.shape[0]):

        # --- Which is the match day?
        match_days_tt = match_days_tt + [games_ss_tt.kickoff_dt[dd][0:10]]


        # --- --- Score of the match?
        # --- --- --- Draw
        if games_ss_tt.outcome[dd] == 'd':
          points_tt = points_tt + [1]
        # --- --- --- Home team wins & 'tt' is the home team
        elif (games_ss_tt.outcome[dd] == 'h') & (games_ss_tt.home_team[dd] == tt):
          points_tt = points_tt + [3]
        # --- --- --- Away team wins & 'tt' is the away team
        elif (games_ss_tt.outcome[dd] == 'a') & (games_ss_tt.away_team[dd] == tt):
          points_tt = points_tt + [3]
        else:
          points_tt = points_tt + [0]


        # --- --- Assign the points:

        # --- --- --- Current Index:
        idx_curr = table_ss[table_ss.match_day == match_days_tt[dd]].index[0]

        # --- --- --- Previous Index:
        idx_prev = table_ss[table_ss.match_day == match_days_tt[dd-1]].index[0]

        # --- --- --- Assign the points of CURRENT match day:
        table_ss.loc[idx_curr,tt] = pd.to_numeric(table_ss.loc[idx_prev,tt]) + points_tt[dd]

        # --- --- --- Fill all INTERIM days:
        table_ss.loc[(idx_prev+1):(idx_curr-1),tt] = table_ss.loc[idx_prev,tt]



      # --- Finally: Assign 'table_ss' to the dictionary 'table_dict'
      table_dict[ss] = table_ss


  return table_dict







# ================================ Create Team-Goals (scored, cumulative) for Each Season ================================ #
# 
# ---- Rows:    Dates
# ---- Columns: Clubs
# ---- Cells:   Goals

def func_goals_scored(seasons,games):


  goals_scored_dict = dict.fromkeys(seasons, [])

  for ss in seasons:

    # --- Extract only the season 'ss' games
    games_ss = games[games.season == ss]

    # --- Order according to date
    games_ss = games_ss.sort_values('kickoff_dt').reset_index(drop=True)

    # --- Which are the match-days?
    match_days_ss = np.unique([i[0:10] for i in games_ss['kickoff_dt']])

    # --- Which teams played in season 'ss'?
    teams_ss = np.unique(games_ss.home_team)

    # --- Create the dataframe: Rows = 'match_days_ss', Columns = 'teams_ss'
    table_ss = pd.DataFrame(data=np.concatenate((np.reshape(match_days_ss,(len(match_days_ss),1)),
                                                np.zeros((len(match_days_ss),len(teams_ss)))),axis=1), 
                            columns=['match_day']+list(teams_ss))
    

    # --- Start the loop for all Teams
    for tt in teams_ss:

      # --- Matches of team 'tt'
      games_ss_tt = games_ss[(games_ss.home_team == tt) | (games_ss.away_team == tt)].reset_index(drop=True)

      # --- Start with the first 'match_days'
      match_days_tt = list()
      match_days_tt = match_days_tt + [games_ss_tt.kickoff_dt[0][0:10]]

      # --- --- Goals of the match?
      goals_tt = list()
      
      if games_ss_tt.home_team[0] == tt:
        goals_tt = goals_tt + [games_ss_tt.home_score[0]]
      elif games_ss_tt.away_team[0] == tt:
        goals_tt = goals_tt + [games_ss_tt.away_score[0]]

      # --- --- Assign the goals
      idx = table_ss[table_ss.match_day == match_days_tt[0]].index[0]
      table_ss.loc[idx,tt] = pd.to_numeric(table_ss.loc[idx,tt]) + goals_tt[0]



      # --- Run loop over remaining 'match_days'
      for dd in range(1,games_ss_tt.shape[0]):

        # --- Which is the match day?
        match_days_tt = match_days_tt + [games_ss_tt.kickoff_dt[dd][0:10]]


        # --- --- Score of the match?
        if games_ss_tt.home_team[dd] == tt:
          goals_tt = goals_tt + [games_ss_tt.home_score[dd]]
        elif games_ss_tt.away_team[dd] == tt:
          goals_tt = goals_tt + [games_ss_tt.away_score[dd]]


        # --- --- Assign the points:

        # --- --- --- Current Index:
        idx_curr = table_ss[table_ss.match_day == match_days_tt[dd]].index[0]

        # --- --- --- Previous Index:
        idx_prev = table_ss[table_ss.match_day == match_days_tt[dd-1]].index[0]

        # --- --- --- Assign the points of CURRENT match day:
        table_ss.loc[idx_curr,tt] = pd.to_numeric(table_ss.loc[idx_prev,tt]) + goals_tt[dd]

        # --- --- --- Fill all INTERIM days:
        table_ss.loc[(idx_prev+1):(idx_curr-1),tt] = table_ss.loc[idx_prev,tt]



      # --- Finally: Assign 'table_ss' to the dictionary 'table_dict'
      goals_scored_dict[ss] = table_ss

  return goals_scored_dict






# ================================ Create Team-Goals (conceded, cumulative) for Each Season ================================ #
# 
# ---- Rows:    Dates
# ---- Columns: Clubs
# ---- Cells:   Goals (conceded)


def func_goals_conceded(seasons,games):


  goals_conceded_dict = dict.fromkeys(seasons, [])

  for ss in seasons:

    # --- Extract only the season 'ss' games
    games_ss = games[games.season == ss]

    # --- Order according to date
    games_ss = games_ss.sort_values('kickoff_dt').reset_index(drop=True)

    # --- Which are the match-days?
    match_days_ss = np.unique([i[0:10] for i in games_ss['kickoff_dt']])

    # --- Which teams played in season 'ss'?
    teams_ss = np.unique(games_ss.home_team)

    # --- Create the dataframe: Rows = 'match_days_ss', Columns = 'teams_ss'
    table_ss = pd.DataFrame(data=np.concatenate((np.reshape(match_days_ss,(len(match_days_ss),1)),
                                                np.zeros((len(match_days_ss),len(teams_ss)))),axis=1), 
                            columns=['match_day']+list(teams_ss))
    

    # --- Start the loop for all Teams
    for tt in teams_ss:

      # --- Matches of team 'tt'
      games_ss_tt = games_ss[(games_ss.home_team == tt) | (games_ss.away_team == tt)].reset_index(drop=True)

      # --- Start with the first 'match_days'
      match_days_tt = list()
      match_days_tt = match_days_tt + [games_ss_tt.kickoff_dt[0][0:10]]

      # --- --- Goals of the match?
      goals_tt = list()
      
      if games_ss_tt.home_team[0] == tt:
        goals_tt = goals_tt + [games_ss_tt.away_score[0]]
      elif games_ss_tt.away_team[0] == tt:
        goals_tt = goals_tt + [games_ss_tt.home_score[0]]

      # --- --- Assign the goals
      idx = table_ss[table_ss.match_day == match_days_tt[0]].index[0]
      table_ss.loc[idx,tt] = pd.to_numeric(table_ss.loc[idx,tt]) + goals_tt[0]



      # --- Run loop over remaining 'match_days'
      for dd in range(1,games_ss_tt.shape[0]):

        # --- Which is the match day?
        match_days_tt = match_days_tt + [games_ss_tt.kickoff_dt[dd][0:10]]


        # --- --- Score of the match?
        if games_ss_tt.home_team[dd] == tt:
          goals_tt = goals_tt + [games_ss_tt.away_score[dd]]
        elif games_ss_tt.away_team[dd] == tt:
          goals_tt = goals_tt + [games_ss_tt.home_score[dd]]


        # --- --- Assign the points:

        # --- --- --- Current Index:
        idx_curr = table_ss[table_ss.match_day == match_days_tt[dd]].index[0]

        # --- --- --- Previous Index:
        idx_prev = table_ss[table_ss.match_day == match_days_tt[dd-1]].index[0]

        # --- --- --- Assign the points of CURRENT match day:
        table_ss.loc[idx_curr,tt] = pd.to_numeric(table_ss.loc[idx_prev,tt]) + goals_tt[dd]

        # --- --- --- Fill all INTERIM days:
        table_ss.loc[(idx_prev+1):(idx_curr-1),tt] = table_ss.loc[idx_prev,tt]



      # --- Finally: Assign 'table_ss' to the dictionary 'table_dict'
      goals_conceded_dict[ss] = table_ss

  return goals_conceded_dict






def func_Player(dict_build):

  # --- Unpack the dictionary:
  N = dict_build['Number of Games']
  Player_lineup = dict_build['Player_lineup']
  lineup = dict_build['Lineup']
  Player_events = dict_build['Events']
  games = dict_build['Games']
  seasons = dict_build['Season']
  table_dict = dict_build['Ladder']
  goals_scored_dict = dict_build['Goals Scored']
  goals_conceded_dict = dict_build['Goals Conceded']

  CR7 = pd.DataFrame({'goal': np.zeros((N,)),'goals_in_match': np.zeros((N,)),
                      'points_team': np.zeros((N,)), 'points_opp': np.zeros((N,)),
                      'goalsscored_cum_team': np.zeros((N,)), 'goalsscored_cum_opp': np.zeros((N,)),
                      'goalsconceded_cum_team': np.zeros((N,)), 'goalsconceded_cum_opp': np.zeros((N,)),
                      'home_pitch': np.zeros((N,)), 'goalsscored_rank_team': np.zeros((N,)), 
                      'goalsconceded_rank_opp': np.zeros((N,)), 'goalsconceded_rank_opp': np.zeros((N,)), 
                      'goalsscored_rank_team_wo_player': np.zeros((N,)), 'goalsscored_cum_player': np.zeros((N,)),
                      'id_match': np.zeros((N,)), 'id_team': np.zeros((N,)), 'id_opp': np.zeros((N,)),
                      'name_team': np.zeros((N,)), 'name_opp': np.zeros((N,)),
                      'season': np.zeros((N,))
                    })


  # --- Fill: id_match
  CR7['id_match'] = Player_lineup['match_id'].values
  CR7['season'] = Player_lineup['season'].values
  CR7['kickoff_dt'] = Player_lineup['kickoff_dt'].values



  # --- Fill: goal; id_team; id_opp; home_pitch;
  for rr in range(CR7.shape[0]):

    # --- Get the current match
    rr_match = Player_events[Player_events.match_id == CR7.loc[rr,'id_match']]

    # --- goal & goals_in_match
    if any(rr_match.type == 'goal'):
      CR7.loc[rr,'goal'] = 1
      CR7.loc[rr,'goals_in_match'] = rr_match[rr_match['type'] == 'goal'].shape[0]


    # --- id_team & name_team
    CR7.loc[rr,'id_team'] = int(Player_lineup.team_id[Player_lineup.match_id == CR7.loc[rr,'id_match']].values[0])

    # --- id_opp & name_opp
    ids_teams_rr = pd.unique(lineup.team_id[lineup.match_id == CR7.loc[rr,'id_match']].values)
    CR7.loc[rr,'id_opp'] = ids_teams_rr[ids_teams_rr != CR7.loc[rr,'id_team']]


    # --- home_pitch
    if CR7.loc[rr,'id_team'] == ids_teams_rr[0]:
      CR7.loc[rr,'home_pitch'] = 1

    
    # --- name_team & name_opp
    if CR7.loc[rr,'home_pitch'] == 1:
      CR7.loc[rr,'name_team'] = games.home_team[games.match_id == CR7.loc[rr,'id_match']].values[0]
      CR7.loc[rr,'name_opp'] = games.away_team[games.match_id == CR7.loc[rr,'id_match']].values[0]
    else:
      CR7.loc[rr,'name_opp'] = games.home_team[games.match_id == CR7.loc[rr,'id_match']].values[0]
      CR7.loc[rr,'name_team'] = games.away_team[games.match_id == CR7.loc[rr,'id_match']].values[0]



  # --- Attach Player-goals by season
  for ss in seasons:

    idx_ss = np.where(CR7.season == ss)[0]
    player_goals = 0

    for ii in idx_ss:

      # --- Get the current match
      ii_match = Player_events[Player_events.match_id == CR7.loc[ii,'id_match']]

      # --- goalsscored_cum_player ---> Add the cumulative number of goals that the player had BEFORE playing the current match-day!
      CR7.loc[ii,'goalsscored_cum_player'] = player_goals

      # --- Did he score in the current match?
      if any(ii_match.type == 'goal'):
        player_goals += len(np.where(ii_match.type == 'goal')[0])




  # --- Remaining Team Statistics: ---> BEWARE: The values will be those that persisted BEFORE the current match was played!

  # --- --- 'points_team', 'points_opp', 'goalsscored_cum_team':, 'goalsconceded_cum_team', 'goalsconceded_cum_opp',
  # --- --- 'goalsscored_rank_team','goalsconceded_rank_team', 'goalsconceded_rank_opp', 'goalsscored_rank_team_wo_player'


  # --- Fill by season
  for ss in seasons:

    idx_ss = np.where(CR7.season == ss)[0]

    # --- Start with the second match! --- Since we use the data that ia available BEFORE the match starts!
    for rr in idx_ss[1:]:

      # --- Filter the CURRENT match_day
      season_rr = ss
      kickoff_rr = CR7.kickoff_dt[rr][0:10]

      # ------------------------------------------------------------------------------------------------------------------------------------ #
      # --- Points
      points_df = table_dict[season_rr]
      # --- --- Filter the PREVIOUS match_day
      idx_rr1 = np.where(points_df.match_day == kickoff_rr)[0][0] - 1
      # --- --- --- goalsscored_cum_team
      CR7.loc[rr,'points_team'] = points_df.loc[idx_rr1,CR7.loc[rr,'name_team']]
      # --- --- --- goalsscored_cum_opp
      CR7.loc[rr,'points_opp'] = points_df.loc[idx_rr1,CR7.loc[rr,'name_opp']]
      # ------------------------------------------------------------------------------------------------------------------------------------ #


      # ------------------------------------------------------------------------------------------------------------------------------------ #
      # --- Goals: scored
      goals_df = goals_scored_dict[season_rr]
      # --- --- Filter the PREVIOUS match_day
      idx_rr1 = np.where(goals_df.match_day == kickoff_rr)[0][0] - 1
      # --- --- --- goalsscored_cum_team
      CR7.loc[rr,'goalsscored_cum_team'] = goals_df.loc[idx_rr1,CR7.loc[rr,'name_team']]
      # --- --- --- goalsscored_cum_opp
      CR7.loc[rr,'goalsscored_cum_opp'] = goals_df.loc[idx_rr1,CR7.loc[rr,'name_opp']]
      # ------------------------------------------------------------------------------------------------------------------------------------ #

      # ------------------------------------------------------------------------------------------------------------------------------------ #
      # --- Goals: conceded
      goals_df = goals_conceded_dict[season_rr]
      # --- --- Filter the PREVIOUS match_day
      idx_rr1 = np.where(goals_df.match_day == kickoff_rr)[0][0] - 1
      # --- --- --- goalsconceded_cum_team
      CR7.loc[rr,'goalsconceded_cum_team'] = goals_df.loc[idx_rr1,CR7.loc[rr,'name_team']]
      # --- --- --- goalsconceded_cum_opp
      CR7.loc[rr,'goalsconceded_cum_opp'] = goals_df.loc[idx_rr1,CR7.loc[rr,'name_opp']]
      # ------------------------------------------------------------------------------------------------------------------------------------ #

      # ------------------------------------------------------------------------------------------------------------------------------------ #
      # --- Goals: scored, Rank
      goals_df = goals_scored_dict[season_rr]
      # --- --- Filter the PREVIOUS match_day
      idx_rr1 = np.where(goals_df.match_day == kickoff_rr)[0][0] - 1
      # --- --- Get the cumulate goals scored by team
      goals_cum_absolute = goals_df.loc[[idx_rr1]].drop('match_day', axis=1)
      # --- --- Get the RANKING
      ranking_goals = rankdata(np.array(goals_cum_absolute.iloc[0].astype(float))).astype(int).reshape((1,goals_cum_absolute.shape[1]))
      # --- --- Transform into REVERSE ranks
      goals_cum_rank = pd.DataFrame(data=np.shape(ranking_goals)[1] - ranking_goals, columns=goals_cum_absolute.columns)
      # --- --- --- goalsscored_rank_team
      CR7.loc[rr,'goalsscored_rank_team'] = goals_cum_rank[CR7.loc[rr,'name_team']][0]
      # --- --- --- goalsscored_rank_opp
      CR7.loc[rr,'goalsscored_rank_opp'] = goals_cum_rank[CR7.loc[rr,'name_opp']][0]
      # ------------------------------------------------------------------------------------------------------------------------------------ #



      # ------------------------------------------------------------------------------------------------------------------------------------ #
      # --- Goals: scored, Rank without Player-Goals
      goals_df = goals_scored_dict[season_rr]
      # --- --- Filter the PREVIOUS match_day
      idx_rr1 = np.where(goals_df.match_day == kickoff_rr)[0][0] - 1
      # --- --- Get the cumulate goals scored by team
      goals_cum_absolute = goals_df.loc[[idx_rr1]].drop('match_day', axis=1) 

      # --- --- Subtract the Player's Scored Goals!
      goals_cum_absolute[CR7.loc[rr,'name_team']] = goals_cum_absolute[CR7.loc[rr,'name_team']] - CR7.loc[rr,'goalsscored_cum_player']

      # --- --- Get the RANKING
      ranking_goals = rankdata(np.array(goals_cum_absolute.iloc[0].astype(float))).astype(int).reshape((1,goals_cum_absolute.shape[1]))
      # --- --- Transform into REVERSE ranks
      goals_cum_rank = pd.DataFrame(data=np.shape(ranking_goals)[1] - ranking_goals, columns=goals_cum_absolute.columns)
      # --- --- --- goalsscored_rank_team
      CR7.loc[rr,'goalsscored_rank_team_wo_player'] = goals_cum_rank[CR7.loc[rr,'name_team']][0]
      # ------------------------------------------------------------------------------------------------------------------------------------ #




      # ------------------------------------------------------------------------------------------------------------------------------------ #
      # --- Goals: conceded, Rank
      goals_df = goals_conceded_dict[season_rr]
      # --- --- Filter the PREVIOUS match_day
      idx_rr1 = np.where(goals_df.match_day == kickoff_rr)[0][0] - 1
      # --- --- Get the cumulate goals scored by team
      goals_cum_absolute = goals_df.loc[[idx_rr1]].drop('match_day', axis=1)
      # --- --- Get the RANKING
      ranking_goals = rankdata(np.array(goals_cum_absolute.iloc[0].astype(float))).astype(int).reshape((1,goals_cum_absolute.shape[1]))
      # --- --- Transform into REVERSE ranks ---> NOT FOR CONCEDED GOALS
      # --- --- --- goalsconceded_rank_team
      CR7.loc[rr,'goalsconceded_rank_team'] = goals_cum_rank[CR7.loc[rr,'name_team']][0]
      # --- --- --- goalsconceded_rank_opp
      CR7.loc[rr,'goalsconceded_rank_opp'] = goals_cum_rank[CR7.loc[rr,'name_opp']][0]
      # ------------------------------------------------------------------------------------------------------------------------------------ #

        


  # --- Finally: Prepare the export
  dict_out = {'Player_df': CR7}

  return dict_out

# ===================== Create URL to download Google-Drive CSV ======================= #

def create_url(my_url):
  return 'https://drive.google.com/uc?export=download&id=' + my_url.split('/')[-2]
