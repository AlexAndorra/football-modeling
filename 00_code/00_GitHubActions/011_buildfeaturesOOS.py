# -*- coding: utf-8 -*-
"""
    Building Features for Matches still to come this season!
"""



# --- The usual Libraries
import pandas as pd
import numpy as np
import pyarrow.feather as feather
from tqdm import tqdm
from scipy.stats import rankdata
import sys

# --- Directory stuff
import os

# --- Libraries for Webscraping
from bs4 import BeautifulSoup as bs
import requests



# ====================================== USER INTERACTION ============================================== #

# --- Set your directory to the main folder:
directory = '.'

# --- Which leage do you want to webscarp? [format: [strings]; e.g. ['premier-league','bundesliga']]
# --- --- a sample of leagues: ['premier-league','bundesliga','2-bundesliga','la-liga','serie-a','ligue-1','a-league','champions-league']
N_leagues = ['premier-league','bundesliga','la-liga','serie-a']


# --- Which season(s) do you want to collect? [format: ['YYYY-YY']; e.g. ['2022-23','2023-24']]
N_seasons = ['2024-25']

# --- Which Players do you want to look at? ---> Deprecated: see list below
#my_player = ['heung-min-son','erling-haaland']


# --- How many match-days shall be discarded at the beginning of the season? (Motivation: claculate meaningful team-/player-statistics)
match_elim = 0


# ====================================== USER INTERACTION ============================================== #

"""## 0. &emsp; Load the Data & Auxiliary Functions"""

# ====================================== 0. Load the Data ============================================== #

# --- For webscraping purposes:
headers_scraping = {'User-Agent':'Safari/537.36'}


# --- Set some league abbreviations:
league_abbreviations = {'premier-league':'PL','bundesliga':'BL1','2-bundesliga':'BL2',
                        'la-liga':'LaL','serie-a':'SEA',
                        'ligue-1':'L1','a-league':'AL','champions-league':'CL'}


# --- Load the data:
games_OOS, games_raw, scorer_raw, players_raw, lineup_raw = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
for ll in N_leagues:

  for ss in N_seasons:

    # --- Extract the season-abbreviation:
    ss_abreviation = ss.replace('-','')[2:]

    ll_games_raw = pd.read_csv(f'{directory}/10_data/100_RawData/{ll}/S{ss_abreviation}_games__OOS.csv')
    # --- Attach the league name:
    ll_games_raw['name_league'] = ll
    # --- Attach the league abbreviation:
    ll_games_raw['match_id'] = league_abbreviations[ll] + '-' + ll_games_raw['match_id'].values
    games_OOS = pd.concat([games_OOS, ll_games_raw], axis=0).reset_index(drop=True)


    ll_games_raw = pd.read_csv(f'{directory}/10_data/100_RawData/{ll}/S{ss_abreviation}_games.csv')
    # --- Attach the league name:
    ll_games_raw['name_league'] = ll
    # --- Attach the league abbreviation:
    ll_games_raw['match_id'] = league_abbreviations[ll] + '-' + ll_games_raw['match_id'].values
    games_raw = pd.concat([games_raw,ll_games_raw], axis=0).reset_index(drop=True)

    ll_scorer_raw = pd.read_csv(f'{directory}/10_data/100_RawData/{ll}/S{ss_abreviation}_scorers.csv', converters = {'name_player': str, 'minute':int})
    # --- Attach the league name:
    ll_scorer_raw['name_league'] = ll
    # --- Attach the league abbreviation:
    ll_scorer_raw['match_id'] = league_abbreviations[ll] + '-' + ll_scorer_raw['match_id'].values
    scorer_raw = pd.concat([scorer_raw,ll_scorer_raw], axis=0).reset_index(drop=True)

    ll_players_raw = pd.read_csv(f'{directory}/10_data/100_RawData/{ll}/S{ss_abreviation}_players.csv', converters = {'name_player': str})
    # --- Attach the league name:
    ll_players_raw['name_league'] = ll
    players_raw = pd.concat([players_raw,ll_players_raw], axis=0).reset_index(drop=True)

    ll_lineup_raw = pd.read_csv(f'{directory}/10_data/100_RawData/{ll}/S{ss_abreviation}_lineup.csv', converters = {'name_player': str})
    # --- Attach the league name:
    ll_lineup_raw['name_league'] = ll
    # --- Attach the league abbreviation:
    ll_lineup_raw['match_id'] = league_abbreviations[ll] + '-' + ll_lineup_raw['match_id'].values
    lineup_raw = pd.concat([lineup_raw,ll_lineup_raw], axis=0).reset_index(drop=True)



# --- Set the Players:
my_player = [ # --- Portugal
              'cristiano-ronaldo','nani','ricardo-quaresma','hugo-almeida','joao-felix','helder-postiga',
              'andre-silva',
              # --- Brasil
              'ronaldo','neymar','pato','ronaldinho','diego-costa','luis-fabiano','adriano','ailton',
              'robinho','roberto-firmino', 'paulo-sergio', 'giovane-elber',
              # --- Argentinia
              'lionel-messi','gonzalo-higuain','sergio-aguero','mauro-icardi','paulo-dybala','diego-milito',
              # --- Spain
              'david-villa','fernando-torres','aritz-aduriz','raul','fernando-llorente','joseba-llorente','joselu',
              'jose-antonio-reyes','alvaro-morata','iago-aspas','nolito','adrian-lopez','jonathan-soriano','manu-del-moral',
              'michu','sergio-garcia',
              # --- Uruguay
              'luis-suarez-2','edinson-cavani','diego-forlan',
              # --- Italy
              'lorenzo-insigne','luca-toni','alberto-gilardino','alessandro-del-piero','filippo-inzaghi','ciro-immobile',
              'emanuele-giaccherini','fabio-quagliarella','antonio-di-natale','giampaolo-pazzini','marco-borriello','graziano-pelle',
              'antonio-cassano','alberto-paloschi','christian-vieri',

              # --- Germany
              'miroslav-klose','mario-gomez','oliver-neuville','carsten-jancker','alexander-zickler','lukas-podolski',
              'nils-petersen','julian-schieber', 'niclas-fuellkrug','stefan-kiessling',

              # --- England
              'wayne-rooney','harry-kane','jamie-vardy','michael-owen','bobby-zamora','emile-heskey','peter-crouch',
              'jermain-defoe','alan-shearer','david-nugent',

              # --- France
              'thierry-henry','karim-benzema','alexandre-lacazette','olivier-giroud','nicolas-anelka','hatem-ben-arfa',
              'antoine-griezmann','bafetimbi-gomis','jimmy-briand','wissam-ben-yedder',

              # --- Ireland
              'robbie-keane','leon-best',

              # --- Wales
              'gareth-bale','craig-bellamy',

              # --- Scotland
              'kenny-miller',

              # --- Poland
              'robert-lewandowski','arkadiusz-milik',

              # --- Netherlands
              'ruud-van-nistelrooy','robin-van-persie','klaas-jan-huntelaar','roy-makaay','arjen-robben',

              # --- Sweden
              'zlatan-ibrahimovic','henrik-larsson','john-guidetti','markus-rosenberg','johan-elmander','marcus-berg',

              # --- Norway
              'john-carew','erling-haaland',

              # --- Denmark
              'ebbe-sand','jon-dahl-tomasson',

              # --- Belgium
              'christian-benteke','romelu-lukaku','eden-hazard','emile-mpenza','dries-mertens',

              # --- Croatia
              'mario-mandzukic','ivica-olic','mladen-petric','davor-suker','ivan-klasnic','nikola-kalinic',

              # --- Paraguay
              'lucas-barrios',

              # --- Colombia
              'falcao','jackson','carlos-bacca',

              # --- Bulgaria
              'dimitar-berbatov',

              # --- Hungary
              'adam-szalai',

              # --- Ukraine
              'andriy-shevchenko',

              # --- Ghana
              'inaki-williams',

              # --- Ivory Coast
              'didier-drogba',

              # --- Cameroon
              'samuel-etoo','eric-maxim-choupo-moting',

              # --- Gaboon
              'pierre-emerick-aubameyang',

              # --- Egypt
              'mohamed-salah',

              # --- Slovakia
              'marek-mintal',

              # --- Korea
              'heung-min-son',

              # --- USA
              'clint-dempsey','jozy-altidore',

              # --- Mexico
              'chicharito','giovani-dos-santos','carlos-vela','raul-jimenez',

              # --- Australia
              'tim-cahill'


              ]

# =================================== 0.1 Auxiliary Functions: Prepare the Ladder ======================================= #


def get_ladder(seasons, directory='./'):

  """

  - seasons:          a 8-digit string, e.g. season Premier League 2023/24 should be in this format: 'PL-S2324'
  - directory:        path to the main folder


  """

  # ------------------------------------- Preparation ------------------------------------- #

  # --- Build a dictionary where to store the ladder for the seasons:
  dict_ladder_leagues_seasons = {}
  for ls in seasons:

    # --- Get the abbreviation of the league:
    ls_league = ls.split('-')[0]
    # --- Get the season:
    ls_season = ls.split('-')[1][1:]

    # --- Append the dictionary:
    if ls_league in list(dict_ladder_leagues_seasons.keys()):
      dict_ladder_leagues_seasons[ls_league][ls_season] = []
    else:
      dict_ladder_leagues_seasons[ls_league] = {ls_season: []}


  # ------------------------------------- Build the ladder for each League & Season ------------------------------------- #
  for ll in list(dict_ladder_leagues_seasons.keys()):

    # --- Get the name of the league:
    name_league = next(key for key, value in league_abbreviations.items() if value == ll)

    print(f'\nCurrent League: {name_league}')

    # ------------------------ Build the ladder for each season and league 'll' ------------------------ #
    for ss in tqdm(list(dict_ladder_leagues_seasons[ll].keys())):


      # --- Load the necessary data:

      games_raw = pd.read_csv(f'{directory}/10_data/100_RawData/{name_league}/S{ss}_games.csv')

      scorer_raw = pd.read_csv(f'{directory}/10_data/100_RawData/{name_league}/S{ss}_scorers.csv')



      # --- Extract the unique game days:
      N_gamedays = [int(gd.replace('GD','')) for gd in pd.unique(games_raw['match_id'].str.split('_',n=3,expand=True).iloc[:,1])]

      # --- Extract the unique teams:
      N_teams = pd.unique(games_raw['team_home'])

      # --- Storages:
      dict_ladder = {}



      # -------------------------------------- Start the Loop -------------------------------------- #


      # --- Run across all game days:
      for gd in N_gamedays:

        # --- 1.0.0 There can be some weird shit going on: e.g. in Season 2022-23, the Premier League had match ay 7 being played
        #           between January 2023 and April 2023, and not somewhere in September 2022. But we need REAL TIME data.
        #           Thus, an entire match day can be missing in the chronological order.
        gd_pos = N_gamedays.index(gd)

        # --- 1.0 Build the ladder:
        #if gd == 1:
        if gd_pos == 0:
          df_ladder = pd.DataFrame({'team':N_teams,'N_games':0,'win':0,'draw':0,'loss':0,
                                    'goals_scored':0,'goals_conceded':0,'goals_balance':0,'points':0})
        else:
          df_ladder = dict_ladder[N_gamedays[gd_pos-1]].copy()


        # --- 1.1 Build the gameday_id:
        gd_id = f'S{ss}_GD{gd}'

        # --- 1.2 Get data of current game day:
        gd_df = games_raw[games_raw['match_id'].str.startswith(f'{gd_id}_')].reset_index(drop=True)

        # --- 1.3 Fill the ladder
        for mm in range(gd_df.shape[0]):

          # --- 1.3A Check for an error during webscraping
          if not np.isnan(gd_df.loc[mm,'score_away_full']):

            # --- 1.4 Index for Home Team:
            idx_team_home = df_ladder[df_ladder['team'] == gd_df.loc[mm,'team_home']].index
            # --- 1.5 Index for Away Team:
            idx_team_away = df_ladder[df_ladder['team'] == gd_df.loc[mm,'team_away']].index

            # --- 1.6 Fill the goals: Home Team
            df_ladder.loc[idx_team_home,'goals_scored'] += gd_df.loc[mm,'score_home_full']
            df_ladder.loc[idx_team_home,'goals_conceded'] += gd_df.loc[mm,'score_away_full']

            # --- 1.7 Fill the goals: Away Team
            df_ladder.loc[idx_team_away,'goals_scored'] += gd_df.loc[mm,'score_away_full']
            df_ladder.loc[idx_team_away,'goals_conceded'] += gd_df.loc[mm,'score_home_full']


            # --- 1.8 Fill the points

            # --- --- Win: Home Team
            if (gd_df.loc[mm,'score_home_full'] > gd_df.loc[mm,'score_away_full']):
              df_ladder.loc[idx_team_home,'win'] += 1
              df_ladder.loc[idx_team_away,'loss'] += 1

              df_ladder.loc[idx_team_home,'points'] += 3

            # --- --- Win: Away Team
            elif (gd_df.loc[mm,'score_home_full'] < gd_df.loc[mm,'score_away_full']):
              df_ladder.loc[idx_team_home,'loss'] += 1
              df_ladder.loc[idx_team_away,'win'] += 1

              df_ladder.loc[idx_team_away,'points'] += 3

            # --- --- Draw
            elif (gd_df.loc[mm,'score_home_full'] == gd_df.loc[mm,'score_away_full']):
              df_ladder.loc[idx_team_home,'draw'] += 1
              df_ladder.loc[idx_team_away,'draw'] += 1

              df_ladder.loc[idx_team_home,'points'] += 1
              df_ladder.loc[idx_team_away,'points'] += 1



            # --- 1.9 Fill number of games:
            df_ladder.loc[idx_team_home,'N_games'] += 1
            df_ladder.loc[idx_team_away,'N_games'] += 1


        # --- 1.10 Fill the goal balance:
        df_ladder['goals_balance'] = df_ladder['goals_scored'] - df_ladder['goals_conceded']



        # --- 1.11 Insert the DataFrame into the dictionary
        dict_ladder[gd] = df_ladder.sort_values(['points','goals_balance'], ascending=False).reset_index(drop=True)


      # --- 1.12 Insert the season 'ss' into the storage:
      dict_ladder_leagues_seasons[ll][ss] = dict_ladder




  return dict_ladder_leagues_seasons

# ================================= 0.2 Auxiliary Functions: Prepare Player Stats ========================================== #



def get_player_stats(dict_data):

  # --- Unpack the dictionary:
  N = dict_data['Number of Games']
  Player_lineup = dict_data['Player_lineup']
  Player_info = dict_data['Player_info']
  Ladder = dict_data['Ladder']
  Games = dict_data['Games']
  Games_OOS = dict_data['Games_OOS']
  Scorers = dict_data['Scorers']
  Player_team = dict_data['Player_team']

  # --- For Compatibility:
  Scorers['season'] = [ss[1:] for ss in Scorers['match_id'].str.split('_', n=3, expand=True).loc[:, 0].str.split('-', n=2, expand=True).loc[:, 1]]

  # --- What's actually the name of the player?
  Player_name = pd.unique(Player_info['name_player'])
  assert len(Player_name) == 1, 'ERROR: you want to prepare data for more than 1 player at once!'
  Player_name = Player_name[0]


  CR7 = pd.DataFrame({'goal': np.nan*np.ones((N,)),'goals_in_match': np.nan*np.ones((N,)),
                      'goals_in_first_half': np.nan*np.ones((N,)),
                      'goals_in_second_half': np.nan*np.ones((N,)),
                      'goals_scored_avg_minutes_left': np.nan*np.ones((N,)),
                      'points_team': np.zeros((N,)), 'points_opp': np.zeros((N,)),
                      'goalsscored_inGame_team': np.nan*np.ones((N,)), 'goalsscored_inGame_opp': np.nan*np.ones((N,)),
                      'goalsscored_cum_team': np.zeros((N,)), 'goalsscored_cum_opp': np.zeros((N,)),
                      'goalsconceded_cum_team': np.zeros((N,)), 'goalsconceded_cum_opp': np.zeros((N,)),
                      'home_pitch': np.zeros((N,)), 'goalsscored_rank_team': np.zeros((N,)),
                      'goalsconceded_rank_team': np.zeros((N,)), 'goalsconceded_rank_opp': np.zeros((N,)),
                      'goalsscored_rank_team_wo_player': np.zeros((N,)), 'goalsscored_cum_player': np.zeros((N,)),
                      'id_match': Games_OOS['match_id'].values,
                      'name_team': Player_team, 'name_opp': 'NA',
                      'name_league': Player_lineup.loc[Player_lineup['name_team'] == Player_team,'name_league'].unique()[0],'id_league': np.zeros((N,)),
                      'season': np.zeros((N,)),'gameday': np.zeros((N,)), 'kick_off': Games_OOS['kick_off'].values
                    })


  # --- Fill some entries
  CR7['id_league'] = Games_OOS['match_id'].str.split('_',n=3, expand=True).loc[:,0].str.split('-',n=2, expand=True).loc[:,0]
  CR7['season'] = [ss[1:] for ss in Games_OOS['match_id'].str.split('_',n=3, expand=True).loc[:,0].str.split('-',n=2, expand=True).loc[:,1]]
  CR7['gameday'] = [int(ss[2:]) for ss in Games_OOS['match_id'].str.split('_',n=3, expand=True).loc[:,1]]


  # --- Merge some general season-specific statistics of the player
  position_pp = pd.DataFrame({'season':[f'{ss[2:4]}{ss[5:7]}' for ss in Player_info['season']],
                            'name_league':Player_info['name_league'],
                            'N_games_left':Player_info['N_games_left'],
                            'N_games_right':Player_info['N_games_right'],
                            'N_games_center':Player_info['N_games_center']})

  CR7 = pd.merge(CR7,position_pp, on=['season','name_league'], how='left')



  # --- Fill: id_team; id_opp; home_pitch;
  for rr in range(CR7.shape[0]):

    # --- Get the current match
    rr_match_id = Games_OOS.loc[Games_OOS['match_id'] == CR7.loc[rr,'id_match'],'match_id'].values[0]



    # --- name_opp
    rr_name_teams = Games_OOS.loc[Games_OOS['match_id'] == rr_match_id,['team_home','team_away']].values[0]
    CR7.loc[rr,'name_opp'] = rr_name_teams[rr_name_teams != CR7.loc[rr,'name_team']][0]


    # --- home_pitch
    if CR7.loc[rr,'name_team'] == rr_name_teams[0]:
      CR7.loc[rr,'home_pitch'] = 1



  # --- Attach cumulative Player-goals by season --- LAGGED by ONE game
  CR7['goalsscored_cum_player'] = Scorers[(Scorers['name_team'] == CR7['name_team'].unique()[0]) & (Scorers['name_player'] == Player_name) & (Scorers['season'] == CR7['season'].unique()[0])].shape[0]




  # --- Remaining Team Statistics: ---> BEWARE: The values will be those that had persisted BEFORE the current match was played!

  # --- --- 'points_team', 'points_opp', 'goalsscored_cum_team':, 'goalsconceded_cum_team', 'goalsconceded_cum_opp',
  # --- --- 'goalsscored_rank_team','goalsconceded_rank_team', 'goalsconceded_rank_opp', 'goalsscored_rank_team_wo_player'



  # --- Fill by league & season
  for ll in pd.unique(CR7['id_league']):

    for ss in pd.unique(CR7['season']):

      idx_ss = np.where((CR7['id_league'] == ll) & (CR7['season'] == ss))[0]

      # --- Run across all Remaining Matches:
      for rr in idx_ss:

        # --- Filter the CURRENT match_day
        season_rr = ss
        gameday_rr = CR7.loc[rr,'gameday']
        kickoff_rr = CR7.loc[rr,'kick_off']

        # --- --- GET the LATEST Match Day
        gameday_rr1 = np.array(list(Ladder[ll][season_rr].keys()))[-1]


        # --- Extract the ladder of the PREVIOUS match day:
        ladder_df = Ladder[ll][season_rr][gameday_rr1].copy()



        # ------------------------------------------------------------- Points ---------------------------------------------------------- #
        # --- --- --- points_team
        CR7.loc[rr,'points_team'] = ladder_df.loc[ladder_df['team'] == CR7.loc[rr,'name_team'],'points'].values[0]
        # --- --- --- points_opp
        CR7.loc[rr,'points_opp'] = ladder_df.loc[ladder_df['team'] == CR7.loc[rr,'name_opp'],'points'].values[0]
        # ------------------------------------------------------------------------------------------------------------------------------------ #



        # --------------------------------------------------------- Goals: scored ------------------------------------------------------------- #
        # --- --- --- goalsscored_cum_team
        CR7.loc[rr,'goalsscored_cum_team'] = ladder_df.loc[ladder_df['team'] == CR7.loc[rr,'name_team'],'goals_scored'].values[0]
        # --- --- --- goalsscored_cum_opp
        CR7.loc[rr,'goalsscored_cum_opp'] = ladder_df.loc[ladder_df['team'] == CR7.loc[rr,'name_opp'],'goals_scored'].values[0]
        # ------------------------------------------------------------------------------------------------------------------------------------ #

        #if (Player_name == 'paulo-dybala') & (rr == 3):
        #    print(ladder_df, gameday_rr1,idx_ss)
        #    sys.exit(-1)

        # ----------------------------------------------------------- Goals: conceded ------------------------------------------------------------ #
        # --- --- --- goalsconceded_cum_team
        CR7.loc[rr,'goalsconceded_cum_team'] = ladder_df.loc[ladder_df['team'] == CR7.loc[rr,'name_team'],'goals_conceded'].values[0]
        # --- --- --- goalsconceded_cum_opp
        CR7.loc[rr,'goalsconceded_cum_opp'] = ladder_df.loc[ladder_df['team'] == CR7.loc[rr,'name_opp'],'goals_conceded'].values[0]
        # ------------------------------------------------------------------------------------------------------------------------------------ #

        # ------------------------------------------------- Goals: scored, Rank ----------------------------------------------------- #
        # --- --- Get the RANKING
        #goals_ranking = ladder_df[['team','goals_scored']].sort_values('goals_scored', ascending=False).reset_index(drop=True)
        goals_ranking = pd.Series(ladder_df.shape[0] - rankdata(ladder_df['goals_scored'].astype(float)), index=ladder_df['team'])
        # --- --- --- goalsscored_rank_team
        #CR7.loc[rr,'goalsscored_rank_team'] = goals_ranking[goals_ranking['team'] == CR7.loc[rr,'name_team']].index[0]
        CR7.loc[rr,'goalsscored_rank_team'] = goals_ranking[CR7.loc[rr,'name_team']]
        # --- --- --- goalsscored_rank_opp
        #CR7.loc[rr,'goalsscored_rank_opp'] = goals_ranking[goals_ranking['team'] == CR7.loc[rr,'name_opp']].index[0]
        CR7.loc[rr,'goalsscored_rank_opp'] = goals_ranking[CR7.loc[rr,'name_opp']]
        # ------------------------------------------------------------------------------------------------------------------------------------ #

        # -------------------------------------  Goals: scored (without Player), Rank --------------------------------------------------------------- #
        # --- --- Subtract the Player's Goals Scored SO FAR! --> 'goalsscored_cum_player' already lagged by ONE GAME
        #idx_team = np.where(goals_ranking['team'] == CR7.loc[rr,'name_team'])[0]
        #goals_ranking.loc[idx_team,'goals_scored'] -= CR7.loc[rr,'goalsscored_cum_player']
        ladder_df_goalsscored = pd.Series(ladder_df['goals_scored'].values, index=ladder_df['team'])
        ladder_df_goalsscored[CR7.loc[rr,'name_team']] -= CR7.loc[rr,'goalsscored_cum_player']
        # --- --- RANK a new!
        #goals_ranking = goals_ranking.sort_values('goals_scored', ascending=False).reset_index(drop=True)
        goals_ranking = pd.Series(len(ladder_df_goalsscored) - rankdata(ladder_df_goalsscored.astype(float)), index=ladder_df_goalsscored.index)
        # --- --- --- goalsscored_rank_team
        #CR7.loc[rr,'goalsscored_rank_team_wo_player'] = goals_ranking[goals_ranking['team'] == CR7.loc[rr,'name_team']].index[0]
        CR7.loc[rr,'goalsscored_rank_team_wo_player'] = goals_ranking[CR7.loc[rr,'name_team']]
        # ------------------------------------------------------------------------------------------------------------------------------------ #

        # ------------------------------------------------- Goals: conceded, Rank ----------------------------------------------------- #
        # --- --- Get the RANKING ---> beware: the lower, the better --> ascending=True
        #goals_ranking = ladder_df[['team','goals_conceded']].sort_values('goals_conceded', ascending=True).reset_index(drop=True)
        goals_ranking = pd.Series(rankdata(ladder_df['goals_conceded'].astype(float)), index=ladder_df['team'])
        # --- --- --- goalsconceded_rank_team
        #CR7.loc[rr,'goalsconceded_rank_team'] = goals_ranking[goals_ranking['team'] == CR7.loc[rr,'name_team']].index[0]
        CR7.loc[rr,'goalsconceded_rank_team'] = goals_ranking[CR7.loc[rr,'name_team']]
        # --- --- --- goalsconceded_rank_opp
        #CR7.loc[rr,'goalsconceded_rank_opp'] = goals_ranking[goals_ranking['team'] == CR7.loc[rr,'name_opp']].index[0]
        CR7.loc[rr,'goalsconceded_rank_opp'] = goals_ranking[CR7.loc[rr,'name_opp']]
        # ------------------------------------------------------------------------------------------------------------------------------------ #



      # --- Finally: Prepare the export
      dict_out = {'Player_df': CR7}

  return dict_out

"""## 1. &emsp; Collect Player Statistics"""

# ====================================== 2.0 Some Preparation ============================================== #

# --- The following 'Factors' will be implemented:
factors_stats = ['goal', 'goals_in_match', 'points_team', 'points_opp',
                'goalsscored_cum_team', 'goalsscored_cum_opp', 'goalsconceded_cum_team',
                'goalsconceded_cum_opp', 'home_pitch', 'goalsscored_rank_team',
                'goalsconceded_rank_opp', 'goalsscored_rank_team_wo_player',
                'goalsscored_cum_player', 'goalsscored_rank_opp',
                'goalsconceded_rank_team', 'goalsscored_diff', 'goal_balance_team',
                'goal_balance_opp', 'goal_balance_diff', 'points_diff',
                'goalsscored_share_player_team','goalsscored_inGame_team','goalsscored_inGame_opp']

# --- Extract the players:
players_list = players_raw['name_player'].values

# --- Check which players are available from the list you requested:
my_player = [p for p in my_player if p in players_list]

# --- Check if 'pp' is still playing in one of the leagues we're looking at:
if len(my_player) == 0:
  print('\nNo Observations on any of your requested players ...')
  exit()

# =============================== Start the Data-Preparation Loop =============================== #

# --- Create a Dictionary where to store each Player's individual Data:
#MY_PLAYER = {key: [] for key in my_player}
MY_PLAYER = {}

# --- Allocate a Dictionary to store information on the Team that the Player is Currently Playing for:
PLAYERS_currentTEAM = {}



for pp in my_player:

    
  print(f'\nCurrent Player: {pp} ...')

  # ================================ Some Information on the Player ====================================== #
  pp_info = players_raw[players_raw['name_player'] == pp]


  # ================================ Which Games is the Team of the Player still going to Play? ================================ #

  # --- Which teams has 'pp' played for?
  pp_lineup = lineup_raw[lineup_raw['name_player'] == pp].reset_index(drop=True)
  # --- --- Extract the LATEST GAMEDAY:
  max_GD = np.argmax([int(g.split('_')[1][2:]) for g in pp_lineup['match_id']])
  # --- --- Get ALL Teams that 'pp' has played for this season:
  pp_teamALL = pd.unique(pp_lineup.iloc[:,2])
  # --- --- Get the Team that 'pp' has played for in the latest match:
  pp_team = pp_lineup.iloc[max_GD,2]

  # --- Check whether he is STILL playing for that team!
  # --- --- Download player specific information:
  url_player = f'https://www.kicker.de/{pp}/spieler'
  # --- --- Scrape the Player's Info:
  response_player = requests.get(url_player, headers=headers_scraping)
  soup_player = bs(response_player.text) #,features="lxml")
  response_player.close()
  # --- --- Extract the Current Team:
  pp_teamCURRENT = str(soup_player.find_all('div', {'class': 'kick__vita__header__team-name'})[0]).split('>')[2].split('<')[0]

  # --- --- Check if Team in the Database is the same as his Current Team:
  if pp_team != pp_teamCURRENT:
      # --- --- Check if he is still playing for a Team in the Top Leagues
      pp_inTOP = pp_teamCURRENT in games_OOS[['team_home','team_away']]

      if pp_inTOP:
          PLAYERS_currentTEAM[pp] = {'team_current': pp_teamCURRENT, 'team_former': pp_team}
          pp_team = pp_teamCURRENT
      else:
          PLAYERS_currentTEAM[pp] = {'team_current':pp_teamCURRENT,'team_former':pp_team}
          continue
  else:
      # --- Check if he has already played for another team this season:
      if len(pp_teamALL) > 1:
          pp_teamPREV = np.array([t for t in pp_teamALL if t != pp_team])[-1]
      else:
          pp_teamPREV = pp_team
      PLAYERS_currentTEAM[pp] = {'team_current': pp_teamCURRENT, 'team_former': pp_teamPREV}

  # --- Which Matches are still to be Played by 'pp_team'?
  pp_matches__OOS = games_OOS[(games_OOS['team_home'] == pp_team) | (games_OOS['team_away'] == pp_team)].reset_index(drop=True)

  # --- Which seasons are we looking at?
  pp_seasons = pd.unique(pp_matches__OOS['match_id'].str.split('_', n=3, expand=True).loc[:, 0])

  # ================================ Which Games Has the Player played in for his current Team? ================================ #

  pp_lineup = lineup_raw[(lineup_raw['name_player'] == pp) & (lineup_raw['name_team'] == pp_team)].reset_index(drop=True)

  # --- Attach the 'kick_off'
  pp_lineup = pd.merge(pp_lineup, games_raw[['match_id', 'kick_off']], on='match_id').sort_values('kick_off',
                                                                                                  ascending=True).reset_index(drop=True)

  # ============================ Prepare the Team Statistics (by League & Season) ================================= #

  print('\nBuilding the Ladder ...')
  # --- Get the Ladder:
  dict_ladder = get_ladder(seasons=pp_seasons, directory=directory)



  # ================ Prepare the Player's individual Data-Frame (all Seasons in one) ======================== #

  # --- Create a dictionary with all relevant data to pass to the function:
  dict_build = {'Number of Games': pp_matches__OOS.shape[0],
                'Player_lineup': pp_lineup,
                'Player_info':pp_info,
                'Player_team':pp_team,
                'Ladder': dict_ladder,
                'Scorers': scorer_raw,
                'Games': games_raw,
                'Games_OOS':pp_matches__OOS
               }

  print('\n\nBuilding the Player\'s Stats ...')
  out = get_player_stats(dict_build)

  # --- Unpack the Output:
  Player_df = out['Player_df']

  # --- For compatibility:
  data = Player_df.copy()



  # ==================================== Some Burn-In match_days per season? ==================================== #

  if match_elim > 0:

    # --- Get the index of the first 'match_elim' gamedays ---> could be done more easily I presume, but just to be sure: sorting on League and Season
    idx_elim = data.groupby(['name_league','season']).apply(lambda x: x.loc[:,'gameday'] > match_elim).values.T

    # --- Delete observations:
    data = data.loc[idx_elim,:].reset_index(drop=True)


  # ==================================== Create Additional Features ======================================= #

  # ---- Difference in goals-scored between 'team' and 'opp': diff > 0 --> higher likelihood of Player scoring (?) --> probably debatable !
  data['goalsscored_diff'] = data['goalsscored_cum_team'].astype(float) - data['goalsscored_cum_opp'].astype(float)

  # ---- Goal-Balance Team: goals-scored - goals-conceded: diff > 0 --> higher likelihood of Player scoring
  data['goal_balance_team'] = data['goalsscored_cum_team'].astype(float) - data['goalsconceded_cum_team'].astype(float)

  # ---- Goal-Balance Opponent: goals-scored - goals-conceded: diff > 0 --> not sure about infering anything about the likelihood of Player scoring
  data['goal_balance_opp'] = data['goalsscored_cum_opp'].astype(float) - data['goalsconceded_cum_opp'].astype(float)

  # ---- Difference of Goal-Balance Team vs Goal-Balance Opponent: diff > 0 --> higher likelihood of Player scoring
  data['goal_balance_diff'] = data['goal_balance_team'].astype(float) - data['goal_balance_opp'].astype(float)

  # ---- Difference in points between 'team' and 'opp': diff > 0 --> higher likelihood of Player scoring
  data['points_diff'] = data['points_team'].astype(float) - data['points_opp'].astype(float)

  # ---- Share of Player's goals within the team: higher share --> higher likelihood of Player scoring
  #      --> but isn't a high share rather an indication of a player's ability?
  data['goalsscored_share_player_team'] = data['goalsscored_cum_player'].astype(float) / data['goalsscored_cum_team'].astype(float)
  # ---- ---- Fill 'inf' with 0:
  data.loc[data['goalsscored_cum_team'].astype(float) == 0.0,'goalsscored_share_player_team'] = 0

  # ============================ THE END IS NEAR! Save your Player's Data ================================= #
  MY_PLAYER[pp] = data


  print(f'\nCongrats! Player {pp}: Done!\n\n\n')

"""## 3 &emsp; Collect the Data

Assemble the data in a single matrix called `data` .
"""




# ========================================================================================================= #
#
#           If you have specified several Players, i.e. if len(my_player) > 1,
#           you may want to row-bind the individual Data-Frames.
#
# ========================================================================================================= #

if len(my_player) > 1:

  data_true = MY_PLAYER[list(MY_PLAYER)[0]].copy()
  # --- Assign Player-Name
  data_true["name_player"] = list(MY_PLAYER)[0]

  for ll in list(MY_PLAYER)[1:]:

    help_ll = MY_PLAYER[ll].copy()
    # --- Assign Player-Name
    help_ll["name_player"] = ll

    data_true = pd.concat([data_true,help_ll], ignore_index=True)

else:

  data_true = MY_PLAYER[list(MY_PLAYER)[0]].copy()
  # --- Assign Player-Name
  data_true["name_player"] = list(MY_PLAYER)[0]



# --- Save your work!
data = data_true.copy()


# --- Make some final adjustment for the 'season'
season_format = data['season'].astype(str).str.zfill(4)
season_format_correct = [f'20{ii[:2]}/{ii[2:]}' for ii in season_format]
data['season'] = season_format_correct

# ==================== The Universe of Features/Factors to Choose from ===================== #
print(data.columns)
print(data.shape)

"""## 4 &emsp; Export the Data


"""

# ======================================== Export the data as a csv-file ==================================== #

# --- Specify the directory where you want to export the data to:
directory_data = '10_data/101_SFM'
directory_dashboard = '00_code/004_Dashboards/0043_SFM/0043_10_data'

# --- Any ID you want to add to the file name?
data_ID = 'SFM_data_byPlayer__OOS'

# --- Load the Existing Data:
if os.path.exists(f'{directory}/{directory_data}/SFM_data_byPlayer__OOS.csv'):
    data_existing = pd.read_csv(f'{directory}/{directory_data}/SFM_data_byPlayer__OOS.csv')

    # --- --- Place a hard-copy of the existing data in folder '00_vintage':
    data_existing.to_csv(f'{directory}/{directory_data}/00_vintage/SFM_data_byPlayer__OOS__{pd.to_datetime("today").strftime("%Y-%m-%d")}.csv')

# ----------------------------------- Export: ----------------------------------- #
# --- Data Folder:
data.to_csv(f'{directory}/{directory_data}/{data_ID}.csv', index=False)
pd.DataFrame.from_dict(PLAYERS_currentTEAM, orient='index').to_csv(f'{directory}/{directory_data}/{data_ID}__currentTEAM.csv', index=True)

# --- Dashboard Folder:
data.to_csv(f'{directory}/{directory_dashboard}/{data_ID}.csv', index=False)
pd.DataFrame.from_dict(PLAYERS_currentTEAM, orient='index').to_csv(f'{directory}/{directory_dashboard}/{data_ID}__currentTEAM.csv', index=True)
