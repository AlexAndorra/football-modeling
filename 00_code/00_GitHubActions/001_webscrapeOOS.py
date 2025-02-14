# -*- coding: utf-8 -*-
"""
  Webscrape Fixtures for Out-Of-Sample Matches, i.e. those, still to come!

"""


# --- The usual Libraries
import pandas as pd
import numpy as np
import pyarrow.feather as feather
from tqdm import tqdm
import time


# --- Libraries for Webscraping
from bs4 import BeautifulSoup as bs
import requests

# --- json
import json

# --- Directory stuff
import os

# ====================================== USER INTERACTION ============================================== #

# --- Set your directory to the main folder:
directory = '.'



if 1==2:
    games_final = pd.DataFrame({'test': [0,1,2]})
    
    ll = 'premier-league'
    if not os.path.isdir(f'{directory}/10_data/100_RawData/{ll}'):
          os.mkdir(f'{directory}/10_data/100_RawData/{ll}')
    
    print(f'Exporting to directory: {directory}/10_data/100_RawData/{ll}/')
    games_final.to_csv(f'{directory}/10_data/100_RawData/{ll}/games.csv', index=False)



# --- Which leage do you want to webscarp? [format: [strings]; e.g. ['premier-league','bundesliga']]
# --- --- a sample of leagues: ['premier-league','bundesliga','2-bundesliga','la-liga','serie-a','ligue-1','a-league','champions-league']
N_leagues = ['premier-league','bundesliga','la-liga','serie-a']

# --- Which season(s) do you want to collect? [format: ['YYYY-YY']; e.g. ['2022-23','2023-24']]
N_seasons = ['2024-25']


# --- Which sequence of game days do you want to extract? [format: [integers,integers], e.g. [range(1,34),range(1,29)], i.e. a list of len(N_leagues)]
N_gamedays = [range(1,39), range(1,35),range(1,39),range(1,39)]


# ====================================== USER INTERACTION ============================================== #

"""## 1. &emsp; Extract the data"""

# ====================================== 1. Extract the Data ============================================== #

"""

The data is stored in dictionaries: each game day has its own entry. The dictionaries are as follows:

- game:           time of kick-off; home team; away team; score home full-time; score away full-time;
                  score home half-time; score away half-time
"""

# --- Some checks before starting:
assert len(N_leagues) == len(N_gamedays), 'ERROR: \'len(N_leagues)\' must equal \'len(N_games)\''



# -------------------------------------- Start the Loop -------------------------------------- #

# --- 1.1 Run across all leagues:
for ll in N_leagues:


  # --- 1.2 Run across all seaons:
  for ss in N_seasons:

    # --- Storages:
    dict_game = {}


    print(f'\nLeague: {ll} --- Season: {ss}')


    # --- Check if we're in the midst of the season, and just run over the matchdays STILL TO COME
    if os.path.exists(f'{directory}/10_data/100_RawData/{ll}/S{ss.replace("-","")[2:]}_games.csv'):
        # --- Load the file to get the latest matchday for which you have data:
        games_done = pd.read_csv(f'{directory}/10_data/100_RawData/{ll}/S{ss.replace("-","")[2:]}_games.csv')
        # --- Extract the latest matchday:
        matchday_done = int(str(games_done.iloc[-1,0]).split('GD')[1].split('_')[0])
        # --- Adjust the Number of Games to run over:
        N_gamedays[N_leagues.index(ll)] = range(matchday_done+1,N_gamedays[N_leagues.index(ll)][-1]+1)
        
            

    
    


    # --- 1.3 Run across all game days:
    for gd in tqdm(N_gamedays[N_leagues.index(ll)]):

      # --- 1.3.1 Instantiate a dataframe for the current gameday for each dictionary:
      dict_game[gd] = pd.DataFrame(columns=['match_id','kick_off','team_home','team_away',
                                            'score_home_full','score_away_full','score_home_half','score_away_half'])


      # --- 1.3.2 Build the url for the current gameday:
      url_gameday = f'https://www.kicker.de/{ll}/spieltag/{ss}/{gd}'


      # --- 1.3.3 Download the current gameday:
      response = requests.get(url_gameday)
      soup = bs(response.text, features="html.parser")
      response.close()


      # --- 1.3.4 Run across all games:
      gd_games = soup.find_all('div',{'class':['kick__v100-gameCell__team__name']})

      # --- --- Since each match requires two teams, this is a sequence of even numbers:
      gd_games_seq = list(range(0,len(soup.find_all('div',{'class':['kick__v100-gameCell__team__name']})), 2))

      for gg in gd_games_seq:


        # --- 1.3.5 Find Home team:
        team_home = str(gd_games[gg]).split('>')[1].split(' <')[0]


        # --- 1.3.6 Find Away team:
        team_away = str(gd_games[gg+1]).split('>')[1].split(' <')[0]


        # --- 1.3.7 Find time of game:
        game_idx = gd_games_seq.index(gg)
        game_kickoff = pd.to_datetime(json.loads(soup.find_all('script', {'type' : ['application/ld+json']})[game_idx+1].text)['startDate'])

          
        # --- --- Extract the string date:
        game_kickoff_date = game_kickoff.strftime('%Y-%m-%d')


        # --- 1.3.8 Create 'match_id': 'S[SEASON]_GD[GAMEDAY]_G[GAME]'
        match_id = 'S' + ss.replace('-','')[2:] + f'_GD{gd}' + f'_G{gd_games_seq.index(gg)+1}'


        # --- 1.3.9 Some metrics are just not available yet!
        score_home_full = np.nan
        score_away_full = np.nan
        score_home_half = np.nan
        score_away_half = np.nan

        # --- 1.3.10 Collect all data in the dataframe:
        dict_game[gd].loc[dict_game[gd].shape[0]+1] = [match_id, game_kickoff_date, team_home, team_away,
                                                       score_home_full, score_away_full, score_home_half, score_away_half]






    # --- 1.6 At the end of the season: some further cleaning might be necessitated: check for duplicates
    games_final = pd.concat(dict_game.values()).reset_index(drop=True)

    # --- --- --- Store the original 'match_id'
    games_IDs_orig = games_final['match_id']

    # --- --- --- Remove duplicated games:
    games_final = games_final.loc[~games_final.duplicated(['team_home','team_away'], keep='last'),:].reset_index(drop=True)

    # --- --- --- Get indices that were removed
    games_IDs_removed = games_IDs_orig[~games_IDs_orig.isin(games_final['match_id'])].values





    # --- 1.8 Export the data '/[LEAGUE]'/S[SEASON]_['games'].csv'

    # --- --- Check if directory exists:
    if not os.path.isdir(f'{directory}/10_data/100_RawData/{ll}'):
      os.mkdir(f'{directory}/10_data/100_RawData/{ll}')

    # --- --- Some cosmetics:
    ss_abreviation = ss.replace('-','')[2:]

    # --- --- Check if we're in the midst of the season, and just run over the matchdays not yet done
    #if os.path.exists(f'{directory}/10_data/100_RawData/{ll}/S{ss_abreviation}_games__OOS.csv'):
        # --- --- --- Load the already existing files and concatenate:
    #    games_existing = pd.read_csv(f'{directory}/10_data/100_RawData/{ll}/S{ss_abreviation}_games__OOS.csv')

        # --- --- --- Concatenate the existing data with the just scraped data:
    #    games_final = pd.concat([games_existing,games_final],axis=0)

      
    print(f'Exporting to directory: {directory}/10_data/100_RawData/{ll}/')
    games_final.to_csv(f'{directory}/10_data/100_RawData/{ll}/S{ss_abreviation}_games__OOS.csv', index=False)
