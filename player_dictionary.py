import pandas as pd

players_df = pd.read_csv('data/raw/players.csv')
kickers_df = players_df[players_df["Position"]=='K']
player_dict = dict(zip(kickers_df["nflId"], kickers_df["displayName"]))

