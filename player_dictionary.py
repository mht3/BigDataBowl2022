import pandas as pd

players_df = pd.read_csv('data/raw/players.csv')
kickers_df = players_df[players_df["Position"]=='K']

heights = (kickers_df.height.str.contains('-'), 'height')
kickers_df.loc[heights] = kickers_df.loc[heights].str.split('-').str[0].astype(int)*12 + \
                        kickers_df.loc[heights].str.split('-').str[1].astype(int)

player_dict = dict(zip(kickers_df["nflId"], kickers_df["displayName"]))
weight_dict = dict(zip(kickers_df["nflId"], kickers_df["weight"]))
height_dict = dict(zip(kickers_df["nflId"], kickers_df["height"]))
