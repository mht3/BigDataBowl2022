import pandas as pd

players_df = pd.read_csv('data/raw/players.csv')
kickers_df = players_df[players_df["Position"]=='K']
kickers_df["height"] = kickers_df["height"]

heights = (kickers_df.height.str.contains('-'), 'height')
kickers_df.loc[heights] = kickers_df.loc[heights].str.split('-').str[0].astype(int)*12 + \
                        kickers_df.loc[heights].str.split('-').str[1].astype(int)


kicker_info = kickers_df[["nflId", "height", "weight"]]
player_dict = dict(zip(kickers_df["nflId"], kickers_df["displayName"]))
