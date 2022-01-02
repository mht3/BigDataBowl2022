import numpy as np 
import pandas as pd

from stage_1 import player_agnostic_scores as pas
from stage_2 import player_dependent_scores as pds

stage_1_dict = dict(zip(pas["Kick Category"], pas["Score"]))
# Fills NANs with baseline values and subtracts stage 2 from stage 1
for category, score in stage_1_dict.items():
    isna = pds["Score"].isna()
    pds.loc[(isna) & (pds["Kick Category"] == category), "Score"] = score
    pds.loc[pds["Kick Category"] == category, "ClutchScore"] = pds["Score"] - score

clutch_scores = pds[["Kick Category", "Name", "Score","ClutchScore"]]

clutch_scores.to_csv("ffnn_results/clutch_scores.csv", index=False)
pas.to_csv("ffnn_results/player_agnostic_scores.csv", index=False)