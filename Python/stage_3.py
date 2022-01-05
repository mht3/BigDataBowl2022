import numpy as np 
import pandas as pd

from stage_1 import pas_kick, pas_score
from stage_2 import pds_kick, pds_score

stage_1_dict_by_distance = dict(zip(pas_kick["Kick Category"], pas_kick["Score"]))
stage_2_dict_by_score = dict(zip(pas_score["Score Category"], pas_score["Score"]))

# BREAKDOWN BY KICK DISTANCE: Fills NANs with baseline values and subtracts stage 2 from stage 1
for category, score in stage_1_dict_by_distance.items():
    isna = pds_kick["Score"].isna()
    pds_kick.loc[(isna) & (pds_kick["Kick Category"] == category), "Score"] = score
    pds_kick.loc[pds_kick["Kick Category"] == category, "KER"] = pds_kick["Score"] - score

clutch_scores_by_distance = pds_kick[["Kick Category", "Name", "Score","KER"]]

clutch_scores_by_distance.to_csv("Python/ffnn_results/clutch_scores_by_distance.csv", index=False)
pas_kick.to_csv("Python/ffnn_results/baseline_by_distance.csv", index=False)

# BREAKDOWN BY SCORE DIFFERENCE: Fills NANs with baseline values and subtracts stage 2 from stage 1
for category, score in stage_2_dict_by_score.items():
    isna = pds_score["Score"].isna()
    pds_score.loc[(isna) & (pds_score["Score Category"] == category), "Score"] = score
    pds_score.loc[pds_score["Score Category"] == category, "KER"] = pds_score["Score"] - score

clutch_scores_by_score_diff = pds_score[["Score Category", "Name", "Score","KER"]]

clutch_scores_by_score_diff.to_csv("Python/ffnn_results/clutch_scores_by_score_difference.csv", index=False)
pas_score.to_csv("Python/ffnn_results/baseline_by_score_difference.csv", index=False)