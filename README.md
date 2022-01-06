# BigDataBowl2022

Clutch Kicker Metric for the Kaggle [Big Data Bowl 2022](https://www.kaggle.com/c/nfl-big-data-bowl-2021) Competition.

The team's report, design, and analysis can be found [here](https://www.kaggle.com/mattt31/evaluating-kicker-performance-using-ker).

## Code Structure
```BigDataBowl2022/``` contains:

```data/```: 

- ```raw/```: NFL Big Data Bowl original data.

- ```preprocessed/```: Contains the CSV used in stages 1 and 2 containing kicker data.

```extras/```: Extra files used for testing and brainstorming that are no longer used.

```Python/```: 

- ```ffnn_results```: Plots, confusion matrices, and data for stages 1, 2, and 3.

- ```play_visualization.ipynb```: Notebook containing code to produce a gif of a field goal.

- ```player_dictionary.py```: Maps a kicker ID to a Name, height, and weight.

- ```stage_1.py```: Code for Stage 1 of the team's algorithm. Outputs baseline probabilities for FGM categorized by field goal distance and score difference..

- ```stage_2.py```: Code for Stage 2 of the team's algorithm. Outputs player dependent probabilities for FGM categorized by field goal distance and score difference..

- ```stage_3.py```: Code for Stage 3 of the team's algorithm. Outputs the raw KER values categorized by field goal distance and score difference.

```R/```: code used to preprocess raw data. Filters by field goal only plays and adds additional features such as weather data.
