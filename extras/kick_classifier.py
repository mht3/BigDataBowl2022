import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from player_dictionary import player_dict
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline



def plot_confusion_matrix(class_names, y_pred, y_test, title="Confusion Matrix"):
    """
    Function to compute a Confusion Matrix and plot a heatmap based on the matrix.
    input: class names, y-predicted, y-test (ground-truth)
    output: None
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap = sns.heatmap(cm, fmt='g', cmap='Blues', annot=True, ax=ax)
    ax.set_xlabel('Predicted class',size=12)
    ax.set_ylabel('True class',size=12)
    ax.set_title(title)
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    plt.show()
    
'''
First, let's start with a dataframe containing kickers along with labels of whether or not the field goal was made.
This first algorithm is going to use a random forest classifier to determine whether or not a FGM based off of certain conditions.

This algorithm does NOT pick unique players, only an accumulation of all players. 
@TODO: Make multiple csv files of data, each csv file representing a player
'''
# Preprocessing Data
# will recieve this from a different file
dataset = pd.read_csv('plays/plays_regularTime_tidy.csv')
kickerIds = dataset["kickerId"]
dataset = dataset.drop(["Unnamed: 0", "gameId", "playId"], axis=1)
data_cols = ["possessionTeam", "Home", "kickLength", "scoreDifference", "secondsRemain"]

# Encode data to represent teams as integers 1-32
le = LabelEncoder()
dataset.loc[:,"possessionTeam"] = le.fit_transform(dataset["possessionTeam"].astype(str))

# Kick length column contains some Nan values when a kick is blocked.
# Remove these rows as they do not reflect a level of clutchness.
dataset = dataset.dropna()

# Getting our labels and data
labels = dataset["Result"]
data = dataset[data_cols]


# # Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,random_state=10, stratify=labels)
# print(X_train.head())

# SVM Comment our random forest classifier and uncomment the SVC pipeline to run.
# pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=0.001, C=8, probability=True))

# Random Forest Classifer
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, random_state=10))
pipeline.fit(X_train, y_train)
predictions = pipeline.predict_proba(X_test)
y_pred = np.argmax(predictions, axis=1)

# # Probability of making a field goal
# P_fgm = [p[1] for p in predictions]
# Probabilities of FGM
# print(P_fgm)

class_names = dataset.Result.unique()
report = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(report)) 
plot_confusion_matrix(class_names, y_pred, y_test, title="Confusion Matrix: Accuracy = {:.3f}".format(report))