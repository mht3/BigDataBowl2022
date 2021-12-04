import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# from sklearn.preprocessing import LabelEncoder
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
dataset = pd.read_csv('data/preprocessed/example_data.csv')
# Last column in dataframe is our labels
labels = dataset.iloc[:, -1]
data = dataset.iloc[:,0:-1]

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,random_state=10)
y_train = y_train.tolist()
y_test = y_test.tolist()

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=15, max_depth=10, random_state=10))
pipeline.fit(X_train, y_train)
predictions = pipeline.predict_proba(X_test)
y_pred = np.argmax(predictions, axis=1)

# Probability of making a field goal
P_fgm = [p[1] for p in predictions]
print(P_fgm)
print(y_pred)
class_names = dataset.FGM.unique()
report = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(report)) 
plot_confusion_matrix(class_names, y_pred, y_test, title="Random Forest Confusion Matrix")