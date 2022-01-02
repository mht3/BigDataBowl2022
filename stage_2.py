import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from player_dictionary import player_dict, weight_dict, height_dict
from sklearn.preprocessing import LabelEncoder


import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter

player_dependent_scores = pd.DataFrame()
VERBOSE = 0

def plot_confusion_matrix(class_names, y_pred, y_test, title="Confusion Matrix"):
    """
    Function to compute a Confusion Matrix and plot a heatmap based on the matrix.
    input: class names, y-predicted, y-test (ground-truth)
    output: None
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap = sns.heatmap(cm, fmt='g', cmap='Blues', annot=True, ax=ax)
    ax.set_ylabel('True class',size=12)
    ax.set_xlabel('Predicted class',size=12)
    ax.set_title(title)
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    fig.savefig('ffnn_results/stage_2_confusion_matrix_5_fold.png')
    
# Preprocessing (splitting into training/testing and standardizing) the data
def preprocess(dataset):
    data_cols = ["Home", "kickLength", "quarter", "scoreDifference", "secondsRemain", "average_temperature", "real_WindSpeed", "condition"]
    cols_to_scale = ["kickLength", "scoreDifference", "secondsRemain", "average_temperature", "real_WindSpeed"]
    # Kick length column contains some Nan values when a kick is blocked.
    # Remove these rows as they do not reflect a level of clutchness.
    dataset = dataset.dropna()

    # Encode data to represent teams as integers 1-32
    le = LabelEncoder()
    dataset["possessionTeam"] = le.fit_transform(dataset["possessionTeam"].astype(str))
    dataset["kickerId"] = le.fit_transform(dataset["kickerId"].astype(str))

    # Getting our labels and data
    labels = dataset["Result"]
    data = dataset[data_cols]

    # Split into training and test sets with k fold validation
    skf = StratifiedKFold(n_splits=5, random_state=12, shuffle=True)
    kfold_sets = []
    for train_index, test_index in skf.split(data, labels):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
        y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
        # Normalizing the data
        scaler = StandardScaler()
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
        X_train = pd.DataFrame(X_train, columns=data_cols)
        X_test = pd.DataFrame(X_test, columns=data_cols)

        kfold_sets.append((X_train, X_test, y_train, y_test))

    return kfold_sets


# Finding class distributions
def peek_majority_prediction(dataset):
    # Baseline Accuracy is 83.5 %. We need to do better than this!
    dist = Counter(dataset['Result'])
    return dist[1]/(dist[0] + dist[1])

# Building the model
def build_model(num_features):
    model = tf.keras.Sequential(name='FieldGoalClassifier')
    model.add(layers.InputLayer(input_shape=(num_features,)))
    # Hidden Layers
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Need a precision recall curve plot. Focuses on accuracy for the minority class
    fp = tf.keras.metrics.FalsePositives(name='fp')
    tn = tf.keras.metrics.TrueNegatives(name='tn')
    sas = tf.keras.metrics.SensitivityAtSpecificity(0.3, name='sas')    
    auc = tf.keras.metrics.AUC(curve='ROC', name='auc')
    metrics = [auc, fp, tn, sas]
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=metrics)
    return model

# Training the model
def train_model(X_train, y_train, model, epochs, batch_size):
    class_weights = {0: 1.7, 1: 1}
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=12, stratify=y_train)
    fitter = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight = class_weights,
                        validation_data=(X_val, y_val), verbose=0)
    return fitter

# plotting categorical and validation accuracy over epochs
def plot_metrics(history):
    fig = plt.figure(figsize=(8,8))
    fig.suptitle("Model Metrics", size=18)
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(history['loss'])
    ax1.plot(history['val_loss'])
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(['train', 'validation'], loc='upper left')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history['auc'])
    ax2.plot(history['val_auc'])
    ax2.set_title('ROC AUC Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.legend(['train', 'validation'], loc='upper left')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(history['sas'])
    ax3.plot(history['val_sas'])
    ax3.set_title('Recall at FPR >= 0.4')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Recall')
    ax3.legend(['train', 'validation'], loc='upper left')

    # false positive rate
    ax4 = fig.add_subplot(2, 2, 4)
    tnr = np.divide(history['tn'], np.add(history['tn'], history['fp']))
    val_tnr = np.divide(history['val_tn'], np.add(history['val_tn'], history['val_fp']))
    ax4.plot(tnr)
    ax4.plot(val_tnr)
    ax4.set_title('Specificity vs Epoch')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('TNR')
    ax4.legend(['train', 'validation'], loc='upper left')

    fig.tight_layout()
    fig.savefig('ffnn_results/stage_2_metrics.png')

def get_clutch_scores(dataset, rows, scores, player_dict=player_dict):
    '''
    Gets the player dependent scores at each kick distance category
    
    Parameters
    ---
    dataset         :       the original dataset used
    rows            :       row numbers used from the dataframe (used to link scores to original dataframe)
    scores          :       scores predicted by our stage 2 model
    player_dict     :       dictionary mapping kicker id to player name
    '''

    final = dataset.filter(items=["kickerId", "kickLength", "Result"])
    final = final.loc[rows,:]
    final["Score"] = scores

    final['Name']= final['kickerId'].map(player_dict)
    categories = ["< 20", "20-29", "30-39", "40-49", "50+"]
    final["Kick Category"] = pd.cut(final["kickLength"], bins=[0,19,29,39,49,75], labels=categories)

    player_groupby = final.groupby(['Kick Category', 'Name'])['Score']

    player_scores = player_groupby.mean()
    player_stds = player_groupby.std()
    player_medians = player_groupby.median()
    player_scores =player_scores.reset_index()

    player_medians =player_medians.reset_index()
    player_stds =player_stds.reset_index()
    player_scores["Median"] = player_medians["Score"]
    player_scores["Std"] = player_stds["Score"]
    return player_scores


def main():
    global player_dependent_scores, VERBOSE
    filename = 'data/preprocessed/kicker_data.csv'
    dataset = pd.read_csv(filename)
    dataset = dataset.drop(["Unnamed: 0", "gameId", "playId", "StadiumName","RoofType","TimeMeasure","TimeStartGame","TimeEndGame"], axis=1)
    dataset['height']= dataset['kickerId'].map(height_dict)
    dataset['weight']= dataset['kickerId'].map(weight_dict)
    # Removing kickers with < 25 kicks
    minimum_kicks = 25
    kicker_instances = dataset.groupby("kickerId").size()
    kickers_to_remove = kicker_instances.loc[kicker_instances < minimum_kicks].reset_index()["kickerId"].tolist()
    # Physically removes all instances of kickers with < 10 kicks
    dataset = dataset[~dataset['kickerId'].isin(kickers_to_remove)]

    folds = preprocess(dataset)
    baseline_accuracy = peek_majority_prediction(dataset)
    if VERBOSE:
        print("------------------------------")
        print("Baseline Accuracy: {:0.3f}".format(baseline_accuracy))
        print("------------------------------")
        print()

    # storage for information from all 5 folds
    fold_num = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    losses = []
    kfold_probabilities = []
    kfold_y_preds = []
    kfold_rows = []
    kfold_y_tests = []
    for fold in folds:
        if VERBOSE:
            print("Fold ", fold_num)
        
        X_train, X_test, y_train, y_test = fold

        # Building the Neural Net
        num_features = X_train.shape[1]
        batch_size = 8
        epochs = 22

        model = build_model(num_features=num_features)
        fitter = train_model(X_train, y_train, model=model, epochs=epochs,batch_size=batch_size)
        loss, auc, fp, tn, recall_sas = model.evaluate(X_test, y_test,verbose=0)
       
        probabilities = model.predict(X_test)
        y_pred = np.round(probabilities)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        if VERBOSE: 
            print("Model Accuracy: {:0.4f}".format(accuracy))
            print("Recall: {:0.4f}".format(recall))
            print("Precision: {:0.4f}".format(precision))
            print("Loss: {:0.4f}".format(loss))

        # Plot the metrics from the last fold to see general trends
        if (fold_num == 1):
            plot_metrics(fitter.history)

        kfold_rows = kfold_rows + list(X_test.index)
        kfold_probabilities = kfold_probabilities + [p[0] for p in probabilities]
        kfold_y_tests = kfold_y_tests + list(y_test)
        kfold_y_preds = kfold_y_preds + list(y_pred)
        accuracy_scores.append(accuracy)
        recall_scores.append(recall)
        precision_scores.append(precision)
        losses.append(loss)
        fold_num += 1

        if VERBOSE:
            print("DONE")
            print("------------------------------")
    
    class_names = ["Miss", "Make"]
    plot_confusion_matrix(class_names, kfold_y_preds, kfold_y_tests, title="Stage 2 Confusion Matrix")
    avg_acc = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_loss = np.mean(losses)
    if VERBOSE:
        print("Model Averages: ")
        print("Accuracy: {:0.4f}\nPrecision: {:0.4f}\nRecall: {:0.4f}\nLoss: {:0.4f}".format(avg_acc, avg_precision, avg_recall,avg_loss))

    player_dependent_scores = get_clutch_scores(dataset, rows=kfold_rows, scores=kfold_probabilities)


# Run the main method
main()
