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
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder


import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter

pas_kick = pd.DataFrame()
pas_score = pd.DataFrame()
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
    fig.savefig('ffnn_results/stage_1_confusion_matrix_5_fold.png')
    
# Preprocessing (splitting into training/testing and standardizing) the data
def preprocess(dataset):
    data_cols = ["Home", "kickLength", "quarter", "scoreDifference", "secondsRemain", "average_temperature", "real_WindSpeed", "condition"]
    cols_to_scale = ["kickLength", "scoreDifference", "secondsRemain", "average_temperature", "real_WindSpeed"]
    dataset['Home'] = dataset['Home'].replace({True: 1, False: 0})

    # Kick length column contains some Nan values when a kick is blocked.
    # Remove these rows as they do not reflect a level of clutchness.
    dataset = dataset.dropna()

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
    # Baseline Accuracy is 84 %. We need to do better than this!
    dist = Counter(dataset['Result'])
    return dist[1]/(dist[0] + dist[1])

# Building the model
def build_model(num_features):
    model = tf.keras.Sequential(name='FieldGoalClassifier')
    model.add(layers.InputLayer(input_shape=(num_features,)))
    # Hidden Layers
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Need a precision recall curve plot. Focuses on accuracy for the minority class
    fp = tf.keras.metrics.FalsePositives(name='fp')
    tn = tf.keras.metrics.TrueNegatives(name='tn')
    sas = tf.keras.metrics.SensitivityAtSpecificity(0.4, name='sas')    
    auc = tf.keras.metrics.AUC(curve='ROC', name='auc')
    metrics = [auc, fp, tn, sas]
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=metrics)
    return model

# Training the model
def train_model(X_train, y_train, model, epochs, batch_size):
    class_weights = {0: 1.62, 1: 1}
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
    fig.savefig('ffnn_results/stage_1_metrics.png')

def get_clutch_scores(dataset, rows, scores):
    '''
    Gets the baseline scores at each kick distance category
    
    Parameters
    ---
    dataset         :       the original dataset used
    rows            :       row numbers used from the dataframe (used to link scores to original dataframe)
    scores          :       scores predicted by our model
    '''

    final = dataset.filter(items=["kickerId", "kickLength", "scoreDifference", "Result"])
    final = final.loc[rows,:]
    final["Score"] = scores

    kick_categories = ["< 25", "25-29", "30-39", "40-49", "50+"]
    final["Kick Category"] = pd.cut(final["kickLength"], bins=[0,24,29,39,49,75], labels=kick_categories)

    player_agnostic_groupby = final.groupby(['Kick Category'])['Score']
    player_agnostic_scores = player_agnostic_groupby.mean()
    player_agnostic_stds = player_agnostic_groupby.std()
    player_agnostic_medians = player_agnostic_groupby.median()
    player_agnostic_scores =player_agnostic_scores.reset_index()
    player_agnostic_medians =player_agnostic_medians.reset_index()
    player_agnostic_stds =player_agnostic_stds.reset_index()
    player_agnostic_scores["Median"] = player_agnostic_medians["Score"]
    player_agnostic_scores["Std"] = player_agnostic_stds["Score"]

    score_categories = ["< -3 ", "-3 - 3", "> 3"]
    final["Score Category"] = pd.cut(final["scoreDifference"], bins=[-50, -4, 4, 50], labels=score_categories)

    player_agnostic_groupby_2 = final.groupby(['Score Category'])['Score']
    player_agnostic_scores_2 = player_agnostic_groupby_2.mean()
    player_agnostic_stds_2 = player_agnostic_groupby_2.std()
    player_agnostic_medians_2 = player_agnostic_groupby_2.median()
    player_agnostic_scores_2 =player_agnostic_scores_2.reset_index()
    player_agnostic_medians_2 =player_agnostic_medians_2.reset_index()
    player_agnostic_stds_2 =player_agnostic_stds_2.reset_index()
    player_agnostic_scores_2["Median"] = player_agnostic_medians_2["Score"]
    player_agnostic_scores_2["Std"] = player_agnostic_stds_2["Score"]

    return player_agnostic_scores, player_agnostic_scores_2


def main():
    global pas_kick, pas_score, VERBOSE
    filename = 'data/preprocessed/kicker_data.csv'
    dataset = pd.read_csv(filename)
    dataset = dataset.drop(["Unnamed: 0", "gameId", "playId", "StadiumName","RoofType","TimeMeasure","TimeStartGame","TimeEndGame"], axis=1)
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
    roc_auc_scores=[]
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
        batch_size = 4
        epochs = 10

        model = build_model(num_features=num_features)
        fitter = train_model(X_train, y_train, model=model, epochs=epochs,batch_size=batch_size)
        loss, auc, fp, tn, recall_sas = model.evaluate(X_test, y_test,verbose=0)

       
        probabilities = model.predict(X_test)
        y_pred = np.round(probabilities)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        if VERBOSE:
            print("Model Accuracy: {:0.4f}".format(accuracy))
            print("Recall: {:0.4f}".format(recall))
            print("Precision: {:0.4f}".format(precision))
            print("ROC-AUC: {:0.4f}".format(roc_auc))
            print("Loss: {:0.4f}".format(loss))
        

        # Plot the metrics from the last fold to see general trends
        if (fold_num == 4):
            plot_metrics(fitter.history)

        kfold_rows = kfold_rows + list(X_test.index)
        kfold_probabilities = kfold_probabilities + [p[0] for p in probabilities]
        kfold_y_tests = kfold_y_tests + list(y_test)
        kfold_y_preds = kfold_y_preds + list(y_pred)
        accuracy_scores.append(accuracy)
        roc_auc_scores.append(roc_auc)
        recall_scores.append(recall)
        precision_scores.append(precision)
        losses.append(loss)
        fold_num += 1
        if VERBOSE:
            print("DONE")
            print("------------------------------")
    
    class_names = ["Miss", "Make"]
    plot_confusion_matrix(class_names, kfold_y_preds, kfold_y_tests, title="Stage 1 Confusion Matrix")
    avg_acc = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_loss = np.mean(losses)
    avg_roc_auc = np.mean(roc_auc_scores)
    if VERBOSE:
        print("Model Averages: ")
        print("Accuracy: {:0.4f}\nPrecision: {:0.4f}\nRecall: {:0.4f}\nROC-AUC: {:0.4f}\nLoss: {:0.4f}".format(avg_acc,avg_precision, avg_recall, avg_roc_auc, avg_loss))

    pas_kick, pas_score = get_clutch_scores(dataset, rows=kfold_rows, scores=kfold_probabilities)

# Run the main method
main()
    
