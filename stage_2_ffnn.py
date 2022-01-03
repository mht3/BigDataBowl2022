import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

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
    fig.savefig('ffnn_results/stage_2_confusion_matrix.png')
    
# Preprocessing (splitting into training/testing and standardizing) the data
def preprocess(dataset):
    data_cols = ["Home", "kickLength", "quarter", "scoreDifference", "secondsRemain", "average_temperature", "real_WindSpeed", "condition"]
    cols_to_scale = ["kickLength", "scoreDifference", "secondsRemain", "average_temperature", "real_WindSpeed"]
    dataset['Home'] = dataset['Home'].replace({True: 1, False: 0})

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

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,random_state=8, stratify=labels)
    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

    # Normalizing the data
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    X_train = pd.DataFrame(X_train, columns=data_cols)
    X_test = pd.DataFrame(X_test, columns=data_cols)
    return X_train, X_test, y_train, y_test

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
    model.add(layers.Dense(6, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Need a precision recall curve plot. Focuses on accuracy for the minority class
    fp = tf.keras.metrics.FalsePositives(name='fp')
    tn = tf.keras.metrics.TrueNegatives(name='tn')
    sas = tf.keras.metrics.SensitivityAtSpecificity(0.4, name='sas')    
    auc = tf.keras.metrics.AUC(curve='ROC')
    metrics = [auc, fp, tn, sas]
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=metrics)
    return model

# Training the model
def train_model(X_train, y_train, model, epochs, batch_size):
    stop = tf.keras.callbacks.EarlyStopping(monitor='auc', mode='max', verbose=0,patience=10)
    class_weights = {0: 1.5, 1: 1}
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=12, stratify=y_train)
    fitter = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight = class_weights,
                        validation_data=(X_val, y_val), callbacks=[stop], verbose=1)
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

def main():
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

    X_train, X_test, y_train, y_test = preprocess(dataset)
    baseline_accuracy = peek_majority_prediction(dataset)

    # Building the Neural Net
    num_features = X_train.shape[1]
    batch_size = 8
    epochs = 30

    model = build_model(num_features=num_features)
    fitter = train_model(X_train, y_train, model=model, epochs=epochs,batch_size=batch_size)
    loss, auc, fp, tn, recall_sas = model.evaluate(X_test, y_test,verbose=0)

    print("------------------------------")
    print("Baseline Accuracy: {:0.3f}".format(baseline_accuracy))
    print("------------------------------")
    print()
    probabilities = model.predict(X_test)
    y_pred = np.round(probabilities)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print("Model Accuracy: {:0.4f}".format(accuracy))
    print("Recall: {:0.4f}".format(recall))
    print("Precision: {:0.4f}".format(precision))
    print("Loss: {:0.4f}".format(loss))

    print("------------------------------")


    plot_metrics(fitter.history)
    class_names = ["Miss", "Make"]
    
    plot_confusion_matrix(class_names, y_pred, y_test, title="Field Goal Confusion Matrix")

if __name__ == '__main__':
    main()