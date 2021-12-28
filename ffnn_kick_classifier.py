import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from player_dictionary import player_dict
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter
import player_dictionary


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
    fig.savefig('ffnn_results/confusion_matrix.png')
    
# Preprocessing (splitting into training/testing and standardizing) the data
def preprocess(dataset):
    dataset = dataset.drop(["Unnamed: 0", "gameId", "playId"], axis=1)

    data_cols = ["possessionTeam", "Home", "kickLength", "scoreDifference", "secondsRemain"]


    # Encode data to represent teams as integers 1-32
    le = LabelEncoder()
    dataset.loc[:,["possessionTeam"]] = le.fit_transform(dataset["possessionTeam"].astype(str))
    # dataset.loc[:,["kickerId"]] = le.fit_transform(dataset["kickerId"].astype(str))

    # Kick length column contains some Nan values when a kick is blocked.
    # Remove these rows as they do not reflect a level of clutchness.
    dataset = dataset.dropna()

    # Getting our labels and data
    labels = dataset["Result"]
    data = dataset[data_cols]

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,random_state=10, stratify=labels)
    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
    # Normalizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=data_cols)
    X_test = pd.DataFrame(X_test, columns=data_cols)
    return X_train, X_test, y_train, y_test

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
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(8,activation='relu'))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Need a precision recall curve plot. Focuses on accuracy for the minority class
    accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    prc = tf.keras.metrics.AUC(name='prc', curve='PR')
    metrics = [accuracy, prc, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return model

# Training the model
def train_model(X_train, y_train, model, epochs, batch_size):
    # stop = tf.keras.callbacks.EarlyStopping(monitor='acc', verbose=1, patience=10, mode='max', restore_best_weights=True)
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_prc', mode='max', verbose=0,patience=10)
    # class_weight ={ 0: 1.5, 1: 1}
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=10, stratify=y_train)
    fitter = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[stop],
                        validation_data=(X_val, y_val), verbose=1)
    return fitter

# plotting categorical and validation accuracy over epochs
def plot_accuracy_loss(history):
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
    ax2.plot(history['prc'])
    ax2.plot(history['val_prc'])
    ax2.set_title('PRC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PRC')
    ax2.legend(['train', 'validation'], loc='upper left')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(history['precision'])
    ax3.plot(history['val_precision'])
    ax3.set_title('Precision')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.legend(['train', 'validation'], loc='upper left')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(history['recall'])
    ax4.plot(history['val_recall'])
    ax4.set_title('Recall')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.legend(['train', 'validation'], loc='upper left')

    fig.tight_layout()
    fig.savefig('ffnn_results/loss_acc_plots.png')


def main():
    dataset = pd.read_csv('plays/plays_regularTime_tidy.csv')
    X_train, X_test, y_train, y_test = preprocess(dataset)
    baseline_accuracy = peek_majority_prediction(dataset)

    # Building the Neural Net
    num_features = X_train.shape[1]
    batch_size = 32
    epochs = 64

    model = build_model(num_features=num_features)
    fitter = train_model(X_train, y_train, model=model, epochs=epochs,batch_size=batch_size)
    loss, accuracy, prc, precision, recall = model.evaluate(X_test, y_test,verbose=0)

    print("------------------------------")
    print("Baseline Accuracy: {:0.3f}".format(baseline_accuracy))
    print("------------------------------")
    print()
    y_pred = np.round(model.predict(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy: {:0.4f}".format(accuracy))
    print("Loss: {:0.4f}".format(loss))
    print("------------------------------")


    plot_accuracy_loss(fitter.history)
    class_names = ["Miss", "Make"]
    
    plot_confusion_matrix(class_names, y_pred, y_test, title="Field Goal Confusion Matrix")

if __name__ == '__main__':
    main()