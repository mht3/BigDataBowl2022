import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from ffnn_kick_classifier import preprocess, peek_majority_prediction 
import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter

import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#from player_dictionary import player_dict

# from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier


dataset = pd.read_csv('plays/plays_regularTime_tidy.csv')
kickerIds = dataset["kickerId"]
X_train, X_test, y_train, y_test = preprocess(dataset)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(X_train.shape[-1],) ),
      keras.layers.Dense(
          16, activation='relu'),
      keras.layers.Dense(
          16, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model


EPOCHS = 5000
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

y_train = np.array(y_train).reshape(len(y_train), )


print(y_train.shape)
# pos, neg = np.bincount( y_train.astype(int) )
total = len(y_train)
neg = 0
for i in y_train:
    if not i:
        neg += 1
pos = total - neg

initial_bias = np.log([pos/neg])


model = make_model(output_bias=initial_bias)

# results = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)

# model.load_weights(initial_weights)
baseline_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping] )


# print(pos, neg)

print("----------")

baseline_results = model.evaluate(X_test, y_test,
                                  batch_size=BATCH_SIZE, verbose=0)
print(baseline_results)



# print(X_train)
# new_X_train = pd.DataFrame()
# new_y_train = []
# counter = 0

# # print(fail_counter)

# for i in range(len(X_train)):
#     if y_train[i]:
#         counter += 1
#     if counter <= fail_counter:
#         # print(X_train.iloc[i])
#         new_X_train = new_X_train.append(X_train.iloc[i])
#         new_y_train.append(y_train[i])

# print(success_counter / len(y_train))

# baseline_accuracy = peek_majority_prediction(dataset)

# MLPlist = []
# X = [[0., 0.], [1., 1.]]
# y = [0, 1]
# print(new_X_train)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                      hidden_layer_sizes=(5, 4, 3, 2), random_state=1,  max_iter=1000)

#MLPlist.append(clf)

# print(X_train.shape)

# clf.fit(X_train, y_train)


# print(clf.score(X_test, y_test))