import tensorflow as tf
from tf import keras
from tf.keras import layers
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

# data cleansing here



# training set made here
df = pd.read_csv("location of data")

X = "input labels"
y = "Output labels"
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.4)

# can optimize by removing the number of nodes in hidden layer and by improving the performance with optimizers
# Feedforward network
model = keras.Sequential()
model.add(layers.Dense(8, activation='sigmoid'))
model.add(layers.Dense(1024, activation='softmax'))
model.add(layers.Dense(8, activation = 'sigmoid'))



