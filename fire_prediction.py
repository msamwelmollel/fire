# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:54:43 2020

@author: Team Mandela 
"""
# Use some functions from tensorflow_docs


from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import interactive
interactive(True)

import numpy as np
import pandas as pd
import seaborn as sns


from sklearn import preprocessing


import tensorflow as tf



#%%

from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

#%%

print(tf.__version__)

path = "forestfires.csv"

dataset = pd.read_csv(path,sep=',')

#print(dataset.head())
dataset = dataset.drop(["month","day","X","Y","FFMC","DMC","DC","ISI"], axis=1)
#print(dataset.head())

dataset.isna().sum()


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


#sns.pairplot(train_dataset[["X", "Y", "FFMC", "DMC","ISI","temp","RH","wind","rain","area"]], diag_kind="kde")

#%%
train_labels = train_dataset.pop('area')
train_labels = np.log((train_labels + 1)) 
test_labels = test_dataset.pop('area')
test_labels= np.log((test_labels + 1))  # purposely to remove the skewness of the labe;

#print(test_labels)

# Normalize the data
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(train_dataset)
X_train = pd.DataFrame(X_train_maxabs, index=range(X_train_maxabs.shape[0]),
                          columns=range(X_train_maxabs.shape[1]))
#print((X_train_maxabs[:2]))


X_test_maxabs = max_abs_scaler.fit_transform(test_dataset)


X_test = pd.DataFrame(X_test_maxabs, index=range(X_test_maxabs.shape[0]),
                          columns=range(X_test_maxabs.shape[1]))

#%%
#build the model 1
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss=root_mean_squared_error,
                optimizer=optimizer,
                metrics=['mae', 'mse',tf.keras.metrics.RootMeanSquaredError(name='rmse')])
  return model

import keras.backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 



#def build_model():
#  model = keras.Sequential([
#    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
#    layers.Dense(128, activation='relu'),
#    layers.Dense(64, activation='relu'),
#    layers.Dense(1)
#  ])
#
#  optimizer = tf.keras.optimizers.RMSprop(0.001)
#
#  model.compile(loss='mse',
#                optimizer=optimizer,
#                metrics=['mae', 'mse'])
#  return model
    
model = build_model()
model.summary()


# test the NN

example_batch = X_train[:10]
example_result = model.predict(example_batch)
print(example_result)

##train the model
EPOCHS = 1000
print((X_train.shape))
print((train_labels.shape))
print(type(train_dataset))
print(type(X_train_maxabs))
labels= train_labels.to_numpy()
print(type(labels))


history = model.fit(X_train_maxabs, labels,epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])


#%%
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [area]')
plt.show()

plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 20])
plt.ylabel('MSE [area^2]')
plt.show()

plotter.plot({'Basic': history}, metric = "rmse")
plt.ylim([0, 20])
plt.ylabel('RMSE [area]')
plt.show()



#%% implenting early stop

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(X_train_maxabs, labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=1, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

# plotter.plot({'Early Stopping': early_history}, metric = "mae")
# plt.ylim([0, 10])
# plt.ylabel('MAE [area]')

#%% evaluating test set
test_labels= test_labels.to_numpy()

print(X_test_maxabs.shape)
print(test_labels.shape)

loss, mae, mse, rmse = model.evaluate(X_test_maxabs, test_labels, verbose=1)
print(mae)
print(mse)
print(rmse)

print("Testing set Mean Abs Error: {:5.2f} area".format(mae))





#%%   SVM

from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)





