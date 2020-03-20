# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:54:43 2020

@author: Team Mandela 
"""
# Use some functions from tensorflow_docs


from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import preprocessing


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#print(tf.__version__)

path = "forestfires.csv"

dataset = pd.read_csv(path,sep=',')

#print(dataset.head())
dataset = dataset.drop(["month","day"], axis=1)
print(dataset.head())

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
print(X_train_maxabs.shape)




