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


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#print(tf.__version__)

path = "forestfires.csv"

dataset = pd.read_csv(path,sep=',')

dataset.head()

dataset.isna().sum()


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


sns.pairplot(train_dataset[["X", "Y", "FFMC", "DMC","ISI","temp","RH","wind","rain","area"]], diag_kind="kde")


