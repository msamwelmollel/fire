# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:18:49 2020

@author: Team Mandela
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import interactive
interactive(True)

import numpy as np
import pandas as pd
import seaborn as sns


from sklearn import preprocessing

path = "train.csv"

train_dataset = pd.read_csv(path,sep=',')

train_dataset.head()

