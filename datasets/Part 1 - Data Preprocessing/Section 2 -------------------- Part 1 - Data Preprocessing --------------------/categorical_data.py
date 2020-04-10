#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:19:06 2019

@author: juangabriel
"""

# Plantilla de Pre Procesado - Datos Categóricos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

##Para codificar datos categóricos ahora se utiliza:

from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
X = onehotencoder.fit_transform(X)
le_y= preprocessing.LabelEncoder()
y=le_y.fit_transform(y)
