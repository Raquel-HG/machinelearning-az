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

#from sklearn import preprocessing
# le_X = preprocessing.LabelEncoder()
# X[:,0] = le_X.fit_transform(X[:,0])


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()
X = np.array(ct.fit_transform(X), dtype=np.float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Para utilizar one hot encoder y crear variables dummy, ya no hace falta utilizar previamente la función label enconder, si no que para aplicar la dummyficación a la primera columna y dejar el resto de columnas como están, lo podemos hacer con:
#
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import make_column_transformer
#onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
#X = onehotencoder.fit_transform(X)


#Cambios de validación cruzada y training/testing

#La función sklearn.grid_search ha cambiado y ya no depende de ese paquete. Ahora debe cargarse con

#from sklearn.model_selection import GridSearchCV

#La función train_test_split ya no forma parte de sklearn.cross_validation, ahora debe cargarse desde el paquete sklearn.model_selection
