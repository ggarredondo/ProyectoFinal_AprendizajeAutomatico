#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import sklearn.utils as sk
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn import metrics

import missingno as msno

#Definimos la semilla
seed = 1
np.random.seed(seed)

#Leemos los datos 
Data = pd.read_csv('datos/communities.data', sep=",", header=None)

#Eliminamos los cinco primeros atributos que como indica el problema no son predictivos
Data = Data.drop(Data.columns[[0,1,2,3,4]], axis='columns')

#Obtenemos los datos y las etiquetas por separado
Y = Data.iloc[:, Data.shape[1]-1]
X = Data.drop(Data.columns[[Data.shape[1]-1]], axis='columns')

#Dividimos el conjunto de datos en datos de entrenamiento y datos de test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)

#---------TRATAMIENTO DE DATOS PERDIDOS--------------

#Pasamos los valores perdidos de una '?' a un 'NaN para que missingno los detecte
x_train = x_train.replace("?", np.nan)

#Pintamos la matriz de datos perdidos
msno.matrix(x_train)




