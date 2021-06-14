#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import missingno as msno
from random import uniform
from sklearn import metrics

#Definimos la semilla
seed = 1
np.random.seed(seed)
pd.set_option('display.max_rows', None)

#Leemos los datos 
Data = pd.read_csv('datos/communities.data', sep=",", header=None)

#Eliminamos los cinco primeros atributos que como indica el problema no son predictivos
Data = Data.drop(Data.columns[[0,1,2,3,4]], axis='columns')

#Obtenemos los datos y las variables objetivo por separado
Y = Data.iloc[:, Data.shape[1]-1]
X = Data.drop(Data.columns[[Data.shape[1]-1]], axis='columns')

#Dividimos el conjunto de datos en datos de entrenamiento y datos de test
#Con el parámetro shuffle nos aseguramos de que los datos se han barajado y no influye
#el orden que tuvieran al inicio en nuestra partición de train y test.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=seed, shuffle=True)

#---------TRATAMIENTO DE DATOS PERDIDOS--------------

#Pasamos los valores perdidos de una '?' a un 'NaN para que missingno los detecte
x_train = x_train.replace("?", np.nan)

#Mostramos para cada atributo los valores perdidos que tiene
print(x_train.isnull().sum())

#Pintamos la matriz de datos perdidos
msno.matrix(x_train)
plt.show()

#Borramos los atributos que superen el 20% de de datos perdidos
x_train = x_train.drop(x_train.columns[[121, 119, 118, 117, 116]], axis='columns')
x_train = x_train.drop(x_train.columns[range(96, 113)], axis='columns')

x_test = x_test.drop(x_test.columns[[121, 119, 118, 117, 116]], axis='columns')
x_test = x_test.drop(x_test.columns[range(96, 113)], axis='columns')

#Tratamos ahora con el único valor perdido que queda
mean = pd.to_numeric(x_train.iloc[:, 25].dropna()).mean()
std = pd.to_numeric(x_train.iloc[:, 25].dropna()).std()

x_train = x_train.replace(np.nan,str(mean + uniform(-1.5*std, 1.5*std)))
x_train.iloc[:,25] = x_train.iloc[:,25].astype(float)


input("\n--- Pulsar tecla para continuar ---\n")

#---------MATRIZ DE DESCRIPCIÓN DE DATOS--------------

df_out = pd.DataFrame()
df_out["Mean"] = x_train.mean(axis=0)
df_out["STD"] = x_train.std(axis = 0)
df_out["Max"] = x_train.max(axis = 0)
df_out["Min"] = x_train.min(axis = 0)
df_out["P1"] = x_train.quantile(0.25, axis = 0)
df_out["P2"] = x_train.quantile(0.5, axis = 0)
df_out["P3"] = x_train.quantile(0.75, axis = 0)

print(tabulate(df_out, headers=df_out.head(), tablefmt="github"))

input("\n--- Pulsar tecla para continuar ---\n")

#---------GRIDSEARCH--------------
 
         #GRADIENTE DESCENDENTE

#Parametros que vamos a usar en GridSearch (regularizacion)
parameters = {'eta0':[0.01,0.1], 'alpha':[0.0001,0.001]}

sgd = SGDRegressor(random_state=seed, penalty='l2', learning_rate='adaptive', max_iter=2000)

clf = GridSearchCV(sgd, parameters, verbose=3, scoring="neg_mean_absolute_error")
clf.fit(x_train, y_train)

print("Parámetros a usar para SGD: ", clf.best_params_)

input("\n--- Pulsar tecla para continuar ---\n")

         #RANDOM FOREST

#Parametros que vamos a usar en GridSearch (regularizacion)
parameters = {'n_estimators':[10, 50], 'min_samples_leaf':[5,10]}

rf = RandomForestRegressor(random_state=seed, max_features="sqrt", oob_score=True)

clf = GridSearchCV(rf, parameters, verbose=3, scoring="neg_mean_absolute_error")
clf.fit(x_train, y_train)

print("Parámetros a usar para Random Forest: ", clf.best_params_)

input("\n--- Pulsar tecla para continuar ---\n")

         #PERCEPTRON MULTICAPA

#Parametros que vamos a usar en GridSearch (regularizacion)
parameters = {'alpha':[0.0001,0.001], 'learning_rate_init':[0.001,0.01]}

mlp = MLPRegressor(random_state=seed, activation='tanh', solver='sgd', learning_rate='adaptive')

clf = GridSearchCV(mlp, parameters, verbose=3, scoring="neg_mean_absolute_error")
clf.fit(x_train, y_train)

print("Parámetros a usar para Perceptron Multicapa: ", clf.best_params_)

input("\n--- Pulsar tecla para continuar ---\n")

#---------GRIDSEARCH--------------

         #GRADIENTE DESCENDENTE

clf = SGDRegressor(penalty = 'l2', random_state=seed, learning_rate='adaptive', alpha=0.001)

scores = cross_val_score(clf, x_train, y_train, cv=5, scoring = "neg_mean_absolute_error")
 
print("SGDRegressor CV: ", abs(scores.mean()))


         #RANDOM FOREST

clf = RandomForestRegressor(random_state=seed, max_features="sqrt", oob_score=True, min_samples_leaf=5, n_estimators=50)

scores = cross_val_score(clf, x_train, y_train, cv=5, scoring = "neg_mean_absolute_error")
 
print("RandomForest CV: ", abs(scores.mean()))

         #PERCEPTRON MULTICAPA
        
clf = MLPRegressor(random_state=seed, activation='tanh', solver='sgd', learning_rate='adaptive', alpha=0.001, learning_rate_init=0.01)

scores = cross_val_score(clf, x_train, y_train, cv=5, scoring = "neg_mean_absolute_error")
 
print("MLPRegressor CV: ", abs(scores.mean()))

input("\n--- Pulsar tecla para continuar ---\n")

#-----------Etest-------------

#Volvemos a hacer el fit con los datos de entrenamiento y probamos que predice para los de test
#Para ello usamos el modelo que mejor resultado nos dió en el apartado anterior

clf = SGDRegressor(penalty = 'l2', random_state=seed, learning_rate='adaptive', alpha=0.001)

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

print("Linear Regression Etest: ", metrics.mean_absolute_error(y_test, prediction))


