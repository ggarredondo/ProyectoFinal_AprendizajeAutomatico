#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.utils as sk
import sklearn.decomposition as skde
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import missingno as msno
from random import uniform

#Definimos la semilla
seed = 1
np.random.seed(seed)
pd.set_option('display.max_rows', None)

#Leemos los datos 
Data = pd.read_csv('datos/communities.data', sep=",", header=None)

#Eliminamos los cinco primeros atributos que como indica el problema no son predictivos
Data = Data.drop(Data.columns[[0,1,2,3,4]], axis='columns')

#Obtenemos los datos y las etiquetas por separado
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

#---------ELIMINAR OUTLIERS--------------

# Utilizamos PCA para reducir la dimensionalidad de los datos a 1D y visualizar el conjunto de entrenamiento.
# Se reduce a 1D pues uno de los ejes debe ser necesariamente la etiqueta continua correspondiente, ya que
# el problema es de regresión.
pca2d = skde.PCA(n_components=1).fit_transform(x_train)
plt.figure(figsize=(16,10))
sns.scatterplot(x=pca2d[:,0], y=y_train)
plt.title("Training reducido a 2D")
plt.show()
