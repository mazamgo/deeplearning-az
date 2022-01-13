#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:38:56 2019
@author: juangabriel
"""
# Redes Neuronales Artificales
# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# Parte 1 - Pre procesado de datos
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

#Seleccionar las Variables Independiente y dependientes.
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#------------------------------------------------------------------------------
# Codificar datos categóricos que conviene recuperar
# Las variables categoricas no vienen representada en numero sino por una palabra o string.
# entonces necesitan ser codificadas por ejem. el Pais,Genero.
# Codigo para codificar las variables categoricas a dami, se trata de traducir todo
# La variables Dami es una columna para cada una de la categorias en el interior un cero o uno respectivamente.
# se trata de convertir todo a valores numericos.
#------------------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#El OneHotEncoder en las nuevas versiones está OBSOLETO
#onehotencoder = OneHotEncoder(categorical_features=[1])
#X = onehotencoder.fit_transform(X).toarray()

#---------------------------------------------------------------------------------------------------
# Ahora deberia de crearce las columnas ficticias (Variables Dami) que separen en uno o un cero respectivamente
# para la primera columna deberia de transformace en la columan que tendra los cero, la columna que tendra los 
# unos y en la columna que tendra 2 y para evitar la trampa de las variables ficticia, hay que colocar una 
# columnas para cada una de las categorias excepto para una de ellas para evitar la multilinealidad.
# la funcion en OneHotEncoder se va encargar de ella
#-----------------------------------------------------------------------------------------------------

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("Churn_Modelling",        # Un nombre de la transformación
         OneHotEncoder(categories='auto'), # La clase a la que transformar
         [1]            # Las columnas a transformar.
         )
    ], remainder='passthrough'
)

X = transformer.fit_transform(X)
#Quitar la primera columna de los Franceses.
X = X[:, 1:]

#------------------------------------------------------------------------------------------------
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
# 0.2 para tener 8,000 observaciones para crear la red neuronal y las 2000 restante para validar.
# asi se tiene mas elemento para la fase de entrenamiento
#-------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#------------------------------------------------------------------------------------------------
# Escalado de variables obligatorio esto no permite que ningua domine sobre el resto.
# si no hace habra mucha confusion debito a que la red neuronal hace suma y productos
# debido a que hay variables que destacan por encima del resto de variables (0,1,80,100000,etc)
# cada uno esta en una escala diferente, esto jhace que el calculo sea mas preciso.
# las var. quedan normalizadas en el rango cercano a cero.
#-------------------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Parte 2 - Construir la RNA para incializar la red neuronal

# Importar Keras y librerías adicionales Video # 32
#-------------------------------------------------------------------------------------------------------------------
# Libreria Dense es para Crear las capas e inicializar los pesos w con numeros pequeños.
# La cantidad de nodos que tendra la casa entrada coincide con el numero de variables independiente
# que resulta ser 11 nodos de entrada en la capa primera.
# luego hay que establecer los nodos de activacion, las funciones de activacion:
# funcion de activacion escalon, sigmoide, Rectificador lineal unitario, Tangente hiperbolica,
# la funcion sigmoide nos da una probalidad, la funcion escalon es mas estricta quien deja y quien no deja el banco
# si la variable dependiente es binaria (1,0) se puede utilizar la funcion sigmoide o escalon
# en cambio el el fn de rectificador lineal es bantante interesante para activar las capas intermedias y para la capa
# final la funcion sigmoide para conocer la probalidad de que un cliente deja o no el banco, esta fn mas interesante.
#--------------------------------------------------------------------------------------------------------------------
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout

# Inicializar la RNA, el objeto a crear para la futura red Neuronal.
classifier = Sequential()

#-----------------------------------------------------------------------------------------------------------------------------------------
# Añadir las capas de entrada y para la primera capa oculta tenemos 11 variables, vamos agregar la primera capa .add, 
# units es el numero de nodos que queremos agregar a la capa oculta (salida) o numero de nodo de entrada para la siguiente capa, 
# entonces Dense es la zona de la signasis donde hay q especificarle el
# tamaño de E/S, Dense es para la conexion entre capas. y cuantos nodos va ha tener la capa oculta una regla puede ser la media de los nodos E/S
# 11 nodos para la capa entrada y un nodo para la capa de salida, entonces la media entre 11 y 1 es 6. 
# el kernel_initializer es para inicializar los pesos estos se deben de asignar de manera aleatoria, la fn uniform es para inicializarlo
# manera uniforme pequeños cercanos a cero. 
# para la fn de activacion vamos a utilizar relu es rectificador lineal unitario pero se pudo utilizar fn sigmoide, etc.
# y para los nodos de entrada input_dim es la dimension de entrada. 
#------------------------------------------------------------------------------------------------------------------------------------------

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu", input_dim = 11))
classifier.add(Dropout(p = 0.1))

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
classifier.add(Dropout(p = 0.1))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)


# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print((cm[0][0]+cm[1][1])/cm.sum())

## Parte 4 - Evaluar, mejorar y Ajustar la RNA

### Evaluar la **RNA**
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
  classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
  classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1, verbose = 1)

mean = accuracies.mean()
variance = accuracies.std()

### Mejorar la RNA
#### Regularización de Dropout para evitar el *overfitting*

### Ajustar la *RNA*
from sklearn.model_selection import GridSearchCV # sklearn.grid_search

def build_classifier(optimizer):
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu", input_dim = 11))
  classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
  classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))
  classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {
    'batch_size' : [25,32],
    'nb_epoch' : [100, 500], 
    'optimizer' : ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
