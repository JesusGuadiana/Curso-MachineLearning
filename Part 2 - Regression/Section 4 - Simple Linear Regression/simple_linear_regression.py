# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:52:15 2018

@author: Jesus
"""

#Regression Linear Simple


#Importacion de librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importar la data
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#Separar la data en training set y test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)

#Fit de Regresion Lineal Simple al Training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predecir el resultado del set de Tests
y_pred = regressor.predict(X_test)

#Visualizar el resultado

#Ver los resultados del con el set que se entreno el modelo
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train,regressor.predict(X_train), color = "black")
plt.title("Salario vs Experiencias (TRAINING SET)")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.show()

#Ver los resultados del set de prueba
plt.scatter(X_test, y_test, color = "Yellow")
plt.plot(X_train, regressor.predict(X_train), color ="Blue")
plt.title("Salario vs Experinecias (Test Set)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Salario")
plt.show()

