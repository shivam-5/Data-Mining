# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 21:33:01 2021

@author: shivam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import datasets
from Difference import Difference
from sklearn import linear_model
from sklearn import metrics

diff = Difference()

# read data from file
filename = "./data/london-borough-profiles-jan2018.csv"
data = pd.read_csv(filename, encoding='unicode_escape')
data = data.drop([0])

# Extract 70,71 columns from the whole dataframe
column1 = "Male life expectancy, (2012-14)"
column2 = "Female life expectancy, (2012-14)"
col7071 = data[[column1,column2]]

#  Remove invalid values and import data as float
col7071 = col7071[~col7071[column1].str.startswith(".")].astype('float')


# Split into train and test
X_train,X_test,y_train,y_test = model_selection.train_test_split(col7071[column1].values, col7071[column2].values, test_size=0.1)

# Plot train,test split points of the extracted columns with different colors
plt.subplot(2,2,1)
plt.xlabel(column1)
plt.scatter(X_train, y_train, color='blue')
plt.scatter(X_test, y_test, color='red')

# Make synthetic dataset of 100 instances with 1 attribute
x_syn,y_syn,p = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=30, coef=True)


# Split synthetic dataset into train and test
X_syn_train,X_syn_test,y_syn_train,y_syn_test = model_selection.train_test_split(x_syn, y_syn, test_size=0.1)

# Plot train,test split points of the synthetic dataset with different colors
plt.subplot(2,2,2)
plt.xlabel("x")
plt.scatter(X_syn_train, y_syn_train, color='blue')
plt.scatter(X_syn_test, y_syn_test, color='red')
plt.show()


# London Borough dataset
# Initialize model
lr = linear_model.LinearRegression()

# Build model by fitting parameters to training data
lr.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

# Output regression equation:
print("Scikit regression equation, y = ")
print(lr.intercept_)
print("+")
print(lr.coef_[0])
print("x")

# Measure training accuracty
y_hat = lr.predict(X_train.reshape(-1, 1))
print("r2: " + str(metrics.r2_score(y_train, y_hat)))
print("MeanSqError: " + str(metrics.mean_squared_error(y_train, y_hat)))

# Measure test accuracy
y_hat = lr.predict(X_test.reshape(-1, 1))
print("r2: " + str(metrics.r2_score(y_test, y_hat)))
print("MeanSqError: " + str(metrics.mean_squared_error(y_test, y_hat)))
    
# Synthetic dataset
# Initialize model
lr = linear_model.LinearRegression()

# Build model by fitting parameters to training data
lr.fit(X_syn_train.reshape(-1, 1), y_syn_train.reshape(-1, 1))

# Output regression equation:
print("Scikit regression equation, y = ")
print(lr.intercept_)
print("+")
print(lr.coef_[0])
print("x")

# Measure training accuracty
y_hat = lr.predict(X_syn_train.reshape(-1, 1))
print("r2: " + str(metrics.r2_score(y_syn_train, y_hat)))
print("MeanSqError: " + str(metrics.mean_squared_error(y_syn_train, y_hat)))

# Measure test accuracy
y_hat = lr.predict(X_syn_test.reshape(-1, 1))
print("r2: " + str(metrics.r2_score(y_syn_test, y_hat)))
print("MeanSqError: " + str(metrics.mean_squared_error(y_syn_test, y_hat)))