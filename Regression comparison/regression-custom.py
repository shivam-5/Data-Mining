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


# Solves linear regression with gradient descent
# M - number of instances
# X - list of variables for M instances
# omega - list of parameter values
# y - list of target values for M instances
# alpha - learning rate
def gradient_descent_2(M, X, omega, y, alpha):
    for j in range(0,M):
        yhat = np.multiply(omega, X[j,:]) # Predict target value
        epsilon = y[j] - yhat # Find difference between prediction and observation
        # Adjust parameters
        omega = omega + alpha * epsilon * X[j,:] / M
    return omega


# Computes the sum of squared errors
# M - number of instances
# X - list of variables for M instances
# omega - list of parameter values
# y - list of target values for M instances
def compute_error(M, X, omega, y):
    yhat = np.dot(X, np.transpose(omega))
    error = diff.SSD(yhat, y)
    error = error / M
    return round(error, 2)


# Computes R^2 for the model
# M - number of instances
# X - list of variables for M instances
# omega - list of parameter values (size of 2)
# y - list of target values for M instances
def compute_r2(M, X, omega, y):
    yhat = np.dot(X, np.transpose(omega))
    u = diff.SSD(yhat,y)
    ymean = np.zeros(y.shape)
    ymean[:] = np.mean(y)
    v = diff.SSD(y,ymean)
    R2 = 1 - (u / v)
    return round(R2,2)

# Adds subplot at given index with label and color
def addSubplot(X, y, line, idx, label, dotColor):
    plt.subplot(3, 3, idx)
    plt.xlabel(label)
    plt.scatter(X, y, color=dotColor)
    plt.plot(X, line)
   
# Train with manual gradient descent given epoch and training set
def train(X_train, y_train, epoch, alpha):
    # Initialize omega and adjust alpha
    omega = [0,0]
    M = X_train.shape[0]
    idx = 1
    
    # Add bias term in feature vector
    X_train_bias = np.c_[np.ones(M), X_train]
    for i in range(0,epoch):
        omega = gradient_descent_2(M, X_train_bias, omega, y_train, alpha)
        
        # Plot on every 1/5th of total iterations
        if i % (int(epoch / 5)) == 0 and i != 0:
            error = compute_error(M, X_train_bias, omega, y_train)
            R2 = compute_r2(M, X_train_bias, omega, y_train)
            
            # Plotting instructions
            label = "Iteration: " + str(idx) + "\n Error: " + str(error) + "\n R2: " + str(R2)
            line = omega[1] * X_train + omega[0]
            addSubplot(X_train, y_train, line, idx, label, 'blue')
            idx = idx + 1
            
    return omega


# Predict the outputs with the omega resulting from train on test set
def test(X_test, y_test, omega):
    M = X_test.shape[0]
    X_test_bias = np.c_[np.ones(M), X_test]
    
    error = compute_error(M, X_test_bias, omega, y_test)
    R2 = compute_r2(M, X_test_bias, omega, y_test)
    
    # Plotting instructions
    label = "Iteration: test \n Error: " + str(error) + "\n R2: " + str(R2)
    line = omega[1] * X_test + omega[0]
    addSubplot(X_test, y_test, line, 6, label, 'red')


# London Borough dataset
epoch = 101
alpha = 0.00001
omega = train(X_train, y_train, epoch, alpha)
test(X_test, y_test, omega)

plt.subplots_adjust(hspace=2, wspace=.5)
plt.show()

# Synthetic dataset
epoch = 10001
alpha = 0.003
omega = train(X_syn_train, y_syn_train, epoch, alpha)
test(X_syn_test, y_syn_test, omega)

plt.subplots_adjust(hspace=2, wspace=.5)
plt.show()
plt.close("all")