#!/usr/bin/env python
# coding: utf-8

from sklearn.ensemble import VotingRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


# DATA SCALING AND TRAIN TEST SPLITTING #

# Load pre-processed data
data = pd.read_csv('data/pre-processed/preprocessed_dataframe.csv')

# Split in test and train data
X = data.drop(["Enedc (g/km)"], axis=1)
y = data["Enedc (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print('X Train: {}'.format(X_train.shape))
print('Y Train: {}'.format(y_train.shape))
print('X Test: {}'.format(X_test.shape))
print('Y Test: {}'.format(y_test.shape))

# Scaling input variables
to_scale = ['Mk', 'm (kg)', 'W (mm)', 'At1 (mm)', 'ec (cm3)', 'ep (KW)']

for feature in to_scale:
    minmax = MinMaxScaler()
    X_train[feature] = minmax.fit_transform(
        X_train[feature].to_numpy().reshape(-1, 1))
    X_test[feature] = minmax.fit_transform(
        X_test[feature].to_numpy().reshape(-1, 1))


# LINEAR REGRESSION #

LineareRegression = LinearRegression()
LineareRegression.fit(X_train, y_train)

# Saving the model
Model_linear_regression = LineareRegression.fit(X_train, y_train)
pickle.dump(Model_linear_regression, open('Model_linear_regression.pkl', 'wb'))

# POLINOMIAL REGRESSION #

polyModel = PolynomialFeatures(degree=2)
xpol_train = polyModel.fit_transform(X_train)
xpol_test = polyModel.fit_transform(X_test)
preg = polyModel.fit(xpol_train, y_train)

xpol_train.shape
xpol_test.shape

liniearModel = LinearRegression()
liniearModel.fit(xpol_train, y_train)

# Saving the model
Model_polynomial_regression = liniearModel.fit(xpol_train, y_train)
pickle.dump(Model_polynomial_regression, open(
    'Model_polynomial_regression.pkl', 'wb'))

# GRADIENT DESCENT #

sgdr = SGDRegressor(penalty=None, random_state=0)
sgdr.fit(X_train, y_train)

# Saving the model
Model_gradient_descent = sgdr.fit(X_train, y_train)
pickle.dump(Model_gradient_descent, open('Model_gradient_descent.pkl', 'wb'))

# KNN REGRESSOR #

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

# Saving the model
Model_knn = knn_model.fit(X_train, y_train)
pickle.dump(Model_knn, open('Model_knn.pkl', 'wb'))

# DECISION TREE REGRESSOR #

Tree = DecisionTreeRegressor(max_depth=None, random_state=0)
Tree.fit(X_train, y_train)

# Saving the model
Model_tree = Tree.fit(X_train, y_train)
pickle.dump(Model_tree, open('Model_tree.pkl', 'wb'))

print(f"Baumtiefe: {Tree.get_depth()}")
print(f"Knotennummer: {Tree.get_n_leaves()}")

# RANDOM FOREST REGRESSOR #

rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)

# Saving the model
Model_forest = rf.fit(X_train, y_train)
pickle.dump(Model_forest, open('Model_forest.pkl', 'wb'))


# SVM REGRESSOR  #

vector = svm.SVR()

vector.fit(X_train, y_train)

# Saving the model
Model_svm = vector.fit(X_train, y_train)
pickle.dump(Model_svm, open('Model_svm.pkl', 'wb'))

# VOTING REGRESSOR #

reg1 = LinearRegression()
reg2 = SGDRegressor(penalty=None, random_state=0)
reg3 = KNeighborsRegressor()
reg4 = DecisionTreeRegressor(random_state=0)
reg5 = RandomForestRegressor(random_state=0)
reg6 = svm.SVR()

ereg = VotingRegressor([("lr", reg1), ("gd", reg2),
                       ("knn", reg3), ("dt", reg4), ("rf", reg5), ("svm", reg6)])
ereg.fit(X_train, y_train)

# Saving the model
Model_voting = ereg.fit(X_train, y_train)
pickle.dump(Model_voting, open('Model_voting.pkl', 'wb'))
