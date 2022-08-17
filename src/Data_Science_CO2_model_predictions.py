#!/usr/bin/env python
# coding: utf-8


import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


# Load test data
X_test = pd.read_csv('data/pre-processed/data_test.csv')
X_train = pd.read_csv('data/pre-processed/data_train.csv')
y_test = pd.read_csv('data/pre-processed/target_test.csv')

# LINEAR REGRESSION #

Model_linear_regression = pickle.load(open(
    'models/Model_linear_regression.pkl', 'rb'))
Model_linear_regression.predict(X_test)

# Saving predictions
y_pred_LR = Model_linear_regression.predict(X_test)
pickle.dump(y_pred_LR, open('Linear_regression_predict.pkl', 'wb'))

# Regression plot
plt.figure(figsize=(10, 10))
plt.title("Linear Regression Model")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
sns.regplot(x=y_test, y=y_pred_LR, ci=None, color='blue')

coefs = pd.DataFrame(Model_linear_regression.coef_, columns=[
                     'Coefficients'], index=X_train.columns)

coefs.plot(kind='barh', figsize=(9, 7))
plt.title('CO2 emissions Linear Regression: Feature Importance')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

# POLINOMIAL REGRESSION #

Model_polinomial_regression = pickle.load(open(
    'models/Model_polynomial_regression.pkl', 'rb'))
polyModel = PolynomialFeatures(degree=2)
xpol_test = polyModel.fit_transform(X_test)
Model_polinomial_regression.predict(xpol_test)

# Saving predictions
y_pred_PR = Model_polinomial_regression.predict(xpol_test)
pickle.dump(y_pred_PR, open('Polinomial_regression_predict.pkl', 'wb'))

# Regression plot
plt.figure(figsize=(10, 10))
plt.title("Polynomial Regression Model")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
sns.regplot(x=y_test, y=y_pred_PR, ci=None, color='blue')

# GRADIENT DESCENT #

Model_gradient_descent = pickle.load(open(
    'models/Model_gradient_descent.pkl', 'rb'))
Model_gradient_descent.predict(X_test)

# Saving predictions
y_pred_GR = Model_gradient_descent.predict(X_test)
pickle.dump(y_pred_GR, open('Gradient_descent_predict.pkl', 'wb'))

plt.figure(figsize=(10, 10))
plt.title("Gradient Descent Model")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
sns.regplot(x=y_test, y=y_pred_GR, ci=None, color='blue')
plt.show()

coefs = pd.DataFrame(Model_gradient_descent.coef_, columns=[
                     'Coefficients'], index=X_train.columns)

coefs.plot(kind='barh', figsize=(9, 7))
plt.title('CO2 emissions Gradient Descent: Feature Importance')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

# KNN REGRESSOR #

Model_knn = pickle.load(open(
    'models/Model_knn.pkl', 'rb'))
Model_knn.predict(X_test)

y_pred_KNN = Model_knn.predict(X_test)
pickle.dump(y_pred_KNN, open('Knn_predict.pkl', 'wb'))

# Regression plot
plt.figure(figsize=(10, 10))
plt.title("K-Nearest Neighbors Model")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
sns.regplot(x=y_test, y=y_pred_KNN, ci=None, color='blue')

# DECISION TREE REGRESSOR #

Model_tree = pickle.load(open(
    'models/Model_tree.pkl', 'rb'))
Model_tree.predict(X_test)

y_pred_DT = Model_tree.predict(X_test)
pickle.dump(y_pred_DT, open('Tree_predict.pkl', 'wb'))

# Regression plot
plt.figure(figsize=(10, 10))
plt.title("Decision Tree Model")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
sns.regplot(x=y_test, y=y_pred_DT, ci=None, color='blue')

coefs = pd.DataFrame(Model_tree.feature_importances_, columns=[
                     'Coefficients'], index=X_train.columns)

coefs.plot(kind='barh', figsize=(9, 7))
plt.title('CO2 emissions decision Tree: Feature Importance')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

# RANDOM FOREST REGRESSOR #

Model_forest = pickle.load(open(
    'models/Model_forest.pkl', 'rb'))
Model_forest.predict(X_test)

y_pred_RF = Model_forest.predict(X_test)
pickle.dump(y_pred_RF, open('Forest_predict.pkl', 'wb'))

coefs = pd.DataFrame(Model_forest.feature_importances_, columns=[
                     'Coefficients'], index=X_train.columns)

coefs.plot(kind='barh', figsize=(9, 7))
plt.title('CO2 emissions Random Forest: Feature Importance')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

# Regression plot
plt.figure(figsize=(10, 10))
plt.title("Random Forest Model")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
sns.regplot(x=y_test, y=y_pred_RF, ci=None, color='blue')

# SVM REGRESSOR  #

Model_svm = pickle.load(open(
    'models/Model_svm.pkl', 'rb'))
Model_svm.predict(X_test)

y_pred_SVM = Model_svm.predict(X_test)
pickle.dump(y_pred_SVM, open('SVM_predict.pkl', 'wb'))

# Regression plot
plt.figure(figsize=(10, 10))
plt.title("Support Vector Machine Model")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
sns.regplot(x=y_test, y=y_pred_SVM, ci=None, color='blue')

# VOTING REGRESSOR  #

Model_voting = pickle.load(open(
    'models/Model_voting.pkl', 'rb'))
Model_voting.predict(X_test)

y_pred_VR = Model_voting.predict(X_test)
pickle.dump(y_pred_VR, open('Voting_predict.pkl', 'wb'))

# Regression plot
plt.figure(figsize=(10, 10))
plt.title("Voting Regressor Model")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
sns.regplot(x=y_test, y=y_pred_VR, ci=None, color='blue')
