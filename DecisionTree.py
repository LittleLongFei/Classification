

# 2023-2-3 written by H.Zhang.


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def DecisionTree(x_train, x_test, y_train, y_test):
    
    # Train.
    
    # GridSearchCV will be used to choose the eest parameters.
    model = DecisionTreeClassifier()
    param = {'criterion': ['gini', 'entropy'], 'max_depth': [30, 50, 60, 100], 'min_samples_leaf': [2, 3, 5, 10], 'min_impurity_decrease': [0.1, 0.2, 0.5]}
    grid = GridSearchCV(model, param_grid=param, cv=5)
    grid.fit(x_train, y_train)

    # Print some informations.
    # print('optimal classifier:', grid.best_estimator_)
    # print('optimal hyperparameters:', grid.best_params_)
    # print('best score:', grid.best_score_)
    
    # Test.
    
    model = grid.best_estimator_
    y_pre = model.predict(x_test)

    return y_pre





