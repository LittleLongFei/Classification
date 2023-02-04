

# 2023-2-3 written by H.Zhang.


from sklearn.linear_model import Perceptron


def PerceptronClassifier(x_train, x_test, y_train, y_test):
    
    # Train.
    
    # GridSearchCV will be used to choose the eest parameters.
    ppn = Perceptron()
    ppn.fit(x_train, y_train)

    # Print some informations.
    # print('optimal classifier:', grid.best_estimator_)
    # print('optimal hyperparameters:', grid.best_params_)
    # print('best score:', grid.best_score_)
    
    # Test.
    
    y_pre = ppn.predict(x_test)

    return y_pre





