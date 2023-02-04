

# 2023-2-3 written by H.Zhang.


from sklearn.neural_network import MLPClassifier


def MLP(x_train, x_test, y_train, y_test):
    
    # Train.
    
    MLP = MLPClassifier(alpha=1, max_iter=1000)

    MLP.fit(x_train, y_train)
    
    # Print some informations.
    # print('optimal classifier:', grid.best_estimator_)
    # print('optimal hyperparameters:', grid.best_params_)
    # print('best score:', grid.best_score_)
    
    # Test.
    
    y_pre = MLP.predict(x_test)

    return y_pre





