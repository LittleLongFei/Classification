

# 2023-2-3 written by H.Zhang.


from sklearn.linear_model import LogisticRegression


def LR(x_train, x_test, y_train, y_test):
    
    # Train.
    
    lr = LogisticRegression(C=1000.0, random_state=0)

    lr.fit(x_train, y_train)
    
    # Print some informations.
    # print('optimal classifier:', grid.best_estimator_)
    # print('optimal hyperparameters:', grid.best_params_)
    # print('best score:', grid.best_score_)
    
    # Test.
    
    y_pre = lr.predict(x_test)

    return y_pre





