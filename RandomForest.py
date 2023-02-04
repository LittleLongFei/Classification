

# 2023-2-3 written by H.Zhang.


from sklearn.ensemble import RandomForestClassifier


def RF(x_train, x_test, y_train, y_test):
    
    # Train.
    
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
    forest.fit(x_train,y_train)
    
    # Print some informations.
    # print('optimal classifier:', grid.best_estimator_)
    # print('optimal hyperparameters:', grid.best_params_)
    # print('best score:', grid.best_score_)
    
    # Test.
    
    y_pre = forest.predict(x_test)

    return y_pre





