

# 2023-2-3 written by H.Zhang.


from sklearn.svm import SVC


def SVM(x_train, x_test, y_train, y_test):
    
    # Train.
    svm = SVC(kernel='rbf',random_state=0,gamma=0.2,C=1.0)
    svm.fit(x_train,y_train)
    
    # Print some informations.
    # print('optimal classifier:', grid.best_estimator_)
    # print('optimal hyperparameters:', grid.best_params_)
    # print('best score:', grid.best_score_)
    
    # Test.
    
    y_pre = svm.predict(x_test)

    return y_pre





