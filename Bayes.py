

# 2023-2-3 written by H.Zhang.

from sklearn.naive_bayes import GaussianNB


def Bayes(x_train, x_test, y_train, y_test):
    
    # Train.
    
    clf=GaussianNB()
    clf.fit(x_train,y_train)
    
    # Print some informations.
    # print('optimal classifier:', grid.best_estimator_)
    # print('optimal hyperparameters:', grid.best_params_)
    # print('best score:', grid.best_score_)
    
    # Test.
    
    y_pre = pre=clf.predict(x_test)

    return y_pre





