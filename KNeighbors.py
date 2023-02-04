

# 2023-2-3 written by H.Zhang.


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def KNeighbors(x_train, x_test, y_train, y_test):
    
    # Train.
    
    # GridSearchCV will be used to choose the eest parameters.
    knn_model = KNeighborsClassifier()
    param_grid = [
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 20)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 20)],
            'p':[i for i in range(1, 6)]
        }
    ]
    
    grid = GridSearchCV(knn_model, param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    # Print some informations.
    # print('optimal classifier:', grid.best_estimator_)
    # print('optimal hyperparameters:', grid.best_params_)
    # print('best score:', grid.best_score_)
    
    # Test.
    
    knn_model = grid.best_estimator_
    y_pre = knn_model.predict(x_test)

    return y_pre

