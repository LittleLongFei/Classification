

# 2023-2-3 written by H.Zhang.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def DatasetLoador():

    iris_data_set = pd.read_csv('./Dataset/iris.csv')
    
    
    x = iris_data_set.iloc[:, 0:4].values
    print(x)
    y = iris_data_set.iloc[:, -1].values
    print(y)
    
    # PCA
    
    transfer = PCA(n_components=2)
    x = transfer.fit_transform(x)




    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    return x_train, x_test, y_train, y_test


