

# 2023-2-3 written by H.Zhang.


# ========================================================== Import Python Toolbox ========================================================== #

from time import *

from DatasetLoador import DatasetLoador
from DecisionTree import DecisionTree
from KNeighbors import KNeighbors
from Perceptron import PerceptronClassifier
from LogisticRegression import LR
from SVM import SVM
from RandomForest import RF
from Bayes import Bayes
from TSKFLS import TSKFLS
from Metrics import Metrics

# =============================================================== Load Dataset ============================================================== #

x_train, x_test, y_train, y_test = DatasetLoador()

# ============================================================== Define Metrics -============================================================ #

MetricsScore = []
MetricsScore.append(["Method", "Precision Score", "Recall Score", "F1 Score", "Training Time"])

# ============================================================= Train Classifiers =========================================================== #


# -------------------------------------------- Decision Tree ->
Method = "Decision Tree"

TrainingTimeBegin = time()
y_pre = DecisionTree(x_train, x_test, y_train, y_test)
TrainingTimeEnd   = time()

''' ____________Calculation Metrics____________ '''

TrainingTime = TrainingTimeEnd - TrainingTimeBegin
Metrics(TrainingTime, y_test, y_pre, Method, MetricsScore)



# -------------------------------------------- K Neighbors ->
Method = "K Neighbors"

TrainingTimeBegin = time()
y_pre = KNeighbors(x_train, x_test, y_train, y_test)
TrainingTimeEnd   = time()

''' ____________Calculation Metrics____________ '''

TrainingTime = TrainingTimeEnd - TrainingTimeBegin
Metrics(TrainingTime, y_test, y_pre, Method, MetricsScore)



# -------------------------------------------- Perceptron Classifier->
Method = "Perceptron"

TrainingTimeBegin = time()
y_pre = PerceptronClassifier(x_train, x_test, y_train, y_test)
TrainingTimeEnd   = time()

''' ____________Calculation Metrics____________ '''

TrainingTime = TrainingTimeEnd - TrainingTimeBegin
Metrics(TrainingTime, y_test, y_pre, Method, MetricsScore)


# -------------------------------------------- Logistic Regression ->
Method = "Logistic Regression"

TrainingTimeBegin = time()
y_pre = LR(x_train, x_test, y_train, y_test)
TrainingTimeEnd   = time()

''' ____________Calculation Metrics____________ '''

TrainingTime = TrainingTimeEnd - TrainingTimeBegin
Metrics(TrainingTime, y_test, y_pre, Method, MetricsScore)


# -------------------------------------------- SVM ->
Method = "SVM"

TrainingTimeBegin = time()
y_pre = SVM(x_train, x_test, y_train, y_test)
TrainingTimeEnd   = time()

''' ____________Calculation Metrics____________ '''

TrainingTime = TrainingTimeEnd - TrainingTimeBegin
Metrics(TrainingTime, y_test, y_pre, Method, MetricsScore)


# -------------------------------------------- Random Forest ->
Method = "Random Forest"

TrainingTimeBegin = time()
y_pre = RF(x_train, x_test, y_train, y_test)
TrainingTimeEnd   = time()

''' ____________Calculation Metrics____________ '''

TrainingTime = TrainingTimeEnd - TrainingTimeBegin
Metrics(TrainingTime, y_test, y_pre, Method, MetricsScore)



# -------------------------------------------- Bayes ->
Method = "Bayes"

TrainingTimeBegin = time()
y_pre = Bayes(x_train, x_test, y_train, y_test)
TrainingTimeEnd   = time()

''' ____________Calculation Metrics____________ '''

TrainingTime = TrainingTimeEnd - TrainingTimeBegin
Metrics(TrainingTime, y_test, y_pre, Method, MetricsScore)




# -------------------------------------------- MLP ->
Method = "MLP"

TrainingTimeBegin = time()
y_pre = Bayes(x_train, x_test, y_train, y_test)
TrainingTimeEnd   = time()

''' ____________Calculation Metrics____________ '''

TrainingTime = TrainingTimeEnd - TrainingTimeBegin
Metrics(TrainingTime, y_test, y_pre, Method, MetricsScore)



# -------------------------------------------- TSKFLS ->
Method = "TSKFLS"

TrainingTimeBegin = time()
y_pre = TSKFLS(x_train, x_test, y_train, y_test)
TrainingTimeEnd   = time()

''' ____________Calculation Metrics____________ '''

TrainingTime = TrainingTimeEnd - TrainingTimeBegin
Metrics(TrainingTime, y_test, y_pre, Method, MetricsScore)








print("\n\n\n\n\n\n\n\n------------------------------------------------------------------------------------------------------------------------")

for i in range(len(MetricsScore)):
    
    print('%20s | %20s | %20s | %20s | %20s '%(MetricsScore[i][0],MetricsScore[i][1],MetricsScore[i][2],MetricsScore[i][3],MetricsScore[i][4]))
    
print("------------------------------------------------------------------------------------------------------------------------")



