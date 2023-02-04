

# 2023-2-3 written by H.Zhang.


from sklearn.metrics import *


def Metrics(TrainingTime, Target_y, Pre_y, Classifier, MetricsScore):
    

    accuracy        = accuracy_score(Target_y, Pre_y)
    recall          = recall_score(Target_y, Pre_y, average="weighted")
    f1              = f1_score(Target_y, Pre_y, average='macro')
    
    print("| ========================================== %s ========================================== |"%(Classifier))
    
    print('%20s | %10f'%("Accuracy Score", accuracy))
    print('%20s | %10f'%("Recall Score", recall))
    print('%20s | %10f'%("F1 Score", f1))

    print('%20s | %10f'%("Training Time", TrainingTime))
    
    MetricsScore.append([Classifier, accuracy, recall, f1, TrainingTime])
























'''
accuracy_score(y_true,y_pre) : 精度

auc(x, y, reorder=False) : ROC曲线下的面积;较大的AUC代表了较好的performance。

average_precision_score(y_true, y_score, average=‘macro’, sample_weight=None):根据预测得分计算平均精度(AP)

brier_score_loss(y_true, y_prob, sample_weight=None, pos_label=None):The smaller the Brier score, the better.

confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):通过计算混淆矩阵来评估分类的准确性 返回混淆矩阵

f1_score(y_true, y_pred, labels=None, pos_label=1, average=‘binary’, sample_weight=None): F1值
F1 = 2 * (precision * recall) / (precision + recall) precision(查准率)=TP/(TP+FP) recall(查全率)=TP/(TP+FN)

log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)：对数损耗，又称逻辑损耗或交叉熵损耗

precision_score(y_true, y_pred, labels=None, pos_label=1, average=‘binary’,) ：查准率或者精度； precision(查准率)=TP/(TP+FP)

recall_score(y_true, y_pred, labels=None, pos_label=1, average=‘binary’, sample_weight=None)：查全率 ；recall(查全率)=TP/(TP+FN)

roc_auc_score(y_true, y_score, average=‘macro’, sample_weight=None)：计算ROC曲线下的面积就是AUC的值，the larger the better

roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)；计算ROC曲线的横纵坐标值，TPR，FPR

TPR = TP/(TP+FN) = recall(真正例率，敏感度) FPR = FP/(FP+TN)(假正例率，1-特异性)



'''
