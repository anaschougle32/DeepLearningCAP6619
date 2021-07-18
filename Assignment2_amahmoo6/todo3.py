class Todo3_A:

    # init method or constructor
    def __init__(self, x, y, model):
        self.y_val = x
        self.x_val = y
        model = y

        self.build()

    def build(self):
        self.build_matrix()
        self.perf_measure(self.y_val, self.model.predict_classes(self.x_val))

    def build_matrix(self):
        plt.figure(figsize=(20, 10))
        sns.heatmap(confusion_matrix(self.y_val, self.model.predict_classes(self.x_val)),
                    cmap=sns.color_palette("pastel", as_cmap=True),
                    annot=True, fmt="d")
        plt.title("Conf Matrix")
        plt.show()

    def perf_measure(self, y_actual, y_pred):
        TruePos = 0
        FalsePos = 0
        TrueNegative = 0
        FalseNeg = 0

        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == 1:
                TruePos += 1
            if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
                FalsePos += 1
            if y_actual[i] == y_pred[i] == 0:
                TrueNegative += 1
            if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
                FalseNeg += 1

        return (TruePos, FalsePos, TrueNegative, FalseNeg)



Todo3_A(y_val, x_val,model)




import tensorflow as tf
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")
def conMatrix():
    plt.figure(figsize=(20,10))
    sns.heatmap(confusion_matrix(y_val, model.predict_classes(x_val)) ,
                cmap=sns.color_palette("pastel", as_cmap=True),
               annot=True, fmt="d")
    plt.title("Conf Matrix")
    plt.show()
conMatrix()

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)
perf_measure(y_val, model.predict_classes(x_val))