import svmutil as libsvm
import numpy as np


class SVC:
    def __init__(self, C=1):
        self.param = libsvm.svm_parameter()
        self.param.kernel_type = libsvm.RBF
        self.param.C = C
        self.model = None


    def fit(self, X_train, y_train):
        X_train = np.array(X_train).tolist()
        y_train = np.array(y_train).tolist()
        problem = libsvm.svm_problem(y_train, X_train)
        self.model = libsvm.svm_train(problem, self.param)


    def predict(self, X_test):
        X_test = np.array(X_test).tolist()
        y_pred, _, _ = libsvm.svm_predict([0]*len(X_test), X_test, self.model)
        return y_pred
