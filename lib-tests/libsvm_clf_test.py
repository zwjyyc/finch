from svmutil import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


X, y = make_classification(5000)
X = X.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

problem = svm_problem(y_train, X_train)
param = svm_parameter()
param.kernel_type = RBF
param.C = 1

model = svm_train(problem, param)
y_pred, _, _ = svm_predict([0]*len(X_test), X_test, model)

acc = np.equal(y_pred, y_test).astype(float).mean()
print('The testing accuracy is %.3f' % acc)
