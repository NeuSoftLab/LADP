from sklearn import svm
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from DataLoader.LoadKwaiDataSVM import getSVMDataLoader
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
svm_C = 100
svm_kernel = 'linear'

'''
C：C-SVC的惩罚参数C?默认值是1.0
C越大，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱;
C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
– 线性：u’v
– 多项式：(gamma*u’v + coef0)^degree
– RBF函数：exp(-gamma|u-v|^2)
–sigmoid：tanh(gammau’*v + coef0)

degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。

原文链接：https://blog.csdn.net/weixin_41990278/article/details/93137009
'''
def SVM():
    clf = svm.SVC(kernel=svm_kernel,C=svm_C)
    train_set , test_set = getSVMDataLoader()
    x_train, y_train = train_set
    x_test, y_test = test_set
    x_test,y_test=x_test[:200],y_test[:200]
    #创建StandardScaler()实例
    standard_s1=StandardScaler()
    #将DataFrame格式的数据按照每一个series分别标准化
    x_train=standard_s1.fit_transform(x_train)
    x_test = standard_s1.transform(x_test)

    clf.fit(x_train,y_train)

    y_train_hat = clf.predict(x_train)
    y_test_hat = clf.predict(x_test)

    # auc
    auc_train = metrics.roc_auc_score(y_train, y_train_hat)
    auc_test = metrics.roc_auc_score(y_test, y_test_hat)
    # rmse
    error_train = y_train - y_train_hat
    rmse_tarin = (error_train ** 2).mean() ** 0.5

    error_test = y_test - y_test_hat
    rmse_test = (error_test ** 2).mean() ** 0.5
    #DF=ABS(Y-Y_HAT)/Y
    #df
    y_train_hat[y_train_hat >= 0.5] = 1.0
    y_train_hat[y_train_hat < 0.5] = 0.0
    df_train = abs(y_train.sum()-y_train_hat.sum())/y_train.sum()

    y_test_hat[y_test_hat >= 0.5] = 1.0
    y_test_hat[y_test_hat < 0.5] = 0.0
    df_test = abs(y_test.sum()-y_test_hat.sum())/y_test.sum()
    print('\n')
    print('train  auc  %.4f  rmse  %.4f  df  %.4f' %(auc_train,rmse_tarin,df_train))
    print('test  auc  %.4f  rmse  %.4f  df  %.4f' %(auc_test,rmse_test,df_test))


