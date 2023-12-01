import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import *
# from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RF
# RandomForestClassifier(n_estimators=100, max_features='auto',n_jobs=1)
from lightgbm import LGBMClassifier as LGBM


np.set_printoptions(threshold=np.inf)

rng = np.random.RandomState(5)

DF = 1
NORMAL = 0

path_DomEye = 'Desktop\\data\\model\\DomEye.pkl'


def get_data():

    df_data = 'Desktop\\data\\dataset\\train\\domain_train_domeye.csv'
    normal_data = 'Desktop\\data\\dataset\\train\\normal_train_domeye.csv'

    data_m = pd.read_csv(df_data, header=None, nrows=1600)
    label_m = pd.Series([DF for i in range(1600)])

    data_n = pd.read_csv(normal_data, header=None, nrows=4000)
    label_n = pd.Series([NORMAL for i in range(4000)])

    X = pd.concat([data_m, data_n], axis=0, join='outer')
    Y = pd.concat([label_m, label_n], axis=0, join='outer')

    return X, Y


def train_model(X, Y):
    """
    train machine learning algorithm
    :param DataFrame X: the matrix of the entire data
    :param Series Y: the vector of the entire labels
    :param str model_name: the classification model (DT, NB or KNN)
    """

    # model = RF(n_estimators=50, random_state=rng)
    model = RF(n_estimators=10, n_jobs=1)
    model_path = path_DomEye

    min_score = 0
    accuracy_list = []
    tpr_list = []
    fpr_list = []
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=rng)
        # print(np.shape(X_train))
        # print(np.shape(X_test))
        model.fit(X_train, Y_train)

        score = model.score(X_test, Y_test)
        accuracy_list.append(score)

        result = get_stat(list(Y_test), model.predict(X_test).tolist())
        tpr_list.append(result[0])
        fpr_list.append(result[1])

        if score > min_score:
            min_score = score
            joblib.dump(model, model_path)
            # print(score)

    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    avg_tpr = sum(tpr_list) / len(tpr_list)
    avg_fpr = sum(fpr_list) / len(fpr_list)
    return avg_accuracy, avg_tpr, avg_fpr


def test_model_tpr(csv_path):
    """
    load and test machine learning algorithm
    :param DataFrame X: the matrix of the entire data
    :param Series Y: the vector of the entire labels
    :param str model_name: the classification model (DT, NB or KNN)
    """

    model_path = path_DomEye

    model = joblib.load(model_path)
    X = pd.read_csv(csv_path, header=None)
    Y = model.predict(X).tolist()

    tpr = sum(Y) / len(Y)

    return tpr


def test_model_fpr():

    model_path = path_DomEye
    model = joblib.load(model_path)

    pcap_archive = 'Desktop\\data\\test\\fpr_data'
    sub_archive = os.listdir(pcap_archive)
    for archive in sub_archive:
        csv_file = archive
        print(csv_file, end=' ')

        csv_path = 'Desktop\\data\\test\\DomEye\\' + csv_file + '.csv'

        X = pd.read_csv(csv_path, header=None)
        Y = model.predict(X).tolist()
        fpr = sum(Y) / len(Y)

        print("模型测试：FPR为{:2.2%}".format(fpr))


def get_stat(ytest, ypred):
    """
    calculate the TPR/FPR/FNR/TNR/PR_AUC
    :param list ytest: the array for the labels of the test instances
    :param list ypred: the array for the predicted labels of the
    test instances
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    label = DF
    for i in range(len(ypred)):
        # print(ypred[i], ytest[i])
        if ypred[i] == label and ytest[i] == label:
            tp += 1
        elif ypred[i] == label and ytest[i] != label:
            fp += 1
        elif ypred[i] != label and ytest[i] == label:
            fn += 1
        elif ypred[i] != label and ytest[i] != label:
            tn += 1
    tpr = float(tp / list(ytest).count(DF))
    fpr = float(fp / list(ytest).count(NORMAL))
    # fnr = float(fn / list(ytest).count(MOAT))
    # tnr = float(tn / list(ytest).count(NORMAL))
    accuracy = float((tp + tn) / (list(ytest).count(DF) + list(ytest).count(NORMAL)))
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = float(tp / (tp + fp))

    return [tpr, fpr, accuracy, precision]


if __name__ == '__main__':

    # train DomEye detector
    X, Y = get_data()
    avg_accuracy, avg_tpr, avg_fpr = train_model(X, Y)
    print(round(avg_accuracy*100, 2), round(avg_tpr*100, 2), round(avg_fpr*100, 2))

    # # test tpr
    # csv_path = 'Desktop\\data\\test\\tpr_data\\test_meek_domeye.csv'
    # tpr = test_model_tpr(csv_path)
    # print("meek测试：TPR为{:2.2%}".format(tpr))
    #
    # csv_path = 'Desktop\\data\\test\\tpr_data\\test_moat_domeye.csv'
    # tpr = test_model_tpr(csv_path)
    # print("moat测试：TPR为{:2.2%}".format(tpr))
    #
    # csv_path = 'Desktop\\data\\test\\tpr_data\\test_snow_domeye.csv'
    # tpr = test_model_tpr(csv_path)
    # print("snowflake测试：TPR为{:2.2%}".format(tpr))
    #
    # # test fpr
    # test_model_fpr()



