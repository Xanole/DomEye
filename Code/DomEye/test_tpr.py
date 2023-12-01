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


if __name__ == '__main__':

    # meek moat snowflake
    # test tpr

    csv_path = 'Desktop\\data\\dataset\\test_tpr\\test_meek_domeye.csv'
    tpr = test_model_tpr(csv_path)
    print("meek测试：TPR为{:2.2%}".format(tpr))

    csv_path = 'Desktop\\data\\dataset\\test_tpr\\test_moat_domeye.csv'
    tpr = test_model_tpr(csv_path)
    print("moat测试：TPR为{:2.2%}".format(tpr))

    csv_path = 'Desktop\\data\\dataset\\test_tpr\\test_snowflake_domeye.csv'
    tpr = test_model_tpr(csv_path)
    print("snowflake测试：TPR为{:2.2%}".format(tpr))




