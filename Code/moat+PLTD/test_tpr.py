import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.naive_bayes import GaussianNB as NB

np.set_printoptions(threshold=np.inf)

DF = 1
NORMAL = 0

model_path = 'Desktop\\data\\model\\moat+PLTD.pkl'

def test_model(csv_path):
    """
    load and test machine learning algorithm
    :param DataFrame X: the matrix of the entire data
    :param Series Y: the vector of the entire labels
    :param str model_name: the classification model (DT, NB or KNN)
    """

    model = joblib.load(model_path)
    X = pd.read_csv(csv_path, header=None)
    Y = model.predict(X).tolist()

    tpr = sum(Y) / len(Y)

    return tpr


if __name__ == '__main__':

    # test tpr
    csv_path = 'Desktop\\data\\dataset\\test_tpr\\test_moat+PLTD.csv'
    tpr = test_model(csv_path)
    print("moat测试：TPR为{:2.2%}".format(tpr))


