import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import *
# from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RF


DF = 1
NORMAL = 0

path_DomEye = 'Desktop\\data\\model\\DomEye.pkl'

if __name__ == '__main__':

    model_path = path_DomEye
    model = joblib.load(model_path)

    name = ['CTU-Mixed-Capture-1', 'CTU-Mixed-Capture-2', 'CTU-Mixed-Capture-3', 'CTU-Mixed-Capture-4',
            'CTU-Mixed-Capture-5', 'CTU-Normal-7', 'CTU-Normal-12', 'CTU-Normal-20', 'CTU-Normal-21', 'CTU-Normal-22',
            'CTU-Normal-23', 'CTU-Normal-24', 'CTU-Normal-25', 'CTU-Normal-26', 'CTU-Normal-27', 'CTU-Normal-28',
            'CTU-Normal-29', 'CTU-Normal-30', 'CTU-Normal-31', 'CTU-Normal-32']

    # pcap_archive = 'Desktop\\data\\dataset\\test_fpr\\data'
    # sub_archive = os.listdir(pcap_archive)
    # for archive in sub_archive:

    for archive in name:
        csv_file = archive
        # print(csv_file, end=' ')

        csv_path = 'Desktop\\data\\dataset\\test_fpr\\DomEye\\' + archive + '.csv'

        X = pd.read_csv(csv_path, header=None)
        Y = model.predict(X).tolist()
        fpr = sum(Y) / len(Y)

        # print("模型测试：FPR为{:2.2%}".format(fpr))
        # print("{:2.2%}".format(fpr))
        print("%2.2f" % (fpr * 100))


