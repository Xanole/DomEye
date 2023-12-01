import math
import os
import warnings
import numpy as np
import torch.optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(threshold=np.inf)

DF = 1
NORMAL = 0

model_path = 'Desktop\\data\\model\\meek+PL.pkl'

df_data = 'Desktop\\data\\dataset\\train\\domain_train_meek+PL.csv'
normal_data = 'Desktop\\data\\dataset\\train\\normal_train_meek+PL.csv'

warnings.simplefilter(action='ignore', category=RuntimeWarning)
torch.manual_seed(2023)


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv1d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 16, 5])
            nn.Conv1d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 32, 1])
            nn.Dropout(0.2),
            nn.Flatten(),  # torch.Size([128, 32])    (假如上一步的结果为[128, 32, 2]， 那么铺平之后就是[128, 64])
        )
        self.model2 = nn.Sequential(
            nn.Linear(in_features=384, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.reshape(-1, 1, 30)   #结果为[128,1,11]  目的是把二维变为三维数据
        x = self.model1(input)
        x = self.model2(x)
        return x


if __name__ == '__main__':

    meekDet = torch.load(model_path)

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

        csv_path = 'Desktop\\data\\dataset\\test_fpr\\meek+PL\\' + archive + '.csv'

        data_test = pd.read_csv(csv_path, header=None)
        X = pd.concat([data_test], axis=0, join='outer')
        X_test = X.values.astype(float)
        X_test = torch.FloatTensor(X_test)
        output = meekDet(X_test)
        pred = output.argmax(axis=1)

        Y_test = pred.tolist()

        # print("模型测试：FPR为{:2.2%}".format(sum(Y_test) / len(Y_test)))
        print("%2.2f" % (sum(Y_test) / len(Y_test) * 100))


