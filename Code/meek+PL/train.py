import math
import os
import warnings
import numpy as np
import torch.optim
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
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


def get_data():

    data_m = pd.read_csv(df_data, header=None, nrows=1000)
    label_m = pd.Series(DF for i in range(1000))
    data_n = pd.read_csv(normal_data, header=None, nrows=4000)
    label_n = pd.Series(NORMAL for i in range(4000))

    X = pd.concat([data_m, data_n], axis=0, join='outer')
    Y = pd.concat([label_m, label_n], axis=0, join='outer')

    X_train = X.values.astype(float)
    Y_train = Y.values
    X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)

    return X_train, Y_train


def train_model(X_train, Y_train):

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    meekDet = CNN1D()
    loss_function = nn.CrossEntropyLoss()

    epochs = 10
    learning_rate = 0.001
    optim = torch.optim.Adam(meekDet.parameters(), lr=learning_rate)

    for i in range(epochs):
        total_acc = []
        total_tpr = []
        total_fpr = []

        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        meekDet.train()
        for data in train_loader:
            X_data, Y_data = data[0], data[1]
            output = meekDet(X_data)
            loss = loss_function(output, Y_data)
            optim.zero_grad()
            loss.backward()
            optim.step()

            pred = output.argmax(axis=1)
            acc = accuracy_score(Y_data, pred)
            result = get_stat(Y_data.tolist(), pred.tolist())

            total_acc.append(acc)
            total_tpr.append(result[0])
            total_fpr.append(result[1])

        total_acc = list(filter(None, total_acc))
        total_tpr = list(filter(None, total_tpr))
        total_fpr = list(filter(None, total_fpr))
        print("第{}轮训练，Accuracy为{:.4f}，tpr为{:.4f}，fpr为{:.4f}".format(i + 1,
                                                                  sum(total_acc) / len(total_acc),
                                                                  sum(total_tpr) / len(total_tpr),
                                                                  sum(total_fpr) / len(total_fpr)))

    torch.save(meekDet, model_path)


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

    if list(ytest).count(DF) == 0:
        tpr = None
    else:
        tpr = float(tp / list(ytest).count(DF))

    if list(ytest).count(NORMAL) == 0:
        fpr = None
    else:
        fpr = float(fp / list(ytest).count(NORMAL))

    accuracy = float((tp + tn) / (list(ytest).count(DF) + list(ytest).count(NORMAL)))
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = float(tp / (tp + fp))

    return [tpr, fpr, accuracy, precision]


if __name__ == '__main__':

    X_train, Y_train = get_data()
    train_model(X_train, Y_train)

