import os
import sys
import dpkt
import socket
import csv
import time
import math
import numpy as np
import pandas as pd
import torch.optim
from torch import nn
torch.set_num_interop_threads(1)
torch.set_num_threads(1)


UPSTREAM = 1
BOTH = 0
DOWNSTREAM = -1
MISSING_ITEM = -1


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


class PacketMeta(object):
    """
    the structure of a packet
    :timestamp the captured time
    :size TCP payload length
    :direction 1: c2s, -1: s2c
    """
    def __init__(self):
        super(PacketMeta, self).__init__()
        self.direction = -1
        self.timestamp = None
        self.flag = 0
        self.payload_len = None


def LocalIP(ip):
    """label local IP, especially of client"""
    if ip[0:3] == "10." or ip[0:4] == "172." or ip[0:4] == "192.":
        return True
    else:
        return False


def pkt_sequence(trace):
    """
    payload length sequence of the packets;
    return the payload length sequence.
    """
    _size = []
    for p in trace:
        _size = _size + [p.payload_len * p.direction]

    return _size


def extract_flow(pcap_path, packet_sum):
    """
    extract every packet's information to form a flow
    :param pcap_path string: a given path of pacp file
    :param packet_sum int: the first n packets
    :return flow list: packet information list of a flow
    """
    flow = []
    packet_count = 0

    f = open(pcap_path, 'rb')
    pcap = dpkt.pcap.Reader(f)
    for ts, buf in pcap:
        time = ts

        eth = dpkt.ethernet.Ethernet(buf)
        ip = eth.data
        tcp = ip.data

        if hasattr(ip, 'src') and hasattr(ip, 'dst'):
            try:
                sip = socket.inet_ntop(socket.AF_INET, ip.src)
                dip = socket.inet_ntop(socket.AF_INET, ip.dst)
            except Exception as e:
                sip = socket.inet_ntop(socket.AF_INET6, ip.src)
                dip = socket.inet_ntop(socket.AF_INET6, ip.dst)

        sport = tcp.sport
        dport = tcp.dport
        payload = tcp.data
        flags = tcp.flags

        packet_count += 1
        if packet_count > packet_sum:
            break

        direction = UPSTREAM if (LocalIP(sip)) else DOWNSTREAM
        length = len(tcp.data)

        pkt = PacketMeta()
        pkt.timestamp = time
        pkt.payload_len = length
        pkt.direction = direction
        pkt.flag = flags

        flow.append(pkt)

    f.close()

    return flow


def extract_features(pcap_path):

    FLOW_LENGTH = 30

    flow = extract_flow(pcap_path, FLOW_LENGTH)
    res = []

    # packet size sequence
    pkts = pkt_sequence(flow)
    if len(pkts) < FLOW_LENGTH:
        pkts += (FLOW_LENGTH - len(pkts)) * [MISSING_ITEM]
    res += pkts

    return res


if __name__ == '__main__':

    model_path = 'Desktop\\data\\model\\meek+PL.pkl'
    model = torch.load(model_path)

    feature_time = []
    model_time = []

    # overhead testing
    pcap_archive = 'Desktop\\data\\dataset\\test_overhead\\meek'
    for pcap in os.listdir(pcap_archive):
        pcap_path = os.path.join(pcap_archive, pcap)

        feature_start_time = time.time_ns()
        ftr = extract_features(pcap_path)
        feature_end_time = time.time_ns()

        feature_time.append(feature_end_time - feature_start_time)

        time.sleep(0.1)

        model_start_time = time.time_ns()
        X = pd.DataFrame([ftr])
        X = X.values.astype(float)
        X = torch.FloatTensor(X)
        output = model(X)
        Y = output.argmax(axis=1)
        model_end_time = time.time_ns()

        model_time.append(model_end_time - model_start_time)

        # print(feature_time[-1], model_time[-1], sys.getsizeof(ftr), sys.getsizeof(X), Y, pcap)
        print(len(ftr), sys.getsizeof(ftr), feature_time[-1], model_time[-1])

    print("特征：{:4.4}, 模型：{:4.4}".format(sum(feature_time) / 1000 / 1000000, sum(model_time) / 1000 / 1000000))


