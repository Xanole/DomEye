import os
import sys
import dpkt
import socket
import csv
import time
import math
import numpy as np
import joblib
import pandas as pd

UPSTREAM = 1
BOTH = 0
DOWNSTREAM = -1
PADDING = -1


class PacketMeta(object):
    """
    the structure of a packet
    :timestamp the captured time
    :size TCP payload length
    :direction 1: c2s, -1: s2c
    """
    def __init__(self):
        super(PacketMeta, self).__init__()
        self.timestamp = None
        self.size = None
        self.direction = 0


def LocalIP(ip):
    """label local IP, especially of client"""
    if ip[0:3] == "10." or ip[0:4] == "172." or ip[0:4] == "192.":
        return True
    else:
        return False


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

        packet_count += 1
        if packet_count > packet_sum:
            break

        direction = UPSTREAM if (LocalIP(sip)) else DOWNSTREAM
        length = len(tcp.data)

        pkt = PacketMeta()
        pkt.timestamp = time
        pkt.size = length
        pkt.direction = direction

        flow.append(pkt)

    f.close()

    return flow


def packet_size(flow, packet_sum):
    """
    print TCP payload length in directions of up (U), down (D) and both (b), respectively.
    :param str pcap_path: the pcap file's path
    :param str directon: the direction label
    :return list packet size statistic
    :return list entropy sequence
    """

    up_total_size = 0
    down_total_size = 0
    up_size_sequence = []
    down_size_sequence = []

    for p in flow:
        # if p.direction == UPSTREAM:
        if p.direction > 0:
            # print(length)
            up_total_size += p.size
            up_size_sequence.append(up_total_size)
        else:
            down_total_size -= p.size
            down_size_sequence.append(down_total_size)

    # padding -1 less than packet sum
    up_size_sequence.extend(down_size_sequence)
    # if len(size_sequence) < packet_sum:
    #         size_sequence += [PADDING] * (packet_sum - len(size_sequence))

    return up_size_sequence


def extract_features(pcap_path):

    PACKET_SUM = 30
    flow = extract_flow(pcap_path, PACKET_SUM)

    res = []
    tmp = packet_size(flow, PACKET_SUM)
    res += tmp

    return res


if __name__ == '__main__':

    path_DomEye = 'Desktop\\data\\model\\DomEye.pkl'
    model = joblib.load(path_DomEye)

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

        X = pd.DataFrame([ftr])
        model_start_time = time.time_ns()
        Y = model.predict(X)
        model_end_time = time.time_ns()

        model_time.append(model_end_time - model_start_time)

        # print(feature_time[-1], model_time[-1], sys.getsizeof(ftr), sys.getsizeof(X), Y, pcap)
        print(len(ftr), sys.getsizeof(ftr), feature_time[-1], model_time[-1])

    print("特征：{:4.4}, 模型：{:4.4}".format(sum(feature_time) / 1000 / 1000000, sum(model_time) / 1000 / 1000000))


