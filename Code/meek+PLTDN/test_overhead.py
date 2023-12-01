import os
import sys
import dpkt
import socket
import csv
import time
import math
import more_itertools
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


def packet_size_features(flow):
    """
    packet size features in a flow
    :param flow list: a flow contain a series of packets
    :param direction int: a given direction
    :return res list: the top 5 most seen packet size
    """
    _size = []
    for p in flow:
        _size.append(p.size)

    max_size = np.max(_size)
    min_size = np.min(_size)
    size_mean = np.mean(_size)
    size_std = np.std(_size)

    return [max_size, min_size, size_mean, size_std]


def packet_time_features(flow):
    """
    percentage of intervals between packets that falls in to a given bin
    :param flow list: a flow contain a series of packets
    :param direction int: a given direction
    :return res list: percentage of every bin
    """
    _time = []
    for p in flow:
        _time.append(p.timestamp)

    intervals = [(y - x) * 1000 for x, y in zip(_time, _time[1:])]

    max_interval = np.max(intervals)
    min_interval = np.min(intervals)
    interval_mean = np.mean(intervals)
    interval_std = np.std(intervals)

    return [max_interval, min_interval, interval_mean, interval_std]


def packet_dir_features(flow):
    """
    direction ratio
    :param flow list: a flow contain a series of packets
    :param direction int: a given direction
    :return res float: down/up
    """
    total_packet = 0
    total_bytes = 0
    packet_up_count = 0
    bytes_up_count = 0
    packet_down_count = 0
    bytes_down_count = 0

    for p in flow:
        if p.direction == UPSTREAM:
            packet_up_count += 1
            bytes_up_count += p.size
        if p.direction == DOWNSTREAM:
            packet_down_count += 1
            bytes_down_count += p.size

    total_packet = packet_up_count + packet_down_count
    total_bytes = bytes_up_count + bytes_down_count

    if total_bytes == 0:
        total_bytes = 0.01

    rate_bytes_send = bytes_up_count / total_bytes
    rate_bytes_receive = bytes_down_count / total_bytes
    rate_packets_send = packet_up_count / total_packet
    rate_packets_receive = packet_down_count / total_packet

    return [rate_bytes_send, rate_bytes_receive, rate_packets_send, rate_packets_receive]


def extract_features(pcap_path):

    FLOW_LENGTH = 30

    flow = extract_flow(pcap_path, FLOW_LENGTH)
    res = []

    # if less than FLOW_LENGTH, padding the last packet
    if len(flow) < FLOW_LENGTH:
        flow += (FLOW_LENGTH - len(flow)) * [flow[-1]]

    # window = 10, step = 4
    flow_wins = more_itertools.windowed(flow, n=10, step=4)
    for sub_flow in flow_wins:
        if None not in sub_flow:
            sub_flow = list(sub_flow)

            # packet size features (4)
            tmp = packet_size_features(sub_flow)
            res += tmp

            # packet time features (4)
            tmp = packet_time_features(sub_flow)
            res += tmp

            # packet direction features (4)
            tmp = packet_dir_features(sub_flow)
            res += tmp

    # print(res)
    return res


if __name__ == '__main__':

    model_path = 'Desktop\\data\\model\\meek+PLTDN.pkl'
    model = joblib.load(model_path)

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
        Y = model.predict(X).tolist()
        model_end_time = time.time_ns()

        model_time.append(model_end_time - model_start_time)

        # print(feature_time[-1], model_time[-1], sys.getsizeof(ftr), sys.getsizeof(X), Y, pcap)
        print(len(ftr), sys.getsizeof(ftr), feature_time[-1], model_time[-1])

    print("特征：{:4.4}, 模型：{:4.4}".format(sum(feature_time)/1000/1000000, sum(model_time)/1000/1000000))


