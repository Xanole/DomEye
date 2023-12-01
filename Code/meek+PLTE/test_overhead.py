import os
import sys
import dpkt
import socket
import csv
import time
import math
from collections import Counter
import numpy as np
import joblib
import pandas as pd


UPSTREAM = 1
BOTH = 0
DOWNSTREAM = -1

MISSING_ITEM = -1
FLAG_ACK = 16


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
        self.entropy = 0


def LocalIP(ip):
    """label local IP, especially of client"""
    if ip[0:3] == "10." or ip[0:4] == "172." or ip[0:4] == "192.":
        return True
    else:
        return False


def cal_entropy(s):
    """
    calcuate the entropy of a string
    :param str s: a string
    """
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())


def pkt_payload_entropy_dist(trace, direction):
    """
    min/max/mean entropies of all the packet payloads
    in a given direction;
    """
    _tmp = []
    for p in trace:
        if p.payload_len == 0:
            continue
        if p.direction == direction:
            if p.entropy: _tmp.append(p.entropy)
    if not _tmp:
        return [MISSING_ITEM] * 3
    return [round(min(_tmp), 2), round(max(_tmp), 2), round(np.average(_tmp), 2)]

def pkt_len_dist(trace, direction):
    """
    payload length distribution of the packets
    in a given direction; only return the top 5
    most seen pcap payload lengths.
    """
    _size = []
    for p in trace:
        if p.direction == direction:
            _size.append(p.payload_len)
        else:
            continue
    r = Counter(_size)
    total = sum(r.values())
    # try:
    #     zero_p = round(float(r.pop(0)) / total, 2)
    # except:
    #     zero_p = 0.0
    res = sorted(r.items(), key=lambda x:x[1], reverse=True)[:5]

    res = [(v[0], round(float(v[1]) / total * 100, 2)) for v in res]
    res = [v[0] for v in res]
    # return zero_p, res
    return res

def pkt_ack_ratio(trace, direction):
    """
    ack percent in a given direction;
    """
    count = 0
    for p in trace:
        if p.flag != FLAG_ACK:
            continue
        if p.direction == direction:
            count += 1
    return round(count / 30, 2)

def pkt_payload_ack_seq(trace, direction):
    """
    percentage of intervals between ACK packets in a given
    direction that falls in to a given range.
    """
    _tmp = []
    for p in trace:
        if p.flag != FLAG_ACK:
            continue
        if p.direction == direction:
            _tmp.append(p.timestamp)
        else:
            continue
    data = [(y - x) * 1000 for x, y in zip(_tmp, _tmp[1:])]
    bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    if not data:
        return [0] * 29
    digitized = np.digitize(data, bins)
    tmp = Counter(digitized)
    total = len(data)
    res = []
    for k in range(1, 30):
        if k not in tmp:
            res.append(0)
        else:
            res.append(round(float(tmp[k]) / total, 2))
    return res


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

        entropy = 0
        if length > 0:
            entropy = cal_entropy(payload)
        pkt.entropy = entropy


        flow.append(pkt)

    f.close()

    return flow


def extract_features(pcap_path):

    FLOW_LENGTH = 30

    flow = extract_flow(pcap_path, FLOW_LENGTH)
    res = []

    for direction in [UPSTREAM, DOWNSTREAM]:
        # entropy
        tmp = pkt_payload_entropy_dist(flow, direction)
        res += tmp

    for direction in [UPSTREAM, DOWNSTREAM]:
        # packet size dist
        top_size = pkt_len_dist(flow, direction)
        if len(top_size) < 5:
            top_size += (5 - len(top_size)) * [MISSING_ITEM]
        res += top_size

    for direction in [UPSTREAM, DOWNSTREAM]:
        # ack percent in each direction
        res.append(pkt_ack_ratio(flow, direction))

    for direction in [UPSTREAM, DOWNSTREAM]:
        # ack_seq dist
        tmp = pkt_payload_ack_seq(flow, direction)
        if tmp:
            res += tmp

    return res


if __name__ == '__main__':

    model_path = 'Desktop\\data\\model\\meek+PLTE.pkl'
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

    print("特征：{:4.4}, 模型：{:4.4}".format(sum(feature_time) / 1000 / 1000000, sum(model_time) / 1000 / 1000000))


