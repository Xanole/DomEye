import os
import sys
import dpkt
import socket
import csv
import time
import math
import numpy as np
from collections import Counter


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
        _size.append(p.payload_len * p.direction)

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


def extract2csv(pcap_archive, csv_file):
    print(pcap_archive, csv_file)

    PACKET_SUM = 30

    f = open(csv_path, 'w', newline='')
    for pcap in os.listdir(pcap_archive):
        pcap_path = os.path.join(pcap_archive, pcap)

        # print(pcap_path)
        flow = extract_flow(pcap_path, PACKET_SUM)
        res = []

        # packet size sequence
        pkts = pkt_sequence(flow)
        if len(pkts) < PACKET_SUM:
            pkts += (PACKET_SUM - len(pkts)) * [MISSING_ITEM]
        res += pkts

        print(res)
        # write res not none
        if res:
            f_csv = csv.writer(f)
            f_csv.writerow(res)
    f.close()


if __name__ == '__main__':

    # for training
    # pcap_archive = 'Desktop\\data\\dataset\\train\\meek'
    # csv_path = 'Desktop\\data\\dataset\\train\\domain_train_meek+PL.csv'
    # extract2csv(pcap_archive, csv_path)

    # pcap_archive = 'Desktop\\data\\dataset\\train\\normal'
    # csv_path = 'Desktop\\data\\dataset\\train\\normal_train_meek+PL.csv'
    # extract2csv(pcap_archive, csv_path)

    # for testing tpr
    # pcap_archive = 'Desktop\\data\\dataset\\test_tpr\\meek'
    # csv_path = 'Desktop\\data\\dataset\\test_tpr\\test_meek+PL.csv'
    # extract2csv(pcap_archive, csv_path)

    # for fpr testing
    pcap_archive = 'Desktop\\data\\dataset\\test_fpr\\data'
    for archive in os.listdir(pcap_archive):
        pcap_dir = os.path.join(pcap_archive, archive)
        csv_path = 'Desktop\\data\\dataset\\test_fpr\\meek+PL\\' + archive + '.csv'
        print(pcap_dir, csv_path)
        extract2csv(pcap_dir, csv_path)




