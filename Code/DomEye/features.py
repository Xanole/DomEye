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
        if p.direction == UPSTREAM:
            # print(length)
            up_total_size += p.size
            up_size_sequence.append(up_total_size)
        else:
            down_total_size -= p.size
            down_size_sequence.append(down_total_size)

    # padding -1 less than packet sum
    size_sequence = up_size_sequence + down_size_sequence
    if len(size_sequence) < packet_sum:
            size_sequence += [PADDING] * (packet_sum - len(size_sequence))

    return size_sequence


def extract2csv(pcap_dir, csv_file):
    print(pcap_dir, csv_file)

    PACKET_SUM = 30

    f = open(csv_file, 'w', newline='')
    for pcap in os.listdir(pcap_dir):
        pcap_path = os.path.join(pcap_dir, pcap)
        # print(pcap_path)

        flow = extract_flow(pcap_path, PACKET_SUM)
        res = []

        tmp = packet_size(flow, PACKET_SUM)
        res += tmp

        # write res if not none
        if res:
            f_csv = csv.writer(f)
            f_csv.writerow(res)

        # print(res)

    f.close()


if __name__ == '__main__':

    # for training
    # pcap_archive = 'Desktop\\data\\dataset\\train\\domain'
    # csv_path = 'Desktop\\data\\dataset\\train\\domain_train_domeye.csv'
    # extract2csv(pcap_archive, csv_path)

    # pcap_archive = 'Desktop\\data\\dataset\\train\\normal'
    # csv_path = 'Desktop\\data\\dataset\\train\\normal_train_domeye.csv'
    # extract2csv(pcap_archive, csv_path)

    # for tpr testing
    # meek moat snowflake
    # pcap_archive = 'Desktop\\data\\dataset\\test_tpr\\meek'
    # csv_path = 'Desktop\\data\\dataset\\test_tpr\\test_meek_domeye.csv'
    # extract2csv(pcap_archive, csv_path)

    # for fpr testing
    pcap_archive = 'Desktop\\data\\dataset\\test_fpr\\data'
    for archive in os.listdir(pcap_archive):
        pcap_dir = os.path.join(pcap_archive, archive)
        csv_path = 'Desktop\\data\\dataset\\test_fpr\\DomEye\\' + archive + '.csv'
        # print(pcap_dir, csv_path)
        # extract2csv(pcap_dir, csv_path)

        number = len(os.listdir(pcap_dir))
        print(number, archive)


