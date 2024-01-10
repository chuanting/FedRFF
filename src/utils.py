# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 15:02
# @Author       : tl22089
# @File         : utils.py
# @Affiliation  : University of Bristol
import copy
from collections import deque
import numpy as np
import torch
import pandas as pd
from numpy.random import standard_normal, uniform


def gaussian_noise(data, snr_range):
    arr = np.zeros(data.shape, dtype=complex)
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0], snr_range[-1], pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        P = sum(abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = np.sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
        arr[pktIdx] = s + n

    return arr


def add_noise(data, args, flag=True):
    data_real = data.iloc[:, :1024]
    data_imag = data.iloc[:, 1024:]

    n = len(data_real)
    arr = np.zeros(data_real.shape, dtype=complex)
    for idx in range(n):
        dp = data_real.iloc[idx].values + 1j * data_imag.iloc[idx].values
        # print(dp)
        arr[idx] = dp
    if args.noise and flag:
        arr = gaussian_noise(arr, snr_range=(args.snr_low, args.snr_high))

    c1 = arr.real
    c2 = arr.imag
    c3 = abs(arr)
    c = np.concatenate([c1, c2, c3], axis=1)
    return c


def data_iid(data, num_users):
    num_items = int(len(data)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(data))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def filter_signal(data, sig):
    data = data[data['radio'] == sig]
    bs_code = data[['radio', 'radio_code', 'bs', 'bs_code']].drop_duplicates()
    print(bs_code)
    data_sig = data.drop(['radio', 'radio_code', 'bs'], axis=1)
    return data_sig


def get_dataset(args):
    data_dir = '../src/'
    train = pd.read_parquet(data_dir + 'All_train.gzip')
    adapt = pd.read_parquet(data_dir + 'All_adapt.gzip')
    test = pd.read_parquet(data_dir + 'All_test.gzip')

    train = filter_signal(train, args.signal)
    adapt = filter_signal(adapt, args.signal)
    test = filter_signal(test, args.signal)

    train_x = train.iloc[:, :-1]
    train_y = train.iloc[:, -1]

    adapt_x = adapt.iloc[:, :-1]
    adapt_y = adapt.iloc[:, -1]

    test_x = test.iloc[:, :-1]
    test_y = test.iloc[:, -1]

    train_samples = add_noise(train_x, args, flag=True)
    adapt_samples = add_noise(adapt_x, args, flag=True)
    test_samples = add_noise(test_x, args, flag=False)

    mean = train_samples.ravel().mean()
    std = train_samples.ravel().std()

    train_samples = (train_samples - mean) / std
    adapt_samples = (adapt_samples - mean) / std
    test_samples = (test_samples - mean) / std

    train_samples = np.append(train_samples, train_y.values[:, np.newaxis], axis=1)
    adapt_samples = np.append(adapt_samples, adapt_y.values[:, np.newaxis], axis=1)
    test_samples = np.append(test_samples, test_y.values[:, np.newaxis], axis=1)

    train_groups = data_iid(train_samples, args.num_users)
    adapt_groups = data_iid(adapt_samples, args.num_users)

    return train_samples, adapt_samples, test_samples, train_groups, adapt_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


