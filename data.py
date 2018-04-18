# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import csv

import numpy as np


# 从dataset.csv中读取数据
def read_from_csv():
    # dataset.csv文件中一共有
    # 2930个数据样本
    # 442个feature

    reader = csv.reader(open('./resource/dataset.csv', encoding='gbk'))
    data = np.zeros([2930, 444])  # 储存全部的数据，先把所有的数据都读进来
    label = np.zeros([2930, 4])  # 所有样本的标签

    line = -1
    for row in reader:
        line += 1
        if line > 0:
            data[line - 1, 0:444] = row[0:444]

    for i in range(2930):
        if data[i, 0] == 0 and data[i, 1] == 0:  # 不缺血，不出血
            label[i, 0] = 1
        elif data[i, 0] == 0 and data[i, 1] == 1:  # 不缺血，出血
            label[i, 1] = 1
        elif data[i, 0] == 1 and data[i, 1] == 0:  # 缺血，不出血
            label[i, 2] = 1
        else:  # 缺血，出血
            label[i, 3] = 1

    sample = np.zeros([2930, 442])  # 只存放样本，不含标签
    for i in range(2930):
        sample[i, 0:442] = data[2:444]
