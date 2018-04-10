# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import numpy as np
import csv

filename = 'dataset.csv'
csvFile = open(filename, 'rb')
NUM_OF_SAMPLE = len(csvFile.readlines()) - 1  # 样本总数为.csv文件行数-1
NUM_OF_FEATURE = 442  # feature总数为442个

input_tensor = np.zeros([NUM_OF_SAMPLE, NUM_OF_FEATURE])
bleeding = np.zeros([NUM_OF_SAMPLE, 1])
ischemic = np.zeros([NUM_OF_SAMPLE, 1])

reader = csv.reader(open(filename, encoding='utf-8'))

# 将数据写入input_tensor，注意表格中的第一行和前两列
lineNo = -1
for row in reader:
    lineNo += 1
    if lineNo > 0:
        input_tensor[lineNo - 1:lineNo, 0:NUM_OF_FEATURE] = row[2:444]

# 将label值存入bleeding和ischemic
for i in range(NUM_OF_SAMPLE):
    if input_tensor[i:i + 1, 7:8] == 1:  # 如果有缺血事件发生则第1位写上1，没有在第0位上写上1
        ischemic[i:i + 1, 1:2] = 1
    else:
        ischemic[i:i + 1, 0:1] = 1

    if input_tensor[i:i + 1, 8:9] == 1:  # 如果有出血事件发生则第1位写上1，没有在第0位上写上1
        bleeding[i:i + 1, 1:2] = 1
    else:
        bleeding[i:i + 1, 0:1] = 1
