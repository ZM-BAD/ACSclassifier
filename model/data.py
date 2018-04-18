# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import csv

import numpy as np


def read_from_csv():
    """
    :return: samples and labels in ndarray
    """
    # There're 2930 samples and 442 features in dataset.csv
    reader = csv.reader(open('../resource/dataset.csv', encoding='gbk'))
    all_data = np.zeros([2930, 444])
    bleed_label = np.zeros([2930, 2])  # bleed happen=(1, 0), not happen=(0, 1)
    ischemic_label = np.zeros([2930, 2])  # ischemic happen=(1, 0), not happen=(0, 1)

    # First line in the file is feature name, ignore it.
    line = -1
    for row in reader:
        line += 1
        if line > 0:
            all_data[line - 1, 0:444] = row[0:444]

    for i in range(2930):
        if all_data[i, 0] == 0:
            ischemic_label[i, 1] = 1
        else:
            ischemic_label[i, 0] = 1

        if all_data[i, 1] == 0:
            bleed_label[i, 1] = 1
        else:
            bleed_label[i, 0] = 1

    # First 2 columns are labels
    sample = np.zeros([2930, 442])  # only samples
    for i in range(2930):
        sample[i, 0:442] = all_data[i, 2:444]

    return sample, bleed_label, ischemic_label


if __name__ == "__main__":
    samples, bleed_labels, ischemic_labels = read_from_csv()
    print(samples)
    print(bleed_labels)
    print(ischemic_labels)
