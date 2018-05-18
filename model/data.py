# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import csv
import numpy as np


def read_from_csv(datafile_path):
    """
    :return: samples and labels in ndarray
    """
    # I know there're 2930 samples and 442 features in dataset.csv
    # But we still have to write some code to get the number
    a = open(datafile_path, 'r', encoding="gbk")
    num_of_sample = len(a.readlines()) - 1
    reader = csv.reader(open(datafile_path, encoding='gbk'))
    columns = 0
    for row in reader:
        columns = len(row)
        break

    all_data = np.zeros([num_of_sample, columns])
    bleed_label = np.zeros([num_of_sample, 2])  # bleed happen=(1, 0), not happen=(0, 1)
    ischemic_label = np.zeros([num_of_sample, 2])  # ischemic happen=(1, 0), not happen=(0, 1)

    # First line in the file is feature name, ignore it.
    line = -1
    for row in reader:
        line += 1
        if line > 0:
            all_data[line - 1, 0:columns] = row[0:columns]

    for i in range(num_of_sample):
        if all_data[i, 0] == 0:
            ischemic_label[i, 1] = 1
        else:
            ischemic_label[i, 0] = 1

        if all_data[i, 1] == 0:
            bleed_label[i, 1] = 1
        else:
            bleed_label[i, 0] = 1

    # First 2 columns are labels
    sample = np.zeros([num_of_sample, columns - 2])  # only samples
    for i in range(num_of_sample):
        sample[i, 0:columns - 2] = all_data[i, 2:columns]

    # Scale the feature values
    for i in range(3, 110):
        array = sample[:, i]
        max_value = max(array)
        min_value = min(array)
        scale = max_value - min_value
        for j in range(num_of_sample):
            sample[j, i] -= min_value
            sample[j, i] /= scale

    return sample, bleed_label, ischemic_label


if __name__ == "__main__":
    samples, bleed_labels, ischemic_labels = read_from_csv("../res/dataset.csv")
