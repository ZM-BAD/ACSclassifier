# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
from model.data import read_from_csv

"""
UI界面与模型之间的交互操作都包含在本模块中
"""


def draw_sample_info_statistics(file_path):
    sample, bleed_label, ischemic_label = read_from_csv(file_path)

    num_of_bleed = 0
    num_of_ischemic = 0
    both = 0
    for i in range(len(sample)):
        if bleed_label[i, 0] == 1:
            num_of_bleed += 1
        if ischemic_label[i, 0] == 1:
            num_of_ischemic += 1
        if bleed_label[i, 0] == 1 and ischemic_label[i, 0] == 1:
            both += 1

    num_of_sample = len(sample)
    num_of_feature = len(sample[0])
    num_of_ischemic = num_of_ischemic
    num_of_bleed = num_of_bleed
    plt.figure(figsize=(4, 3.6), dpi=100)
    venn2(subsets=(num_of_bleed - both, num_of_ischemic - both, both), set_labels=('bleeding', 'ischemic'))
    string = 'A total of ' + str(num_of_sample) + ' samples with ' + str(num_of_feature) + ' features'
    plt.title(string)
    plt.savefig("../res/venn.png")


# 将选择的文件的路径传到函数中，返回f1_score等结果
def calc_numerical_result(dataset_file_path):
    f1_score = "f1_score"
    precision = "precision"
    recall = "recall"
    return f1_score, precision, recall


def get_sample_info(dataset_path):
    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)

    num_of_bleed = 0
    num_of_ischemic = 0
    for i in range(len(sample)):
        if bleed_label[i, 0] == 1:
            num_of_bleed += 1
        if ischemic_label[i, 0] == 1:
            num_of_ischemic += 1

    num_of_sample = str(len(sample)) + ' samples'
    num_of_feature = str(len(sample[0])) + ' features'
    num_of_ischemic = str(num_of_ischemic) + ' cases'
    num_of_bleed = str(num_of_bleed) + ' cases'
    return num_of_sample, num_of_feature, num_of_ischemic, num_of_bleed


if __name__ == "__main__":
    draw_sample_info_statistics("C:/Users/ZM-BAD/Projects/ACSclassifier/res/dataset.csv")
