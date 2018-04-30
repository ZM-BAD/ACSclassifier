# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import time
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from model.data import read_from_csv

"""
UI界面与模型之间的交互操作都包含在本模块中
"""


def draw_event_graph():
    pass


# Draw venn diagram
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
    plt.figure(figsize=(4, 3.35), dpi=100)
    venn2(subsets=(num_of_bleed - both, num_of_ischemic - both, both), set_labels=('bleeding', 'ischemic'))
    string = 'A total of ' + str(num_of_sample) + ' samples with ' + str(num_of_feature) + ' features'
    plt.title(string)
    plt.savefig("../res/venn.png")


# 将选择的文件的路径传到函数中，返回f1_score等结果
def calc_numerical_result(dataset_file_path):
    auc = "fuck auc"
    f1_score = "f1_score"
    precision = "precision"
    recall = "recall"
    return auc, f1_score, precision, recall


# Do LR train
def lr_experiment(dataset_path, epoch):
    """
    :param dataset_path: <string>
    :param epoch: <string>
    :return:
    """
    time.sleep(5)
    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)

    pass


# Do SDAE train
def sdae_experiment(dataset_path, epoch, hiddens):
    """
    :param dataset_path: <string>
    :param epoch: <string>
    :param hiddens: <list>
    :return:
    """
    print(dataset_path)
    print(epoch)
    print(hiddens)
    time.sleep(5)


if __name__ == "__main__":
    draw_sample_info_statistics("C:/Users/ZM-BAD/Projects/ACSclassifier/res/dataset.csv")
