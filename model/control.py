# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'
"""
UI界面与模型之间的交互操作都包含在本模块中
"""


# 将选择的文件的路径传到函数中，返回f1_score等结果
def calc_numerical_result(dataset_file_path):
    f1_score = "f1_score"
    precision = "precision"
    recall = "recall"
    return f1_score, precision, recall


def get_sample_info(dataset_file_path):
    num_of_sample = str(2930) + ' samples'
    num_of_feature = str(422) + ' features'
    num_of_ischemic = str(45) + ' cases'
    num_of_bleed = str(156) + ' cases'
    return num_of_sample, num_of_feature, num_of_ischemic, num_of_bleed
