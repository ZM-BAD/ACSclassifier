# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from matplotlib_venn import venn2
from model.data import read_from_csv


def is_float_number(number):
    try:
        float(number)
        return True
    except:
        return False


# Calculate acc, auc, f1-score, recall, precision
def evaluate(tol_label, tol_pred):
    """
    Evaluate the predictive performance of the model
    :param tol_label: real labels for test samples
    :param tol_pred: predicted probability distribution of test samples
    :return: acc, auc, f1-score, recall, precision
    """
    assert tol_label.shape == tol_pred.shape

    y_true = np.argmax(tol_label, axis=1)
    y_pred = np.argmax(tol_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(tol_label, tol_pred, average=None)

    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f_score = f1_score(y_true, y_pred, average=None)

    return accuracy, auc, f_score, recall, precision


# Draw acc, auc, f1-score, recall, precision graph
def draw_event_graph(result, event, model):
    """
    :param result: <tuple> (acc, auc, f1-score, recall, precision)
    :param event: <string> bleeding event or ischemic event
    :param model: <string> lr HotPink, sdae CornflowerBlue
    :return:
    """
    plt.figure(figsize=(4, 3.35), dpi=100)
    plt.grid(True)

    pic_name = event + ".png"

    color = "CornflowerBlue"
    if model == "lr":
        color = "HotPink"

    result = (result[0], result[1][0], result[2][0], result[3][0], result[4][0])
    plt.bar(range(len(result)), result, color=color)
    plt.xticks(range(len(result)), (u"ACC", u"AUC", u"F1-score", u"Recall", u"Precision"))
    for a, b in zip(range(len(result)), result):
        plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom', fontsize=8)

    if not os.path.exists("../res/output"):
        os.mkdir("../res/output")
    plt.savefig("../res/output/" + pic_name)


# Handle the loss array
def weighted_mean(loss, k):
    new_length = len(loss) / k
    new_loss = [0] * int(new_length)

    for i in range(len(loss)):
        index = int(i % new_length)
        new_loss[index] += loss[i]

    for i in new_loss:
        i /= k

    return new_loss


# Draw loss curve
def draw_loss_curve(bleeding_loss, ischemic_loss, bleeding_epoch, ischemic_epoch, sample_quantity, k=5):
    bleeding_loss = weighted_mean(bleeding_loss, k=k)
    ischemic_loss = weighted_mean(ischemic_loss, k=k)
    bleeding_step = bleeding_epoch // sample_quantity
    bleeding_x = []
    for i in range(sample_quantity):
        bleeding_x.append(i * bleeding_step)

    ischemic_step = ischemic_epoch // sample_quantity
    ischemic_x = []
    for i in range(sample_quantity):
        ischemic_x.append(i * ischemic_step)

    fig = plt.figure(figsize=(4, 3.35), dpi=100)
    ax = fig.add_subplot(111)

    plt.xlabel("Epoch")
    plt.ylabel('Loss Value')

    plt.plot(bleeding_x, bleeding_loss, 'r', label='bleeding')
    plt.plot(ischemic_x, ischemic_loss, 'b', label='ischemic')
    plt.xticks(np.linspace(0, max(bleeding_epoch, ischemic_epoch), 10, endpoint=True), fontsize='xx-small')
    plt.yticks(fontsize='xx-small')

    plt.legend(loc='upper right', fontsize='large', frameon=False)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    if not os.path.exists("../res/output"):
        os.mkdir("../res/output")
    plt.savefig("../res/output/loss_curve.png")


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
