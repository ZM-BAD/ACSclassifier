# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score


def evaluate(real_label, pred_label):
    """
    Evaluate the predictive performance of the model
    :param real_label:
    :param pred_label:
    :return: f1-score, recall, precision, auc
    """
    y_true = np.argmax(real_label, axis=2)
    y_pred = np.argmax(pred_label, axis=2)

    f_score = f1_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    auc = roc_auc_score(real_label, pred_label, average=None)

    return f_score, recall, precision, auc
