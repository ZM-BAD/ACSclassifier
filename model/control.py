# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from matplotlib_venn import venn2
from model.data import read_from_csv
from model.sdae import SDAE


# Calculate acc, auc, f1-score, recall, precision
def evaluate(tol_label, tol_pred):
    """
    对模型的预测性能进行评估
    :param tol_label: 测试样本的真实标签
    :param tol_pred: 测试样本的预测概率分布
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

    pic_name = "ischemic.png"
    if event == "Bleeding events":
        pic_name = "bleeding.png"

    color = "CornflowerBlue"
    if model == "lr":
        color = "HotPink"

    result = (result[0], result[1][0], result[2][0], result[3][0], result[4][0])
    plt.bar(range(len(result)), result, color=color)
    plt.xticks(range(len(result)), (u"ACC", u"AUC", u"F1-score", u"Recall", u"Precision"))

    if not os.path.exists("../res/output"):
        os.mkdir("../res/output")
    plt.savefig("../res/output/" + pic_name)


# Draw loss curve
def draw_loss_curve(bleeding_loss, ischemic_loss, epoch, sample_quantity):
    step = epoch // sample_quantity
    x = []
    for i in range(sample_quantity):
        x.append(i * step)

    fig = plt.figure(figsize=(4, 3.35), dpi=100)
    ax = fig.add_subplot(111)

    plt.xlabel("Epoch")
    plt.ylabel('Loss Value')

    plt.plot(x, bleeding_loss, 'r', label='bleeding')
    plt.plot(x, ischemic_loss, 'b', label='ischemic')
    plt.xticks(np.linspace(0, epoch, 10, endpoint=True), fontsize='xx-small')
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


# LR train as benchmark
def lr_experiment(dataset_path, epoch):
    """
    :param dataset_path: <string>
    :param epoch: <string>
    :return:
    """

    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)
    n_class = 2
    n_feature = len(sample[0])
    bleeding_loss = []
    ischemic_loss = []

    # Collect 50 loss values regardless of the value of epoch
    sample_quantity = 50
    epoch = int(epoch)
    step = epoch // sample_quantity

    # Bleeding events
    x = tf.placeholder(tf.float32, [None, n_feature])
    w = tf.Variable(tf.zeros([n_feature, n_class]))
    b = tf.Variable(tf.zeros([n_class]))

    # y is prediction
    y = tf.matmul(x, w) + b
    pred = tf.nn.softmax(y)

    # y_ is real
    y_ = tf.placeholder(tf.float32, [None, n_class])
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_, y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        split_label = bleed_label[:, 0]
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(sample, split_label):
            x_train = []
            y_train = []
            for i in train_index:
                x_train.append(sample[i])
                y_train.append(bleed_label[i])
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = []
            y_test = []
            for i in test_index:
                x_test.append(sample[i])
                y_test.append(bleed_label[i])
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            for i in range(epoch):
                _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_train, y_: y_train})
                if i % step == 0:
                    print(loss, i)
                    bleeding_loss.append(loss)

            p = sess.run(pred, feed_dict={x: x_test})

            bleeding_result = evaluate(y_test, p)
        draw_event_graph(bleeding_result, event="Bleeding events", model="lr")

    ##########################################################################
    # Ischemic events
    x = tf.placeholder(tf.float32, [None, n_feature])
    w = tf.Variable(tf.zeros([n_feature, n_class]))
    b = tf.Variable(tf.zeros([n_class]))

    # y is prediction
    y = tf.matmul(x, w) + b
    pred = tf.nn.softmax(y)

    # y_ is real
    y_ = tf.placeholder(tf.float32, [None, n_class])
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_, y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        x_train, x_test, y_train, y_test = train_test_split(sample, ischemic_label, test_size=0.3, random_state=0)

        for i in range(epoch):
            _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_train, y_: y_train})
            if i % step == 0:
                print(loss, i)
                ischemic_loss.append(loss)

        p = sess.run(pred, feed_dict={x: x_test})

        ischemic_result = evaluate(y_test, p)
        draw_event_graph(ischemic_result, event="Ischemic events", model="lr")

    if len(bleeding_loss) > sample_quantity:
        bleeding_loss = bleeding_loss[:-1]
    if len(ischemic_loss) > sample_quantity:
        ischemic_loss = ischemic_loss[:-1]
    draw_loss_curve(bleeding_loss, ischemic_loss, epoch, sample_quantity)


# Do SDAE train
def sdae_experiment(dataset_path, epoch, hiddens_str):
    """
    :param dataset_path: <string>
    :param epoch: <string>
    :param hiddens_str: <list>
    :return:
    """
    epoch = int(epoch)
    hiddens = []
    for i in hiddens_str:
        hiddens.append(int(i))
    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)
    origin_n_input = len(sample[0])
    n_class = 2
    # 抽取后的feature数量
    extract_feature_n = hiddens[-1]
    # loss曲线的采样数量
    sample_quantity = 50
    step = epoch // sample_quantity
    bleeding_loss = []
    ischemic_loss = []

    # Bleeding events
    x_train, x_test, y_train, y_test = train_test_split(sample, bleed_label, test_size=0.3, random_state=0)
    # SDAE本质上是无监督的，所以不需要label，训练SDAE用x_train即可
    # 对特征抽取后，有一层Softmax，但是对于二分类而言，Softmax退化为LR
    # LR是监督学习，需要训练。LR训练的样本是x_train抽取出来的x_extract_train，而样本标签依旧为y_train

    x = tf.placeholder(tf.float32, [None, extract_feature_n])
    w = tf.Variable(tf.zeros([extract_feature_n, n_class]))
    b = tf.Variable(tf.zeros([n_class]))

    # y is prediction
    y = tf.matmul(x, w) + b
    pred = tf.nn.softmax(y)

    # y_ is real
    y_ = tf.placeholder(tf.float32, [None, n_class])
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_, y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sdae = SDAE(origin_n_input, hiddens)
        sdae.pre_train(x_train)
        x_extract_train = sdae.encode(x_train)
        x_extract_test = sdae.encode(x_test)
        for i in range(epoch):
            _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_extract_train, y_: y_train})
            if i % step == 0:
                bleeding_loss.append(loss)

        p = sess.run(pred, feed_dict={x: x_extract_test})

        bleeding_result = evaluate(y_test, p)
        draw_event_graph(bleeding_result, event="Bleeding events", model="sdae")

    ##########################################################################
    # Ischemic events
    x_train, x_test, y_train, y_test = train_test_split(sample, ischemic_label, test_size=0.3, random_state=0)

    x = tf.placeholder(tf.float32, [None, extract_feature_n])
    w = tf.Variable(tf.zeros([extract_feature_n, n_class]))
    b = tf.Variable(tf.zeros([n_class]))

    # y is prediction
    y = tf.matmul(x, w) + b
    pred = tf.nn.softmax(y)

    # y_ is real
    y_ = tf.placeholder(tf.float32, [None, n_class])
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_, y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sdae = SDAE(origin_n_input, hiddens)
        sdae.pre_train(x_train)
        x_extract_train = sdae.encode(x_train)
        x_extract_test = sdae.encode(x_test)

        for i in range(epoch):
            _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_extract_train, y_: y_train})
            if i % step == 0:
                print(loss, i)
                ischemic_loss.append(loss)

        p = sess.run(pred, feed_dict={x: x_extract_test})

        ischemic_result = evaluate(y_test, p)
        draw_event_graph(ischemic_result, event="Ischemic events", model="sdae")

    if len(bleeding_loss) > sample_quantity:
        bleeding_loss = bleeding_loss[:-1]
    if len(ischemic_loss) > sample_quantity:
        ischemic_loss = ischemic_loss[:-1]
    draw_loss_curve(bleeding_loss, ischemic_loss, epoch, sample_quantity)


if __name__ == "__main__":
    hiddens = [256, 128, 64]
    sdae_experiment("C:/Users/ZM-BAD/Projects/ACSclassifier/res/dataset.csv", 1000, hiddens)
    # lr_experiment("C:/Users/ZM-BAD/Projects/ACSclassifier/res/dataset.csv", 1000)
