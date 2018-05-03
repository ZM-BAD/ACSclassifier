# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from matplotlib_venn import venn2
from model.data import read_from_csv
from model.sdae import SDAE


# TODO: Calculate acc, auc, f1-score, recall, precision
# TODO: Add SDAE model
# TODO: Debug the whole fucking system

# Calculate acc, auc, f1-score, recall, precision
def evaluate(real, pred):
    r = np.reshape(real, [-1])
    p = np.reshape(pred, [-1])
    acc = accuracy_score(r, p)
    auc = roc_auc_score(r, p)
    f_one_score = f1_score(r, p, average='weighted')
    recall = recall_score(r, p, average='macro')
    precision = precision_score(r, p, average='macro')

    return acc, auc, f_one_score, recall, precision


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
    if event == "Bleeding events":
        pic_name = "bleeding.png"
    else:
        pic_name = "ischemic.png"

    if model == "lr":
        color = "HotPink"
    else:
        color = "CornflowerBlue"

    plt.bar(range(len(result)), result, color=color)
    plt.xticks(range(len(result)), (u"ACC", u"AUC", u"F1-score", u"Recall", u"Precision"))
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

        x_train, x_test, y_train, y_test = train_test_split(sample, bleed_label, test_size=0.3, random_state=0)

        for i in range(epoch):
            _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_train, y_: y_train})
            if i % step == 0:
                print(loss, i)
                bleeding_loss.append(loss)

        p = sess.run(pred, feed_dict={x: x_test})
        for i in range(len(p)):
            if p[i, 0] >= 0.5:
                p[i, 0] = 1
            else:
                p[i, 0] = 0
            if p[i, 1] >= 0.5:
                p[i, 1] = 1
            else:
                p[i, 1] = 0

        bleeding_result = evaluate(real=y_test, pred=p)
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
        for i in range(len(p)):
            if p[i, 0] >= 0.5:
                p[i, 0] = 1
            else:
                p[i, 0] = 0
            if p[i, 1] >= 0.5:
                p[i, 1] = 1
            else:
                p[i, 1] = 0

        ischemic_result = evaluate(real=y_test, pred=p)
        draw_event_graph(ischemic_result, event="Ischemic events", model="lr")

    if len(bleeding_loss) > sample_quantity:
        bleeding_loss = bleeding_loss[:-1]
    if len(ischemic_loss) > sample_quantity:
        ischemic_loss = ischemic_loss[:-1]
    draw_loss_curve(bleeding_loss, ischemic_loss, epoch, sample_quantity)


# Do SDAE train
def sdae_experiment(dataset_path, epoch, hiddens):
    """
    :param dataset_path: <string>
    :param epoch: <string>
    :param hiddens: <list>
    :return:
    """
    epoch = int(epoch)
    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)
    origin_n_input = len(sample[0])
    n_class = 2
    # 抽取后的feature数量
    extract_feature_n = hiddens[-1]
    # loss曲线的采样数量
    sample_quantity = 50
    step = sample_quantity // epoch
    bleeding_loss = []
    ischemic_loss = []

    # Bleeding events
    x_train, x_test, y_train, y_test = train_test_split(sample, bleed_label, test_size=0.3, random_state=0)
    # SDAE本质上是无监督的，所以不需要label，训练SDAE用x_train即可
    # 对特征抽取后，有一层Softmax，但是对于二分类而言，Softmax退化为LR
    # LR是监督学习，需要训练。LR训练的样本是x_train抽取出来的x_extract_train，而样本标签依旧为y_train
    print("1")
    sdae = SDAE(origin_n_input, hiddens)
    print("2")
    sdae.pre_train(x_train)
    print("3")
    x_extract_train = sdae.encode(x_train)
    print("4")
    x_extract_test = sdae.encode(x_test)
    print("5")
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
        for i in range(epoch):
            _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_extract_train, y_: y_train})
            if i % step == 0:
                bleeding_loss.append(loss)

        p = sess.run(pred, feed_dict={x: x_extract_test})
        for i in range(len(p)):
            if p[i, 0] >= 0.5:
                p[i, 0] = 1
            else:
                p[i, 0] = 0
            if p[i, 1] >= 0.5:
                p[i, 1] = 1
            else:
                p[i, 1] = 0

        bleeding_result = evaluate(real=y_test, pred=p)
        draw_event_graph(bleeding_result, event="Bleeding events", model="sdae")

    ##########################################################################
    # Ischemic events
    x_train, x_test, y_train, y_test = train_test_split(sample, ischemic_label, test_size=0.3, random_state=0)
    sdae.pre_train(x_train)
    x_extract_train = sdae.encode(x_train)
    x_extract_test = sdae.encode(x_test)
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

        for i in range(epoch):
            _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_extract_train, y_: y_train})
            if i % step == 0:
                print(loss, i)
                ischemic_loss.append(loss)

        p = sess.run(pred, feed_dict={x: x_extract_test})
        for i in range(len(p)):
            if p[i, 0] >= 0.5:
                p[i, 0] = 1
            else:
                p[i, 0] = 0
            if p[i, 1] >= 0.5:
                p[i, 1] = 1
            else:
                p[i, 1] = 0
        ischemic_result = evaluate(real=y_test, pred=p)
        draw_event_graph(ischemic_result, event="Ischemic events", model="sdae")
    if len(bleeding_loss) > sample_quantity:
        bleeding_loss = bleeding_loss[:-1]
    if len(ischemic_loss) > sample_quantity:
        ischemic_loss = ischemic_loss[:-1]
    draw_loss_curve(bleeding_loss, ischemic_loss, epoch, sample_quantity)


if __name__ == "__main__":
    hiddens = [256, 128, 64]
    sdae_experiment("C:/Users/ZM-BAD/Projects/ACSclassifier/res/dataset.csv", 1000, hiddens)
