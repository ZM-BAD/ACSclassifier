# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from model.data import read_from_csv
from model.sdae import SDAE


def evaluate(real, pred):
    auc = 0
    f1_score = 0
    recall = 0
    precision = 0

    return auc, f1_score, recall, precision


def draw_event_graph(result, event, model):
    """
    :param result: tuple (auc, f1-score, recall, precision)
    :param event: bleeding event or ischemic event
    :param model: lr HotPink, sdae CornflowerBlue
    :return:
    """

    pass


def draw_loss_curve(bleeding_loss, ischemic_loss, epochs):

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


# Do LR train
def lr_experiment(dataset_path, epoch):
    """
    :param dataset_path: <string>
    :param epoch: <string>
    :return:
    """
    print("I am here")
    #
    # sample, bleed_label, ischemic_label = read_from_csv(dataset_path)
    # n_class = 2
    # n_feature = len(sample[0])
    # bleeding_loss = []
    # ischemic_loss = []
    #
    # # Bleeding events
    # sess = tf.InteractiveSession()
    # x = tf.placeholder(tf.float32, [None, n_feature])
    #
    # W = tf.Variable(tf.zeros([n_feature, n_class]))
    # b = tf.Variable(tf.zeros([n_class]))
    #
    # # y is prediction
    # y = tf.matmul(x, W) + b
    # pred = tf.nn.sigmoid(y)
    #
    # # y_ is real
    # y_ = tf.placeholder(tf.float32, [None, n_class])
    # cross_entropy = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_, y))
    # train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    # tf.global_variables_initializer.run()
    #
    # x_train, x_test, y_train, y_test = train_test_split(sample, bleed_label, test_size=0.3, random_state=0)
    #
    # for i in range(epoch):
    #     _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_train, y_: y_train})
    #     if i % 50 == 0:
    #         print(loss)
    #         bleeding_loss.append(loss)
    #
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    #
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    # print(accuracy.eval({x: x_test, y_: y_test}))

    # bleeding_result = evaluate(real=None, pred=None)
    # draw_event_graph(bleeding_result, event="bleeding", model="lr")
    #
    # # Ischemic events
    # ischemic_result = evaluate(real=None, pred=None)
    # draw_event_graph(ischemic_result, event="ischemic", model="lr")
    #
    # draw_loss_curve(bleeding_loss, ischemic_loss, epoch)


# Do SDAE train
def sdae_experiment(dataset_path, epoch, hiddens):
    """
    :param dataset_path: <string>
    :param epoch: <string>
    :param hiddens: <list>
    :return:
    """
    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)
    n_input = len(sample[0])
    sdae = SDAE(n_input, hiddens)

