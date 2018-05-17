# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import itertools
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from model.sdae import SDAE
from model.utils import *


# LR train as benchmark
def lr_experiment(dataset_path, epoch, learning_rate):
    """
    :param dataset_path: <string>
    :param epoch: <string>
    :param learning_rate: <string>
    :return:
    """
    learning_rate = float(learning_rate)
    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)
    n_class = 2
    n_feature = len(sample[0])
    bleeding_loss = []
    ischemic_loss = []

    # Collect 50 loss values regardless of the value of epoch
    # 最后采集的loss点可能会多一个，比如sample_quantity为101
    # step为2，最后会采集到0, 2, 4, ..., 100共51个点
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
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    with tf.Session() as sess:
        split_label = bleed_label[:, 0]
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        count = 0
        all_y_test = []
        all_p = []
        for train_index, test_index in kf.split(sample, split_label):
            sess.run(tf.global_variables_initializer())
            count += 1

            x_train = sample[train_index]
            y_train = bleed_label[train_index]

            x_test = sample[test_index]
            y_test = bleed_label[test_index]

            if count == 1:
                all_y_test = y_test
            else:
                all_y_test = np.append(all_y_test, y_test, axis=0)
            for i in range(epoch):
                _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_train, y_: y_train})
                if i % step == 0:
                    print(loss, i)
                    bleeding_loss.append(loss)
            # 可能会多一个
            if len(bleeding_loss) % sample_quantity > 0:
                bleeding_loss = bleeding_loss[:-1]

            p = sess.run(pred, feed_dict={x: x_test})
            if count == 1:
                all_p = p
            else:
                all_p = np.append(all_p, p, axis=0)

        bleeding_result = evaluate(all_y_test, all_p)
        draw_event_graph(bleeding_result, event="Bleeding events", model="lr")

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
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    with tf.Session() as sess:
        split_label = ischemic_label[:, 0]
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        count = 0
        all_y_test = []
        all_p = []

        for train_index, test_index in kf.split(sample, split_label):
            sess.run(tf.global_variables_initializer())
            count += 1
            x_train = sample[train_index]
            y_train = ischemic_label[train_index]

            x_test = sample[test_index]
            y_test = ischemic_label[test_index]

            if count == 1:
                all_y_test = y_test
            else:
                all_y_test = np.append(all_y_test, y_test, axis=0)

            for i in range(epoch):
                _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_train, y_: y_train})
                if i % step == 0:
                    print(loss, i)
                    ischemic_loss.append(loss)
            if len(ischemic_loss) % sample_quantity > 0:
                ischemic_loss = ischemic_loss[:-1]
            p = sess.run(pred, feed_dict={x: x_test})
            if count == 1:
                all_p = p
            else:
                all_p = np.append(all_p, p, axis=0)

        ischemic_result = evaluate(all_y_test, all_p)
        draw_event_graph(ischemic_result, event="Ischemic events", model="lr")
    draw_loss_curve(bleeding_loss, ischemic_loss, epoch, sample_quantity)


# Do SDAE train
def sdae_experiment(dataset_path, epoch, hiddens, learning_rate):
    """
    :param dataset_path: <string>
    :param epoch: <string>
    :param hiddens: <list>
    :param learning_rate: <string>
    """
    epoch = int(epoch)
    learning_rate = float(learning_rate)
    hidden_layers = []
    for i in hiddens:
        hidden_layers.append(int(i))

    # Get samples and labels
    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)
    origin_n_input = len(sample[0])
    # Binary classification
    n_class = 2
    # loss曲线采样数量
    sample_quantity = 20
    bleeding_loss = []
    ischemic_loss = []

    # Bleeding event
    with tf.Session() as sess:
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        split_label = bleed_label[:, 0]
        # count用来计算当前是K折中的哪一折
        count = 0
        all_y_test = []
        all_p = []
        sdae = SDAE(origin_n_input, hidden_layers, n_class=n_class, sess=sess)

        for train_index, test_index in kf.split(sample, split_label):
            sess.run(tf.global_variables_initializer())
            count += 1
            # Get train set and test set
            x_train = sample[train_index]
            y_train = bleed_label[train_index]
            x_test = sample[test_index]
            y_test = bleed_label[test_index]

            if count == 1:
                all_y_test = y_test
            else:
                all_y_test = np.append(all_y_test, y_test, axis=0)

            sdae.train_model(x_train=x_train, y_train=y_train, x_test=x_test, epochs=epoch, learning_rate=learning_rate,
                             sample_quantity=sample_quantity)
            p = sdae.get_pred()

            if count == 1:
                all_p = p
            else:
                all_p = np.append(all_p, p, axis=0)
            bleeding_loss.append(sdae.get_loss())
        bleeding_result = evaluate(all_y_test, all_p)
        draw_event_graph(bleeding_result, event="Bleeding events", model="sdae")

    # # Ischemic events
    with tf.Session() as sess:
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        split_label = ischemic_label[:, 0]
        count = 0
        all_y_test = []
        all_p = []
        sdae = SDAE(origin_n_input, hidden_layers, n_class=n_class, sess=sess)

        for train_index, test_index in kf.split(sample, split_label):
            sess.run(tf.global_variables_initializer())
            count += 1
            # Get train set and test set
            x_train = sample[train_index]
            y_train = ischemic_label[train_index]
            x_test = sample[test_index]
            y_test = ischemic_label[test_index]

            if count == 1:
                all_y_test = y_test
            else:
                all_y_test = np.append(all_y_test, y_test, axis=0)

            sdae.train_model(x_train=x_train, y_train=y_train, x_test=x_test, epochs=epoch, learning_rate=learning_rate,
                             sample_quantity=sample_quantity)
            p = sdae.get_pred()

            if count == 1:
                all_p = p
            else:
                all_p = np.append(all_p, p, axis=0)
            ischemic_loss.append(sdae.get_loss())
        ischemic_result = evaluate(all_y_test, all_p)
        draw_event_graph(ischemic_result, event="Ischemic events", model="sdae")
    bleeding_loss = list(itertools.chain.from_iterable(bleeding_loss))
    ischemic_loss = list(itertools.chain.from_iterable(ischemic_loss))
    draw_loss_curve(bleeding_loss, ischemic_loss, epoch, sample_quantity)


if __name__ == "__main__":
    hidden = [128, 64]
    sdae_experiment("../res/dataset.csv", epoch=40, hiddens=hidden, learning_rate=0.001)
    # lr_experiment("C:/Users/ZM-BAD/Projects/ACSclassifier/res/dataset.csv", 500, 0.001)
