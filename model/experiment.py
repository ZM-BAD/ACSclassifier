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
    There're 2 LR experiments, bleeding event and ischemic event.
    :param dataset_path: <string>
    :param epoch: <string>
    :param learning_rate: <string>
    :return:
    """
    # Collect 50 loss values
    sample_quantity = 50
    epoch = int(epoch)
    learning_rate = float(learning_rate)
    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)
    bleeding_loss = lr_train(sample, bleed_label, epoch, learning_rate, sample_quantity, event='bleeding')
    ischemic_loss = lr_train(sample, ischemic_label, epoch, learning_rate, sample_quantity, event='ischemic')

    draw_loss_curve(bleeding_loss, ischemic_loss, epoch, sample_quantity)


def lr_train(sample, label, epoch, learning_rate, sample_quantity, event):
    """
    Do a single train
    :param sample:
    :param label:
    :param epoch:
    :param learning_rate:
    :param sample_quantity:
    :param event: <string>, 'bleeding' or 'ischemic'
    :return: loss array
    """
    n_class = 2
    n_feature = len(sample[0])
    loss_points = []

    # epoch must larger than sample quantity
    # 最后采集的loss点可能会多一个，比如sample_quantity为101, step为2，最后会采集到0, 2, 4, ..., 100共51个点

    step = epoch // sample_quantity

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
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        split_label = label[:, 0]
        # count用来计算当前是K折中的哪一折
        count = 0
        all_y_test = []
        all_p = []
        for train_index, test_index in kf.split(sample, split_label):
            sess.run(tf.global_variables_initializer())
            count += 1

            x_train = sample[train_index]
            y_train = label[train_index]
            x_test = sample[test_index]
            y_test = label[test_index]

            if count == 1:
                all_y_test = y_test
            else:
                all_y_test = np.append(all_y_test, y_test, axis=0)

            for i in range(epoch):
                _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_train, y_: y_train})
                if i % step == 0:
                    print(loss, i)
                    loss_points.append(loss)
            # 可能会多一个
            if len(loss_points) % sample_quantity > 0:
                loss_points = loss_points[:-1]

            p = sess.run(pred, feed_dict={x: x_test})
            if count == 1:
                all_p = p
            else:
                all_p = np.append(all_p, p, axis=0)

        result = evaluate(all_y_test, all_p)
        draw_event_graph(result, event=event, model="lr")

    return loss_points


# SDAE train
def sdae_experiment(dataset_path, epoch, hidden_layers, learning_rate):
    """
    There're 2 SDAE experiments, bleeding event and ischemic event.
    :param dataset_path: <string>
    :param epoch: <string>
    :param hidden_layers: <list>
    :param learning_rate: <string>
    """
    epoch = int(epoch)
    learning_rate = float(learning_rate)
    hiddens = []
    for i in hidden_layers:
        hiddens.append(int(i))

    # Get samples and labels
    sample, bleed_label, ischemic_label = read_from_csv(dataset_path)

    # Collect 50 loss values
    sample_quantity = 50

    bleeding_loss = sdae_train(sample, bleed_label, epoch, hiddens, learning_rate, sample_quantity, event='bleeding')
    ischemic_loss = sdae_train(sample, ischemic_label, epoch, hiddens, learning_rate, sample_quantity, event='ischemic')

    draw_loss_curve(bleeding_loss, ischemic_loss, epoch, sample_quantity)


def sdae_train(sample, label, epoch, hidden_layers, learning_rate, sample_quantity, event):
    """
    Do a single SDAE train
    :param sample:
    :param label:
    :param epoch:
    :param hidden_layers:
    :param learning_rate:
    :param sample_quantity:
    :param event: <string> 'bleeding' or 'ischemic'
    :return: loss array
    """
    origin_n_input = len(sample[0])
    n_class = 2
    loss_points = []

    with tf.Session() as sess:
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        split_label = label[:, 0]
        # count用来计算当前是K折中的哪一折
        count = 0
        all_y_test = []
        all_p = []
        sdae = SDAE(origin_n_input, hidden_layers, n_class=n_class, sess=sess, learning_rate=learning_rate)

        for train_index, test_index in kf.split(sample, split_label):
            sess.run(tf.global_variables_initializer())
            count += 1

            x_train = sample[train_index]
            y_train = label[train_index]
            x_test = sample[test_index]
            y_test = label[test_index]

            if count == 1:
                all_y_test = y_test
            else:
                all_y_test = np.append(all_y_test, y_test, axis=0)

            sdae.train_model(x_train=x_train, y_train=y_train, x_test=x_test, epochs=epoch,
                             sample_quantity=sample_quantity)
            loss_points.append(sdae.get_loss())

            p = sdae.get_pred()
            if count == 1:
                all_p = p
            else:
                all_p = np.append(all_p, p, axis=0)

        result = evaluate(all_y_test, all_p)
        draw_event_graph(result, event=event, model="sdae")

    loss_points = list(itertools.chain.from_iterable(loss_points))
    return loss_points


if __name__ == "__main__":
    hidden = [8, 4]
    sdae_experiment("../res/dataset.csv", epoch=40, hidden_layers=hidden, learning_rate=0.001)
    # lr_experiment("C:/Users/ZM-BAD/Projects/ACSclassifier/res/dataset.csv", 500, 0.001)
