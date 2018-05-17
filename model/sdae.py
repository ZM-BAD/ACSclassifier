# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import tensorflow as tf
from model.dae import DAE
from model.utils import *


# SDAE is based on DAE, which adds additive Gaussian noise
class SDAE(object):
    def __init__(self, n_input, hiddens, n_class=2, transfer_function=tf.nn.softplus, scale=0.1, name='sdae', sess=None,
                 optimizer=tf.train.AdamOptimizer()):
        """
        :param n_input: number of input nodes
        :param hiddens: list, number of nodes in every hidden layer
        :param n_class: number of final classification
        :param transfer_function: transfer(activation) function
        :param scale: Gaussian noise figure
        :param optimizer:
        :param name:
        :param sess:
        """
        self.name = name
        with tf.variable_scope(self.name):
            self.n_input = n_input
            self.n_class = n_class
            self.stacks = len(hiddens) + 1
            self.scale = scale
            self.transfer = transfer_function
            self.optimizer = optimizer
            self.loss = []
            self.p = None
            self.sess = sess or tf.Session()

            init = tf.global_variables_initializer()
            self.sess.run(init)

            # 网络结构
            self.sdae = self._init_sdae(self.n_input, hiddens)
            self.x = tf.placeholder(tf.float32, [None, n_input], name="input")
            self.hidden = self.sdae[self.stacks - 1].hidden

            self.rec = self.hidden
            for dae in reversed(self.sdae):
                self.rec = dae.decode_func(self.rec)

    def _init_sdae(self, n_input, hiddens):
        """
        多个dae叠加形成sdae，叠加的方式为建立一个list
        :param n_input: number of input
        :param hiddens: list of num of hidden layers
        :return: layers of dae
        """
        stacked_dae = []
        for i in range(len(hiddens) + 1):
            if i is 0:
                dae = DAE(n_input, hiddens[i],
                          transfer_function=self.transfer,
                          scale=self.scale,
                          optimizer=self.optimizer,
                          name="dae_{}".format(i),
                          sess=self.sess)
                stacked_dae.append(dae)
            elif i < len(hiddens):
                dae = DAE(hiddens[i - 1], hiddens[i],
                          transfer_function=self.transfer,
                          scale=self.scale,
                          optimizer=self.optimizer,
                          name="dae_{}".format(i),
                          sess=self.sess)
                stacked_dae.append(dae)
            else:
                dae = DAE(hiddens[i - 1], self.n_class,
                          transfer_function=self.transfer,
                          scale=self.scale,
                          optimizer=self.optimizer,
                          name="dae_{}".format(i),
                          sess=self.sess)
                stacked_dae.append(dae)
        return stacked_dae

    def train_model(self, x_train, y_train, x_test, epochs=500, learning_rate=0.001, sample_quantity=50):
        """
        函数中包含了对SDAE模型的训练，以及在SDAE基础上的LR，训练的时候需要训练集的特征、标签，但是测试集就不需要标签了
        :param x_train: 用于训练的数据，是二阶张量，是原始数据, without feature extraction
        :param y_train: labels of train set
        :param x_test: origin data without feature extraction
        :param epochs: epoch of training
        :param learning_rate:
        :param sample_quantity: loss曲线采样数量
        """

        # Clear the loss array before train
        self.loss.clear()
        step = epochs // sample_quantity
        temp_train = x_train
        for index in range(self.stacks):
            for i in range(epochs):
                self.sdae[index].partial_fit(temp_train)
            temp_train = self.sdae[index].encode_func(temp_train).eval(session=self.sess)

        y = self.get_hidden(self.x)
        pred = tf.nn.softmax(y)
        y_ = tf.placeholder(tf.float32, [None, 2])
        cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_, y))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        self.sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            _, p, loss = self.sess.run((train_step, pred, cross_entropy), feed_dict={self.x: x_train, y_: y_train})
            if i % step == 0:
                self.loss.append(loss)

        if len(self.loss) > sample_quantity:
            self.loss = self.loss[:-1]

        self.p = self.sess.run(pred, feed_dict={self.x: x_test})

    def get_hidden(self, x):
        """
        get latest hidden layer
        :param x: input data
        :return: a tensor
        """
        y = self.sdae[0].encode_func_without_noise(x)
        for i in range(1, self.stacks):
            y = self.sdae[i].encode_func_without_noise(y)

        return y

    def encode(self, x):
        """
        get the hidden representation
        :param x: data input
        :return: hidden representation
        """
        return self.sess.run(self.hidden, feed_dict={self.x: x})

    def reconstruct(self, x):
        """
        get the reconstructed data
        :param x: data input
        :return: reconstructed data
        """
        return self.sess.run(self.rec, feed_dict={self.x: x})

    def get_loss(self):
        """
        loss points of every fold in K-fold cross validation
        :return: list
        """
        return self.loss

    def get_pred(self):
        """
        prediction
        :return:
        """
        return self.p


# if __name__ == "__main__":
#     input_n = 20
#     hiddens_n = [10, 5]
#     sdae = SDAE(input_n, hiddens_n)
#     train_x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]])
#     test_x = np.array([[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1]])
#     hidden = sdae.encode(test_x)
#     rec_x = sdae.reconstruct(test_x)
#     # print(rec_x)
#     print(hidden)
