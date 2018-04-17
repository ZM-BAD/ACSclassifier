# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import tensorflow as tf
import numpy as np
from dae import DAE


# SDAE是基于DAE实现的，其中DAE使用的噪声为加性高斯噪声
class SDAE(object):
    def __init__(self, n_input, hiddens, transfer_function=tf.nn.softplus, scale=0.1, name='sdae', sess=None,
                 optimizer=tf.train.AdamOptimizer(), epochs=1000):
        """
        :param n_input: 输入节点数
        :param hiddens: 每个隐藏层中的神经元数，是一个list
        :param transfer_function: transfer(activation) function
        :param scale: 高斯噪声系数
        :param optimizer: 优化器
        :param epochs: 训练的迭代轮数
        :param name: 命名
        :param sess: 会话，加入此参数用来确保所有的DAE都被同一个Session管理
        """
        self.name = name
        with tf.variable_scope(self.name):
            self.n_input = n_input
            self.stacks = len(hiddens)
            self.scale = scale
            self.epochs = epochs
            self.transfer = transfer_function
            self.optimizer = optimizer
            self.sess = sess if sess is not None else tf.Session()

            # 网络结构
            self.sdae = self._init_sdae(self.n_input, hiddens)
            self.x = tf.placeholder(tf.float32, [None, n_input], name="input")
            self.hidden = self.x
            for dae in self.sdae:
                self.hidden = dae.decode_func(self.hidden)

            self.rec = self.hidden
            for dae in reversed(self.sdae):
                self.rec = dae.decode_func(self.rec)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    # def __call__(self, x):
    #     """
    #     as a component of parent model
    #
    #     :param x: input tensor
    #     :return: hidden representation tensor
    #     """
    #     x_copy = x
    #     for dae in self.sdae:
    #         x_copy = dae(x_copy)
    #     return x_copy

    def _init_sdae(self, n_input, hiddens):
        """
        多个sdae叠加形成dae，叠加的方式为建立一个list
        :param n_input: 输入节点数
        :param hiddens: list of num of hidden layers
        :return: layers of dae
        """
        stacked_dae = []
        for i in range(len(hiddens)):
            if i is 0:
                dae = DAE(n_input, hiddens[i],
                          transfer_function=self.transfer,
                          scale=self.scale,
                          optimizer=self.optimizer,
                          name="dae_{}".format(i),
                          sess=self.sess)
                stacked_dae.append(dae)
            else:
                dae = DAE(hiddens[i - 1], hiddens[i],
                          transfer_function=self.transfer,
                          scale=self.scale,
                          optimizer=self.optimizer,
                          name="dae_{}".format(i),
                          sess=self.sess)
                stacked_dae.append(dae)
        return stacked_dae

    def pre_train(self, data_set, batch_size=128):
        """
        预训练模型，所谓"预"是针对提取特征之后的训练
        :param data.read_data.DataSet data_set: the training data set
        :param batch_size: `batch size` of data
        """
        for i in range(self.stacks):
            while data_set.epoch_completed < self.epochs:
                x, _ = data_set.next_batch(batch_size)
                self.sdae[i].partial_fit(x)
            x = self.sdae[i].encode(data_set.examples)
            data_set = DataSet(x, data_set.labels)

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


if __name__ == "__main__":
    input_n = 20
    hiddens_n = [10, 5]
    sdae = SDAE(input_n, hiddens_n)
    train_x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]])
    text_x = np.array([[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1]])
    data = DataSet(train_x, np.arange(train_x.shape[0]))
    sdae.pre_train(data, 10)

    rec_x = sdae.reconstruct(text_x)
    print(rec_x)
