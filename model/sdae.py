# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import numpy as np
import tensorflow as tf

from model.dae import DAE


# SDAE is based on DAE, which adds additive Gaussian noise
class SDAE(object):
    def __init__(self, n_input, hiddens, transfer_function=tf.nn.softplus, scale=0.1, name='sdae', sess=None,
                 optimizer=tf.train.AdamOptimizer()):
        """
        :param n_input: 输入节点数
        :param hiddens: 每个隐藏层中的神经元数，是一个list
        :param transfer_function: transfer(activation) function
        :param scale: 高斯噪声系数
        :param optimizer: 优化器
        :param name: 命名
        :param sess: 会话，加入此参数用来确保所有的DAE都被同一个Session管理
        """
        self.name = name
        with tf.variable_scope(self.name):
            self.n_input = n_input
            self.stacks = len(hiddens)
            self.scale = scale
            self.transfer = transfer_function
            self.optimizer = optimizer
            self.sess = sess or tf.Session()

            # 网络结构
            self.sdae = self._init_sdae(self.n_input, hiddens)
            self.x = tf.placeholder(tf.float32, [None, n_input], name="input")
            self.hidden = self.x
            for dae in self.sdae:
                # self.hidden = dae.encode_func(self.hidden)
                self.hidden = dae.encode_func_without_noise(self.hidden)

            self.rec = self.hidden
            for dae in reversed(self.sdae):
                self.rec = dae.decode_func(self.rec)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _init_sdae(self, n_input, hiddens):
        """
        多个sdae叠加形成dae，叠加的方式为建立一个list
        :param n_input: number of input
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

    def pre_train(self, train_data, epochs=1000):
        """
        对SDAE进行预训练。所谓'预'是相对于抽取特征之后的训练而言。函数作用类似于DAE中的partial_fit
        :param train_data: 用于训练的数据，是二阶张量
        :param epochs: epoch of training
        :return: return nothing
        """
        temp_train = train_data
        for index in range(self.stacks):
            for i in range(epochs):
                self.sdae[index].partial_fit(temp_train)
            temp_train = self.sdae[index].encode_func(temp_train).eval(session=self.sess)

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
    test_x = np.array([[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1]])
    sdae.pre_train(train_x)
    hidden = sdae.encode(test_x)
    rec_x = sdae.reconstruct(test_x)
    # print(rec_x)
    print(hidden)
