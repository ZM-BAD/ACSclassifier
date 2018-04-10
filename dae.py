# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf


def xavier_init(fan_in, fan_out, constant=1):
    """
    :param fan_in: 输入节点数量
    :param fan_out: 输出节点数量
    :param constant:
    :return:
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


# DAE中最常使用的噪声是加性高斯噪声(Additive Gaussian Noise)
class DAE(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        """
        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数
        :param optimizer: 优化器
        :param scale: 高斯噪声系数
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # 网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']),
                                           self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 自编码器的损失函数
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 参数初始化函数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 定义损失cost及执行一步训练的函数partial_fit
    def partial_fit(self, x):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: x, self.scale: self.training_scale})
        return cost

    # 只求损失cost的函数
    def calc_total_cost(self, x):
        return self.sess.run(self.cost, feed_dict={self.x: x, self.scale: self.training_scale})

    def transform(self, x):
        return self.sess.run(self.hidden, feed_dict={self.x: x, self.scale: self.training_scale})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x, self.scale: self.training_scale})

    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_biases(self):
        return self.sess.run(self.weights['b1'])


def standard_scale(x_train, x_test):
    preprocessor = prep.StandardScaler().fit(x_train)
    x_train = preprocessor.transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data), batch_size)
    return data[start_index:(start_index + batch_size)]
