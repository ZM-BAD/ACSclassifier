# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import numpy as np
import tensorflow as tf


def xavier_init(fan_in, fan_out, constant=1):
    """
    :param fan_in: 输入节点数量
    :param fan_out: 输出节点数量
    :param constant: 系数常量(个人理解)
    :return: 标准的均匀分布
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


# DAE中最常使用的噪声是加性高斯噪声(Additive Gaussian Noise)
class DAE(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1, name='dae', sess=None):
        """
        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数，即精简、抽取后的特征数
        :param transfer_function: 隐含层激活函数，transfer(activation)function
        :param optimizer: 优化器
        :param scale: 高斯噪声系数
        :param name: 命名
        :param sess: 会话，加入此参数用来确保所有的DAE都被同一个Session管理
        """
        self.name = name
        with tf.variable_scope(self.name):
            self.n_input = n_input
            self.n_hidden = n_hidden
            self.transfer = transfer_function
            self.scale = tf.placeholder(tf.float32)
            self.training_scale = scale
            self.weights = self.initialize_weights()

            # 网络结构
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
            # w1 隐含层的权重，b1 隐含层的偏置，w2 输出层的权重，b2 输出层的偏置
            self.hidden = self.transfer(
                tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']),
                       self.weights['b1']))
            self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

            # 自编码器的损失函数
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
            self.optimizer = optimizer.minimize(self.cost)

            init = tf.global_variables_initializer()
            self.sess = sess if sess is not None else tf.Session()
            self.sess.run(init)

    # 参数初始化函数，要初始化4个参数，w1，b1，w2，b2
    def initialize_weights(self):
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

    # 整个DAE其实就两点，encode提取特征，decode通过高阶特征还原数据
    def encode(self, x):
        return self.sess.run(self.hidden, feed_dict={self.x: x, self.scale: self.training_scale})

    def decode(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def encode_func(self, x):
        # different with 'encode', this method accepts tensor and return tensor
        return self.transfer(
            tf.add(tf.matmul(x + self.training_scale * tf.random_normal((self.n_input,)), self.weights['w1']),
                   self.weights['b1']))

    def decode_func(self, hidden):
        # different with 'decode', this method accepts tensor and return tensor
        return tf.add(tf.matmul(hidden, self.weights['w2']), self.weights['b2'])

    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x, self.scale: self.training_scale})

    # 获取隐藏层权重w1
    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐藏层偏置b1
    def get_biases(self):
        return self.sess.run(self.weights['b1'])


if __name__ == "__main__":
    print("DAE test")
    input_n = 10
    hidden_n = 5
    train_x = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]])
    test_x = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

    dae = DAE(input_n, hidden_n)

    for i in range(1000):
        dae.partial_fit(train_x)

    new = dae.reconstruct(test_x)
    print(new)
