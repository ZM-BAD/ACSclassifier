# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import tensorflow as tf
from model.dae import DAE


# SDAE is based on DAE, which adds additive Gaussian noise
class SDAE(object):
    def __init__(self, n_input, hiddens, n_class=2, learning_rate=0.001, transfer_function=tf.nn.softplus, scale=0.1,
                 name='sdae', sess=None, optimizer=tf.train.AdamOptimizer()):
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

            # 网络结构
            self.sdae = self._init_sdae(self.n_input, hiddens)
            self.x = tf.placeholder(tf.float32, [None, n_input], name="input")
            self.hidden = self.x
            for dae in self.sdae:
                self.hidden = dae.encode_func_without_noise(self.hidden)

            self.rec = self.hidden
            for dae in reversed(self.sdae):
                self.rec = dae.decode_func(self.rec)

            y = self.hidden
            self.pred = tf.nn.softmax(y)
            self.y_ = tf.placeholder(tf.float32, [None, 2])
            self.cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_, y))
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

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

    def train_model(self, x_train, y_train, x_test, epochs=500, sample_quantity=50):
        """
        This function contains training SDAE model and softmax based on SDAE
        :param x_train: origin train data without feature extraction
        :param y_train: labels of train set
        :param x_test: origin data without feature extraction
        :param epochs: epoch of training
        :param sample_quantity: loss curve sample quantity
        """

        self.sess.run(tf.global_variables_initializer())
        # Clear the loss array before train
        self.loss.clear()
        step = epochs // sample_quantity
        temp_train = x_train
        for index in range(self.stacks):
            for i in range(epochs):
                self.sdae[index].partial_fit(temp_train)
            temp_train = self.sdae[index].encode_func(temp_train).eval(session=self.sess)

        for i in range(epochs):
            _, p, loss = self.sess.run((self.train_step, self.pred, self.cross_entropy),
                                       feed_dict={self.x: x_train, self.y_: y_train})
            # self.print_w_b()
            # print("*************")
            if i % step == 0:
                self.loss.append(loss)

        if len(self.loss) > sample_quantity:
            self.loss = self.loss[:-1]

        self.p = self.sess.run(self.pred, feed_dict={self.x: x_test})

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

    def print_w_b(self):
        """
        print weights and biases to check if the weights and biases changed
        """
        for dae in self.sdae:
            w = dae.get_weights()
            b = dae.get_biases()
            print(w)
            print(b)
