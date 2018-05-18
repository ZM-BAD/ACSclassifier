#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score


# TODO: 跑好LR的数据，大约3~4个左右吧
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

    # Train DAE with noise, but use DAE without noise
    def encode_func_without_noise(self, x):
        return self.transfer(tf.add(tf.matmul(x, self.weights['w1']), self.weights['b1']))

    def decode_func(self, hidden):
        # different with 'decode', this method accepts tensor and return tensor
        return tf.add(tf.matmul(hidden, self.weights['w2']), self.weights['b2'])

    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x, self.scale: self.training_scale})

    # Get hidden layer's weights, w1
    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    # Get hidden layer's biases, b1
    def get_biases(self):
        return self.sess.run(self.weights['b1'])


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
            print(loss)
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


def read_from_csv(datafile_path):
    """
    :return: samples and labels in ndarray
    """
    # I know there're 2930 samples and 442 features in dataset.csv
    # But we still have to write some code to get the number
    a = open(datafile_path, 'r', encoding="gbk")
    num_of_sample = len(a.readlines()) - 1
    reader = csv.reader(open(datafile_path, encoding='gbk'))
    columns = 0
    for row in reader:
        columns = len(row)
        break

    all_data = np.zeros([num_of_sample, columns])
    bleed_label = np.zeros([num_of_sample, 2])  # bleed happen=(1, 0), not happen=(0, 1)
    ischemic_label = np.zeros([num_of_sample, 2])  # ischemic happen=(1, 0), not happen=(0, 1)

    # First line in the file is feature name, ignore it.
    line = -1
    for row in reader:
        line += 1
        if line > 0:
            all_data[line - 1, 0:columns] = row[0:columns]

    for i in range(num_of_sample):
        if all_data[i, 0] == 0:
            ischemic_label[i, 1] = 1
        else:
            ischemic_label[i, 0] = 1

        if all_data[i, 1] == 0:
            bleed_label[i, 1] = 1
        else:
            bleed_label[i, 0] = 1

    # First 2 columns are labels
    sample = np.zeros([num_of_sample, columns - 2])  # only samples
    for i in range(num_of_sample):
        sample[i, 0:columns - 2] = all_data[i, 2:columns]

    return sample, bleed_label, ischemic_label


# Calculate acc, auc, f1-score, recall, precision
def evaluate(tol_label, tol_pred):
    """
    Evaluate the predictive performance of the model
    :param tol_label: real labels for test samples
    :param tol_pred: predicted probability distribution of test samples
    :return: acc, auc, f1-score, recall, precision
    """
    assert tol_label.shape == tol_pred.shape

    y_true = np.argmax(tol_label, axis=1)
    y_pred = np.argmax(tol_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(tol_label, tol_pred)

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f_score = f1_score(y_true, y_pred, average='weighted')

    return accuracy, auc, f_score, recall, precision


# Save acc, auc, f1-score, recall, precision data
def draw_event_graph(result, event, model, learning_rate, epoch, hiddens=None):
    """
    :param result: <tuple> (acc, auc, f1-score, recall, precision)
    :param event: <string> "Bleeding" or "Ischemic"
    :param model: <string> "lr" or "sdae"
    :param learning_rate: float
    :param epoch: int
    :param hiddens: <string>
    :return:
    """
    file_name = "result.txt"
    result = (result[0], result[1], result[2], result[3], result[4])
    with open(file_name, 'a') as f:
        f.write(model + " model " + event + "\n")

        f.write('epoch = ' + str(epoch) + "\n")
        f.write('learning_rate = ' + str(learning_rate) + "\n")
        if model == "sdae":
            f.write('hiddens = ' + str(hiddens) + '\n')
        else:
            f.write('hiddens = None\n')

        f.write("acc, auc, f1-score, recall, precision\n")
        for i in result:
            f.write(str(i))
            f.write("\n")


# Handle the loss array
def weighted_mean(loss, k):
    new_length = len(loss) / k
    new_loss = [0] * int(new_length)

    for i in range(len(loss)):
        index = int(i % new_length)
        new_loss[index] += loss[i]

    for i in new_loss:
        i /= k

    return new_loss


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
    lr_train(sample, bleed_label, epoch, learning_rate, sample_quantity, event='bleeding')
    lr_train(sample, ischemic_label, epoch, learning_rate, sample_quantity, event='ischemic')


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
                print(loss)
                if i % step == 0:
                    # print(loss, i)
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
        draw_event_graph(result, event=event, model="lr", learning_rate=learning_rate, epoch=epoch)


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

    sdae_train(sample, bleed_label, epoch, hiddens, learning_rate, sample_quantity, event='bleeding')
    sdae_train(sample, ischemic_label, epoch, hiddens, learning_rate, sample_quantity, event='ischemic')


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
        draw_event_graph(result, event, "sdae", learning_rate=learning_rate, epoch=epoch, hiddens=hidden_layers)


if __name__ == "__main__":
    dataset = "../res/dataset.csv"
    hidden = [8, 4]
    sdae_experiment(dataset, epoch=50, hidden_layers=hidden, learning_rate=0.001)
    # lr_experiment(dataset, 50, 0.001)
