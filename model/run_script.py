#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score


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

    # 获取隐藏层权重w1
    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐藏层偏置b1
    def get_biases(self):
        return self.sess.run(self.weights['b1'])


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
            all_data[line - 1, 0:444] = row[0:444]

    for i in range(2930):
        if all_data[i, 0] == 0:
            ischemic_label[i, 1] = 1
        else:
            ischemic_label[i, 0] = 1

        if all_data[i, 1] == 0:
            bleed_label[i, 1] = 1
        else:
            bleed_label[i, 0] = 1

    # First 2 columns are labels
    sample = np.zeros([2930, 442])  # only samples
    for i in range(2930):
        sample[i, 0:442] = all_data[i, 2:444]

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
    auc = roc_auc_score(tol_label, tol_pred, average=None)

    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f_score = f1_score(y_true, y_pred, average=None)

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
    result = (result[0], result[1][0], result[2][0], result[3][0], result[4][0])
    with open(file_name, 'a') as f:
        f.write(model + " model " + event + "\n")

        f.write('epoch = ' + str(epoch) + "\n")
        f.write('learning_rate = ' + str(learning_rate) + "\n")
        if model == "sdae":
            f.write('hiddens = ' + hiddens + '\n')
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


# Save loss curve data
def draw_loss_curve(bleeding_loss, ischemic_loss, k=5):
    bleeding_loss = weighted_mean(bleeding_loss, k=k)
    ischemic_loss = weighted_mean(ischemic_loss, k=k)
    file_name = 'result.txt'
    with open(file_name, 'a') as f:
        f.write('bleeding loss:\n')
        for i in bleeding_loss:
            f.write(str(i) + ' ')
        f.write("\n")
        f.write('ischemic loss:\n')
        for i in ischemic_loss:
            f.write(str(i) + ' ')
        f.write("\n")
        f.write("##########################################" + "\n")


# LR train as benchmark
def lr_experiment(epoch, learning_rate, sample, bleed_label, ischemic_label):
    """
    :param epoch: <string>
    :param learning_rate:
    :param sample:
    :param bleed_label:
    :param ischemic_label:
    :return:
    """

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
        sess.run(tf.global_variables_initializer())
        split_label = bleed_label[:, 0]
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        count = 0
        all_y_test = []
        all_p = []
        for train_index, test_index in kf.split(sample, split_label):
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
                    # print(loss, i)
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
        draw_event_graph(bleeding_result, event="Bleeding", model="lr", learning_rate=learning_rate, epoch=epoch)

    ##########################################################################
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
        sess.run(tf.global_variables_initializer())
        split_label = ischemic_label[:, 0]
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        count = 0
        all_y_test = []
        all_p = []

        for train_index, test_index in kf.split(sample, split_label):
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
                    # print(loss, i)
                    ischemic_loss.append(loss)
            if len(ischemic_loss) % sample_quantity > 0:
                ischemic_loss = ischemic_loss[:-1]
            p = sess.run(pred, feed_dict={x: x_test})
            if count == 1:
                all_p = p
            else:
                all_p = np.append(all_p, p, axis=0)

        ischemic_result = evaluate(all_y_test, all_p)
        draw_event_graph(ischemic_result, event="Ischemic", model="lr", learning_rate=learning_rate, epoch=epoch)

    draw_loss_curve(bleeding_loss, ischemic_loss)


# Do SDAE train
def sdae_experiment(epoch, hiddens, learning_rate, sample, bleed_label, ischemic_label):
    """
    :param epoch: <string>
    :param hiddens: <list>
    :param learning_rate:
    :param sample:
    :param bleed_label:
    :param ischemic_label:
    :return:
    """
    epoch = int(epoch)
    origin_n_input = len(sample[0])
    n_class = 2
    # 抽取后的feature数量
    extract_feature_n = hiddens[-1]
    # loss曲线的采样数量
    sample_quantity = 50
    step = epoch // sample_quantity
    bleeding_loss = []
    ischemic_loss = []

    # Bleeding events
    x = tf.placeholder(tf.float32, [None, extract_feature_n])
    w = tf.Variable(tf.zeros([extract_feature_n, n_class]))
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
        split_label = bleed_label[:, 0]
        count = 0
        all_y_test = []
        all_p = []
        sdae = SDAE(origin_n_input, hiddens)

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

            # SDAE本质上是无监督的，所以不需要label，训练SDAE用x_train即可
            # 对特征抽取后，有一层Softmax，但是对于二分类而言，Softmax退化为LR
            # LR是监督学习，需要训练。LR训练的样本是x_train抽取出来的x_extract_train，而样本标签依旧为y_train
            sdae.pre_train(x_train)
            x_extract_train = sdae.encode(x_train)
            x_extract_test = sdae.encode(x_test)
            for i in range(epoch):
                _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_extract_train, y_: y_train})
                if i % step == 0:
                    bleeding_loss.append(loss)

            if len(bleeding_loss) % sample_quantity > 0:
                bleeding_loss = bleeding_loss[:-1]

            p = sess.run(pred, feed_dict={x: x_extract_test})
            if count == 1:
                all_p = p
            else:
                all_p = np.append(all_p, p, axis=0)

        bleeding_result = evaluate(all_y_test, all_p)
        draw_event_graph(bleeding_result, event="Bleeding", model="sdae", learning_rate=learning_rate,
                         epoch=epoch, hiddens=str(hiddens))

    # ##########################################################################
    # # Ischemic events
    x = tf.placeholder(tf.float32, [None, extract_feature_n])
    w = tf.Variable(tf.zeros([extract_feature_n, n_class]))
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
        split_label = ischemic_label[:, 0]
        count = 0
        all_y_test = []
        all_p = []
        sdae = SDAE(origin_n_input, hiddens)

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

            sdae.pre_train(x_train)
            x_extract_train = sdae.encode(x_train)
            x_extract_test = sdae.encode(x_test)
            for i in range(epoch):
                _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: x_extract_train, y_: y_train})
                if i % step == 0:
                    ischemic_loss.append(loss)

            if len(ischemic_loss) % sample_quantity > 0:
                ischemic_loss = ischemic_loss[:-1]

            p = sess.run(pred, feed_dict={x: x_extract_test})
            if count == 1:
                all_p = p
            else:
                all_p = np.append(all_p, p, axis=0)

        ischemic_result = evaluate(all_y_test, all_p)
        draw_event_graph(ischemic_result, event="Ischemic", model="sdae", learning_rate=learning_rate,
                         epoch=epoch, hiddens=str(hiddens))

    draw_loss_curve(bleeding_loss, ischemic_loss)


if __name__ == "__main__":
    dataset = "dataset.csv"
    sample, bleed_label, ischemic_label = read_from_csv(dataset)
    epochs = [300, 500, 1000]
    learning_rates = [0.001, 0.0001, 0.00001]
    hiddens = [[256, 128],                          # model 1, 2 layers
               [32, 8, 2],                          # model 2, 3 layers
               [64, 16, 8],                         # model 3, 3 layers, more nodes
               [128, 64, 16],                       # model 4, 3 layers, moore nodes
               [256, 128, 64],                      # model 5, 3 layers, mooore nodes
               [256, 128, 64, 32],                  # model 6, 4 layers
               [256, 128, 64, 32, 16]]              # model 7, 5 layers
    # model 1, 5, 6, 7形成对照，探究层数对效果的影响
    # model 2, 3, 4, 5形成对照，探究层数一定，节点数对效果的影响

    for i in epochs:
        for j in learning_rates:
            lr_experiment(i, j, sample, bleed_label, ischemic_label)
            # for hidden in hiddens:
            #     sdae_experiment(i, hidden, j, sample, bleed_label, ischemic_label)

    # So, we train 72 models in total
