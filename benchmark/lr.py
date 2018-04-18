# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import tensorflow as tf
from sklearn.model_selection import train_test_split

from model.data import read_from_csv

# 读入样本以及出血事件的标签，暂时不训练缺血事件的预测
# ischemic_label这里就用 _ 打个酱油
sample, bleed_label, _ = read_from_csv()

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 442])
W_bleed = tf.Variable(tf.zeros([442, 2]))
b_bleed = tf.Variable(tf.zeros([2]))

y_bleed = tf.nn.softmax(tf.matmul(x, W_bleed) + b_bleed)

y_bleed_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_bleed_ * tf.log(y_bleed), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

tf.global_variables_initializer().run()

# 将样本随机划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(sample, bleed_label, test_size=0.3, random_state=0)

for i in range(1000):
    batch_xs, batch_ys = x_train, y_train
    train_step.run({x: batch_xs, y_bleed_: batch_ys})
    if i % 50 == 0:
        print(sess.run(cross_entropy, feed_dict={x: batch_xs, y_bleed_: batch_ys}))

correct_prediction = tf.equal(tf.argmax(y_bleed, 1), tf.argmax(y_bleed_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: x_test, y_bleed_: y_test}))
