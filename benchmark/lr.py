# -*- coding:utf-8 -*-
__author__ = 'ZM-BAD'

import tensorflow as tf
from model.data import read_from_csv
from sklearn.cross_validation import train_test_split

# 对出血时间进行二分类
sample, bleed_label, _ = read_from_csv()
n_class = 2
n_feature = 442
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, n_feature])

W = tf.Variable(tf.zeros([n_feature, n_class]))
b = tf.Variable(tf.zeros([n_class]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, n_class])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

tf.global_variables_initializer().run()

x_train, x_test, y_train, y_test = train_test_split(sample, bleed_label, test_size=0.3, random_state=0)

for i in range(1000):
    batch_xs, batch_ys = x_train, y_train
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: x_test, y_: y_test}))
