# -*- coding:utf-8 -*-
__author__ = 'ZM-BAD'

import tensorflow as tf
from model.data import read_from_csv
from sklearn.model_selection import train_test_split

# Classification of bleeding events
sample, bleed_label, _ = read_from_csv("C:/Users/ZM-BAD/Projects/ACSclassifier/res/dataset.csv")
n_class = 2
n_feature = 442
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, n_feature])

W = tf.Variable(tf.zeros([n_feature, n_class]))
b = tf.Variable(tf.zeros([n_class]))

y = tf.matmul(x, W) + b
pred = tf.nn.sigmoid(y)

y_ = tf.placeholder(tf.float32, [None, n_class])
cross_entropy = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_, y))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

tf.global_variables_initializer().run()

# split train set and test set
x_train, x_test, y_train, y_test = train_test_split(sample, bleed_label, test_size=0.3, random_state=0)

for i in range(1000):
    batch_xs, batch_ys = x_train, y_train
    _, p, loss = sess.run((train_step, pred, cross_entropy), feed_dict={x: batch_xs, y_: batch_ys})
    print(loss)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: x_test, y_: y_test}))
