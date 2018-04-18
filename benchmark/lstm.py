# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def rnn(x, weights, biases):
    # hidden layer for input
    x = tf.reshape(x, [-1, n_inputs])
    x_in = tf.matmul(x, weights['in']) + biases['in']
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])

    # cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    # results = tf.matmul(states[1], weights['out']) + biases['out']
    # or
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters init
l_r = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

# define placeholder for input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define w and b
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

pred = rnn(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(l_r).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for i in range(training_iters):
for i in range(training_iters):
    # get batch to learn easily
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
    sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
    if i % 50 == 0:
        print(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, }))
# test_data = mnist.test.images.reshape([-1, n_steps, n_inputs])
# test_label = mnist.test.labels
# print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
