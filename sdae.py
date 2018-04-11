# -*- coding: utf-8 -*-
__author__ = 'ZM-BAD'

import numpy as np
import tensorflow as tf
from dae import DAE

# 定义训练参数
training_epochs = 5
batch_size = 1000
display_step = 1
stack_size = 3  # 栈中包含3个dae
hidden_size = [20, 20, 20]
input_n_size = [3, 200, 200]


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


# 建立sdae图，构建SDAE的一种想法
sdae = []
for i in range(stack_size):
    if i == 0:
        dae = DAE(n_input=2,
                  n_hidden=hidden_size[i],
                  activation_function=tf.nn.softplus,
                  optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                  scale=0.01)
        dae.initialize_weights()
        sdae.append(dae)
    else:
        dae = DAE(n_input=hidden_size[i - 1],
                  n_hidden=hidden_size[i],
                  activation_function=tf.nn.softplus,
                  optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                  scale=0.01)
        dae.initialize_weights()
        sdae.append(dae)

W = []
b = []
Hidden_feature = []  # 保存每个dae的特征
X_train = np.array([0])
for j in range(stack_size):
    # 输入
    if j == 0:
        X_train = np.array(pd.train_set)
        X_test = np.array(pd.test_set)
    else:
        X_train_pre = X_train
        X_train = sdae[j - 1].transform(X_train_pre)
        print(X_train.shape)
        Hidden_feature.append(X_train)

    # 训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[1] / batch_size)
        # Loop over all batches
        for k in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            # Fit training using batch data
            cost = sdae[j].partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / X_train.shape[1] * batch_size

        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # 保存每个ae的参数
    weight = sdae[j].get_weights()
    # print (weight)
    W.append(weight)
    b.append(sdae[j].get_biases())
