import tensorflow as tf
import pandas as pd
import numpy as np

x = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.float32, [None, 10])
droprate = tf.placeholder(tf.float32)
weight = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[10]))
hidelayer = []
for i in range(4):
    tempcell = tf.nn.rnn_cell.BasicLSTMCell(64)
    tempcell = tf.nn.rnn_cell.DropoutWrapper(tempcell, output_keep_prob=droprate)
    hidelayer.append(tempcell)
lstmcells = tf.nn.rnn_cell.MultiRNNCell(hidelayer)
initstate = lstmcells.zero_state(100, dtype=tf.float32)
output, _ = tf.nn.dynamic_rnn(lstmcells, x, initial_state=initstate, dtype=tf.float32, time_major=False)
lstmy = tf.nn.softmax(tf.matmul(output[:, 1, :], weight) + bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=lstmy, labels=y))
trainop = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
