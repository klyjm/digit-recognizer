import tensorflow as tf
import tensorflow.contrib as contrib
import pandas as pd
import numpy as np
import os
import time


def train(filename):
    traindata = pd.read_csv(filename).values
    imagedata = traindata[:, 1:]
    imagedata = imagedata.astype(np.float)
    imagedata = np.multiply(imagedata, 1.0 / 255)
    verifyimg = imagedata[40000:, :]
    imagedata = imagedata[0:40000, :]
    labeldata = traindata[:, 0]
    verifylab = labeldata[40000:]
    labeldata = labeldata[0:40000]
    onehotlabel = [[0 for i in range(10)] for i in range(len(labeldata))]
    for i in range(len(labeldata)):
        onehotlabel[i][labeldata[i]] = 1
    onehotlabel = np.array(onehotlabel, dtype=np.uint8)
    label = [[0 for i in range(10)] for i in range(len(verifylab))]
    for i in range(len(verifylab)):
        label[i][verifylab[i]] = 1
    label = np.array(label, dtype=np.uint8)
    size = 100
    x = tf.placeholder(tf.float32, [None, 28, 28])
    y = tf.placeholder(tf.float32, [None, 10])
    batchsize = tf.placeholder(tf.int32, [])
    droprate = tf.placeholder(tf.float32, [])
    weight = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[10]))
    hidelayer = []
    for i in range(3):
        tempcell = tf.nn.rnn_cell.BasicLSTMCell(512)
        #tempcell = contrib.cudnn_rnn.CudnnCompatibleLSTMCell(64)
        tempcell = tf.nn.rnn_cell.DropoutWrapper(tempcell, output_keep_prob=droprate)
        hidelayer.append(tempcell)
    lstmcells = tf.nn.rnn_cell.MultiRNNCell(hidelayer)
    initstate = lstmcells.zero_state(batchsize, dtype=tf.float32)
    # lstm = contrib.cudnn_rnn.CudnnLSTM(4, 64, dtype=tf.float32)
    # output, (_, _) = lstm(x)
    output, _ = tf.nn.dynamic_rnn(lstmcells, x, initial_state=initstate, dtype=tf.float32, time_major=False)
    lstmy = tf.nn.softmax(tf.matmul(output[:, -1, :], weight) + bias)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=lstmy, labels=y))
    trainop = tf.train.AdadeltaOptimizer(0.01).minimize(loss)
    correctpredict = tf.equal(tf.argmax(lstmy, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctpredict, tf.float32))
    saver = tf.train.Saver(max_to_keep=1500)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(config=config) as sess:
        max = 0.0
        maxi = 0
        tf.global_variables_initializer().run()
        for i in range(2000):
            for j in range(400):
                x1 = imagedata[j * size:(j + 1) * size]
                x1 = np.reshape(x1, (size, 28, 28))
                y1 = onehotlabel[j * size:(j + 1) * size]
                _, lossval = sess.run([trainop, loss], feed_dict={x: x1, y: y1, batchsize: size, droprate: 0.5})
            saver.save(sess, '/ckpt/lstm/lstm' + str(i) + '.ckpt')
            verifyimg = np.reshape(verifyimg, (2000, 28, 28))
            accuracyrate = sess.run(accuracy, feed_dict={x: verifyimg, y: label, batchsize: 2000, droprate: 1.0})
            if accuracyrate > max:
                max = accuracyrate
                maxi = i
            print(str(accuracyrate) + str(i))
        print(str(max))
        print(str(maxi))


def datatest(filename):
    testdata = pd.read_csv(filename).values
    imagedata = testdata.astype(np.float)
    imagedata = np.multiply(imagedata, 1.0 / 255)
    x = tf.placeholder(tf.float32, [None, 28, 28])
    batchsize = tf.placeholder(tf.int32)
    droprate = tf.placeholder(tf.float32)
    weight = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[10]))
    hidelayer = []
    for i in range(4):
        tempcell = tf.nn.rnn_cell.BasicLSTMCell(64)
        tempcell = tf.nn.rnn_cell.DropoutWrapper(tempcell, output_keep_prob=droprate)
        hidelayer.append(tempcell)
    lstmcells = tf.nn.rnn_cell.MultiRNNCell(hidelayer)
    initstate = lstmcells.zero_state(batchsize, dtype=tf.float32)
    output, _ = tf.nn.dynamic_rnn(lstmcells, x, initial_state=initstate, dtype=tf.float32, time_major=False)
    lstmy = tf.nn.softmax(tf.matmul(output[:, 1, :], weight) + bias)
    y = tf.nn.softmax(lstmy)
    saver = tf.train.Saver()
    yout = tf.arg_max(y, 1)
    with tf.Session() as sess:
        saver.restore(sess, '/ckpt/lstm/lenet5220.ckpt')
        n = int(len(imagedata) / 1000)
        data = [0] * len(imagedata)
        k = 0
        for i in range(n):
            x1 = imagedata[i * 1000:(i + 1) * 1000]
            x1 = np.reshape(x1, (1000, 28, 28))
            result = sess.run(yout, feed_dict={x: x1, batchsize: 1000, droprate: 1.0})
            for j in range(1000):
                data[k] = result[j]
                k += 1
        index = list(range(1, len(data) + 1))
        pd.DataFrame(data=data, index=index, columns=['Label']).to_csv('result.csv')


if not os.path.exists('/ckpt/lstm/'):
    os.mkdir('/ckpt/lstm/')
inputstr = input('请输入命令：\n')
while 1:
    if inputstr == 'exit':
        exit()
    inputstr = inputstr.strip().split()
    if len(inputstr) != 2:
        inputstr = input('非法命令！请重新输入命令：\n')
        continue
    if inputstr[0] == 'train':
        train(inputstr[1])
    if inputstr[0] == 'test':
        datatest(inputstr[1])
    inputstr = input('请输入命令：\n')
tf.reset_default_graph()
