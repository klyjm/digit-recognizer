import tensorflow as tf
import pandas as pd
import numpy as np
import os


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
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    droprate = tf.placeholder(tf.float32, [])
    w1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
    b1 = tf.Variable(tf.zeros([300]))
    w2 = tf.Variable(tf.truncated_normal([300, 100], stddev=0.1))
    b2 = tf.Variable(tf.zeros([100]))
    w3 = tf.Variable(tf.truncated_normal([100, 30], stddev=0.1))
    b3 = tf.Variable(tf.zeros([30]))
    w4 = tf.Variable(tf.zeros([30, 10]))
    b4 = tf.Variable(tf.zeros([10]))
    hy1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    hy1 = tf.nn.dropout(hy1, droprate)
    hy2 = tf.nn.relu(tf.matmul(hy1, w2) + b2)
    hy2 = tf.nn.dropout(hy2, droprate)
    hy3 = tf.nn.relu(tf.matmul(hy2, w3) + b3)
    hy3 = tf.nn.dropout(hy3, droprate)
    mlpy = tf.nn.softmax(tf.matmul(hy3, w4) + b4)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=mlpy))
    trainop = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
    saver = tf.train.Saver(max_to_keep=400)
    label = [[0 for i in range(10)] for i in range(len(verifylab))]
    for i in range(len(verifylab)):
        label[i][verifylab[i]] = 1
    label = np.array(label, dtype=np.uint8)
    correctpredict = tf.equal(tf.argmax(lenet5y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctpredict, tf.float32))
    with tf.Session() as sess:
        max = 0.0
        maxi = 0
        ckpt = tf.train.get_checkpoint_state('/ckpt/mlp/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()
        for i in range(400):
            for j in range(400):
                x1 = imagedata[j * batchsize:(j + 1) * batchsize]
                y1 = onehotlabel[j * batchsize:(j + 1) * batchsize]
                _, lossval, step = sess.run([trainop, loss, globalstep], feed_dict={x: x1, y: y1, droprate: 0.5})
            saver.save(sess, '/ckpt/mlp/mlp' + str(i) + '.ckpt')
            accuracyrate = sess.run(accuracy, feed_dict={x: verifyimg, y: label, droprate: 1.0})
            if accuracyrate > max:
                max = accuracyrate
                maxi = i
            print(str(accuracyrate) + str(i))
        print(str(max))
        print(str(maxi))
    tf.reset_default_graph()


def datatest(filename):
    testdata = pd.read_csv(filename).values
    imagedata = testdata.astype(np.float)
    imagedata = np.multiply(imagedata, 1.0 / 255)
    x = tf.placeholder(tf.float32, [None, 784])
    droprate = tf.placeholder(tf.float32, [])
    w1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
    b1 = tf.Variable(tf.zeros([300]))
    w2 = tf.Variable(tf.truncated_normal([300, 100], stddev=0.1))
    b2 = tf.Variable(tf.zeros([100]))
    w3 = tf.Variable(tf.truncated_normal([100, 30], stddev=0.1))
    b3 = tf.Variable(tf.zeros([30]))
    w4 = tf.Variable(tf.zeros([30, 10]))
    b4 = tf.Variable(tf.zeros([10]))
    hy1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    hy1 = tf.nn.dropout(hy1, droprate)
    hy2 = tf.nn.relu(tf.matmul(hy1, w2) + b2)
    hy2 = tf.nn.dropout(hy2, droprate)
    hy3 = tf.nn.relu(tf.matmul(hy2, w3) + b3)
    hy3 = tf.nn.dropout(hy3, droprate)
    mlpy = tf.nn.softmax(tf.matmul(hy3, w4) + b4)
    saver = tf.train.Saver()
    yout = tf.arg_max(mlpy, 1)
    with tf.Session() as sess:
        saver.restore(sess, '/ckpt/mlp/lenet5220.ckpt')
        n = int(len(imagedata) / 1000)
        data = [0] * len(imagedata)
        k = 0
        for i in range(n):
            x1 = imagedata[i * 1000:(i + 1) * 1000]
            x1 = np.reshape(x1, (1000, 28, 28))
            result = sess.run(yout, feed_dict={x: x1, droprate: 1.0})
            for j in range(1000):
                data[k] = result[j]
                k += 1
        index = list(range(1, len(data) + 1))
        pd.DataFrame(data=data, index=index, columns=['Label']).to_csv('result.csv')


if not os.path.exists('/ckpt/mlp/'):
    os.mkdir('/ckpt/mlp/')
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

