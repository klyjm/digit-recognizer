import tensorflow as tf
import pandas as pd
import numpy as np


def train(filename):
    traindata = pd.read_csv(filename).as_matrix()
    imagedata = traindata[:, 1:]
    imagedata = imagedata.astype(np.float)
    verifyimg = imagedata[40000:, :]
    imagedata = imagedata[0:39999, :]
    imagedata = np.multiply(imagedata, 1.0/255)
    labeldata = traindata[:, 0]
    verifylab = labeldata[40000:]
    labeldata = labeldata[0:39999]
    onehotlabel = [[0 for i in range(10)] for i in range(len(labeldata))]
    for i in range(len(labeldata)):
        onehotlabel[i][labeldata[i]] = 1;
    onehotlabel = np.array(onehotlabel, dtype=np.uint8)
    batchsize = 100
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    inputdata = tf.reshape(x, [-1, 28, 28, 1])
    conv1w = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    conv1b = tf.Variable(tf.constant(0.1, shape=[32]))
    conv1 = tf.nn.conv2d(inputdata, conv1w, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1b))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2w = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    conv2b = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.conv2d(pool1, conv2w, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2b))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    poolshape = pool2.get_shape().as_list()
    nodes = poolshape[1] * poolshape[2] * poolshape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])
    fc1w = tf.Variable(tf.truncated_normal([nodes, 512], stddev=0.1))
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.0001)(fc1w))
    fc1b = tf.Variable(tf.constant(0.1, shape=[512]))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1w) + fc1b)
    fc1 = tf.nn.dropout(fc1, 0.5)
    fc2w = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.0001)(fc2w))
    fc2b = tf.Variable(tf.constant(0.1, shape=[10]))
    lenet5y = tf.matmul(fc1, fc2w) + fc2b
    globalstep = tf.Variable(0, trainable=False)
    varave = tf.train.ExponentialMovingAverage(0.99, globalstep)
    varaveop = varave.apply(tf.trainable_variables())
    crossentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=lenet5y)
    crossentrogymean = tf.reduce_mean(crossentropy)
    loss = crossentrogymean + tf.add_n(tf.get_collection('loss'))
    learnrate = tf.train.exponential_decay(0.8, globalstep, 400, 0.99)
    trainstep = tf.train.GradientDescentOptimizer(learnrate).minimize(loss, global_step=globalstep)
    trainop = tf.group(trainstep, varaveop)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state('/ckpt/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(20):
            for j in range(400):
                x1 = imagedata[j * batchsize:(j + 1) * batchsize]
                y1 = onehotlabel[j * batchsize:(j + 1) * batchsize]
                sess.run([trainop, loss, globalstep], feed_dict={x: x1, y: y1})
        saver.save(sess, '/ckpt/lenet5.ckpt')


def datatest(filename):
    traindata = pd.read_csv(filename).as_matrix()
    imagedata = traindata.iloc[:, :].values
    imagedata = imagedata.astype(np.float)
    imagedata = np.multiply(imagedata, 1.0 / 255)


inputstr = input('请输入命令：\n')
while 1:
    if inputstr == 'exit':
        exit()
    inputstr = inputstr.strip().split()
    if len(inputstr) != 2 :
        inputstr = input('非法命令！请重新输入命令：\n')
        continue
    if inputstr[0] == 'train':
        train(inputstr[1])
    if inputstr[0] == 'test':
        datatest(inputstr[1])
    inputstr = input('请输入命令：\n')