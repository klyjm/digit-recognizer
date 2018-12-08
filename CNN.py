import tensorflow as tf
import pandas as pd
import numpy as np
import os


def train(filename):
    y = tf.placeholder(tf.float32, [None, 10])
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
    batchsize = 100
    globalstep = tf.Variable(0, trainable=False)
    varave = tf.train.ExponentialMovingAverage(0.99, globalstep)
    varaveop = varave.apply(tf.trainable_variables())
    crossentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=lenet5y)
    crossentrogymean = tf.reduce_mean(crossentropy)
    loss = crossentrogymean + tf.add_n(tf.get_collection('loss'))
    learnrate = tf.train.exponential_decay(0.001, globalstep, 400, 0.99)
    #trainop = tf.train.GradientDescentOptimizer(learnrate).minimize(loss, global_step=globalstep)
    trainstep = tf.train.GradientDescentOptimizer(learnrate).minimize(loss, global_step=globalstep)
    #trainstep = tf.train.AdadeltaOptimizer(0.01).minimize(loss, global_step=globalstep)
    #trainstep = tf.train.AdamOptimizer(0.01).minimize(loss, global_step=globalstep)
    trainop = tf.group(trainstep, varaveop)
    saver = tf.train.Saver(max_to_keep=1000)
    label = [[0 for i in range(10)] for i in range(len(verifylab))]
    for i in range(len(verifylab)):
        label[i][verifylab[i]] = 1
    label = np.array(label, dtype=np.uint8)
    correctpredict = tf.equal(tf.argmax(lenet5y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctpredict, tf.float32))
    with tf.Session() as sess:
        max = 0.0
        maxi = 0
        ckpt = tf.train.get_checkpoint_state('/ckpt/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.global_variables_initializer().run()
        for i in range(1000):
            for j in range(400):
                x1 = imagedata[j * batchsize:(j + 1) * batchsize]
                x1 = np.reshape(x1, (100, 28, 28, 1))
                y1 = onehotlabel[j * batchsize:(j + 1) * batchsize]
                _, lossval, step = sess.run([trainop, loss, globalstep], feed_dict={x: x1, y: y1, droprate: 0.5})
            saver.save(sess, '/ckpt/lenet5'+str(i)+'.ckpt')
            verifyimg = np.reshape(verifyimg, (2000, 28, 28, 1))
            accuracyrate = sess.run(accuracy, feed_dict={x: verifyimg, y: label, droprate: 1.0})
            if accuracyrate > max:
                max = accuracyrate
                maxi = i
            print(str(accuracyrate)+str(i))
        print(str(max))
        print(str(maxi))
        # saver.save(sess, '/ckpt/lenet5.ckpt')
    # with tf.Session() as sess:
    #     ckpt = tf.train.get_checkpoint_state('/ckpt/')
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #     verifyimg = np.reshape(verifyimg, (2000, 28, 28, 1))
    #     accuracyrate = sess.run(accuracy, feed_dict={x: verifyimg, y: label})
    #     print(str(accuracyrate))


def datatest(filename):
    testdata = pd.read_csv(filename).values
    imagedata = testdata.astype(np.float)
    imagedata = np.multiply(imagedata, 1.0 / 255)
    # batchsize = 100
    # globalstep = tf.Variable(0, trainable=False)
    # varave = tf.train.ExponentialMovingAverage(0.99, globalstep)
    # varaveop = varave.apply(tf.trainable_variables())
    # crossentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=lenet5y)
    # crossentrogymean = tf.reduce_mean(crossentropy)
    # loss = crossentrogymean + tf.add_n(tf.get_collection('loss'))
    # learnrate = tf.train.exponential_decay(0.01, globalstep, 400, 0.99)
    # trainstep = tf.train.GradientDescentOptimizer(learnrate).minimize(loss, global_step=globalstep)
    # trainop = tf.group(trainstep, varaveop)
    y = tf.nn.softmax(lenet5y)
    saver = tf.train.Saver()
    yout = tf.arg_max(y, 1)
    with tf.Session() as sess:
        saver.restore(sess, '/ckpt/lenet5113.ckpt')
        n = int(len(imagedata) / 1000)
        data = [0] * len(imagedata)
        k = 0
        for i in range(n):
            x1 = imagedata[i * 1000:(i + 1) * 1000]
            x1 = np.reshape(x1, (1000, 28, 28, 1))
            result = sess.run(yout, feed_dict={x: x1, droprate: 1.0})
            for j in range(1000):
                data[k] = result[j]
                k += 1
        index = list(range(1, len(data) + 1))
        pd.DataFrame(data=data, index=index, columns=['Label']).to_csv('result.csv')


if not os.path.exists('/ckpt/'):
    os.mkdir('/ckpt/')
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
droprate = tf.placeholder(tf.float32)
inputdata = tf.reshape(x, [-1, 28, 28, 1])
conv1w = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
conv1b = tf.Variable(tf.constant(0.1, shape=[32]))
conv1 = tf.nn.conv2d(inputdata, conv1w, strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1b))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv2w = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
conv2b = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.conv2d(pool1, conv2w, strides=[1, 1, 1, 1], padding='SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2b))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv3w = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1))
conv3b = tf.Variable(tf.constant(0.1, shape=[128]))
conv3 = tf.nn.conv2d(pool2, conv3w, strides=[1, 1, 1, 1], padding='SAME')
relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3b))
pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
poolshape = pool3.get_shape().as_list()
nodes = poolshape[1] * poolshape[2] * poolshape[3]
reshaped = tf.reshape(pool3, [-1, nodes])
fc1w = tf.Variable(tf.truncated_normal([nodes, 1024], stddev=0.1))
tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.0001)(fc1w))
fc1b = tf.Variable(tf.constant(0.1, shape=[1024]))
fc1 = tf.nn.relu(tf.matmul(reshaped, fc1w) + fc1b)
fc1 = tf.nn.dropout(fc1, droprate)
fc2w = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.0001)(fc2w))
fc2b = tf.Variable(tf.constant(0.1, shape=[10]))
lenet5y = tf.matmul(fc1, fc2w) + fc2b
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