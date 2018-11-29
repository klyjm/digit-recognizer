import tensorflow as tf
import pandas as pd
import numpy as np


def train(filename):
    traindata = pd.read_csv(filename)
    imagedata = traindata.iloc[:, 1:].values
    imagedata = imagedata.astype(np.float)
    imagedata = np.multiply(imagedata, 1.0/255)
    labeldata = traindata.iloc[:, 0].values


def test(filename):
    traindata = pd.read_csv(filename)
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
        test(inputstr[1])
    inputstr = input('请输入命令：\n')