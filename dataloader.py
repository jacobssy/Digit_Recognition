from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from skimage import  transform ,io
import os

def prepare_data():
    index = np.zeros(3000, dtype='int')
    for i in range(2999):
        index[i] = i * 15 + np.random.randint(0, 15)
    (X, Y), (X_test, y_test) = mnist.load_data(path='mnist.npz')
    X = X[:, :, :, np.newaxis]
    X_test = X_test[:, :, :, np.newaxis]
    Y = np_utils.to_categorical(Y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    X_val = X[index]
    y_val = Y[index]
    # creat the training index1
    index1 = np.setdiff1d(np.array(range(60000)), index, assume_unique=True)
    X_train = X[index1]
    y_train = Y[index1]
    print 'Train data shape: ', X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Validation labels shape: ', y_val.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape
    return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data1(IMG_SIZE):
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    with open('./label.txt') as  f:
        for line in f.readlines():
            img_path,label = line.split(',')
            img_path = os.path.join('/data/sunsiyuan/2018learning/handWriting/pictures',img_path)
            X_train.append(transform.resize(io.imread(img_path),(IMG_SIZE, IMG_SIZE)))
            y_train.append(int(label))
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # X_train = X_train[:, :, :, np.newaxis]
    # X_train = X_train.reshape(10, 784)
    y_train = np_utils.to_categorical(y_train, 10)
    # with open('./label.txt.bak') as  f:
    #     for line in f.readlines():
    #         img_path,label = line.split(',')
    #         img_path = os.path.join('/data/sunsiyuan/2018learning/handWriting/pictures3',img_path)
    #         X_valid.append(transform.resize(io.imread(img_path), (IMG_SIZE, IMG_SIZE)))
    #         y_valid.append(int(label))
    # X_valid = np.array(X_valid)
    # y_valid = np.array(y_valid)
    # X_valid = X_valid[:, :, :, np.newaxis]
    # X_valid = X_valid.reshape(29, 784)
    # y_valid = np_utils.to_categorical(y_valid, 10)
    print 'Train data shape: ', X_train.shape
    print 'Train labels shape: ', y_train.shape
    # print 'valid data shape: ', X_valid.shape
    # print 'valid labels shape: ', y_valid.shape
    return X_train, y_train

