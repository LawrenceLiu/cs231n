#!/usr/bin/env python
# encoding: utf-8

#import os
#import sys
import logging

import numpy as np

from heap_sort import min_k

class K_NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.X_train = np.array(X, dtype=np.int8)
        self.Y_train = np.array(Y, dtype=np.int8)
        pass

    def predict(self, X, k=1):
        #logging.debug("K = %d", k)
        num_test = X.shape[0]
        Y_pred = np.zeros(num_test, dtype=self.Y_train.dtype)

        for i in xrange(num_test):
            dis = np.sum(np.abs(self.X_train - X[i, :]), axis = 1)
            min_idx = np.argmin(dis)
            Y_pred[i] =  self.Y_train[min_idx]
            if i % 10 == 0:
                print "processed %d\r" % i
        print "Done !"
        print len(Y_pred)

def unpickle(filename):
    import cPickle

    fin = open(filename, 'rb')
    data = cPickle.load(fin)
    fin.close()
    return data

def load_train_data():
    prefix = "../data/cifar-10-batches-py/data_batch_"
    total = 5
    X = []
    labels = []
    for idx in range(1, total+1):
        filename = prefix + str(idx)
        print filename
        temp = unpickle(filename)
        if X == [] :
            X = temp['data']
            labels = temp['labels']
        else:
            X = np.concatenate((X, temp['data']))
            labels = np.concatenate((labels, temp['labels']))
    return X, labels


def load_test_data():
    filename = "../data/cifar-10-batches-py/test_batch"
    data = unpickle(filename)
    return data['data'], data['labels']

def main():
    logging.basicConfig(format="[%(levelname)s] %(asctime)s : %(message)s", level=logging.DEBUG)
    logging.info("Launch KNN model for CIFAR-10:")

    path="../data/cifar-10-batches-py/batches.meta"
    meta = unpickle(path)

    # load all data
    train_X, train_Y = load_train_data()
    test_X, test_Y = load_test_data()

    # make K_NearestNeighbor classifier
    knn_model = K_NearestNeighbor()
    knn_model.train(train_X, train_Y)

    # do prediction
    knn_model.predict(test_X)
    #knn_model.predict(None, k=5)

if __name__ == "__main__":
    main()
