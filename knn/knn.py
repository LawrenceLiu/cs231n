#!/usr/bin/env python
# encoding: utf-8

#import os
#import sys
import logging

#import numpy as np

class K_NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        pass

    def predict(self, X, k=1):
        logging.debug("K = %d", k)
        pass

def unpickle(filename):
    import cPickle

    fin = open(filename, 'rb')
    data = cPickle.load(fin)
    fin.close()
    return data

def main():
    logging.basicConfig(format="[%(levelname)s] %(asctime)s : %(message)s", level=logging.DEBUG)
    logging.info("Launch KNN model for CIFAR-10:")
    #path="../data/cifar-10-batches-py/batches.meta"

    # load all data

    # make K_NearestNeighbor classifier
    knn_model = K_NearestNeighbor()
    # maybe do some validations

    # load test data

    # do prediction
    knn_model.predict(None, k=5)

if __name__ == "__main__":
    main()
