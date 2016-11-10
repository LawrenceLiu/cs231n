#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import numpy as np


def unpickle(filename):
    import cPickle
    fin = open(filename, 'rb')
    data = cPickle.load(fin)
    fin.close()
    return data

def main():

    #path="../data/cifar-10-batches-py/batches.meta"

    # load all data

    # make K_NearestNeighbor classifier

    # maybe do some validations

    # load test data

    # do prediction

if __name__ == "__main__":
    main()
