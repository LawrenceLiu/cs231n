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
    #data = unpickle(path)

    path="../data/cifar-10-batches-py/data_batch_1"
    data = unpickle(path)
    print len(data['labels'])
    print data['labels'][:10]

if __name__ == "__main__":
    main()
