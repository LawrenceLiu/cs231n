#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def sift_down(array, array_idx, start, end):
    root = start
    while True:
        child = root * 2 + 1
        if child > end:
            break
        if child + 1 <= end and array[child + 1] > array[child]:
            child = child + 1
        if array[root] < array[child]:
            array[root], array[child] = array[child], array[root]
            array_idx[root], array_idx[child] = array_idx[child], array_idx[root]
            root = child
        else:
            break

def sift_up(array, array_idx, start, end):
    root = start
    while True:
        child = root * 2 + 1
        if child > end:
            break
        if child + 1 <= end and array[child + 1] < array[child]:
            child = child + 1
        if array[root] > array[child]:
            array[root], array[child] = array[child], array[root]
            array_idx[root], array_idx[child] = array_idx[child], array_idx[root]
            root = child
        else:
            break

def min_heap_sort(array, array_idx):
    if len(array) <= 1:
        return

    for index in xrange((len(array) // 2 -1), -1, -1):
        sift_down(array, array_idx, index, len(array) - 1)

    for end in xrange(len(array) - 1, 0, -1):
        array[0], array[end] = array[end], array[0]
        array_idx[0], array_idx[end] = array_idx[end], array_idx[0]
        sift_down(array, array_idx, 0, end - 1)



def min_k(array, k=1):
    if len(array) <= k:
        return

    top_k = array[:k]
    top_k_idx = range(k)
    for index in xrange((len(top_k) // 2 -1), -1, -1):
        sift_down(top_k, top_k_idx, index, len(top_k) - 1)
    for idx in xrange(k, len(array)-1):
        if array[idx] >= top_k[0]:
            continue
        else:
            top_k[0] = array[idx]
            top_k_idx[0] = idx
            sift_down(top_k, top_k_idx, 0, k-1)
    min_heap_sort(top_k, top_k_idx)

    return top_k, top_k_idx


def main():
    np.random.seed(1107)
    test_data = np.random.randint(0, 10, 10)
    #test_data = [83, 6, 18, 50, 53, 28, 75, 27, 25, 43]

    #array = test_data
    #array_idx = range(len(test_data))
    print "Raw: %s" % (test_data)
    #print "Raw index: %s" % (array_idx)
    #min_heap_sort(array, array_idx)
    #print "After sort: %s" % (array)
    #print "Index: %s" % (array_idx)

    array = test_data
    top_k, top_k_idx = min_k(array, 5)
    print "Min K sort: %s" % (top_k)
    print "Min K index: %s" % (top_k_idx)


if __name__ == "__main__":
    main()





