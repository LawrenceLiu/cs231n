#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def sift_down(array, start, end):
    root = start
    while True:
        child = root * 2 + 1
        if child > end:
            break
        if child + 1 <= end and array[child + 1] > array[child]:
            child = child + 1
        if array[root] < array[child]:
            array[root], array[child] = array[child], array[root]
            root = child
        else:
            break

def sift_up(array, start, end):
    root = start
    while True:
        child = root * 2 + 1
        if child > end:
            break
        if child + 1 <= end and array[child + 1] < array[child]:
            child = child + 1
        if array[root] > array[child]:
            array[root], array[child] = array[child], array[root]
            root = child
        else:
            break

def min_heap_sort(array):
    if len(array) <= 1:
        return

    for index in xrange((len(array) // 2 -1), -1, -1):
        sift_down(array, index, len(array) - 1)

    for end in xrange(len(array) - 1, 0, -1):
        array[0], array[end] = array[end], array[0]
        sift_down(array, 0, end - 1)



def min_k(array, k=1):
    if len(array) <= k:
        return

    top_k = array[:k]
    sift_down(top_k, 0, k-1)
    #print top_k
    for idx in xrange(k, len(array)-1):
        if array[idx] >= top_k[0]:
            continue
        else:
            top_k[0] = array[idx]
            sift_down(top_k, 0, k-1)
        #print top_k
    min_heap_sort(top_k)

    return top_k


def main():
    np.random.seed(1107)
    test_data = np.random.randint(0, 1000000, 100000)
    #print "Raw: %s" % (test_data)

    #array = test_data
    #min_heap_sort(array)
    #print "After sort: %s" % (array)

    array = test_data
    top_k = min_k(array, 5)
    print "Min K sort: %s" % (top_k)


if __name__ == "__main__":
    main()





