#!/usr/bin/env python
# encoding: utf-8


import sys
import numpy as np

def quick_sort_native(array):
    if len(array) <= 1:
        return array
    return quick_sort_native([x for x in array[1:] if x <= array[0]]) \
        + array[0] \
        + quick_sort_native([x for x in array[1:] if x > array[0]])


def quick_sort_as_c(array, start, end):
    if start >= end:
        return
    key = array[start]
    #print >> sys.stdout, "\nkey: %d start: %d end: %d" % (key, start, end)
    #print array
    #i = i + 1
    i = start + 1
    j = end
    while i < j:
        #print "i:%d, j:%d" % (i, j)
        while array[i] <= key and i < j:
            i += 1
        while array[j] > key and i < j:
            j -= 1
        #print "i:%d, j:%d" % (i, j)
        array[i], array[j] = array[j], array[i]
        j -= 1
        #print array
    if array[i] < array[start]:
        array[i], array[start] = array[start], array[i]
    #print array
    quick_sort_as_c(array, start, i - 1)
    quick_sort_as_c(array, i + 1, end)
    return

def quick_sort(array):
    print >> sys.stdout, "Raw: %s" % (array)

    test = array
    quick_sort_as_c(test, 0, len(test) - 1)
    print >> sys.stdout, "After sort: %s" % (array)

    test = array
    quick_sort_native(test)
    print >> sys.stdout, "After sort: %s" % (test)

if __name__ == "__main__":
    np.random.seed(1106)
    test_data = np.random.randint(0, 100, 20)
    quick_sort(test_data)
