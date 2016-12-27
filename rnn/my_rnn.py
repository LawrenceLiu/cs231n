#!/usr/bin/env python
# encoding: utf-8

import logging
import time

import numpy as np

class RNN_onehot(object):
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


        self.reset_state()
        pass

    def reset_state(self):
        self.state = np.zeros(shape=(self.hidden_size, 1))

    def predict(self):
        return []

    def fit(self):
        return 0.0

def load_data(filename):
    with open(filename, 'r') as fin:
        data = fin.read().decode('utf-8')
        data_size = len(data)
        vocab = list(set(data))
        vocab_size = len(vocab)
        chr2id = {ch:i for i, ch in enumerate(vocab)}
        id2chr = {i:ch for i, ch in enumerate(vocab)}
    return data, data_size, vocab_size, chr2id, id2chr

def main():
    logging.info("My RNN go!")

    # load data
    test_file = "input_101.txt"
    data, data_size, vocab_size, chr2id, id2chr = load_data(test_file)

    # hyper params
    hidden_size = 10
    seq_len = 14
    my_rnn = RNN_onehot(vocab_size, hidden_size)
    fout = open("output", 'w')

    # main train loop
    max_epoch = 1
    epoch = 0
    cur_idx = 0
    tic = time.time()
    while True:
        # get data
        if cur_idx + seq_len + 1 >= data_size or epoch == 0:
            my_rnn.reset_state()
            input_txt = [x for x in data[cur_idx:cur_idx+seq_len]]
            target_txt = [x for x in data[cur_idx+1:cur_idx+seq_len+1]]
            #input_txt = [chr2id[x] for x in data[cur_idx:cur_idx+seq_len]]
            #target_txt = [chr2id[x] for x in data[cur_idx+1:cur_idx+seq_len+1]]
        print input_txt
        print target_txt

        # make some test
        if epoch % 100 == 0:
            predictions = my_rnn.predict()
            out = ''.join([id2chr[x] for x in predictions])
            print >> fout, out

        # train and update params
        loss = my_rnn.fit()

        # output training info
        if epoch % 10 == 0:
            toc = time.time()
            print epoch
            print loss
            print toc-tic
            logging.debug("epoch: %d, loss: %.6f time cost:%.6f", epoch, loss, toc-tic)
            tic = toc

        # train and update params
        epoch += 1
        if epoch >= max_epoch:
            break
        fout.close()


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s][%(asctime)s] %(message)s", \
                        level=logging.DEBUG)
    main()
