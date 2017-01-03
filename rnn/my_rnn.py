#!/usr/bin/env python
# encoding: utf-8

import logging
import time

import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(x))

class RNNOneHot(object):
    def __init__(self, vocab_size, hidden_size, learning_rate):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.alpha = -1.0*learning_rate

        self.Wxh = np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, vocab_size))
        self.Whh = np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, hidden_size))
        self.bh = np.zeros(shape=(hidden_size, 1))
        self.Why = np.random.uniform(low=-0.01, high=0.01, size=(vocab_size, hidden_size))
        self.by = np.zeros(shape=(vocab_size, 1))

        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mbh = np.zeros_like(self.bh)
        self.mWhy = np.zeros_like(self.Why)
        self.mby = np.zeros_like(self.by)

        self.reset_state()

    def reset_state(self):
        self.state = np.zeros(shape=(self.hidden_size, 1))

    def predict(self, start_seed, out_len):
        vocab = range(self.vocab_size)
        out_idxs = [0] * out_len
        h = np.copy(self.state)
        x = np.zeros(shape=(self.vocab_size, 1))
        x[start_seed] = 1
        for i in range(out_len):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            exp_y = np.exp(y)
            p = exp_y / np.sum(exp_y)
            pred_idx = np.random.choice(vocab, p=p.ravel())
            x = np.zeros(shape=(self.vocab_size, 1))
            x[pred_idx] = 1
            out_idxs[i] = pred_idx

        return out_idxs

    def fit(self, input_ids, target_ids):
        loss = 0.0
        steps = len(input_ids)
        xs, hs, ys, ps = {}, {}, {}, {}
        # forward
        hs[-1] = np.copy(self.state)
        for i in xrange(steps):
            xs[i] = np.zeros(shape=(self.vocab_size, 1))
            xs[i][input_ids[i], 0] = 1
            hs[i] = np.tanh(np.dot(self.Wxh, xs[i]) + np.dot(self.Whh, hs[i-1]) + self.bh)
            ys[i] = np.dot(self.Why, hs[i]) + self.by
            exp_y = np.exp(ys[i])
            ps[i] = exp_y / np.sum(exp_y)
            loss += -np.log(ps[i][target_ids[i], 0])
        # backward
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(xrange(steps)):
            dy = np.copy(ps[t])
            dy[target_ids[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        # params clip
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        # update params
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem += dparam**2
            param += self.alpha * dparam / np.sqrt(mem + 1e-8)
            # p aram += self.alpha * dparam
        self.state = hs[steps-1]
        return loss

class GRUOneHot(object):
    def __init__(self, vocab_size, hidden_size, learning_rate):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.alpha = -1.0*learning_rate

        # update gate
        self.Wxz = np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, vocab_size))
        self.Whz = np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, hidden_size))
        self.bz = np.zeros(shape=(hidden_size, 1))

        # reset gate
        self.Wxr = np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, vocab_size))
        self.Whr = np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, hidden_size))
        self.br = np.zeros(shape=(hidden_size, 1))

        # hidden state
        self.Wxh = np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, vocab_size))
        self.Whh = np.random.uniform(low=-0.01, high=0.01, size=(hidden_size, hidden_size))
        self.bh = np.zeros(shape=(hidden_size, 1))

        # output
        self.Why = np.random.uniform(low=-0.01, high=0.01, size=(vocab_size, hidden_size))
        self.by = np.zeros(shape=(vocab_size, 1))

        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mbh = np.zeros_like(self.bh)
        self.mWhy = np.zeros_like(self.Why)
        self.mby = np.zeros_like(self.by)

        self.state = np.zeros(shape=(self.hidden_size, 1))

    def reset_state(self):
        self.state = np.zeros(shape=(self.hidden_size, 1))

    def predict(self, start_seed, out_len):
        vocab = range(self.vocab_size)
        out_idxs = [0] * out_len
        h = np.copy(self.state)
        x = np.zeros(shape=(self.vocab_size, 1))
        x[start_seed] = 1
        for i in range(out_len):
            z = sigmoid(np.dot(self.Wxz, x) + np.dot(self.Whz, h) + self.bz)
            r = sigmoid(np.dot(self.Wxr, x) + np.dot(self.Whr, h) + self.br)
            h_hat = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, (h * r)) + self.bh)
            h = (1 - z) * h_hat + z * h
            y = np.dot(self.Why, h) + self.by
            exp_y = np.exp(y)
            p = exp_y / np.sum(exp_y)
            pred_idx = np.random.choice(vocab, p=p.ravel())
            x = np.zeros(shape=(self.vocab_size, 1))
            x[pred_idx] = 1
            out_idxs[i] = pred_idx

        return out_idxs

    def fit(self, input_ids, target_ids):
        loss = 0.0
        steps = len(input_ids)
        xs, zs, rs, h_hats, hs, ys, ps = {}, {}, {}, {}, {}, {}, {}
        # forward
        hs[-1] = np.copy(self.state)
        for i in xrange(steps):
            xs[i] = np.zeros(shape=(self.vocab_size, 1))
            xs[i][input_ids[i], 0] = 1
            zs[i] = sigmoid(np.dot(self.Wxz, xs[i]) + np.dot(self.Whz, hs[i-1]) + self.bz)
            rs[i] = sigmoid(np.dot(self.Wxr, xs[i]) + np.dot(self.Whr, hs[i-1]) + self.br)
            h_hats[i] = np.tanh(np.dot(self.Wxh, xs[i]) + np.dot(self.Whh, (hs[i-1] * rs[i])) + self.bh)
            hs[i] = (1 - zs[i]) * h_hats[i] + zs[i] * hs[i-1]
            ys[i] = np.dot(self.Why, hs[i]) + self.by
            exp_y = np.exp(ys[i])
            ps[i] = exp_y / np.sum(exp_y)
            loss += -np.log(ps[i][target_ids[i], 0])
        # # backward
        # dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        # dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        # dhnext = np.zeros_like(hs[0])
        # for t in reversed(xrange(steps)):
        #     dy = np.copy(ps[t])
        #     dy[target_ids[t]] -= 1
        #     dWhy += np.dot(dy, hs[t].T)
        #     dby += dy
        #     dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
        #     dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        #     dbh += dhraw
        #     dWxh += np.dot(dhraw, xs[t].T)
        #     dWhh += np.dot(dhraw, hs[t-1].T)
        #     dhnext = np.dot(self.Whh.T, dhraw)
        # # params clip
        # for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        #     np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        # # update params
        # for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
        #                               [dWxh, dWhh, dWhy, dbh, dby],
        #                               [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
        #     mem += dparam**2
        #     param += self.alpha * dparam / np.sqrt(mem + 1e-8)
        #     # p aram += self.alpha * dparam
        self.state = hs[steps-1]
        return loss


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
    test_file = "input_102.txt"
    # test_file = "min-char-rnn.py"
    # test_file = "input_101.txt"
    data, data_size, vocab_size, chr2id, id2chr = load_data(test_file)

    # hyper params
    hidden_size = 200
    seq_len = 25
    learning_rate = 0.01
    # my_rnn = RNNOneHot(vocab_size, hidden_size, learning_rate)
    my_rnn = GRUOneHot(vocab_size, hidden_size, learning_rate)
    fsample = open("output", 'w')

    # main train loop
    max_epoch = 1000
    epoch = 0
    cur_idx = 0
    tic = time.time()
    smooth_loss = -np.log(1.0/vocab_size) * seq_len
    while True:
        # get data
        if cur_idx + seq_len + 1 >= data_size or epoch == 0:
            my_rnn.reset_state()
            cur_idx = 0
        input_ids = [chr2id[x] for x in data[cur_idx:cur_idx+seq_len]]
        target_ids = [chr2id[x] for x in data[cur_idx+1:cur_idx+seq_len+1]]

        # make some test
        # if epoch % 100 == 0:
        #     start_seed = chr2id['[']
        #     predictions = my_rnn.predict(start_seed, 200)
        #     # predictions = my_rnn.sample(start_seed, 200)
        #     out = ''.join([id2chr[x] for x in predictions])
        #     # print >> fsample, "-----epoch:%-8d-----" % epoch
        #     # print >> fsample, out
        #     print "----\n%s\n----" % out

        # train and update params
        loss = my_rnn.fit(input_ids, target_ids)
        # loss = my_rnn.lossFun(input_ids, target_ids)
        smooth_loss = 0.99*smooth_loss + 0.01*loss
        # output training info
        if epoch % 100 == 0:
            toc = time.time()
            logging.debug("epoch: %d, loss: %.6f time cost:%.6f", epoch, smooth_loss, toc-tic)
            tic = toc

        # train and update params
        cur_idx += seq_len
        epoch += 1
        if epoch > max_epoch:
            break
    fsample.close()


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s][%(asctime)s] %(message)s",
                        level=logging.DEBUG)
    np.random.seed(870905)
    main()
