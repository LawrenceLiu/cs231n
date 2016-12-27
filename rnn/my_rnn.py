#!/usr/bin/env python
# encoding: utf-8

import logging


def main():
    logging.info("My RNN go!")

if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s][%(asctime)s] %(message)s", \
                        level=logging.DEBUG)
    main()
