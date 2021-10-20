# -*- coding: utf-8 -*-

# import os

import atomic_generator as atg

if __name__ == '__main__':
    filename = 'datasets/dialogues_test.csv'
    atg.run_generator(filename)
    # os.chdir('../')
    # print(os.getcwd())
