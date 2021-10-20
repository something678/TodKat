# -*- coding: utf-8 -*-

# import os

import atomic_phrase_generator as apg

if __name__ == '__main__':
    filename = 'datasets/dialogues_test.csv'
    apg.run_generator(filename)
    # os.chdir('../')
    # print(os.getcwd())
