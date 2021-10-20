# -*- coding: utf-8 -*-

import os
import sys
import torch
from src.data import config as cfg
from src.interactive import functions as utilfuncs
import csv


def run_generator(filename):
    saved_pretrained_model_file = \
        'datasets/comet_pretrained_models/atomic_pretrained_model.pickle'
    device = 'cpu'
    sampling_algorithm = 'topk-3'
    opt, state_dict = utilfuncs.load_model_file(saved_pretrained_model_file)
    data_loader, text_encoder = utilfuncs.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = utilfuncs.make_model(opt, n_vocab, n_ctx, state_dict)

    if device != "cpu":
        cfg.device = int(device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    data = csv.reader(
        open(filename, encoding="utf-8"),
        delimiter='\t', quoting=csv.QUOTE_NONE)
    dialogues = [[uttr for uttr in row if uttr not in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}] for row in data]
    data = csv.reader(
        open(filename, encoding="utf-8"),
        delimiter='\t', quoting=csv.QUOTE_NONE)
    emotions = [[label for label in row if label in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}] for row in data]
    category = ["oReact", "xIntent", "xReact"]
    sampler = utilfuncs.set_sampler(opt, sampling_algorithm, data_loader)

    new_dialogues = []
    for dialogue in dialogues:
        newuttrs = []
        for uttr in dialogue:
            concatenate_str = ''
            results = utilfuncs.get_atomic_sequence(
                uttr, model, sampler, data_loader, text_encoder, category)
            xintentions = [anevent for anevent in results['xIntent']['beams'] if anevent != 'none']
            if len(xintentions) > 0:
                if len(xintentions) == 1:
                    xintention = 'PersonX wanted %s.' % xintentions[0]
                else:
                    xintention = 'PersonX wanted %s.' % ' and '.join(xintentions)
                concatenate_str = xintention
            else:
                xintention = ''
            xreactions = [anevent for anevent in results['xReact']['beams'] if anevent != 'none']
            if len(xreactions) > 0:
                if len(xreactions) == 1:
                    xreaction = 'PersonX will feel %s.' % xreactions[0]
                else:
                    xreaction = 'PersonX will feel %s.' % ' and '.join(xreactions)
                concatenate_str += ' ' + xreaction
            else:
                xreaction = ''
            oreactions = [anevent for anevent in results['oReact']['beams'] if anevent != 'none']
            if len(oreactions) > 0:
                if len(oreactions) == 1:
                    oreaction = 'PersonY will feel %s.' % oreactions[0]
                else:
                    oreaction = 'PersonY will feel %s.' % ' and '.join(oreactions)
                concatenate_str += ' ' + oreaction
            else:
                oreaction = ''
            if concatenate_str == '':
                newuttrs.append(uttr)
            else:
                newuttrs.append(uttr + ' ' + concatenate_str)
        new_dialogues.append(newuttrs)

    datawriter = csv.writer(
        open(filename[:-4] + '_ext' + '.csv', 'wt', encoding="utf-8"),
        delimiter='\t', quoting=csv.QUOTE_NONE)
    for idx, dialogue in enumerate(new_dialogues):
        datawriter.writerow(dialogue + emotions[idx])
