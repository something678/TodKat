# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
import scipy.spatial
import pandas as pd
import json
import os
import csv

# read the atomic dataset
df = pd.read_csv("./datasets/v4_atomic_all.csv", index_col=0)
df.iloc[:, :9] = df.iloc[:, :9].apply(
    lambda col: col.apply(json.loads))

embedder = SentenceTransformer('bert-base-nli-mean-tokens')
index_values = df.index.values[:]
atomicdict = {}
for idx, anitem in enumerate(index_values):
    if anitem not in atomicdict:
        atomicdict[anitem] = [idx]
    else:
        atomicdict[anitem].append(idx)

# ########## Multiple intentions to keep.
atomicevents = list(atomicdict.keys())

# #########Convert the atomicevents
event_embeddings = embedder.encode(atomicevents)

# uttrs = ['I have to work on Saturday.',
#            'She sent me a gift.',
#            'I passed the exam.',
#            'He made me a cup of coffee.']

filename = 'datasets/dialogues_test.csv'

data = csv.reader(
    open(filename, encoding="utf-8"),
    delimiter='\t', quoting=csv.QUOTE_NONE)
dialogues = [[uttr for uttr in row if uttr not in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}] for row in data]
data = csv.reader(
    open(filename, encoding="utf-8"),
    delimiter='\t', quoting=csv.QUOTE_NONE)
emotions = [[label for label in row if label in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}] for row in data]

dialogue_embeddings = [embedder.encode(utters)
                       for utters in dialogues]

# Find the closest 3 event of the knowledgebase
#  for each utterance based on cosine similarity
top_k = 3
new_dialogues = []
for dialogue, dialogue_embedding in zip(dialogues, dialogue_embeddings):
    # compute cosine similarity as the sentence distance
    newuttrs = []
    for uttr, utter_embedding in zip(dialogue, dialogue_embedding):
        distances = scipy.spatial.distance.cdist(
            [utter_embedding], event_embeddings, "cosine")[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        counter = 0
        ext_str = ''
        for idx, distance in results[0: top_k]:
            if counter >= top_k:
                break
            for ex_idx in atomicdict[atomicevents[idx]]:
                if counter >= top_k:
                    break
                if df.iloc[ex_idx].name != 'PersonX realizes it was saturday':
                    continue
                concatenate_str = ''
                if len(df.iloc[ex_idx].values[5]) > 0 and df.iloc[ex_idx].values[5][0] != 'none':
                    if len(df.iloc[ex_idx].values[5]) == 1:
                        if df.iloc[ex_idx].values[5][0].endswith('.'):
                            xintention = 'PersonX wanted %s.' % df.iloc[ex_idx].values[5][0][:-1]
                        else:
                            xintention = 'PersonX wanted %s.' % df.iloc[ex_idx].values[5][0]
                    else:
                        xintention = 'PersonX wanted %s.' % ' and '.join([aclause[:-1] if aclause.endswith('.') else aclause for aclause in df.iloc[ex_idx].values[5]])
                    concatenate_str += xintention
                else:
                    xintention = ''
                if len(df.iloc[ex_idx].values[7]) > 0 and df.iloc[ex_idx].values[7][0] != 'none':
                    if len(df.iloc[ex_idx].values[7]) == 1:
                        if df.iloc[ex_idx].values[7][0].endswith('.'):
                            xreaction = 'PersonX will feel %s.' % df.iloc[ex_idx].values[7][0][:-1]
                        else:
                            xreaction = 'PersonX will feel %s.' % df.iloc[ex_idx].values[7][0]
                    else:
                        xreaction = 'PersonX will feel %s.' % ' and '.join([aclause[:-1] if aclause.endswith('.') else aclause for aclause in df.iloc[ex_idx].values[7]])
                    concatenate_str += ' ' + xreaction
                else:
                    xreaction = ''
                if len(df.iloc[ex_idx].values[1]) > 0 and df.iloc[ex_idx].values[1][0] != 'none':
                    if len(df.iloc[ex_idx].values[1]) == 1:
                        if df.iloc[ex_idx].values[1][0].endswith('.'):
                            oreaction = 'PersonY will feel %s.' % df.iloc[ex_idx].values[1][0][:-1]
                        else:
                            oreaction = 'PersonY will feel %s.' % df.iloc[ex_idx].values[1][0]
                    else:
                        oreaction = 'PersonY will feel %s.' % ' and '.join([aclause[:-1] if aclause.endswith('.') else aclause for aclause in df.iloc[ex_idx].values[1]])
                    concatenate_str += ' ' + oreaction
                else:
                    oreaction = ''
                ext_str += ' ' + concatenate_str
                counter += 1
        if ext_str == '':
            newuttrs.append(uttr)
        else:
            newuttrs.append(uttr + ' ' + ext_str)
    new_dialogues.append(newuttrs)
datawriter = csv.writer(
    open(filename[:-4] + '_ext' + '.csv', 'wt', encoding="utf-8"),
    delimiter='\t', quoting=csv.QUOTE_NONE)
for idx, dialogue in enumerate(new_dialogues):
    datawriter.writerow(dialogue + emotions[idx])
