# -*- coding: utf-8 -*-

"""
This files contains various pytorch dataset classes, that provide
data to the Transformer model
"""
from torch.utils.data import Dataset
# IterableDataset is available after version 1.2, However, here the version is
# 1.1 so it is still unavailable
# from torch.utils.data import IterableDataset
from typing import List
import torch
import logging
from tqdm import tqdm
from input_instance import InputInstance
from csv_reader import CSVDataReader
# from transformers import BertTokenizer
# from transformers import BertModel
from roberta_with_finetune import ROBERTA
from torch.utils.data import DataLoader
# import sys
# from torch.nn import TransformerEncoder

# from transformers import TransfoXLConfig, TransfoXLModel


class TorchWrappedDataset(Dataset):
    """
    Dataset for smart batching, that is
     each batch is only padded to its
     longest sequence instead of padding all
     sequences to the max length.
    The SentenceBertEncoder.smart_batching_collate is
     required for this to work.
     SmartBatchingDataset does *not* work without it.

    inherit torch.utils.data.Dataset
     and must implement __getitem__ or __iter__

    Be careful, there is a sentence label dataset.
    You have to implement
    torch.utils.data.IterableDataset to use iter_next,
    if not, iter_next will iterate over the labels.
    """

    def __init__(
            self, instances: List[InputInstance],
            model: ROBERTA,
            # tokenizer_model_name_or_path: str = 'bert-base-uncased',
            max_seq_length: int = 25,
            show_progress_bar: bool = None,
            do_lower_case: bool = True):
        """
        Create a new TorchWrappedDataset with the
         tokenized texts and the labels as Tensor
        """
        if show_progress_bar is None:
            show_progress_bar = \
                (logging.getLogger().getEffectiveLevel() == logging.INFO or
                 logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.tokens = None
        self.labels = None
        self.lengths = None
        self.textlengths = None
        self.textmasks = None
        self.max_seq_length = max_seq_length
        self.max_text_seq_length = -1
        self.max_dataset_seq_length = -1
        self.max_dataset_text_seq_length = -1
        self.convert_input_instances_and_pad(
            instances,
            model,
            do_lower_case)
        print(self.max_dataset_seq_length)

    def convert_input_instances_and_pad(
            self, instances: List[InputInstance],
            model: ROBERTA,
            do_lower_case: bool = True):
        """
        Converts input instances to a SmartBatchingDataset
         together with the calling of torch.utils.data.DataLoader
         usable to train the model with
         .smart_batching_collate as the collate_fn for the DataLoader

        smart_batching_collate as collate_fn is required because
         it transforms the tokenized texts to the tensors.

        The unaligned utterances are padded so that they can match

        :param instances:
            the input instances for the train

        :return: a SmartBatchingDataset usable to train the model
         with .smart_batching_collate as the collate_fn
         for the DataLoader
        """

        # uppder text seq length limit
        max_text_seq_length = model.get_max_seq_length()
        # print('---max_text_seq_length', max_text_seq_length)
        # current max_seq_length within this dataset
        max_dataset_seq_length = -1
        max_dataset_text_seq_length = -1

        inputs = []
        # [[] for _ in range(num_instances)]
        labels = []
        # [[] for _ in range(num_instances)]
        lengths = []
        text_lengths = []
        text_masks = []

        label_type = None
        iterator = instances

        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert dataset")

        # a full zero list for copy
        # actually I don't think there would be a manipulation of the fullzero
        # read-access only
        # rather, the tokens should all be in the basic elements
        # So it won't be a problem
        fullzerotokens_4_copy = model.tokenize_and_pad(
            '',
            # add_special_tokens=False) # Originally add_special_tokens=False, then pad to all zero token, but with the uptade we have to set it to True otherwise the padding will not be performed
            add_special_tokens=True)
        fullzeromask_4_copy = [0] * max_text_seq_length
        # print(max_text_seq_length)
        for ex_index, instance in enumerate(iterator):
            # the internal/extra index within this scope,
            #  not the global unique index
            if label_type is None:
                if isinstance(instance.labels[0], int):
                    # long currently means int32, althought python's
                    # int is unlimited
                    label_type = torch.long
                elif isinstance(instance.labels[0], float):
                    # float means float32
                    label_type = torch.float

            num_texts = len(instance.texts)
            if num_texts > self.max_seq_length:
                logging.info("Instance {} longer than max_seq_length, actual length {}".format(ex_index, num_texts))
                logging.info("Set instence {} to max_seq_length {}".format(ex_index, self.max_seq_length))
                num_texts = self.max_seq_length
            if num_texts > max_dataset_seq_length:
                max_dataset_seq_length = num_texts
            too_long = [0] * num_texts
            instance_text_lengths = [0] * num_texts
            # text_masks = [fullzero_4_copy.copy()] * self.max_seq_length

            # dangerous! the list() within [] will share one object,
            # thus accumulating! see tentative tests for information
            # instance_text_masks = [list()] * self.max_seq_length
            instance_text_masks = []
            # cannot be append([]) then append fullzeromask_4_copy
            # since [] is also an element. So the fullzeromask_4_copy
            # will be attached outside the self.max_seq_length
            # for _ in range(self.max_seq_length):
            for _ in range(num_texts):
                instance_text_masks.append([])

            tokenized_text_lens = []
            tokenized_texts = []
            text_i = 0
            for text in instance.texts:
                # 2 special tokens
                tokenized_text_lens.append(len(model.tokenize(text)) + 2)
                tokenized_texts.append(model.tokenize_and_pad(text))
                text_i += 1
                if text_i == num_texts:
                    break
            # tokenized_textpairs = [
            #     (len(model.tokenize(text)), model.tokenize_and_pad(text))
            #     for text in instance.texts]
            # print(len(tokenized_texts[0]))

            for i, tokens in enumerate(tokenized_texts):
                len_tokens = tokenized_text_lens[i]
                # if len_tokens > 128:
                #     print('len_tokens', len_tokens)
                if max_dataset_text_seq_length < len_tokens:
                    max_dataset_text_seq_length = len_tokens
                instance_text_lengths[i] = len_tokens
                if len_tokens > max_dataset_text_seq_length:
                    max_dataset_text_seq_length = len_tokens
                if max_text_seq_length > 0 and\
                   len_tokens >= max_text_seq_length:
                    # too_long[i] = 1 means that i is exceeding the length
                    too_long[i] += 1
                    logging.info(
                        "Instance {} utterance {} longer than max_text_seq_length: {}, actual length {}".format(ex_index, i, max_text_seq_length, len_tokens))
                # text masks, each [i] append with [[1][1][1]...[0]] within seq-len
                # [1] * len_tokens
                if len_tokens < max_text_seq_length:
                    for _ in range(len_tokens):
                        instance_text_masks[i].append(1)
                    for _ in range(len_tokens, max_text_seq_length):
                        instance_text_masks[i].append(0)
                else:
                    for _ in range(max_text_seq_length):
                        instance_text_masks[i].append(1)
                # print(instance_text_masks[i])
                # if len(instance_text_masks[i]) > 128:
                #     print('Error: instance_text_masks len > 128')
                # print(len(instance_text_masks[0]))

            if num_texts == self.max_seq_length:
                instance_labels = instance.labels[:num_texts].copy()
            else:
                instance_labels = instance.labels.copy()
            # padding for both text_mask and texts, outside seq-len
            # append for max_seq_len - num_texts times
            for i in range(num_texts, self.max_seq_length):
                tokenized_texts.append(fullzerotokens_4_copy.copy())
                instance_text_masks.append(fullzeromask_4_copy.copy())
                # if len(instance_text_masks[i]) > 128:
                #     print('Error: instance_text_masks len > 128')
                # print(len(fullzeromask_4_copy.copy()))
                if label_type is torch.long:
                    instance_labels.append(0)
                elif label_type is torch.float:
                    instance_labels.append(0.0)
                else:
                    # default label type equals 0
                    instance_labels.append(0)
                instance_text_lengths.append(0)

            for i in range(num_texts, self.max_seq_length):
                if len(instance_text_masks[i]) > max_text_seq_length:
                    print('Error: instance_text_masks len > {}'.format(max_text_seq_length))
                if len(instance_text_masks[i]) < max_text_seq_length:
                    print('Error: instance_text_masks len < {}'.format(max_text_seq_length))
                # print(instance_text_masks[i])
            # print(len(tokenized_texts))
            # print('max_seq_length=', self.max_seq_length)
            # print(tokenized_texts)
            inputs.append(tokenized_texts)
            labels.append(instance_labels)
            lengths.append(num_texts)
            text_lengths.append(instance_text_lengths)
            text_masks.append(instance_text_masks)

        # # padding is integated into the tokenization
        # iterator = inputs
        # if self.show_progress_bar:
        #     iterator = tqdm(iterator, desc="Perform utterance padding")
        # for ex_index, ainput in enumerate(iterator):
        #     for _ in range(self.max_seq_length - lengths[ex_index]):
        #         inputs[ex_index].append(model.tokenize_and_pad(
        #             '', add_special_tokens=False))
        #         labels[ex_index].append(0)
            # print(len(inputs[ex_index]))
            # print(max_dataset_seq_length)

        # LongTensor inherently call asarray, which doen't treat array as element but an array
        self.tokens = torch.LongTensor(inputs)
        # long tensor in our dataet
        self.labels = torch.LongTensor(labels)
        # use both the list or LongTensor are OK
        # since the dataloader only transforms the first 2 tenses
        # self.lengths = lengths
        self.lengths = torch.LongTensor(lengths)
        # for masking

        # self.textlengths = torch.LongTensor(text_lengths)
        self.textlengths = torch.LongTensor(text_lengths)
        self.textmasks = torch.LongTensor(text_masks)
        self.max_text_seq_length = max_text_seq_length
        self.max_dataset_seq_length = max_dataset_seq_length
        self.max_dataset_text_seq_length = max_dataset_text_seq_length

    def get_max_seq_len(self):
        max_len = 0
        for alist_of_tokens in self.tokens:
            the_length = len(alist_of_tokens)
            if max_len < the_length:
                max_len = the_length
        return max_len

    def get_label_category_count(self):
        # # equivalent to size(0)
        # return self.labels.unique().size()[0]
        return self.labels.unique().size(0)

    def __getitem__(self, item):

        # return [self.tokens[i][item] for
        # i in range(len(self.tokens))], self.labels[item]
        # be careful, you can only return

        return self.tokens[item], self.labels[item], self.lengths[item], self.textlengths[item], self.textmasks[item]
        # self.textlengths[item]
        # self.textlengths[item]
        # use LongTensor to solve 'list' object has no attribute 'size'
        # equals to (self.tokens[item], self.labels[item])

    def __len__(self):
        return len(self.tokens)
