# -*- coding: utf-8 -*-

from torch import nn
from transformers import BertModel, BertTokenizer
import json
from typing import List
import os
import numpy as np
import logging
from torch import LongTensor, tensor
import torch


class BERT(nn.Module):
    """BERT model to generate word embeddings.
    actually makes little differences.
    """

    def __init__(
            self, model_name_or_path: str,
            max_seq_length: int = 128, do_lower_case: bool = True):
        super(BERT, self).__init__()

        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case
        if max_seq_length > 510:
            '''
            The max_seq_length has been defined with the customized model.
            '''
            logging.warning("BERT only allows a max_seq_length of 510 (512 with"
                            "special tokens). Value will be set to 510")
            max_seq_length = 510
        logging.info("BERT max_seq_length set to {}".format(max_seq_length))
        self.max_seq_length = max_seq_length

        # similar to load, however, this
        # function will first download and the load
        # config can still be loaded and overwrite the original config
        # default bertConfiguration is not to be fine-tuned.
        self.bert = BertModel.from_pretrained(
            model_name_or_path)
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=do_lower_case,
            max_seq_length=self.max_seq_length)
        # others like pad_token='[PAD]' can be set here.
        # self.cls_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        # self.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])[0]

    def forward(self, features):
        """
        If you do not have parameters, you don't have to rewrite this.
        forward just make it easier for model() and inherently easier to move
        all parameters to cuda by model.to(device num)
        """
        '''
        Here, the authors even reimplemented the forward by adding a parameter,
         showing that this works.

        Not sure whether the bert is to be fine-tuned or to be fixed,
        search the bookmark 'The values in kwargs' you will see that that the
        default bertConfiguration is not to be fine-tuned.

        We have to re-implement forward, since the bert module need to be put into
         the nn.Sequential.
        '''
        '''[CLS] tokens of 1st sentence [SEP] tokens of 2nd sentence... [SEP]'''

        # output_tokens = self.bert(input_ids=features['input_ids'], token_type_ids=features['token_type_ids'], attention_mask=features['input_mask'])[0]
        # cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        # features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'input_mask': features['input_mask']})
        batch_size, max_seq_len_ex, max_text_seq_len = features[0].size()
        # padding enables the reshape. Mask reduces the computation.
        # after computing with bert, we will change the values with paddings
        # print(features[0].size())
        tokens_flattened = features[0].view(
            batch_size * max_seq_len_ex, max_text_seq_len)
        # Here, you can already use stack&assign
        masks_flattened = features[4].view(
            batch_size * max_seq_len_ex, max_text_seq_len)
        # index can only be used in cpu, if in GPU, you have to use
        #  index_select
        #  However, seems that you can use [:, 0, :] to avoid this
        # No, now the emotion_transformer_test has proven that this is feasible
        #  since bert returns a list and in each list there's a cuda tensor
        output_tokens = self.bert(
            input_ids=tokens_flattened,
            attention_mask=masks_flattened)[0]
        cls_tokens = output_tokens[:, 0, :]
        we_dim = self.get_word_embedding_dimension()
        cls_tokens = cls_tokens.view(batch_size, max_seq_len_ex, we_dim)
        # default dtype=torch.float
        fullzeropad4assign = torch.zeros(we_dim)
        seqlens = features[2]
        for ibatch in range(batch_size):
            for iseq in range(seqlens[ibatch], max_seq_len_ex):
                cls_tokens[ibatch, iseq, :] = fullzeropad4assign
        # output_tokens = self.bert(
        #     input_ids=features[0])
        features[0] = cls_tokens
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.bert.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        Without adding special tokens
        """
        return self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(
                text))
        # return self.tokenizer.convert_tokens_to_ids(
        #     self.tokenizer.tokenize(
        #         text,
        #         add_special_tokens=True))

    def tokenize_and_pad(
            self, text: str, add_special_tokens=True) -> List[int]:
        '''
        Tokenize a text, convert tokens to ids, and pad them
        add special tokens
        '''
        return self.tokenizer.encode(
            text,
            max_length=self.max_seq_length,
            add_special_tokens=add_special_tokens,
            pad_to_max_length=True)
        # with add_special_tokens=True you will be automatically charged
        # 2 more tokens

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length)

        tokens = tokens[:pad_seq_length]
        input_ids = [self.cls_token_id] + tokens + [self.sep_token_id]
        sentence_length = len(input_ids)

        pad_seq_length += 2
        # # Add Space for CLS + SEP token

        token_type_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length. BERT: Pad to the right
        padding = [0] * (pad_seq_length - len(input_ids))
        input_ids += padding
        token_type_ids += padding
        input_mask += padding

        assert len(input_ids) == pad_seq_length
        assert len(input_mask) == pad_seq_length
        assert len(token_type_ids) == pad_seq_length

        return {'input_ids': np.asarray(input_ids, dtype=np.int64), 'token_type_ids': np.asarray(token_type_ids, dtype=np.int64), 'input_mask': np.asarray(input_mask, dtype=np.int64), 'sentence_lengths': np.asarray(sentence_length, dtype=np.int64)}

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.bert.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    def get_max_seq_length(self):
        '''
        '''
        # if hasattr(self, 'max_seq_length'):
        #    return self._first_module().max_seq_length
        return self.max_seq_length
        # return None

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
            config = json.load(fIn)
        return BERT(model_name_or_path=input_path, **config)
