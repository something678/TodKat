# -*- coding: utf-8 -*-

from torch import nn
from transformers import RobertaModel, RobertaTokenizer
import json
from typing import List, Dict, Union
import os
import numpy as np
import logging
from torch import LongTensor, tensor
import torch


class ROBERTA(nn.Module):
    """LM model to generate embeddings.
    interit nn.Module actually enables possible add-on parameters,
    actually makes little differences.
    """

    def __init__(
            self, model_name_or_path: str,
            max_seq_length: int = 128, do_lower_case: bool = True,
            devicepad: str = None):
        super(ROBERTA, self).__init__()
        '''
        Here, we use the "load from confiuration" structure
        '''

        # configure keys are to be saved, whose valus are retrieved
        #  from self.xxx
        # need the devicepad, otherwise it will still be None when
        #  loading although saved with cuda
        self.config_keys = ['max_seq_length', 'do_lower_case', 'devicepad']
        self.do_lower_case = do_lower_case
        if max_seq_length > 510:
            '''
            The max_seq_length has been defined with the customized model.
            '''
            logging.warning("ROBERTA only allows a max_seq_length of 510 (512 with"
                            "special tokens). Value will be set to 510")
            max_seq_length = 510
        logging.info("ROBERTA max_seq_length set to {}".format(max_seq_length))
        self.max_seq_length = max_seq_length

        # similar to load, however, this
        # function will first download and the load
        # config can still be loaded and overwrite the original config
        self.roberta = RobertaModel.from_pretrained(
            model_name_or_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=do_lower_case,
            max_seq_length=self.max_seq_length)
        self.devicepad = devicepad
        self.devicepad_device = torch.device(self.devicepad)
        # torch.device(devicepad)
        # others like pad_token='[PAD]' can be set here.
        # self.cls_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        # self.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])[0]

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        """
        If you do not have parameters, you don't have to rewrite this.
        forward just make it easier for model() and inherently easier to move
        all parameters to cuda by model.to(device num)
        """
        '''
        [CLS] token can be regarded as sentence embeddings for sentiment classification

        Here, the authors even reimplemented the forward by adding a parameter,
         showing that this works.


        We have to re-implement forward
        '''
        '''[CLS] tokens of 1st sentence [SEP] tokens of 2nd sentence... [SEP]'''

        '''
        '''

        # output_tokens = self.bert(input_ids=features['input_ids'], token_type_ids=features['token_type_ids'], attention_mask=features['input_mask'])[0]
        # cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        # features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'input_mask': features['input_mask']})
        batch_size, max_seq_len_ex, max_text_seq_len = features[0].size()
        seqlens = features[2]
        lst_uttrs = []
        lst_masks = []
        for ibatch, seqlen in enumerate(seqlens):
            lst_uttrs.append(features[0][ibatch, :seqlen, :])
            lst_masks.append(features[4][ibatch, :seqlen, :])
        tokens_flattened = torch.cat(lst_uttrs, dim=0)
        masks_flattened = torch.cat(lst_masks, dim=0)
        # logging.info(
        #     'tokens_flattened size: {}'.format(tokens_flattened.size()))
        # print(tokens_flattened.type())
        output_tokens = self.roberta(
            input_ids=tokens_flattened,
            attention_mask=masks_flattened)[0]
        cls_tokens = output_tokens[:, 0, :]
        cls_tokens = cls_tokens.to(self.devicepad_device)
        # becareful, since this is a reference, this will change the device, as well as the
        # print(tokens_flattened.type())

        # The single index will automatically unsqueeze it
        # logging.info('cls_tokens size: {}'.format(cls_tokens.size()))
        we_dim = self.get_word_embedding_dimension()
        for ibatch in range(batch_size):
            # not sure if the newly-created tensor will be on cuda
            # test showed that the newly-created tensor will be on cpu
            # So it cannot be concatenated with the tokens_flattened
            fullzeropad4insert = torch.zeros(
                max_seq_len_ex - seqlens[ibatch], we_dim).to(
                self.devicepad_device)
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            cls_tokens = torch.cat(
                [cls_tokens[:index4insert],
                 fullzeropad4insert,
                 cls_tokens[index4insert:]], dim=0)
        cls_tokens = cls_tokens.view(batch_size, max_seq_len_ex, we_dim)
        # output_tokens = self.bert(
        #     input_ids=features[0])
        return (cls_tokens, features[1], features[2], features[3], features[4])

    def get_word_embedding_dimension(self) -> int:
        return self.roberta.config.hidden_size

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
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=add_special_tokens,
            # pad_to_max_length=True) # depreciated
            padding='max_length')
        # with add_special_tokens=True you will be automatically charged
        # 2 more tokens

    # def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
    #     """
    #     Convert tokenized sentence in its embedding ids, segment ids and mask

    #     :param tokens:
    #         a tokenized sentence
    #     :param pad_seq_length:
    #         the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
    #     :return: embedding ids, segment ids and mask for the sentence
    #     """
    #     pad_seq_length = min(pad_seq_length, self.max_seq_length)

    #     tokens = tokens[:pad_seq_length]
    #     input_ids = [self.cls_token_id] + tokens + [self.sep_token_id]
    #     sentence_length = len(input_ids)

    #     pad_seq_length += 2
    #     # # Add Space for CLS + SEP token

    #     token_type_ids = [0] * len(input_ids)
    #     input_mask = [1] * len(input_ids)

    #     # Zero-pad up to the sequence length. BERT: Pad to the right
    #     padding = [0] * (pad_seq_length - len(input_ids))
    #     input_ids += padding
    #     token_type_ids += padding
    #     input_mask += padding

    #     assert len(input_ids) == pad_seq_length
    #     assert len(input_mask) == pad_seq_length
    #     assert len(token_type_ids) == pad_seq_length

    #     return {'input_ids': np.asarray(input_ids, dtype=np.int64), 'token_type_ids': np.asarray(token_type_ids, dtype=np.int64), 'input_mask': np.asarray(input_mask, dtype=np.int64), 'sentence_lengths': np.asarray(sentence_length, dtype=np.int64)}

    def get_topic_vec(self, features, split_layer: int = 6):
        # return layer 6 by default, can be 12 if large
        # self.roberta()
        batch_size, max_seq_len_ex, max_text_seq_len = features[0].size()
        seqlens = features[2]
        lst_uttrs = []
        lst_masks = []
        for ibatch, seqlen in enumerate(seqlens):
            lst_uttrs.append(features[0][ibatch, :seqlen, :])
            lst_masks.append(features[4][ibatch, :seqlen, :])
        tokens_flattened = torch.cat(lst_uttrs, dim=0)
        masks_flattened = torch.cat(lst_masks, dim=0)
        # tuple index
        # print(tokens_flattened.type())
        # print(masks_flattened.type())
        output_tokens = self.roberta(
            input_ids=tokens_flattened,
            attention_mask=masks_flattened,
            output_hidden_states=True)[2][split_layer]
        topic_vecs = output_tokens[:, 0, :]

        topic_vecs = topic_vecs.to(self.devicepad_device)

        # The single index will automatically unsqueeze it
        # logging.info('cls_tokens size: {}'.format(cls_tokens.size()))
        we_dim = self.get_word_embedding_dimension()
        for ibatch in range(batch_size):
            # not sure if the newly-created tensor will be on cuda
            # test showed that the newly-created tensor will be on cpu
            # So it cannot be concatenated with the tokens_flattened
            fullzeropad4insert = torch.zeros(
                max_seq_len_ex - seqlens[ibatch], we_dim).to(
                self.devicepad_device)
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            topic_vecs = torch.cat(
                [topic_vecs[:index4insert],
                 fullzeropad4insert,
                 topic_vecs[index4insert:]], dim=0)
        topic_vecs = topic_vecs.view(batch_size, max_seq_len_ex, we_dim)

        return topic_vecs

    def get_attentions_vec(self, features, split_layer: int = 6):
        # return layer 6 by default, can be 12 if large
        # self.roberta()
        batch_size, max_seq_len_ex, max_text_seq_len = features[0].size()
        seqlens = features[2]
        lst_uttrs = []
        lst_masks = []
        for ibatch, seqlen in enumerate(seqlens):
            lst_uttrs.append(features[0][ibatch, :seqlen, :])
            lst_masks.append(features[4][ibatch, :seqlen, :])
        tokens_flattened = torch.cat(lst_uttrs, dim=0)
        masks_flattened = torch.cat(lst_masks, dim=0)
        # tuple index
        # print(tokens_flattened.type())
        # print(masks_flattened.type())
        output_tokens = self.roberta(
            input_ids=tokens_flattened,
            attention_mask=masks_flattened,
            output_attentions=True)[2][split_layer]
        attention_vecs = output_tokens[:, :, 0, :]
        _, heads_count, m_text_seq_len = attention_vecs.size()
        assert max_text_seq_len == m_text_seq_len

        # Be careful, in the evaluation_test process
        attention_vecs = attention_vecs.to(self.devicepad_device)

        # The single index will automatically unsqueeze it
        # logging.info('cls_tokens size: {}'.format(cls_tokens.size()))
        # max_text_seq_len = self.get_word_embedding_dimension()
        for ibatch in range(batch_size):
            # not sure if the newly-created tensor will be on cuda
            # test showed that the newly-created tensor will be on cpu
            # So it cannot be concatenated with the tokens_flattened
            fullzeropad4insert = torch.zeros(
                max_seq_len_ex - seqlens[ibatch], heads_count, max_text_seq_len).to(
                self.devicepad_device)
            index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
            attention_vecs = torch.cat(
                [attention_vecs[:index4insert],
                 fullzeropad4insert,
                 attention_vecs[index4insert:]], dim=0)
        attention_vecs = attention_vecs.view(batch_size, max_seq_len_ex, heads_count, max_text_seq_len)

        return attention_vecs

    def get_config_dict(self) -> Dict[str, Union[int, bool, str]]:
        # in the saving and loading, to indicate the configurations
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.roberta.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'my_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    def get_max_seq_length(self) -> int:
        '''
        '''
        # if hasattr(self, 'max_seq_length'):
        #    return self._first_module().max_seq_length
        return self.max_seq_length
        # return None

    # Only static method can be used to retrieve an object
    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'my_bert_config.json')) as fIn:
            config = json.load(fIn)
        return ROBERTA(model_name_or_path=input_path, **config)
