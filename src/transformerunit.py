# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import TransformerEncoderLayer
import logging
import os
import torch
import json


class TransformerUnit(nn.Module):
    """Transformer customized
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            out_features: int = -1):
        super(TransformerUnit, self).__init__()

        self.config_keys = ['d_model', 'n_heads', 'out_features']
        self.d_model = d_model
        self.n_heads = d_model
        self.out_features = out_features
        # activation by default the GELU
        self.transformerlayer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            activation='gelu')
        self.linear = nn.Linear(
            d_model,
            out_features,
            bias=True)

    def forward(self, features):
        """Returns embeddings"""
        '''
        '''
        uttrs = features[0]
        uttrs_tr = self.transformerlayer(uttrs)
        uttrs_ln = self.linear(uttrs_tr)
        # features[0] = uttrs_ln
        # return features
        return (uttrs_ln, features[1], features[2], features[3], features[4])

    def get_config_dict(self):
        # get current values of several keys you have mentioned
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        # logging.info('The transformer accepts a save file: '
        #              'save to {}'.format(output_path))
        torch.save(self.state_dict(), os.path.join(
            output_path, 'pytorch_model.bin'))

        with open(os.path.join(
                output_path,
                'transformerunit_config.json'), 'w')\
                as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    # Static is not necessary, here, we use static since the load method
    # is called in the transformer __init__
    @staticmethod
    def load(input_path: str, device_load: str = 'cpu'):
        '''
        not necessarily load to GPU actually.

        You can always load the model to cpu and transfer it to
         cuda:0 in the sequential model. This is what bert is
         actually adopting. No need to directly load the model
         to GPU.
        '''
        with open(os.path.join(
                input_path, 'transformerunit_config.json')) as fIn:
            config = json.load(fIn)

        # Here, we don't adopt the "load from the initialization
        # of the model" scheme. We load the model from the
        # .bin, so we don't have to implement the model_name_or_path
        # in the __init__()

        # **dict
        # *list
        transformerunit = TransformerUnit(**config)
        device = torch.device(device_load)
        transformerunit.load_state_dict(torch.load(os.path.join(
            input_path, 'pytorch_model.bin'), map_location=device))
        return transformerunit
