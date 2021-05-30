# -*- coding: utf-8 -*-

from torch import nn
from typing import Union

# for debug only
import sys
sys.path.append("..")
from csv_reader import CSVDataReader
# from ..torchdataset_wrapper import TorchWrappedDataset
from torchdataset_wrapper import TorchWrappedDataset

from bert import BERT
from torch.utils.data import DataLoader
# enables relative import
# from ..csv_reader import CSVDataReader
# relative imports ../csv_reader


class Padder(nn.Module):
    """
    A sequential padder to pad sequence of varying lengths
    Usually we use pad_packed_sequence to pack batched sequences
    the total_length param in pad_packed_sequence() is used to
    define the padded length of the sequences. If total_length is
    shorter than the max_length of the sequences, then the util
    function will throw/raise an error/exception

    amendament: See book marks deep learning - why do we "pack" the
     sequences in pytorch? - Stack Overflow, to save the computation,
     LSTMs will no longer compute the zero vlaues with pack. LSTM did
     those by gradient-stop
     We change to use pad_sequence here now.
    """

    def __init__(
            self, batch_first: bool=False,
            padding_value: Union[int, float]=0.0,
            # total_length=36,
            needs_packing: bool=False):
        '''
        Seems that batch_first = False is more natural but here we
        will later specify batch_first = True.

        I realized that we do not need to specify the total_length,
        We just ensure that within each batch all the sequence are
        in the same length

        if RNN, then better needs_packing=True
        '''
        self._batch_first = batch_first
        self._padding_value = padding_value
        self._needs_packing = needs_packing
        # if total_length > 36:
        #     '''
        #     The max_seq_length has been defined with the customized model.
        #     '''
        #     logging.warning("padder only allows a max_seq_length"
        #                     " of 36 (36 with"
        #                     "special tokens). Value will be set to 36")
        #     self._total_length = 36
        # else:
        #     self._total_length = total_length

    def forward(self, sequences):
        """Returns padded embeddings, cls_token"""
        '''The inputs should themselves be a list of tensor'''
        if self._batch_first:
            list_lengths = [ts.size()[1] for ts in sequences]
        else:
            list_lengths = [ts.size()[0] for ts in sequences]
        padded_sequences = nn.utils.rnn.pad_sequence(
            sequences=sequences,
            batch_first=self._batch_first,
            padding_value=self._padding_value)

        if self._needs_packing:
            # # the inputs are batched
            # nn.utils.rnn.pack_padded_sequence(
            #     sequence=padded_sequences,
            #     batch_first=self._batch_first,
            #     padding_value=self._padding_value,
            #     total_length=self.total_length)
            packed_padded_sequence = nn.utils.rnn.pack_padded_sequence(
                input=padded_sequences,
                lengths=list_lengths,
                batch_first=self._batch_first,
                enforce_sorted=False)
            return packed_padded_sequence
        else:
            return padded_sequences

    # def get_word_embedding_dimension(self) -> int:
    #     return self.bert.config.hidden_size
    #     you can see how they use the typing to hint the return type

    # def tokenize(self, text: str) -> List[int]:
    #     """
    #     Tokenizes a text and maps tokens to token-ids
    #     """
    #     return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))


if __name__ == '__main__':
    # solution: add sys.path.append("..") and use from csv_reader import CSVDataReader
    # you cannot use from ..csv_reader import CSVDataReader, since
    #  you may be running main from an outside file, and this file is loading an outside
    #  file, this will cause a confusion

    csvDataReader = CSVDataReader('../../datasets/')
    instances = csvDataReader.get_instances('dialogues_train.csv')
    model_name = 'bert-base-uncased'
    model = BERT(model_name)
    print('Read train dataset')
    train_data = TorchWrappedDataset(instances, model)

    print(train_data.__getitem__(0))

    pdr = Padder(batch_first=True)
    train_batch_size = 16
    data_iterator = iter(
        DataLoader(train_data, shuffle=True, batch_size=train_batch_size))
    first_batch = next(data_iterator)

    print(first_batch[0])
    # print(first_batch[1])
