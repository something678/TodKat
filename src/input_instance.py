# -*- coding: utf-8 -*-

from typing import Union, List


class InputInstance:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(
            self, guid: str, texts: List[str],
            labels: List[Union[int, float]]):
        """
        Creates one InputInstance with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """

        '''
        I have learnt something. Actually, the python can predefine
         the type in the
         parameters definition like C, but you need to import typing,
         List corresponds
         to list, Set correspond to set, Union correspond to either
         into or float, etc.
        '''

        '''
        Actually, you can also use guid: str=None to
         spcify the type and initialize the
         param
        '''
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.labels = labels
