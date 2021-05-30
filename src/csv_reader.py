# -*- coding: utf-8 -*-

from input_instance import InputInstance
import csv
import os


class CSVDataReader:
    """
    Reads in the CSV-format dataset.
     Each line contains several sentences
     (utterance_1, utterance_2) and 2 labels (label_1, label_2)
    """

    def __init__(self, dataset_folder, delimiter="\t",
                 quoting=csv.QUOTE_NONE):
        '''
        the parameters denote the index number of different properties
        '''
        self.dataset_folder = dataset_folder
        self.delimiter = delimiter
        self.quoting = quoting

    def get_instances(self, filename, max_instances=0):
        """
        filename specified which data
         split to use (train.csv, dev.csv, test.csv).
        """
        data = csv.reader(
            open(os.path.join(self.dataset_folder, filename),
                 encoding="utf-8"),
            delimiter=self.delimiter, quoting=self.quoting)
        instances = []

        for id, row in enumerate(data):
            # print(row)
            column_count = len(row)
            # print(column_count)
            uttrances = row[:column_count // 2]
            # defaultly int <=> int32
            labels = list(map(int, row[column_count // 2:]))
            instances.append(InputInstance(
                guid=filename + str(id),
                texts=uttrances,
                labels=labels))
            if max_instances > 0 and len(instances) >= max_instances:
                break

        return instances
