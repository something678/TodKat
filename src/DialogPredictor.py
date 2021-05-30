# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
import logging
import torch
from util import batch_to_device
# tqdm: turn a iterater into a progress bar iterator
# so you can do the enumerate
from tqdm import tqdm, trange
from torch.nn import functional
import os
# pandas can read quicker
import csv


class DialogPredictor:
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(
            self,
            dataloader: DataLoader,
            name: str = "",
            # softmax_model=None,
            device: str = None):
        '''
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        :param name:
            model save name, usually the model name
        #：softmax_model:
        #    the model, here without softmax for CE loss
        '''
        self.dataloader = dataloader
        # this allows for a different device for evaluation
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        # self.device = device
        self.device = torch.device(device)
        self.name = name
        # self.softmax_model = softmax_model
        # self.softmax_model.to(self.device)

        if name:
            name = "_" + name

        self.csv_file = "all_utterance_prediction" + name + "_results.csv"
        self.csv_headers = ["predictions"]

    def __call__(
            self,
            model,
            output_path: str = None,
            epoch: int = -1,
            steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation
         with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation,
             i.e., batchsize
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher
             score indicating a better result
        """
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        # logging.info("Prediction on the " + self.name + " dataset" + out_txt)
        # self.dataloader.collate_fn = model.smart_batching_collate
        logging.info("Prediction on the " + self.name + " dataset" + out_txt)
        # self.dataloader.collate_fn = model.smart_batching_collate
        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    # the same as (epoch, steps, accuracy)
                    # writer.writerow([epoch, steps, accuracy])
                    for step, batch in enumerate(tqdm(self.dataloader, desc="Predicting")):
                        features = batch_to_device(batch, self.device)
                        # batch_tokens = features[0]
                        # label_ids = features[1]
                        with torch.no_grad():
                            modeled_features = model(features)
                        batch_uttrs = modeled_features[0]
                        b_size, seq_size, emb_size = batch_uttrs.size()
                        label_ids = modeled_features[1]
                        lengths = modeled_features[2]
                        # is_humors = modeled_features[3]
                        # # the softmax has been embodied in the CrossEntropy already
                        # softmaxed_texts = functional.softmax(batch_texts[:, 0: 2], dim=1)

                        # total += softmaxed_texts.size(0)
                        # # .eq equals to ==
                        # correct += torch.argmax(softmaxed_texts, dim=1).eq(
                        #     is_humors).sum().item()
                        softmaxed_uttrs = functional.softmax(batch_uttrs, dim=2)
                        # lst_softmaxed_uttrs = []
                        # lst_labels = []
                        for i_dim in range(b_size):
                            temp_softmaxed_uttrs = softmaxed_uttrs[i_dim, :lengths[i_dim], :]
                            temp_label_ids = label_ids[i_dim, :lengths[i_dim]]
                            total += lengths[i_dim]
                            argmaxed_uttrs = torch.argmax(temp_softmaxed_uttrs, dim=1)
                            corrects = argmaxed_uttrs.eq(temp_label_ids)
                            correct += corrects.sum().item()
                            line_left = '\t'.join(map(str, argmaxed_uttrs.tolist()))
                            # print(argmaxed_uttrs.tolist())
                            # print(line_left)
                            # line_right = '\t'.join(corrects.tolist())
                            writer.writerow([line_left])
            else:
                # exist, then append for the rest epochs
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for step, batch in enumerate(tqdm(self.dataloader, desc="Predicting")):
                        features = batch_to_device(batch, self.device)
                        # batch_tokens = features[0]
                        # label_ids = features[1]
                        with torch.no_grad():
                            modeled_features = model(features)
                        batch_uttrs = modeled_features[0]
                        b_size, seq_size, emb_size = batch_uttrs.size()
                        label_ids = modeled_features[1]
                        lengths = modeled_features[2]
                        # is_humors = modeled_features[3]
                        # # the softmax has been embodied in the CrossEntropy already
                        # softmaxed_texts = functional.softmax(batch_texts[:, 0: 2], dim=1)

                        # total += softmaxed_texts.size(0)
                        # # .eq equals to ==
                        # correct += torch.argmax(softmaxed_texts, dim=1).eq(
                        #     is_humors).sum().item()
                        softmaxed_uttrs = functional.softmax(batch_uttrs, dim=2)
                        # lst_softmaxed_uttrs = []
                        # lst_labels = []
                        for i_dim in range(b_size):
                            temp_softmaxed_uttrs = softmaxed_uttrs[i_dim, :lengths[i_dim], :]
                            temp_label_ids = label_ids[i_dim, :lengths[i_dim]]
                            total += lengths[i_dim]
                            argmaxed_uttrs = torch.argmax(temp_softmaxed_uttrs, dim=1)
                            corrects = argmaxed_uttrs.eq(temp_label_ids)
                            correct += corrects.sum().item()
                            line_left = '\t'.join(map(str, argmaxed_uttrs.tolist()))
                            # line_right = '\t'.join(corrects.tolist())
                            # print(line_left)
                            # print(argmaxed_uttrs.tolist())
                            writer.writerow([line_left])
        accuracy = correct / total
        logging.info("Accuracy: {:.4f} ({}/{})\n".format(
            accuracy, correct, total))
        return accuracy
