# -*- coding: utf-8 -*-

import json
import logging
import os
import shutil
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type
from zipfile import ZipFile
import sys
from util import batch_to_device

import numpy as np
import transformers
import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# # import from __init__.py the __DOWNLOAD_SERVER__ string
# from . import __DOWNLOAD_SERVER__
# from . import __version__
# # only if the class name is the same as the file name, similar to java
# # .evaluation can only be used when called in the package from outside main
# # from .evaluation import SentenceEvaluator
# # from .util import import_from_string, batch_to_device, http_get
# # __init__.py will be automatically called when calling a class within
# #  the package, with an __init__.py you can directly call a .py
# # as the __init__.py implicitly import the class
# # from evaluation import SentenceEvaluator
# from DialogEvaluator_in_presentation import DialogEvaluator
from DialogEvaluator_iemocap import DialogEvaluator
from DialogPredictor import DialogPredictor
# you can import function from a .py, or use util.batch_to_device to call it
# or use from util import *
# from util import import_from_string, batch_to_device, http_get
# import util.batch_to_device/class but no import util.class.method
from util import batch_to_device, import_from_string
from roberta_with_finetune import ROBERTA
from torch.utils.data import DataLoader
from input_instance import InputInstance
from csv_reader import CSVDataReader
from torchdataset_wrapper_roberta_with_finetune import TorchWrappedDataset
import sys
from transformerunit import TransformerUnit
from datetime import datetime
import math
from loggingHandler import LoggingHandler
import time
# from attention_and_topics import extract_attended_topic_words_and_its_topics
# from topics_and_emotions_iemocap import plot_topics_and_emotions


class DialogTransformer(nn.Sequential):
    '''
    Here, the modules will be either an LSTM or a Transformer
    '''

    def __init__(
            self,
            model_name_or_path: str = None,
            modules: Iterable[nn.Module] = None, device: str = None):
        '''
        In the very beginning, the modules shouldn't be null,
        If modules is not None, then train the model. Else, load the model and
        predict
        '''
        '''
        Here we employ the load from model initialization structure
        '''
        if modules is not None and not isinstance(modules, OrderedDict):
            # if orderedDict then use it
            modules = OrderedDict(
                [(str(idx), module) for idx, module in enumerate(modules)])

        if model_name_or_path is not None and model_name_or_path != "":
            logging.info("Load pretrained DialogTransformer: {}".format(
                model_name_or_path))

            # #### Load from server
            # if '/' not in model_name_or_path and '\\' not in model_name_or_path and not os.path.isdir(model_name_or_path):
            #     logging.info("Did not find a / or \\ in the name. Assume to download model from server")
            #     model_name_or_path = __DOWNLOAD_SERVER__ + model_name_or_path + '.zip'

            # if model_name_or_path.startswith('http://') or model_name_or_path.startswith('https://'):
            #     model_url = model_name_or_path
            #     folder_name = model_url.replace("https://", "").replace("http://", "").replace("/", "_")[:250]

            #     # print('===================')

            #     try:
            #         from torch.hub import _get_torch_home
            #         torch_cache_home = _get_torch_home()

            #         # print('=================== didnt enter exception')
            #     # os.getenv(key=, default=), and the TORCH_HOME, XDG_CACHE_HOME
            #     # does not exist, so expanduser change the ~/.cache/torch
            #     os.makedirs(model_path, exist_ok=True)
            # else:
            #     model_path = model_name_or_path
            model_path = model_name_or_path

            # #### Load from disk
            if model_path is not None:
                logging.info("Load DialogTransformer from folder: {}".format(
                    model_path))

                # if os.path.exists(os.path.join(model_path, 'config.json')):
                #     with open(os.path.join(model_path, 'config.json')) as fIn:
                #         config = json.load(fIn)
                #         if config['__version__'] > __version__:
                #             logging.warning("You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(config['__version__'], __version__))

                with open(os.path.join(model_path, 'modules.json')) as fIn:
                    contained_modules = json.load(fIn)

                # the modules are bert, LSTM and so-on
                modules = OrderedDict()
                for module_config in contained_modules:
                    module_class = import_from_string(module_config['type'])
                    module = module_class.load(
                        os.path.join(model_path, module_config['path']))
                    modules[module_config['name']] = module

        # instantialize self._modules, therefore can conduct the basic function
        #  of the modules
        # register the modules so you can directly call it.
        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # print(device)
            logging.info("Use pytorch device: {}".format(device))
        self.device = torch.device(device)
        # put the modules to device
        self.to(device)
        # for feature_name in features:
        #     features[feature_name] = torch.tensor(np.asarray(features[feature_name])).to(self.device)

    def save(self, path):
        """
        Saves all elements for this seq. sentence embedder
        into different sub-folders

        Store the total config only
        """
        if path is None:
            return

        logging.info("Save model to {}".format(path))
        contained_modules = []

        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            # __name__ is the class or function name,
            # __module__ is the .py name, possibly including the relative
            #  path if executed from the outside folder.
            # logging.info("module.__name__: {}".format(
            #     type(module).__name__))
            model_path = os.path.join(
                path,
                str(idx) + "_" + type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            # modules are saved here, using the save in modules respectively
            module.save(model_path)
            # __module__ name of module in which this class was defined
            #  sometimes you will import the module from folders,
            #  for instance, you run the __main__ outside sentence_transformers
            #  folder, in which case the relative import is allowed,
            #  the __module__ is the relative path, that
            # logging.info('type(module).__module__ :{}'.format(
            #     type(module).__module__))
            # If you use __init__ and the classname is the same as the
            #  .py file. So if you use __init__, and the module when saved
            #  is imported as the Module_Folder.Classname, then
            #  when saved, however, you can use both the __module__.__name__
            #  and the __modulefolder__.__module__ to import the
            #  filename. So __init__ + save meets both. However,
            #  If you don't use __init__, then you need to save the path
            #  as __module__.__name__, and load it use __module__.__name__,
            #  so loading using __module__.__name__ meets the both
            contained_modules.append(
                {'idx': idx,
                 'name': name,
                 'path': os.path.basename(model_path),
                 'type': (
                     type(module).__module__ + '.' + type(module).__name__)})

        # the sequential configuration is saved as the modules.json in
        # the out-most folder. The contained_modules dict are saved in
        # modules.json. Whilst sequential has no modules to save.
        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

        # with open(os.path.join(path, 'config.json'), 'w') as fOut:
        #     json.dump({'__version__': __version__}, fOut, indent=2)

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: DialogEvaluator,
            epochs: int = 1,
            steps_per_epoch=None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {
                'lr': 2e-5,
                'eps': 1e-6,
                'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            fp16: bool = False,
            fp16_opt_level: str = 'O1',
            local_rank: int = -1
            ):
        """
        Train the model with the given training objective

        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as
        there are in the smallest one
        to make sure of equal training with each dataset.

        :param weight_decay:
        :param scheduler:
        :param warmup_steps:
        :param optimizer:
        :param evaluation_steps:
        :param output_path:
        :param save_best_model:
        :param max_grad_norm:
        :param fp16:
        :param fp16_opt_level:
        :param local_rank:
        :param train_objectives:
            Tuples of DataLoader and LossConfig
        :param evaluator:
        :param epochs:
        :param steps_per_epoch: Train for x steps in each epoch. If set to None, the length of the dataset will be used
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            if os.listdir(output_path):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(output_path))

        # retrieve dataloaders
        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # # Use smart batching
        # # Originally the tensorize was done in the smart batch,
        # #  We did this in the dataset_wrapper initialize
        # # Actually it converts instances to the batches
        # #  reshape the datasets and convert them to the tensors
        # #  The dataloader has default collate_fn, that is,
        # #  each batch is a list,
        # #  and [0] is feature[0], [1] is feature[1], etc., see collate_fn in
        # #  dataloader.py for detailed usages
        # for dataloader in dataloaders:
        #     dataloader.collate_fn = self.smart_batching_collate

        models = [amodel for _, amodel in train_objectives]

        # retrieve the loss_models
        # each loss_model is actually a module, for parallel training
        # on one dataset, enables parallel computation with distributed batches
        device = self.device
        for amodel in models:
            amodel.to(device)

        self.best_score = -9999999

        # num_of_batches
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = \
                min([len(dataloader) for dataloader in dataloaders])
            # the smallerest dataset determines the steps_per_epoch, that is
            # the num_of_batches per epoch

        # total number of training steps
        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        # the schedulers seem to be useless
        # Oh, it's useful. It's used to change the learning rate
        # from the scheduler, we can see that _get_scheduler actually
        # wraps the optimizer, and privides a learning decay
        # for each epoch
        # >>> lambda1 = lambda epoch: epoch // 30
        # >>> lambda2 = lambda epoch: 0.95 ** epoch
        # >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        for model in models:
            # return the names and the parameters
            param_optimizer = list(model.named_parameters())
            '''
            Choose parameters to optimize
            Second way to pass parameters as groups
            '''
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            # optimizer_grouped_parameters[0] are params with weightdecay
            #  similar requires_grad=True
            # optimizer_grouped_parameters[1] are params without weightdecay
            t_total = num_train_steps
            # allow distribution, each machine execute
            #  t_total // torch.distributed.get_world_size() epochs
            if local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()

            # **: from dict to params
            optimizer = optimizer_class(
                optimizer_grouped_parameters, **optimizer_params)

            # scheduler is the one which linearly/linear-linearly
            # /constantly updates the learning rate in each epoch
            scheduler_obj = self._get_scheduler(
                optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                t_total=t_total)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        if fp16:
            # training with half precision
            # if you are not training on small devices, please disable this
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            for train_idx in range(len(models)):
                '''
                loss_models includes the nn.Module successor which implements
                the Forward function
                loss_models[train_index] subscribes/points to/retrieves a model
                and accompany it with an optimizer, then they are copied to
                two parallel arrays
                '''
                model, optimizer = amp.initialize(
                    models[train_idx],
                    optimizers[train_idx],
                    opt_level=fp16_opt_level)
                models[train_idx] = model
                optimizers[train_idx] = optimizer

        global_step = 0
        # steps_per_epoch * number_of_loss_models
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        # the number of models and affiliated dataloaders
        num_train_objectives = len(train_objectives)

        # criterion, LossFunction, suitable for multi model with the
        #  same training criterion. For multiple please modulize 
        #  the loss or wrap the loss outside this self module.
        # Both the model, loss and the values should be put to device
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)

        # iterate over all epochs
        for epoch in trange(epochs, desc="Epoch"):
            # enable the progress bar, description is "Epoch"
            # trange(0, epochs, ncols=47, ascii=True) means the bar is
            # 47-length and the bar expression is the '#'
            training_steps = 0

            # model.zero_grad() and optimizer.zero_grad()
            #  are the same IF all your model parameters are in
            #  that optimizer. I found it is safer to call model.zero_grad()
            #  to make sure all grads are zero, e.g. if you have two
            #  or more optimizers for one model.
            # call model.zero_grad() before train() so that gradients
            # can be erased safer, confirm that gradients are erased
            for model in models:
                model.zero_grad()
                model.train()

            # iterate over all batches
            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
                # iterate over all the models
                for train_idx in range(num_train_objectives):
                    # each model is trained per batch
                    model = models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        # each model is trained per batch
                        # fetch the next batch
                        data = next(data_iterator)
                    except StopIteration:
                        # logging.info("Restart data_iterator")
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        # next is the built-in function for an iterable
                        # That is, who has implemented the __iter__() interface
                        data = next(data_iterator)
                        # print('Exception:', data)
                    # print(data)
                    # # It's a procedure
                    # features = batch_to_device(data, self.device)
                    # features = data
                    features = batch_to_device(data, self.device)
                    # print(features)

                    # logging.info('if on cuda {}'.format(features[0].is_cuda))

                    # logging.info(model.is_cuda)
                    # Sequential doesn't have is_cuda, but will response to
                    #  to()

                    # both model.to() or model = model.to()
                    #  can put the model to cuda. BUT only var = var.to()
                    #  can put a variable to cuda!
                    tfboyed_features = model(features)

                    # this shouldn't be matter actually
                    # but we'd better view and stack them
                    batched_uttrs = tfboyed_features[0]
                    # tensor
                    batched_labels = tfboyed_features[1]
                    # previously list of scalars now a tensor
                    batched_lengths = tfboyed_features[2]
                    b_size, seq_size, emb_size = batched_uttrs.size()

                    lst_uttrs = []
                    lst_labels = []
                    for i_dim in range(b_size):
                        # lst_uttrs.append(
                        #     batched_uttrs[
                        #         i_dim,
                        #         :batched_lengths[i_dim],
                        #         :].view(
                        #             batched_lengths[i_dim],
                        #             emb_size))
                        # # The single index will automatically squeeze it
                        # lst_uttrs.append(
                        #     batched_uttrs[
                        #         i_dim,
                        #         :batched_lengths[i_dim],
                        #         :].squeeze())
                        # lst_labels.append(
                        #     batched_labels[
                        #         i_dim,
                        #         :batched_lengths[i_dim]].squeeze())
                        lst_uttrs.append(
                            batched_uttrs[
                                i_dim,
                                :batched_lengths[i_dim],
                                :])
                        lst_labels.append(
                            batched_labels[
                                i_dim,
                                :batched_lengths[i_dim]])
                    var_uttrs = torch.cat(lst_uttrs, dim=0)
                    var_labels = torch.cat(lst_labels, dim=0)
                    loss_value = criterion(
                        var_uttrs,
                        var_labels)

                    if fp16:
                        # if (...
                        #        and ...)
                        with amp.scale_loss(loss_value, optimizer) \
                                as scaled_loss:
                            # scale the loss_value by the amplifier
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer),
                            max_grad_norm)
                    else:
                        # perform backward for the loss_value
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_grad_norm)

                    # mandatory for an optimizer
                    # optimizer optimizes the grads for the selected params
                    optimizer.step()
                    # learning rate starts with = 1e-5
                    # warmup_step decays by one each step.
                    scheduler.step()
                    # mandatory for an optimizer
                    optimizer.zero_grad()

                # training step denotes the batch_idx
                #  to the epoch here.
                # global steps = epochs * batch_num + training_steps
                training_steps += 1
                global_step += 1

                # Avoid zero every time when denumerated/divided
                # evaluate the model every evaluation_steps' batches
                if evaluation_steps > 0 and \
                        training_steps % evaluation_steps == 0:
                    # the evaluation loss is different from the model loss
                    #  and is used to save the model.
                    self._eval_during_training(
                        evaluator,
                        output_path,
                        save_best_model,
                        epoch,
                        training_steps)
                    # evaluate after each batch
                    # evaluation during training
                    for model in models:
                        model.zero_grad()
                        model.train()
            # evaluate after each epoch
            self._eval_during_training(
                evaluator,
                output_path,
                save_best_model,
                epoch,
                -1)

    def evaluate(self, evaluator: DialogEvaluator, output_path: str = None):
        """
        Evaluate the model
        evaluate after training

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

    def _eval_during_training(
            self, evaluator, output_path, save_best_model, epoch, steps):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(
                self, output_path=output_path, epoch=epoch, steps=steps)
            if score > self.best_score and save_best_model:
                self.save(output_path)
                self.best_score = score

    def _get_scheduler(
            self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler

        # the learning rate optimisation is wrapped in a scheduler
        # constantlr means lr is fixed.
        # Warmupconstant means lr is accerating constantly
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            # this doesn't include warmup
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            # this uses warmup
            return transformers.get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            # This uses warmup + lr-decay with t_total lr-decays
            # only this wrapper accepts num_trianing_steps
            # if you open the function get_linear_schedule_with_warmup
            # you will find that return
            # LambdaLR(optimizer, lr_lambda, last_epoch)
            # and last_epoch=-1. So it will end till
            # num_training_steps consume up
            # and you will see that each .step() consumes 1 training
            # step. You can see from
            # https://pytorch.org/docs/stable/optim.html
            # 'How to adjust Learning Rate'
            # When you call the step, the counter will reduce by 1,
            # and the learning rate will be adjusted accordingly.
            # Initial learning rate is given in the Optimiser.
            # Initially, at each step, the learning rate will be
            # adjusted by param_group in optimizer.param_groups:
            #    param_group['lr'] = lr
            # Now it is implicitly adjusted by the scheduler
            return transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))


if __name__ == '__main__':

    # #### Just some code to print debug information to stdout
    # emit must be implemented by Handler subclasses
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        # handlers=[logging.Handler(level=logging.NOTSET)])
                        handlers=[LoggingHandler()])
    # # #########Training the model
    # logging.info("Begin reading the dataset")
    # csvDataReader = CSVDataReader('../datasets/')
    # instances = csvDataReader.get_instances('iemocap_train_generated.csv')
    # tokenizer_name = '../datasets/topic-language-model-iemocap'
    # # shorter max_text_seq_length save memory, although masking, the input to bert always consumes memory
    # # shorter max_dialogue_seq_len don't save memory, since empty seq are not input to bert
    # tokenizer_roberta = ROBERTA(tokenizer_name, max_seq_length=100, devicepad='cpu')
    # # 160 for daily dialogue
    # # 108 for emory
    # # devicepad is only used for the evaluation of the model.

    # logging.info('Read train dataset')
    # # emory max_seq_length is the default 25
    # train_data = TorchWrappedDataset(instances, tokenizer_roberta, max_seq_length=36)
    # train_batch_size = 2
    # # #########change to true when in operation
    # train_dataloader = DataLoader(
    #     train_data, shuffle=True, batch_size=train_batch_size)

    # # #########Test on cuda whether pass a listed cudatensor to
    # #          a cudamodel will be fine

    # model_tfe = TransformerUnit(
    #     d_model=tokenizer_roberta.get_word_embedding_dimension(),
    #     n_heads=8,
    #     out_features=train_data.get_label_category_count())
    # # berted_batch1 = model_bert(first_batch)
    # # print('begin feeding TransformerEncoderLayer')

    # # transed_batch1 = model_tfe(berted_batch1)
    # # print(transed_batch1[0].size())

    # # If you run .device("cuda"), your tensor will be routed to
    # #  the CUDA current device, which by default is the 0 device.
    # # !!!!!Maybe I need to add a softmax layer here since model_tfe output hasn't been normalized and has been sent to calculate the loss.
    # # !!!!!So I also have to change the Dialogue Evaluator since there is an extra softmax
    # # !!!!!No need to apply softmax since the softmax has been embodied in the CrossEntropy already
    # modelDlg = DialogTransformer(
    #     modules=(
    #         tokenizer_roberta,
    #         model_tfe),
    #     device='cuda:0')
    # # print("modelDlg on cuda?", next(modelDlg.parameters()).is_cuda)
    # #    device='cpu')
    # # atoken = input()

    # # Read the dataset
    # model_name = 'dialogtransformer'
    # dev_batch_size = 2
    # num_epochs = 100
    # # num_epochs = 2
    # model_save_path = \
    #     '../save/training_dialogtransformer_coldstart_training-' + \
    #     model_name + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # logging.info("Read dev dataset")
    # valid_instances = csvDataReader.get_instances('iemocap_val_generated.csv')
    # valid_data = TorchWrappedDataset(valid_instances, tokenizer_roberta, max_seq_length=36)
    # dev_dataloader = DataLoader(
    #     valid_data, shuffle=False, batch_size=dev_batch_size)
    # evaluator = DialogEvaluator(dev_dataloader, name='', device='cuda:0')
    # # 'cuda:0')
    # # print("evaluator on cuda?", next(evaluator.parameters()).is_cuda)
    # # evaluator is not an nn.Module

    # warmup_steps = math.ceil(
    #     len(train_data) * num_epochs / train_batch_size * 0.5)
    # # len(train_data) / train_batch_size : train steps, i.e., batch number or count
    # #  since the num_training_steps is t_total, i.e., batch number * epoch number
    # # 10% of training steps for warm-up, that is, batch_count * epoch_num * 0.1
    # # warm up until the ten percent of the data
    # logging.info("Warmup-steps: {}".format(warmup_steps))
    # # Train the model
    # # using hugging face (bert) will trigger the warning of
    # #  "This overload of add_ is deprecated"
    # # each step times lr, lr doesn't change
    # # evaluation steps not used, see training_steps % evaluation_steps == 0:

    # # small learning rate corresponds to small batch
    # # large learning rate corresponds to large batch
    # modelDlg.fit(
    #     train_objectives=[(train_dataloader, modelDlg)],
    #     evaluator=evaluator,
    #     epochs=num_epochs,
    #     evaluation_steps=1000,
    #     warmup_steps=warmup_steps,
    #     optimizer_params={
    #         'lr': 2e-5,
    #         'eps': 1e-6,
    #         'correct_bias': False},
    #     output_path=model_save_path)

    # # atoken = input()

    # ######### Load the stored model to evaluate it on the test set
    logging.info("Begin reading the dataset")
    csvDataReader = CSVDataReader('../datasets/')
    instances = csvDataReader.get_instances('iemocap_test_generated.csv')

    tokenizer_name = '../save/topic-language-model-iemocap'

    tokenizer_bert = ROBERTA(tokenizer_name, max_seq_length=122, devicepad='cuda:0')

    logging.info('Read test dataset')
    # daily dialogue max_seq_length = 36, emory: 25, iemocap: 112 180
    test_data = TorchWrappedDataset(instances, tokenizer_bert, max_seq_length=96)
    # can be the same the train_batch_size
    train_batch_size = 2

    # If loading it's the saving configuration of model_dlg's bert that determines
    #  it's device
    model_save_path = ('../save/saved-model-iemocap')
    model_dlg = DialogTransformer(
        model_save_path,
        device='cuda:0')

    # dataloader will automatically allocate the size-1 last batch
    #  So you'd better restrict the batch_size within the epoch
    test_dataloader = DataLoader(
        test_data, shuffle=False, batch_size=train_batch_size)

    logging.info('steps_per_epoch {}'.format(
        len(test_dataloader)))

    evaluator = DialogEvaluator(test_dataloader, name='', device='cuda:0')
    model_dlg.evaluate(evaluator, output_path='../save/saved-model-iemocap')

    # for prediction
    # predictor = DialogPredictor(test_dataloader, name='', device='cuda:0')
    # predictor(model_dlg, output_path=model_save_path)

    # ~~~~~~~~~~~~~~~~~~~~~ fluffy things
    # data_iterator = iter(test_dataloader)
    # first_batch = next(data_iterator)
    # # first_batch.to(torch.device('cuda:1'))
    # first_batch = batch_to_device(first_batch, torch.device('cuda:0'))
    # print('first_dimension=', len(first_batch))
    # # first_batch = torch.LongTensor(first_batch)
    # print('shape of first instance/tokens=', first_batch[0].size())
    # print('shape of second instance/labels=', first_batch[1].size())
    # print('shape of third instance/lengths=', first_batch[2].size())
    # print('shape of fourth instance/textlengths=', first_batch[3].size())
    # print('shape of fifth instance=/masks', first_batch[4].size())
    # topic_roberta = model_dlg.__getitem__(0)
    # tensor_topic_vec = topic_roberta.get_topic_vec(first_batch)
    # print(tensor_topic_vec.size())

    # # , skip_special_tokens=True, clean_up_tokenization_spaces=False
    # tensor_attention_vec = topic_roberta.get_attentions_vec(first_batch)
    # # a_dialogue_flattened = [model_bert.tokenizer.convert_ids_to_tokens(a_tokenized_uttr)
    # #                         for a_tokenized_uttr in first_batch[0].view(-1, 112)]
    # a_dialogue_flattened = model_bert.tokenizer.batch_decode(first_batch[0].view(-1, 36), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print('recovery of first instance/tokens', a_dialogue_flattened)

    # fpIn = open('../datasets/topic_list.json', 'rt', encoding='utf8')
    # extract_attended_topic_words_and_its_topics(a_dialogue_flattened, tensor_attention_vec, json.load(fpIn))
    # fpIn.close()

    # model_dlg.to(torch.device('cpu'))
    # plot_topics_and_emotions(test_dataloader, topic_roberta, torch.device('cpu'))

    # test_data_visual = TorchWrappedDataset(instances, model_bert, max_seq_length=72)
    # test_dataloader_visual = DataLoader(
    #     test_data_visual, shuffle=False, batch_size=train_batch_size)
    # plot_topics_and_emotions(test_dataloader_visual, topic_roberta, torch.device('cpu'))
    # first_batch = batch_to_device(first_batch, torch.device('cpu'))
    # tensor_topic_vec = None
    # tensor_attention_vec = None
    # a_dialogue_flattened = None
    # torch.cuda.empty_cache()
    # plot_topics_and_emotions(test_dataloader, topic_roberta, torch.device('cuda:0'))
