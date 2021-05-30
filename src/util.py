import requests
from torch import Tensor, device
from typing import Tuple, List
from tqdm import tqdm
import sys
import importlib


def batch_to_device(batch, target_device: device):
    """
    send a batch to a device
    Originally the tensorize was done in the smart batch,
    We did this in the dataset_wrapper

    :param batch: column * batchsize * other, column is a list
    :param target_device:
    :return: the batch sent to the device
    """
    # features = batch['features']
    # labels = batch['labels'].to(target_device)
    # return features, labels
    # for i in range(len(batch)):
    #     batch[i].to(target_device)

    # better not use batch[i] = batch[i].to(target_device)
    # this will left reference to cudas preventing the deletion
    # for i in range(len(batch)):
    #     batch[i] = batch[i].to(target_device)
    features = []
    for i in range(len(batch)):
        features.append(batch[i].to(target_device))
    return features


def http_get(url, path):
    file_binary = open(path, "wb")
    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()

    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total, unit_scale=True)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            file_binary.write(chunk)
    progress.close()


def fullname(o):

  module = o.__class__.__module__
  if module is None or module == str.__class__.__module__:
    return o.__class__.__name__  # Avoid reporting __builtin__
  else:
    return module + '.' + o.__class__.__name__

def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.

    module_path indicates the .py
    class_name indicates the classname
    """
    # if dotted_path.find('.') != -1:
    #     try:
    #         # reversed split
    #         module_path, class_name = dotted_path.rsplit('.', 1)
    #     except ValueError:
    #         msg = "%s doesn't look like a module path" % dotted_path
    #         raise ImportError(msg)

    #     module = importlib.import_module(module_path)
    #     try:
    #         return getattr(module, class_name)
    #     except AttributeError:
    #         msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
    #         raise ImportError(msg)
    # else:
    #     module_path = dotted_path
    #     print('module_path: {}'.format(module_path))
    #     module = importlib.import_module(module_path)
    #     return module

    try:
        # reversed split
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)
