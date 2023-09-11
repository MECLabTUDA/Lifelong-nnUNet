##########################################################################################################
#------This module contains useful functions that are used throughout the nnUNet_extensions project.-----#
##########################################################################################################

import os, shutil
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.nnunet.training.model_restore import restore_model
import torch
import torch.nn as nn
def delete_dir_con(path):
    r"""This function deletes the whole content in the folder specified by the path and then deletes the empty folder
        if it still exists.
        :param path: Full path to the folder for which the content needs to be deleted
    """
    # -- Delete content of the folder -- #
    shutil.rmtree(path)

    # -- If the folder still exists, remove it because it is just empty -- #
    if os.path.exists(path):
        os.rmdir(path)

def copy_dir(source, dest):
    r"""This function copies the content of the source folder into dest.
        :param source: Full path to the folder containing the desired content, e.g. ../data/XY
        :param dest: Full path to which source will be copied, e.g. ../copied_data/XY
        This function will then copy everything from ../data/XY/.. into ../copied_data/XY/.
    """
    # -- Check if dest exists, if yes, than empty the folder since, otherwise create the folder -- #
    if os.path.exists(dest):
        # -- Delete directory -- #
        delete_dir_con(dest)

    # -- Copy content from source to dest -- #
    shutil.copytree(source, dest)

def move_dir(source, dest):
    r"""This function moves the content of the source folder into dest.
        :param source: Full path to the folder containing the desired content, e.g. ../data/XY
        :param dest: Full path to which source will be moved, e.g. ../moved_data/XY
        This function will then move everything from ../data/XY/.. into ../moved_data/XY/.
    """
    # -- Check if dest exists, if yes, than empty the folder since, otherwise create the folder -- #
    if not os.path.exists(dest):
        os.makedirs(dest)

    # -- Move content from source to dest -- #
    shutil.move(source, dest, copy_function=shutil.copytree)

    # -- Delete empty folder from which everything has been moved -- #
    delete_dir_con(source)

def join_texts_with_char(texts, combine_with):
    r"""This function takes a list of strings and joins them together with the combine_with between each text from texts.
        :param texts: List of strings
        :param combine_with: A char or string with which each element in the list will be connected with
        :return: A String consisting of every string from the list combined using combine_with
    """
    assert isinstance(combine_with, str), "The character to combine the string series from the list with needs to be from type <string>."
    # -- Combine the string series from the list with the combine_with -- #
    return combine_with.join(texts)

def get_nr_parameters(model):
    r"""This function returns the number of parameters and trainable parameters of a network.
        Based on: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    # -- Extract and count nr of parameters -- #
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # -- Return the information -- #
    return total_params, trainable_params

def get_model_size(model):
    r"""This function return the size in MB of a model.
        Based on: https://discuss.pytorch.org/t/finding-model-size/130275
    """
    # -- Extract parameter and buffer sizes -- #
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    # -- Transform into MB -- #
    size_all_mb = (param_size + buffer_size) / 1024**2
    # -- Return the size -- #
    return size_all_mb

def dumpDataFrameToCsv(data, path, name, sep='\t'):
    r"""This function dumps a DataFrame in form of a .csv file.
        :param data: A DataFrame.
        :param path: Path as a string indicating where to store the file.
        :param name: String indicating the name of the file.
        :param sep: String indicating the seperator for the csv file, ie. how to seperate the content."""
    # -- Check if csv is in the name -- #
    if '.csv' not in name:
        # -- Add it if its missing -- #
        name = name + '.csv'
    # -- Build the absolute path based on the path and name -- #
    path = os.path.join(path, name)
    # -- Dump the DataFrame using the path and seperator without using the index from the frame -- #
    data.to_csv(path, index=False, sep=sep)

def flattendict(data, delim):
    r"""This function can be used to flatten any dictionary/json no matter how nested the dict is.
        It is specifically used for transforming nested dicts into a flat form to store the dict then
        as a .csv file. --> see function nestedDictToFlatTable(...).
        Extracted from: https://stackoverflow.com/questions/1871524/how-can-i-convert-json-to-csv.
        :param data: Nested dictionary where valueas are dicts of dicts etc.
        :param delim: String indicating the delimeter that is used to concatenate between the different layers
                      of the nested dict (key1 + delim + key11 + delim + ... + delim + key11111..),
                      
    """
    # -- Define result dictionary that will be flat, ie. key is a string and value is of primitive type -- #
    val = {}
    # -- Loop through the first layer of keys -- #
    for i in data.keys():
        # -- If the value is a dict then go one layer deeper --> nested -- #
        if isinstance(data[i], dict):
            # -- Do recursive call until the leaf node is reached -- #
            get = flattendict(data[i], delim)
            # -- Loop through keys again -- #
            for j in get.keys():
                # -- Register the joint keys using the delimeter as key and add the leaf value as correpsonding value -- #
                val[i + delim + j] = get[j]
        else:
            # -- If there is no dict left, leaf node is reached add it to the dict -- #
            val[i] = data[i]
    # -- Return the flattened dictionary -- #
    return val
  
def flatteneddict_to_df(flatteneddict, cols, delim):
    r"""This function can be used to transform a flattened dictionary into a DataFrame, ie. table like structure
        (tidy data). Use the flattened dict from the flattendict(...) function with the used delimeter.
        Note that the cols should be of the same length than the length of the keys split used the delimeter + the value.
        :param flatteneddict: A flattened dictionary, ie. key and value whereas value is primitve type like int, str, etc.
        :param cols: List of column names for the DataFrame --> Should have the same length as the extracted elements, ie.
                     splitted key based on delimeter + the value mapped to the key.
        :param delim: String representing the delimeter that has been used during the falltening process of a nested dict.
        """
    # -- Check that the number of cols matches as expected -- #
    assert len(cols) == len(list(flatteneddict.keys())[0].split(delim))+1,\
    "The number of columns in the json does not match with the provided list of columns."
    # -- Define an empty DataFrame based on the cols -- #
    res = pd.DataFrame([], columns=cols)
    # -- Loop through the flattened dict and build the table -- #
    for k, v in flatteneddict.items():
        # -- Split the key using the delimeter to get the first n-1 values -- #
        row = k.split(delim)
        # -- Add the corresponding value to the list -- #
        row.append(v)
        # -- Append this list as a row to the DataFrame -- #
        res.loc[len(res)] = row
    # -- Return the build DataFrame while resetting the indices of the frame -- #
    return res.reset_index(drop=True)

def nestedDictToFlatTable(nested_dict, cols):
    r"""This function can be used to transform a nested dictionary into a DataFrame. The function uses
        flattendict and flatteneddict_to_df to realize this.
        :param nested_dict: The nested dictionary no matter how deep ;)
        :param cols: List of column names for the DataFrame --> Should have the same length as the depth
                     of the provided nested_dict + 1 (all keys + value).
        """
    # -- Build the DataFrame from the dictionary and return the Frame -- #
    return flatteneddict_to_df(flattendict(nested_dict, "__"), cols, '__')
def getIndexOfBottleneckLayer(folder,checkpoint_name="model_final_checkpoint",model_type='3d_fullres',mcdo:int=-1,folds=None,mixed_precision=True):
    if isinstance(folds, str):
        folds = [join(folder, "all")]
        assert isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [join(folder, "all")]
        else:
            folds = [join(folder, "fold_%d" % i) for i in folds]
        assert all([isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [join(folder, "fold_%d" % folds)]
        assert all([isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds. Type: %s. Expected: list of int, int, str or None", str(type(folds)))
    trainer = restore_model(join(folds[0], "%s.model.pkl" % checkpoint_name), fp16=mixed_precision)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)

    if model_type in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']:
        trainer.initialize(False, mcdo=mcdo, network_arch='generic')
    else:
        trainer.initialize(False, mcdo=mcdo, network_arch=model_type)
        

    conv_blocks_context = nn.ModuleList(trainer.network.conv_blocks_context)
    print("length of modulelist:",len(conv_blocks_context))
    isMultiBlock=False
    if isinstance(conv_blocks_context[-1],torch.nn.modules.container.Sequential):
        isMultiBlock=True
        return len(conv_blocks_context)-1,isMultiBlock,len(conv_blocks_context[-1])-1
    return len(conv_blocks_context)-1,isMultiBlock,None