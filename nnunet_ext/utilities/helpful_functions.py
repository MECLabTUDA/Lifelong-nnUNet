##########################################################################################################
#------This module contains useful functions that are used throughout the nnUNet_extensions project.-----#
##########################################################################################################

import copy
import torch
import pandas as pd
from types import ModuleType
import sys, os, shutil, importlib
from torch.cuda.amp import autocast
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import join
from nnunet_ext.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir

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
    try:
        delete_dir_con(source)
    except: # Folder does not exist anymore
        pass

def join_texts_with_char(texts, combine_with):
    r"""This function takes a list of strings and joins them together with the combine_with between each text from texts.
        :param texts: List of strings
        :param combine_with: A char or string with which each element in the list will be connected with
        :return: A String consisting of every string from the list combined using combine_with
    """
    assert isinstance(combine_with, str), "The character to combine the string series from the list with needs to be from type <string>."
    # -- Combine the string series from the list with the combine_with -- #
    return combine_with.join(texts)

def delete_task(task_id):
    r"""This function can be used to sucessfully delete a task in all nnU-Net directories.
        If it does not exists there is nothign to delete.
        :param task_id: A string representing the ID of the task to delete, e.g. '002' for the Decathlong Heart Dataset.
    """
    # -- Ensure that the task_id is of type string since an integer with a beginning 0 is not accepted as integer -- #
    # -- Further ensure that the provided task_id is three digits long so we can be sure it is unique and only one task is removed -- #
    assert isinstance(task_id, str) and len(task_id) == 3, "The provided task_id should be a string and exactly three digits long."
    
    # -- Extract all existing tasks as long as the desired id is in there -- #
    task = [x for x in os.listdir(nnUNet_raw_data) if task_id in x] # Only the one with the mapping id
    
    # -- Check that the list only includes one element, if not raise an error since we do not know what to delete now -- #
    assert len(task) == 1, "The task does not exist or there are multiple tasks with the same task_id: {}.".format(join_texts_with_char(task, ' '))

    # -- Delete the task in raw_data, cropped_data and preprocessed_data -- #
    # -- For each deletion make a seperate try, since some script might crashed in between there and some parts are missing/deleted! -- #
    try:
        delete_dir_con(join(nnUNet_raw_data, task[0]))
    except Exception as e:
        print(e)
    try:
        delete_dir_con(join(nnUNet_cropped_data, task[0]))
    except Exception as e:
        print(e)
    try:
        delete_dir_con(join(preprocessing_output_dir, task[0]))
    except Exception as e:
        print(e)

def refresh_mod_imports(mod, reload=False):
    r"""This function can be used especially during generic testing, when a specific import
        needs to be refreshed, ie. resetted or reloaded.
        :param mod: String specifying module name or distinct string (stem like 'nnunet') that is
                    included in a bunch of modules that should all be refreshed/removed from sys modules.
                    If reload is True than mod needs to be a module otherwise the reload fails.
        :param reload: Boolean specifying if using importlib to reload a module.
        Example: All imported modules during a test that are connnected to nnUNet or Lifelong-nnUNet
                 need to be refreshed, then simply provide mod = 'nnunet'.
                 If a specific module needs to be reloaded like MultiHead_Module, then mod = MultiHead_Module
                 and reload = True. Note that mod needs to be of type Module for this to work.
    """
    # -- When the user wants to reload a Module, then use importlib -- #
    if reload == True:
        # -- Check that mod is a Module, otherwise importlib will throw an error -- #
        assert isinstance(mod, ModuleType), "When trying to reload a module, then please provide a module."
        # -- Reload/Reimport mod -- #
        importlib.reload(mod)
    # -- User does not want to use importlib, ie. the modules that contain mod in their name are removed from the sys modules -- #
    else:
        # -- Check that mod is of type string in this case -- #
        assert isinstance(mod, str), "When removing all modules based on a mapping string, than mod should be a string."
        # -- loop through all present modules in sys -- #
        for key in list(sys.modules.keys()):
            # -- If mod is part of the key name in any shape or form -- #
            if mod in key:
                # -- Remove it from the sys modules -- #
                del sys.modules[key]

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

def calculate_target_logits(mh_network, gen, num_batches_per_epoch, fp16, gpu_id=0):
    r"""This function is used to calculate the target_logits based on a transmitted generator.
        The function returns a dictionary representing the target_logits based on the mh_network.
        This function is essential for the LwF Trainer.
        :param mh_network: A MultiHead Network that is used to generate the target logits with (every head is used)
        :param gen: The generator for which the target_logits are extracted
        :param num_batches_per_epoch: Represents the number of batches per epoch
        :param fp16: Specify if using floating point 16 or not
        :param gpu_id: Specify the CUDA ID to put the model and data on. If set to -1, the CPU will be used
        :return: A dictionary with the target_logits (list of tensors) per task (head)
    """
    # -- Define where to put the data and model during the calculation -- #
    if gpu_id == -1:
        device = 'cpu'
    else:
        device = 'cuda:'+str(gpu_id)

    # -- Loop through tasks and build the corresponding model to make predictions -- #
    target_logits = dict()
    for task in list(mh_network.heads.keys()):
        # -- Build the corresponding network -- #
        network = mh_network.assemble_model(task)
        # -- Remove the softmax layer at the end by replacing the corresponding element with an identity function -- #
        network.inference_apply_nonlin = lambda x: x
        # -- Put netowrl to CPU or GPU device as desired -- #
        network.to(device)
        # -- Set network to eval -- #
        network.eval()
        # -- Add the task to the dict -- #
        target_logits[task] = list()

        # -- Make the predictions and store them in a dictionary to use during the LwF loss -- #
        for _ in range(num_batches_per_epoch):
            # -- Extract the current batch from data transform to tensor and push to GPU -- #
            data_dict = next(gen)
            x = maybe_to_torch(data_dict['data'])
            # -- Put data on GPU if no CPU is desired --> currently x is on CPU -- #
            if device != 'cpu':
                x = to_cuda(x, gpu_id=gpu_id)

            # -- Make predictions using the loaded model and data -- #
            if fp16:
                with autocast():
                    output = network(x)[0]
            else:
                output = network(x)[0]
                
            task_logit = copy.deepcopy(output.detach().cpu())   # --> To cut any links or references
            del x, output

            # -- Append the result to target_logits -- #
            target_logits[task].extend(task_logit)
            del task_logit

    # -- Empty the GPU cache if a GPU was used -- #
    if device != 'cpu':
        torch.cuda.empty_cache()

    # -- Return the target_logits -- #
    return target_logits