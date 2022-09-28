##########################################################################################################
#------This module contains useful functions that are used throughout the nnUNet_extensions project.-----#
##########################################################################################################

import math, torch
import numpy as np
import pandas as pd
from torch import nn
from types import ModuleType
from datetime import datetime
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
import copy, torch, time, sys, os, shutil, importlib
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir

# --------------------------------------- Useful classes --------------------------------------- #
class SamePad(nn.Module):
    r"""Extracted from https://github.com/xxxnell/spatial-smoothing/blob/48aaeaae2446a1861828f4c9296ca87419c9caf4/models/layers.py#L42.
    """
    def __init__(self, filter_size, pad_mode="constant", **kwargs):
        r"""Extracted from """
        super(SamePad, self).__init__()

        self.pad_size = [
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
        ]
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, self.pad_size, mode=self.pad_mode)
        return x

class Blur(nn.Module):
    r"""Extracted from https://github.com/xxxnell/spatial-smoothing/blob/48aaeaae2446a1861828f4c9296ca87419c9caf4/models/layers.py#L62.
    """
    def __init__(self, in_filters, sfilter=(1, 1), pad_mode="replicate", **kwargs):
        super(Blur, self).__init__()

        filter_size = len(sfilter)
        self.pad = SamePad(filter_size, pad_mode=pad_mode)
        self.filter_proto = torch.tensor(sfilter, dtype=torch.float, requires_grad=False)
        self.filter = torch.einsum("i, j -> i j", self.filter_proto, self.filter_proto)
        self.filter = self.filter / torch.sum(self.filter)
        self.filter = self.filter.repeat([in_filters, 1, 1, 1])
        self.filter = torch.nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, self.filter, groups=x.size()[1])
        return x
    
# --------------------------------------- Useful classes --------------------------------------- #
    
# --------------------------------------- Useful methods --------------------------------------- #
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

# -- Modified version of https://www.geeksforgeeks.org/common-divisors-of-two-numbers/ -- # 
# Function to calculate all common divisors
# of two given numbers
# a, b --> input integer numbers
def gcd(a, b):
    if a == 0:
        return b
    return gcd(b % a, a)

def commDiv(a, b):
    # -- GCD of a, b -- #
    n = gcd(a, b)

    # -- Extract divisors of n -- #
    result = []
    for i in range(1, n+1):
        if n % i == 0:
            result.append(i)
    return result

def get_ViT_LSA_SPT_scale_folder_name(do_LSA, do_SPT, FeatScale, AttnScale,
                                      filter_with=None, nth_filter=None, filter_rate=None,
                                      useFFT=False, f_map_type=None, conv_smooth=None, ts_msa=False,
                                      cross_attn=False, cbam=False, registration=False):
    r"""Use this function when the ViT_U-Net is used and the output folder needs to be build.
    """
    # -- Specify the folder name based on do_LSA and do_SPT -- #
    folder_n = 'reg' if registration else 'seg'
    folder_n += str(os.sep)
    if do_SPT:
        folder_n += 'SPT'
    if do_LSA:
        folder_n += 'LSA' if len(folder_n) == 0 else '_LSA'
    if FeatScale:
        folder_n += 'FeatScale' if len(folder_n) == 0 else '_FeatScale'
    if AttnScale:
        folder_n += 'AttnScale' if len(folder_n) == 0 else '_AttnScale'
    if useFFT:
        folder_n += 'FFT' if len(folder_n) == 0 else '_FFT'
    if len(folder_n) == 0:
        folder_n = 'traditional'

    # -- Build ViT FFT filtering part -- #
    if filter_with is not None:
        if filter_with == 'high_basic':
            folder_f = filter_with + '_' + str(filter_rate) + '_' + str(nth_filter)
        else:
            folder_f = filter_with + '_' + str(nth_filter)
        folder_n = os.path.join(folder_n, folder_f)
    
    if f_map_type is not None and f_map_type != 'none':
        folder_n = os.path.join(folder_n, f_map_type)
        
    if cross_attn:
        folder_n = os.path.join(folder_n, 'cross_attn')
    
    if ts_msa and not cross_attn:
        folder_n = os.path.join(folder_n, 'TS_MSA')
        
    if conv_smooth is not None:
        folder_n = os.path.join(folder_n, 'Conv_Smoothing_' + str(conv_smooth[1]) + '_blocks_' + str(conv_smooth[0]) + '_MSAs_temp_' + str(conv_smooth[-1]))

    if cbam:
        folder_n = os.path.join(folder_n, 'CBAM_MSA')

    # -- Return the folder name -- #
    return folder_n

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

#-------------------------- Copied from nnU-Net implementation but changed -----------------------------------#
def print_to_log_file(log_file, output_folder=None, prefix_name='', *args):
    r"""This function can be used to log information into a txt file."""
    # -- Get the current timestamp -- #
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)

    # -- Extract the arguments -- #
    args = ("%s:" % dt_object, *args)

    # -- Create the log file if it does not exist -- #
    if log_file is None:
        assert output_folder is not None and len(prefix_name) > 0,\
            'When no log_file path is provided, then set the output_folder and prefix_name so we can create one..'
        maybe_mkdir_p(output_folder)
        timestamp = datetime.now()
        log_file = join(output_folder, prefix_name+"_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                       (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                        timestamp.second))
                        
    # -- Write everything form args into the log file -- #
    with open(log_file, 'a+') as f:
        for a in args:
            f.write(str(a))
            f.write(" ")
        f.write("\n")

    # -- Return the log file since we do not use global variables and then the file will be overwritten over and over again since log_file is always None -- #
    return log_file
#-------------------------- Copied from nnU-Net implementation but changed -----------------------------------#

# https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/
def calculate_2dft(input, do_torch=False):
    r"""Use this to calculate the 2D FFT with numpy or PyTorch."""
    if do_torch:
        ft = torch.fft.ifftshift(input)
        ft = torch.fft.fft2(ft)
        return torch.fft.fftshift(ft)
    else:
        ft = np.fft.ifftshift(input)
        ft = np.fft.fft2(ft)
        return np.fft.fftshift(ft)

def high_pass_filter(img_size, filter_rate):
    r"""Use this to apply extract a binary high pass filtering mask to multiply with FFT shifted volume."""
    h, w = img_size[-2:] # height and width
    mask = np.full([h, w], 1) # empty array
    cy, cx = int(h/2), int(w/2) # centerness
    rh, rw = int(filter_rate * cy), int(filter_rate * cx) # filter_size
    # the value of center pixel is zero.
    mask[cy-rh:cy+rh, cx-rw:cx+rw] = 0
    overall_size = 1
    for i in img_size[:-2]:
        overall_size *= i
    mask = np.concatenate([[mask]] * overall_size, axis=0) # stack mask n times based on img_size to get a whole volume
    mask = mask.reshape(img_size)
    return mask

def low_pass_filter(img_size, filter_rate):
    r"""Use this to apply extract a binary high pass filtering mask to multiply with FFT shifted volume."""
    h, w = img_size[-2:] # height and width
    mask = np.full([h, w], 0) # empty array
    cy, cx = int(h/2), int(w/2) # centerness
    rh, rw = int((1-filter_rate) * cy), int((1-filter_rate) * cx) # filter_size
    # the value of center pixel is 1.
    mask[cy-rh:cy+rh, cx-rw:cx+rw] = 1
    overall_size = 1
    for i in img_size[:-2]:
        overall_size *= i
    mask = np.concatenate([[mask]] * overall_size, axis=0) # stack mask n times based on img_size to get a whole volume
    mask = mask.reshape(img_size)
    return mask

# Fourier feature mapping from https://arxiv.org/pdf/2006.10739.pdf, https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb#scrollTo=OcJUfBV0dCww
def input_mapping(x, B, torch=False):
    r"""B is the mapping that should be used:   
    No mapping:  gamma(v)=v .
    Basic mapping:  gamma(v)=[cos(2𝜋v),sin(2𝜋v)]^T .
    Gaussian Fourier feature mapping:  gamma(v)=[cos(2𝜋Bv),sin(2𝜋Bv)]^T , where each entry in  𝐁∈ℝ𝑚×𝑑  is sampled from  N(0,sig^2) 
    """
    if B is None:
        return x
    else:
        if torch:
            x_proj = (2.*torch.from_numpy(np.pi)*x) @ B.T
            return torch.concatenate([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        else:
            x_proj = (2.*np.pi*x) @ B.T
            return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

def smooth_blur(in_filters, sfilter=(1, 1), pad_mode="constant"):
    r"""Blurring for smoothing Conv layers extracted from https://github.com/xxxnell/spatial-smoothing/blob/48aaeaae2446a1861828f4c9296ca87419c9caf4/models/layers.py#L34."""
    if tuple(sfilter) == (1, 1) and pad_mode in ["constant", "zero"]:
        layer = nn.AvgPool2d(kernel_size=1, stride=1)
        # layer = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
    else:
        layer = Blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)
    return layer

def Identity(x):
    return x

def softmax_helper(x):
    return F.softmax(x, 1)

@contextmanager
def suppress_stdout():
    r"""This can be used to surpress the output when executing a function.
        Extracted from: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
def single_dice_coef(y_true, y_pred_bin, do_torch=False):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin) if not do_torch else torch.sum(y_true * y_pred_bin)
    if do_torch:
        if (torch.sum(y_true)==0) and (torch.sum(y_pred_bin)==0):
            return 1
        return ((2*intersection) / (torch.sum(y_true) + torch.sum(y_pred_bin))).item()
    else:
        if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
            return 1
        return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

# def mean_dice_coef(y_true, y_pred_bin, num_classes=1, do_torch=False):
#     # from: https://www.codegrepper.com/code-examples/python/dice+similarity+coefficient+python
#     # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
#     batch_size = y_true.shape[0]
#     channel_num = y_true.shape[-1]
#     mean_dice_channel = 0.
#     for i in range(batch_size):
#         for j in range(channel_num):
#             channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j], num_classes, do_torch)
#             mean_dice_channel += channel_dice/(channel_num*batch_size)
#     return mean_dice_channel

def mean_dice_coef(y_true, y_pred_bin, num_classes=1, do_torch=False):
    # from: https://www.codegrepper.com/code-examples/python/dice+similarity+coefficient+python
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    # channel_num = y_true.shape[-1]
    mean_dice_channel = 0.
    # dict contains label: dice per batch
    channel_dices_per_batch = {i+1:list() for i in range(num_classes)}
    for i in range(batch_size):
        for j in range(1, num_classes+1):
            y_t = y_true[i, :].clone() if do_torch else copy.deepcopy(y_true[i, :])
            y_p = y_pred_bin[i, :].clone() if do_torch else copy.deepcopy(y_pred_bin[i, :])
            y_t[y_t != j] = 0
            y_t[y_t == j] = 1
            y_p[y_p != j] = 0
            y_p[y_p == j] = 1
            channel_dice = single_dice_coef(y_t, y_p, do_torch)
            channel_dices_per_batch[j].append(channel_dice)
            # channel_dice = single_dice_coef(y_true[i, :, :, j], y_pred_bin[i, :, :, j], num_classes, do_torch)
            mean_dice_channel += channel_dice/(num_classes*batch_size)
    return mean_dice_channel, channel_dices_per_batch
# --------------------------------------- Useful methods --------------------------------------- #