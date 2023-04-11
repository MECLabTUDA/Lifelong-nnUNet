#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

from _warnings import warn
from typing import Tuple

import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.neural_network import SegmentationNetwork
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler

matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime
from tqdm import trange
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

import pickle
from nnunet_ext.network_architecture.generic_UNet import Generic_UNet
import random
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
import torch.nn.functional as F
import torch

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

NUM_FEATURE_SAMPLES = 20
FEATURE_PATH = "extracted_features"
REHEARSAL_PROBABILITY = 30      # 30% training samples are rehearsed


class nnUNetTrainerFeatureRehearsal(nnUNetTrainerMultiHead):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='feature_rehearsal', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. --> Note that the only
            difference to the Multi-Head Trainer is the transfer_heads flag which should always be True for this Trainer!
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)
        

    def run_training(self, task, output_folder, build_folder=True):

        ## clear feature folder on first task!
        if self.tasks_list_with_char[0][0] == task:
            self.print_to_log_file("first task. deleting feature sets")
            path = join(self.trained_on_path, self.extension,  FEATURE_PATH)
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                for f in os.listdir(path):
                    os.remove(join(path,f))
        else:
            ## freeze encoder
            self.freeze_encoder()


        #print("before training: ", self.network.conv_blocks_context[0].blocks[0].conv.weight)

        self.network.__class__ = Generic_UNet
        ret = super().run_training(task, output_folder, build_folder)
        #ret = None
        #print("after training: ", self.network.conv_blocks_context[0].blocks[0].conv.weight)
        

        ## freeze encoder
        self.freeze_encoder()

        ## compute features and store them
        self.store_features(task)

        return ret
    

    def freeze_encoder(self):
        # TODO these layers might be influenced by weight decay and/or (nesterov) momentum
        for name, param in self.network.named_parameters():
            if 'conv_blocks_context' in name or 'td' in name:
                param.requires_grad = False

    def store_features(self, task):
        self.print_to_log_file("extract features!")


        # TODO make this in a normal way. This is kinda messed up 
        with torch.no_grad():
            last_feature_set_keys = []
            for i in range(NUM_FEATURE_SAMPLES):
                for _ in range(100): # this way we have to test at most 100 * NUM_FEATURE_SAMPLES for a valid feature set
                    data_dict = next(self.tr_gen)

                    if np.any(last_feature_set_keys != data_dict['keys']):
                        break
                
                data = data_dict['data']
                target = data_dict['target']
                last_feature_set_keys = data_dict['keys']
                self.print_to_log_file(data_dict['keys'])

                data = maybe_to_torch(data)
                target = maybe_to_torch(target)

                if torch.cuda.is_available():
                    data = to_cuda(data)
                    target = to_cuda(target)

                output, features = self.network(data, return_features=True)


                l = self.loss(output, target)
                self.print_to_log_file("sanity check loss during feature extraction: "+ str(l.item()))


                
                ## store features and target
                # target: List of torch.Tensor
                # output: Tuple of torch.Tensor (?)
                # output[0]: torch.Tensor (full resolution)

                #print(output[0])
                #print(target)
                #print(len(output))
                #print(len(target))

                pseudo_labels = []
                for i in range(len(target)):
                    pseudo_labels.append(torch.argmax(output[i],1, keepdim=True)) #perform inference

                feature_train_pair = (features, pseudo_labels)
                #feature_train_pair = (features, target)


                #pickle.dump(feature_train_pair, FEATURE_PATH + str(i) + ".pkl")
                #write_pickle(feature_train_pair, join(self.output_folder, FEATURE_PATH, task + str(i) + ".pkl"))
                write_pickle(feature_train_pair, join(self.trained_on_path, self.extension,  FEATURE_PATH, task + str(i) + ".pkl"))


    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #

        rehearse = False
        if do_backprop and len(self.mh_network.heads.keys()) > 1: # only enable the chance of rehearsal when training (not during evaluation) and when trainind not training the first task
            rehearse = random.randint(0,99) < REHEARSAL_PROBABILITY


        if rehearse:
            rehearsal_pair_file = random.choice(os.listdir(join(self.trained_on_path, self.extension,  FEATURE_PATH)))
            data, target = load_pickle(join(self.trained_on_path, self.extension,  FEATURE_PATH, rehearsal_pair_file))
        else:
            data_dict = next(data_generator)
            data = data_dict['data']
            target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                if rehearse:
                    output = self.network.feature_forward(data)
                else:
                    output = self.network(data)
                del data
                if not no_loss:
                    l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            if rehearse:
                output = self.network.feature_forward(data)
            else:
                output = self.network(data)
            del data
            if not no_loss:
                l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        
        # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
        if do_backprop:
            self.mh_network.update_after_iteration()
        
        # -- Return the loss -- #
        if not no_loss:
            if detach:
                l = l.detach().cpu().numpy()
            return l
