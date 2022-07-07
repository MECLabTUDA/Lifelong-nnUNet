#########################################################################################################
#------------------This class represents the nnUNet trainer for agnostic training.--------------------#
#########################################################################################################




# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

# class nnUNetTrainerExpertGate(nnUNetTrainerMultiHead):
#     autoencoder: ExpertGate_Autoencoder

#     # -- Trains n tasks agnostic using transfer learning -- #
#     def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
#                  unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
#                  identifier=default_plans_identifier, extension='expert_gate', tasks_list_with_char=None, mixed_precision=True,
#                  save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
#                  ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
#         r"""Constructor of Agnostic trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. --> Note that the only
#             difference to the Multi-Head Trainer is the transfer_heads flag which should always be True for this Trainer!
#         """
#         assert(transfer_heads == True)
#         assert(use_vit == False)
#         assert(vit_type == 'base')
#         assert(ViT_task_specific_ln == False)

#         # -- Initialize using parent class -- #
#         super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
#                          fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
#                          save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
#                          network, use_param_split)

#     def initialize_network(self):
#         #run init of the mh network
#         super().initialize_network()

        
#         #init the autoencoder
#         self.autoencoder = ExpertGate_Autoencoder(self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

#     def reinitialize(self, task, print_loss_info=True):
#         super().reinitialize(self, task, print_loss_info)

#         #create new autoencoder
#         self.autoencoder = ExpertGate_Autoencoder(self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

"""
from typing import OrderedDict
import torch
from nnunet_ext.network_architecture.ExpertGate_Autoencoder import ExpertGate_Autoencoder
from nnunet.training.network_training.network_trainer import NetworkTrainer
from nnunet_ext.paths import default_plans_identifier, network_training_output_dir, preprocessing_output_dir
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation

class nnUNetTrainerExpertGate(NetworkTrainer):
    def __init__(self, deterministic: bool, fp16: bool, num_epochs: int) -> None:
        super().__init__(deterministic, fp16)
        print("__init")
        self.loss = torch.nn.CrossEntropyLoss()
        self.max_num_epochs = num_epochs
        return



    def initialize(self, output_folder: str, fold: int, dataset_directory: str, 
    folder_with_preprocessed_data: str, patch_size: np.array):
        print("initialize")

        self.output_folder = output_folder
        self.fold = fold
        self.dataset_directory = dataset_directory

        self.dataset: OrderedDict = load_dataset(folder_with_preprocessed_data)
        basic_generator_patch_size: np.array = get_patch_size(patch_size)
        self.tr_gen = DataLoader3D(self.dataset,,patch_size)
        
        self.tr_gen = self.val_gen = get_moreDA_augmentation(dl_tr, dl_val,
                                                            #self.data_aug_params['patch_size_for_spatialtransform'],
                                                            patch_size,
                                                            data_aug_params,
                                                            deep_supervision_scales=deep_supervision_scales,
                                                            pin_memory=pin_memory,
                                                            use_nondetMultiThreadedAugmenter=False)
        
        self.network = None
        self.optimizer = None
        self.lr_scheduler = None


        self.was_initiliazed = True
        return

    def load_dataset(self):
        print("load dataset")
        # -- Extract the folder with the preprocessed data in it -- #
        #folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                 #"_stage%d" % stage)
        #self.dataset = load_dataset(folder_with_preprocessed_data)
        pass

    def validate(self, *args, **kwargs):
        print("validate:")
        print(args)
        print(kwargs)

    def run_training(self, task, output_folder, build_folder=True):
        print("run_training", task) 
        print(output_folder)
        folder_with_preprocessed_data: str = join(preprocessing_output_dir,)
        dataset: OrderedDict = load_dataset()
        #load data for that task and convert it using a pretrained alexnet
        #self.tr_gen = self.val_gen = None
        self.dl_tr, self.dl_val = self.get_basic_generators()
        self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                self.data_aug_params['patch_size_for_spatialtransform'],
                                                                self.data_aug_params,
                                                                deep_supervision_scales=self.deep_supervision_scales,
                                                                pin_memory=self.pin_memory,
                                                                use_nondetMultiThreadedAugmenter=False)

        #set output path
        self.output_folder = join(network_training_output_dir, "autoencoder", task)

        #create new autoencoder
        self.network = ExpertGate_Autoencoder(1000)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.001, momentum=0.5)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        #call training
        super().run_training()
        pass

"""



from typing import OrderedDict
import torch
from nnunet_ext.network_architecture.expert_gate_autoencoder import expert_gate_autoencoder
from nnunet_ext.paths import default_plans_identifier, network_training_output_dir, preprocessing_output_dir
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join


from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer



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


class nnUNetTrainerExpertGate(nnUNetTrainer):
    def initialize(self, training=True, force_load_plans=False, num_epochs: int=1000):
        super().initialize(training, force_load_plans)
        self.max_num_epochs = num_epochs


    def initialize_network(self):
        print(self.patch_size)
        input_size = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        code_dims = 100 #int(0.003 * input_size)
        self.network = expert_gate_autoencoder(input_size, code_dims)

        self.loss = torch.nn.CrossEntropyLoss()
        
        self.use_progress_bar = True

        if torch.cuda.is_available():
            self.network.cuda()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        data = torch.sigmoid(data)
        #print(data.shape)
        target = torch.flatten(data, start_dim=2)
        target = torch.squeeze(target)
        #print(target.shape)

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            #print(output.shape)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def run_online_evaluation(self, output, target):
        #print("online eval")
        pass
        
    def finish_online_evaluation(self):
        #print("finish eval")
        pass