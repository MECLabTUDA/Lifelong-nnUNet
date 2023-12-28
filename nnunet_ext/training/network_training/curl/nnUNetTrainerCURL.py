#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

import itertools
import pandas as pd
import sklearn
import tqdm
from nnunet_ext.network_architecture.VariationalUNet import VariationalUNetNoSkips
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.EarlyStop import EarlyStop
from nnunet_ext.training.FeatureRehearsalDataset2 import FeatureRehearsalDataset2
from nnunet_ext.training.FeatureRehearsalDataset2Analyzer import FeatureRehearsalDataset2Analyzer
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
import calendar
import time
import socket
from datetime import datetime
from _warnings import warn
from typing import Tuple
import timeit
import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.neural_network import SegmentationNetwork
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from nnunet.network_architecture.initialization import InitWeights_He

from nnunet_ext.utilities.helpful_functions import add_folder_before_filename, flattendict, nestedDictToFlatTable
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
from itertools import tee
import pickle
from nnunet_ext.network_architecture.generic_UNet import Generic_UNet
import random
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
import torch.nn.functional as F
import torch
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from multiprocessing import Process, Queue
from nnunet_ext.inference import predict
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.augmentations.utils import pad_nd_image
import torchvision
from nnunet_ext.network_architecture.VAE import *
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalConcatDataset, FeatureRehearsalMultiDataset, FeatureRehearsalTargetType, FeatureRehearsalDataLoader, InfiniteIterator
import torch.utils.data, torch.utils.data.sampler
from nnunet_ext.training.investigate_vae import visualize_latent_space, visualize_second_stage_latent_space
from nnunet_ext.training.IncrementalSummaryWriter import IncrementalSummaryWriter
from nnunet.utilities.nd_softmax import softmax_helper

#########################################################################################################
#----------This class represents the nnUNet Multiple Head Trainer. Implementation-----------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import numpy as np
import os, copy, torch
from itertools import tee
from torch.cuda.amp import autocast
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from nnunet_ext.utilities.helpful_functions import *
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet_ext.training.model_restore import restore_model
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.network_architecture.MultiHead_Module import MultiHead_Module
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet_ext.paths import default_plans_identifier, evaluation_output_dir, preprocessing_output_dir, default_plans_identifier





# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}


GENERATED_FEATURE_PATH_TR = "generated_features_tr"
EXTRACTED_FEATURE_PATH_TR = "extracted_features_tr"
EXTRACTED_FEATURE_PATH_VAL = "extracted_features_val"

class nnUNetTrainerCURL(nnUNetTrainerMultiHead):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='curl', tasks_list_with_char=None, 
                 #custom args
                 #target_type: FeatureRehearsalTargetType = FeatureRehearsalTargetType.GROUND_TRUTH,
                 num_rehearsal_samples_in_perc: float= 1.0,
                 layer_name_for_feature_extraction: str="",

                 mixed_precision=True,
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
        self.num_rehearsal_samples_in_perc = num_rehearsal_samples_in_perc
        self.layer_name_for_feature_extraction = layer_name_for_feature_extraction
        assert self.num_rehearsal_samples_in_perc > 0 and self.num_rehearsal_samples_in_perc <= 1, "samples_in_perc should be between 0 and 1: (0, 1]."
        assert self.layer_name_for_feature_extraction.startswith(("conv_blocks_context","td", "tu", "conv_blocks_localization"))
        assert self.layer_name_for_feature_extraction.count('.') == 1, "layer_name must have exactly 1 dot"
        self.num_feature_rehearsal_cases = 0        # store amount of feature_rehearsal cases. init with 0
        self.rehearse = False                       # variable for alternating schedule


        self.dict_from_file_name_to_task_idx = {}
        self.task_label_to_task_idx = []

        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                          tasks_list_with_char, num_rehearsal_samples_in_perc, layer_name_for_feature_extraction, 
                          mixed_precision, save_csv, del_log, use_vit, self.vit_type,
                          version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT)
        
        self.UNET_CLASS = VariationalUNetNoSkips

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr / 10, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        
        #self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None

    def super_initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = VariationalUNetNoSkips(input_channels=self.num_input_channels, 
                                              base_num_features=self.base_num_features,
                                              num_classes=self.num_classes, 
                                              num_pool=len(self.net_num_pool_op_kernel_sizes),
                                              patch_size=self.patch_size.tolist(), 
                                              num_conv_per_stage=self.conv_per_stage,
                                              feat_map_mul_on_downscale=2,
                                              conv_op=conv_op,
                                                norm_op=norm_op,
                                                norm_op_kwargs=norm_op_kwargs,
                                                dropout_op=dropout_op,
                                                dropout_op_kwargs=dropout_op_kwargs,
                                                nonlin=net_nonlin,
                                                nonlin_kwargs=net_nonlin_kwargs,
                                                deep_supervision=True,
                                                dropout_in_localization=False,
                                                final_nonlin=lambda x : x,
                                                weightInitializer=InitWeights_He(1e-2),
                                                pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
                                                conv_kernel_sizes=self.net_conv_kernel_sizes,
                                                upscale_logits=False,
                                                convolutional_pooling=True,
                                                convolutional_upsampling=True,
                                              vit_version=self.version, 
                                              vit_type=self.vit_type,
                                              split_gpu=self.split_gpu, 
                                              ViT_task_specific_ln=self.ViT_task_specific_ln,
                                              first_task_name=self.tasks_list_with_char[0][0], 
                                              do_LSA=self.LSA, 
                                              do_SPT=self.SPT)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_network(self):
        r"""Extend Initialization of Network --> Load pre-trained model (specified to setup the network).
            Optimizer and lr initialization is still the same, since only the network is different.
        """
        if self.trainer_path is None:
            # -- Specify if 2d or 3d plans is necessary -- #
            _2d_plans = "_plans_2D.pkl" in self.plans_file
            # -- Load the correct plans file, ie. the one from the first task -- #
            self.process_plans(load_pickle(join(preprocessing_output_dir, self.tasks_list_with_char[0][0], self.identifier + "_plans_2D.pkl" if _2d_plans else self.identifier + "_plans_3D.pkl")))
            # -- Backup the patch_size since we need to restore it again -- #
            patch_size = self.patch_size
            net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes

            # -- Create a Multi Head Generic_UNet from the current network using the provided split and first task name -- #
            # -- Do not rely on self.task for initialization, since the user might provide the wrong task (unintended), -- #
            # -- however for self.plans, the user needs to extract the correct plans_file path by himself using always the -- #
            # -- first task from a list of tasks since the network is build using the plans_file and thus the structure might vary -- #
            if self.use_vit:
                self.first_task_name = self.tasks_list_with_char[0][0]
                # -- Initialize from beginning and start training, since no model is provided using ViT architecture -- #
                nnViTUNetTrainer.initialize_network(self) # --> This updates the corresponding variables automatically since we inherit this class
                self.mh_network = MultiHead_Module(VariationalUNetNoSkips, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.network,
                                                   input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                                   num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes),\
                                                   patch_size=patch_size.tolist(), vit_version=self.version, vit_type=self.vit_type,\
                                                   split_gpu=self.split_gpu, ViT_task_specific_ln=self.ViT_task_specific_ln,\
                                                   first_task_name=self.tasks_list_with_char[0][0], do_LSA=self.LSA, do_SPT=self.SPT)
            else:
                # -- Initialize from beginning and start training, since no model is provided -- #
                self.super_initialize_network() # --> This updates the corresponding variables automatically since we inherit this class

                self.mh_network = MultiHead_Module(VariationalUNetNoSkips, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.network,
                                                   input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                                   num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes))
                
            # -- Load the correct plans file again, ie. the one from the current task -- #
            self.process_plans(load_pickle(self.plans_file))
            # -- Update the loss so there will be no error during forward function -- #
            self._update_loss_after_plans_change(net_num_pool_op_kernel_sizes, patch_size)  # patch_size etc. is restored in here
            del patch_size, net_num_pool_op_kernel_sizes

            # -- Add the split to the already_trained_on since it is simplified by now -- #
            self.already_trained_on[str(self.fold)]['used_split'] = self.mh_network.split
            # -- Save the updated dictionary as a json file -- #
            write_pickle(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.pkl'))
            # -- Update self.init_tasks so the storing works properly -- #
            self.update_init_args()
            return  # Done with initialization

        # -- Load the model and parameters and initialize it -- #
        # -- NOTE: self.trainer_model can be a nnUNetTrainerV2 or a Multi Head Network with a model, body and heads. -- #
        print("Loading trainer and setting the network for training")
        if 'fold_' in self.trainer_path:
            # -- Do not add the addition of fold_X to the path -- #
            checkpoint = join(self.trainer_path, "model_final_checkpoint.model")
        else:
            # -- fold_X is not in the path so add it to be in the correct location for the checkpoint loading -- #
            checkpoint = join(self.trainer_path, 'fold_'+str(self.fold), "model_final_checkpoint.model")
        pkl_file = checkpoint + ".pkl"
        use_extension = not 'nnUNetTrainerV2' in self.trainer_path
        self.trainer_model = restore_model(pkl_file, checkpoint, train=False, fp16=self.mixed_precision,\
                                           use_extension=use_extension, extension_type=self.extension,\
                                           param_search=self.param_split, network=self.network_name)
        self.trainer_model.initialize(True)

        # -- Delete the created log_file from the restored model and set it to None since we don't need it (eg. during eval) -- #
        if self.del_log:
            os.remove(self.trainer_model.log_file)
            self.trainer_model.log_file = None

        # -- Update the loss so there will be no error during forward function -- #
        self._update_loss_after_plans_change(self.trainer_model.net_num_pool_op_kernel_sizes, self.trainer_model.patch_size)

        # -- Some sanity checks and loads.. -- #
        # -- Check if the trainer contains plans.pkl file which it should have after sucessfull training -- #
        if 'fold_' in self.trainer_model.output_folder:
            # -- Remove the addition of fold_X from the output_folder, since the plans.pkl is outside of the fold_X directories -- #
            plans_dir = self.trainer_model.output_folder.replace('fold_', '')[:-1]
        else:
            # -- If no fold_ in output_folder, everything is fine -- #
            plans_dir = self.trainer_model.output_folder
        
        assert isfile(join(plans_dir, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file.."

        # -- Check that the trainer type is as expected -- #
        assert isinstance(self.trainer_model, (nnUNetTrainerV2, nnUNetTrainerMultiHead)), "The trainer needs to be nnUNetTrainerV2 or nnUNetTrainerMultiHead.."

        # -- Set mh_network -- #
        # -- Make it to Multi Head Network first and then update the heads if the model was of Multi Head type -- #
        # -- Use the first task in tasks_joined_name, since this represents the corresponding task name, whereas self.task -- #
        # -- is the task to train on, which is not equal to the one that will be initialized now using a pre-trained network -- #
        # -- (prev_trainer). -- #
        if self.trainer_model.__class__.__name__ == nnUNetTrainerV2.__name__:   # Important when doing evaluation, since nnUNetTrainerV2 has no mh_network
            self.mh_network = MultiHead_Module(VariationalUNetNoSkips, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.trainer_model.network,
                                               input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                               num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes))
        elif self.trainer_model.__class__.__name__ == nnViTUNetTrainer.__name__:   # Important when doing evaluation, since nnViTUNetTrainer has no mh_network
            self.mh_network = MultiHead_Module(VariationalUNetNoSkips, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.trainer_model.network,
                                               input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                               num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes),\
                                               patch_size=self.patch_size.tolist(), vit_version=self.version, vit_type=self.vit_type,\
                                               split_gpu=self.split_gpu, ViT_task_specific_ln=self.ViT_task_specific_ln,\
                                               first_task_name=self.tasks_list_with_char[0][0], do_LSA=self.LSA, do_SPT=self.SPT)
        else:
            self.mh_network = self.trainer_model.mh_network

        if not self.trainer_model.__class__.__name__ == nnUNetTrainerV2.__name__\
        and not self.trainer_model.__class__.__name__ == nnViTUNetTrainer.__name__: # Do not use isinstance, since nnUNetTrainerMultiHead is instance of nnUNetTrainerV2 
            # -- Ensure that the split that has been previously used and the current one are equal -- #
            # -- NOTE: Do this after initialization, since the splits might be different before but still lead to the same level after -- #
            # -- Split simplification. -- #
            prev_split = self.already_trained_on[str(self.fold)]['used_split']
            assert self.mh_network.split == prev_split,\
                "To continue training on the fold {} the same split, ie. \'{}\' needs to be provided, not \'{}\'.".format(self.fold, prev_split, self.mh_network.split)
            # -- Delete the prev_split --> not necessary anymore -- #
            del prev_split
            # -- Reset the mh_network using the base models -- #
            self.mh_network = self.trainer_model.mh_network
        
        # -- Set self.network to the model in mh_network --> otherwise the network is not initialized and not in right type -- #
        self.network = self.trainer_model.network    # Does not matter what the model is, will be updated in run_training anyway

    def run_training(self, task, output_folder, build_folder=True):
        #self.num_batches_per_epoch = 5

        ## clear feature folder on first task!
        if self.tasks_list_with_char[0][0] == task:
            self.print_to_log_file("first task. deleting feature sets")
            self.clean_up()

            assert self.was_initialized
            self.save_checkpoint(join(self.output_folder, "before_training.model"), False)
        else:
            ## freeze encoder
            self.network.__class__ = self.UNET_CLASS

        #compute the sum of all weights in the network
        
        self.network.__class__ = self.UNET_CLASS
        ret = super().run_training(task, output_folder, build_folder)
        #ret = None

        if task == self.tasks_list_with_char[0][-1]:
            self.print_to_log_file("last task. stopping already and do not train VAE.")
            self.clean_up()
            return

        ## compute features, store them and update feature_rehearsal_dataset
        self.store_features(task)
        self.store_features(task, False)
        self.update_dataloader()

        analyzer = FeatureRehearsalDataset2Analyzer(self.extracted_features_dataset_tr)
        analyzer.compute_statistics()
        self.print_to_log_file("extracted_features_dataset_tr:")
        self.print_to_log_file(str(analyzer))
        del analyzer

        ## delete old features
        self.clean_up([EXTRACTED_FEATURE_PATH_TR]) #after training, delete extracted features -> preserve privacy
        self.clean_up([GENERATED_FEATURE_PATH_TR]) #berfore generating, make sure the previous samples are deleted!
        self.generate_features()


        analyzer = FeatureRehearsalDataset2Analyzer(self.generated_feature_rehearsal_dataset)
        analyzer.compute_statistics()
        self.print_to_log_file("generated_feature_rehearsal_dataset:")
        self.print_to_log_file(str(analyzer))
        del analyzer


        return ret
    


    def store_features(self, task: str, train: bool=True):
        self.print_to_log_file("extract features!")

        with torch.no_grad():
            # preprocess training cases and put them in a queue

            input_folder = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task, 'imagesTr')
            
            expected_num_modalities = load_pickle(self.plans_file)['num_modalities'] # self.plans
            case_ids = predict.check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

            if train:
                ## take train cases only
                #print(case_ids)                 # <- all cases from this dataset
                #print(self.dataset_tr.keys())   # <- all train cases from this dataset
                assert set(self.dataset_tr.keys()).issubset(set(case_ids)), "idk what, but something is wrong " + str(self.dataset_tr.keys()) + " " + str(case_ids)
                case_ids = list(self.dataset_tr.keys())

                ## take train cases subset
                case_ids = random.sample(case_ids, round(len(case_ids) * self.num_rehearsal_samples_in_perc))
                self.num_feature_rehearsal_cases += len(case_ids)

                self.print_to_log_file("the following cases will be used for feature rehearsal training:" + str(case_ids))
                output_folder = join(self.trained_on_path, self.extension,  EXTRACTED_FEATURE_PATH_TR)
            else:
                assert set(self.dataset_val.keys()).issubset(set(case_ids)), "idk what, but something is wrong " + str(self.dataset_val.keys()) + " " + str(case_ids)
                case_ids = list(self.dataset_val.keys())
                self.print_to_log_file("the following cases will be used for feature rehearsal validation:" + str(case_ids))
                output_folder = join(self.trained_on_path, self.extension,  EXTRACTED_FEATURE_PATH_VAL)


            maybe_mkdir_p(output_folder)

            output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
            all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
            list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
            
            output_filenames = output_files[0::1]
            output_filenames = [add_folder_before_filename(f, task) for f in output_filenames]


            assert len(list_of_lists) == len(output_filenames)

            for o in output_filenames:
                dr, f = os.path.split(o)
                if len(dr) > 0:
                    maybe_mkdir_p(dr)


            ground_truth_segmentations = []
            for input_path_in_list in list_of_lists:
                input_path = input_path_in_list[0]
                # read segmentation file and place it in ground_truth_segmentations
                input_path_array = input_path.split('/')
                assert(input_path_array[-2] == "imagesTr")
                input_path_array[-2] = "labelsTr"
                assert(input_path_array[-1].endswith('_0000.nii.gz'))
                input_path_array[-1] = input_path_array[-1][:-12] + '.nii.gz'

                segmentation_path = join(*input_path_array)
                segmentation_path = "/" + segmentation_path
                ground_truth_segmentations.append(segmentation_path)

            print(ground_truth_segmentations)

            #print(list_of_lists[0::1])
            #exit()
            #preprocessing_generator = predict.preprocess_multithreaded(self, list_of_lists[0::1], output_filenames)
            preprocessing_generator = self._preprocess_multithreaded(list_of_lists[0::1], output_filenames, segs_from_prev_stage=ground_truth_segmentations)
            
            # for all preprocessed training cases with seg mask
            for preprocessed in preprocessing_generator:
                output_filename, (d,dct), gt_segmentation = preprocessed

                if isinstance(d, str):
                    assert isinstance(gt_segmentation, str)
                    data = np.load(d)
                    os.remove(d)
                    d = data

                    s = np.load(gt_segmentation)
                    os.remove(gt_segmentation)
                    gt_segmentation = s

                assert np.all(d.shape[1:] == gt_segmentation.shape), str(d.shape) + " " + str(gt_segmentation.shape)
                #unpack channel dimension on data

                # turn off deep supervision ???
                #step_size = 0.5 # TODO verify!!!
                #pad_border_mode = 'constant'
                #pad_kwargs = {'constant_values': 0}
                #mirror_axes = self.data_aug_params['mirror_axes']
                current_mode = self.network.training
                self.network.eval()

                ds = self.network.do_ds
                
                output_filename = output_filename
                
                self.network.do_ds = False                           #default: False    TODO
                do_mirroring = False                                #default: True      only slight performance gain anyways
                mirror_axes = self.data_aug_params['mirror_axes']   #hard coded
                use_sliding_window = True                           #hard coded
                step_size = 0.5                                     #default        (?)
                use_gaussian = True                                 #hard coded
                pad_border_mode = 'constant'                        #default        (unset)
                pad_kwargs = None                                   #default        (unset)
                all_in_gpu = False                                  #default        (?)
                verbose = True                                      #default        (unset)
                mixed_precision = True                              #default        (?)
                

                ret = self.network.predict_3D(d, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision, 
                                      ground_truth_segmentation=gt_segmentation, feature_dir=output_filename[:-7],
                                      layer_name_for_feature_extraction=self.layer_name_for_feature_extraction)

                self.network.do_ds = ds
                self.network.train(current_mode)
            # END: for preprocessed


    def update_dataloader(self):
        ## update dataloader
        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)

        if layer == "conv_blocks_context":
            num_features = id + 1
        elif layer == "td":
            num_features = id + 2
        else:
            num_features = len(self.network.conv_blocks_context)

        self.extracted_features_dataset_tr = FeatureRehearsalDataset2(join(self.trained_on_path, self.extension,  EXTRACTED_FEATURE_PATH_TR), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, self.tasks_list_with_char[0], load_skips=False, load_meta=True)
        
        self.extracted_features_dataset_val = FeatureRehearsalDataset2(join(self.trained_on_path, self.extension,  EXTRACTED_FEATURE_PATH_VAL), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, self.tasks_list_with_char[0], load_skips=False, load_meta=True)
        
        
        dataset = FeatureRehearsalDataset2(join(self.trained_on_path, self.extension,  GENERATED_FEATURE_PATH_TR), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, self.tasks_list_with_char[0], load_skips=False,
                                            constant_skips = self.extracted_features_dataset_val.constant_skips, load_meta=True)
        if len(dataset) > 0:
            self.generated_feature_rehearsal_dataset = dataset
        else:
            del dataset

    def update_generated_dataloader(self):
        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)
        if layer == "conv_blocks_context":
            num_features = id + 1
        elif layer == "td":
            num_features = id + 2
        else:
            num_features = len(self.network.conv_blocks_context)
            
        dataset = FeatureRehearsalDataset2(join(self.trained_on_path, self.extension,  GENERATED_FEATURE_PATH_TR), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, self.tasks_list_with_char[0], load_skips=False,
                                            constant_skips = self.extracted_features_dataset_val.constant_skips, load_meta=True)
        self.generated_feature_rehearsal_dataset = dataset
        
        dataloader = FeatureRehearsalDataLoader(dataset, batch_size=int(self.batch_size), num_workers=8, pin_memory=True, deep_supervision_scales=self.deep_supervision_scales, persistent_workers=True)
        self.generated_feature_rehearsal_dataiter = InfiniteIterator(dataloader)




    def _preprocess_multithreaded(self, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
        #mostly copied from inference/predict
        def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes,
                             transpose_forward):

            errors_in = []
            for i, l in enumerate(list_of_lists):
                try:
                    output_file = output_files[i]
                    print("preprocessing", output_file)
                    d, _, dct = preprocess_fn(l)
                    # print(output_file, dct)
                    if segs_from_prev_stage[i] is not None:
                        assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                            ".nii.gz"), "segs_from_prev_stage" \
                                        " must point to a " \
                                        "segmentation file " + str(segs_from_prev_stage[i])
                        seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                        # check to see if shapes match
                        img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                        assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                        "stage don't have the same pixel array " \
                                                                                        "shape! image: %s, seg_prev: %s" % \
                                                                                        (l[0], segs_from_prev_stage[i])
                        seg_prev = seg_prev.transpose(transpose_forward)
                        seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1)
                        assert(np.all(seg_reshaped.shape == d.shape[1:])), "error in segmentation shape probably"

                        #seg_reshaped = to_one_hot(seg_reshaped, classes)
                        #d = np.vstack((d, seg_reshaped)).astype(np.float32)
                    """There is a problem with python process communication that prevents us from communicating objects 
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                    filename or np.ndarray and will handle this automatically"""
                    print(d.shape)
                    seg_shape = np.prod(seg_reshaped.shape) if seg_reshaped is not None else 0
                    if np.prod(d.shape) + seg_shape > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                        print(
                            "This output is too large for python process-process communication. "
                            "Saving output temporarily to disk")
                        np.save(output_file[:-7] + ".npy", d)
                        d = output_file[:-7] + ".npy"
                        np.save(output_file[:-7] + "_seg.npy", seg_reshaped)
                        seg_reshaped = output_file[:-7] + "_seg.npy"

                    q.put((output_file, (d, dct), seg_reshaped))
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    print("error in", l)
                    print(e)
                    errors_in.append(l)
            q.put("end")
            if len(errors_in) > 0:
                print("There were some errors in the following cases:", errors_in)
                print("These cases were ignored.")
            else:
                print("This worker has ended successfully, no errors to report")
            # restore output
            # sys.stdout = sys.__stdout__
        ###########################################################################################################################
        
        
        if segs_from_prev_stage is None:
            segs_from_prev_stage = [None] * len(list_of_lists)

        num_processes = min(len(list_of_lists), num_processes)

        classes = list(range(1, self.num_classes))
        assert isinstance(self, nnUNetTrainer)
        q = Queue(1)
        processes = []
        for i in range(num_processes):
            pr = Process(target=preprocess_save_to_queue, args=(self.preprocess_patient, q,
                                                                list_of_lists[i::num_processes],
                                                                output_files[i::num_processes],
                                                                segs_from_prev_stage[i::num_processes],
                                                                classes, self.plans['transpose_forward']))
            pr.start()
            processes.append(pr)

        try:
            end_ctr = 0
            while end_ctr != num_processes:
                item = q.get()
                if item == "end":
                    end_ctr += 1
                    continue
                else:
                    yield item

        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()  # this should not happen but better safe than sorry right
                p.join()

            q.close()


    def clean_up(self, to_be_deleted=[EXTRACTED_FEATURE_PATH_TR, EXTRACTED_FEATURE_PATH_VAL, GENERATED_FEATURE_PATH_TR]):
        assert isinstance(to_be_deleted, list)
        for feature_folder in to_be_deleted:
            for task in self.tasks_list_with_char[0]:
                if os.path.isfile(join(self.trained_on_path, self.extension,  feature_folder, task, "meta.pkl")):
                    os.remove(join(self.trained_on_path, self.extension,  feature_folder, task, "meta.pkl"))

                for folder in ["gt", "features", "predictions", "feature_pkl"]:
                    path = join(self.trained_on_path, self.extension,  feature_folder, task, folder)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    else:
                        for f in os.listdir(path):
                            os.remove(join(path,f))
                    assert len(os.listdir(path)) == 0

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #
        if not isinstance(self.network, self.UNET_CLASS):
            self.network.__class__ = self.UNET_CLASS
        
        rehearse = False
        
        if do_backprop and len(self.mh_network.heads.keys()) > 1: # only enable the chance of rehearsal when training (not during evaluation) and when trainind not training the first task
            rehearse = self.rehearse
            self.rehearse = not self.rehearse

        if rehearse:
            data_dict = next(self.generated_feature_rehearsal_dataiter)
        else:
            data_dict = next(data_generator)

        data = data_dict['data']        # torch.Tensor (normal),    list[torch.Tensor] (rehearsal)
        target = data_dict['target']    # list[torch.Tensor]


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
                    output, (mean, log_var) = self.network(data, return_mean_log_var=True)
                del data
                if not no_loss:
                    l = self.loss(output, target)
                    if not rehearse:
                        if isinstance(output, tuple):
                            batch_size = output[0].shape[0]
                        else:
                            batch_size = output.shape[0]
                        # -- Add KL Divergence loss to the loss function -- #
                        current_task_idx = len(self.mh_network.heads.keys()) - 1
                        expected_mean = 8*current_task_idx # 0, 8, 16, ...
                        assert len(mean.shape) == 4, f"mean.shape: {mean.shape}"
                        KL_divergence = 0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + (mean-expected_mean)**2) / (mean.shape[2:].numel() * batch_size)
                        l += KL_divergence


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
                output, (mean, log_var) = self.network(data, return_mean_log_var=True)
            del data

            if not no_loss:
                l = self.loss(output, target)
                if not rehearse:
                    if isinstance(output, tuple):
                        batch_size = output[0].shape[0]
                    else:
                        batch_size = output.shape[0]
                    # -- Add KL Divergence loss to the loss function -- #
                    current_task_idx = len(self.mh_network.heads.keys()) - 1
                    expected_mean = 8*current_task_idx # 0, 8, 16, ...
                    assert len(mean.shape) == 4, f"mean.shape: {mean.shape}"
                    KL_divergence = 0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + (mean-expected_mean)**2) / (mean.shape[2:].numel() * batch_size)
                    l += KL_divergence

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

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True, 
                                                         mirror_axes: Tuple[int] = None, 
                                                         use_sliding_window: bool = True, step_size: float = 0.5, 
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant', 
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False, 
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        
        if not isinstance(self.network, self.UNET_CLASS):
            self.network.__class__ = self.UNET_CLASS

        return super().predict_preprocessed_data_return_seg_and_softmax(data, do_mirroring, mirror_axes, use_sliding_window, step_size, use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision)
    


    @torch.no_grad()
    def generate_features(self, num_samples_per_task=None):
        if num_samples_per_task is None:
            num_samples_per_task = len(self.extracted_features_dataset_tr)
        self.print_to_log_file(f"generating {num_samples_per_task} features per task")


        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)
        if layer == "conv_blocks_context":
            num_features = id + 1
        elif layer == "td":
            num_features = id + 2
        else:
            num_features = len(self.network.conv_blocks_context)
        
        output_folder = join(self.trained_on_path, self.extension,  GENERATED_FEATURE_PATH_TR)
        maybe_mkdir_p(output_folder)
        
        prev_do_ds = self.network.do_ds
        self.network.do_ds = False
        self.network.eval()
        self.network.cuda()
        
        dummy = torch.zeros(1, self.num_input_channels, *self.patch_size).cuda()
        _, dummy_features_and_skips = self.network(dummy, layer_name_for_feature_extraction=self.layer_name_for_feature_extraction)

        for y, task in enumerate(self.already_trained_on[str(self.fold)]['finished_training_on']):
            self.print_to_log_file(f"generating for task {task} where {y} is the task index")


            assert not os.path.isfile(join(output_folder, task, "meta.pkl"))
            # _dict = load_pickle(join(output_folder, task, "meta.pkl"))
            _dict = dict()

            for sample_idx in tqdm.tqdm(range(num_samples_per_task), total=num_samples_per_task):#20000
                # sample rand from N(8*current_task_idx, I)
                dummy_features_and_skips[-1] = torch.normal(8*y, 1, dummy_features_and_skips[-1].shape).cuda()
                output = self.network.generate(dummy_features_and_skips, self.layer_name_for_feature_extraction)
                
                features = dummy_features_and_skips[-1]
                #features = features[0] # remove batch dimension
                output = output[0] # remove batch dimension
                segmentation = output.argmax(0)

                file_name = f"{task}_{sample_idx}_{0}_{0}_{0}"
                _dict[f"{task}_{sample_idx}"] = {
                            'max_x': 0,
                            'max_y': 0,
                            'max_z': 0
                        }
                
                np.save(join(output_folder, task, "features", file_name + "_" + str(num_features-1) + ".npy"), features.cpu().numpy())
                np.save(join(output_folder, task, "predictions", file_name + "_0.npy"), segmentation.cpu().numpy())
                open(join(output_folder, task, "gt", file_name + ".txt"), 'a').close()
            write_pickle(_dict, join(output_folder, task, "meta.pkl"))


        self.network.do_ds = prev_do_ds


        dataset = FeatureRehearsalDataset2(join(self.trained_on_path, self.extension,  GENERATED_FEATURE_PATH_TR), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, self.tasks_list_with_char[0], load_skips=False,
                                            constant_skips = self.extracted_features_dataset_val.constant_skips, load_meta=True)
        self.generated_feature_rehearsal_dataset = dataset
        
        dataloader = FeatureRehearsalDataLoader(dataset, batch_size=int(self.batch_size), num_workers=8, pin_memory=True, deep_supervision_scales=self.deep_supervision_scales, persistent_workers=True)
        #dataloader = FeatureRehearsalDataLoader(dataset, batch_size=3, num_workers=8, pin_memory=True, deep_supervision_scales=self.deep_supervision_scales, persistent_workers=True)
        self.generated_feature_rehearsal_dataiter = InfiniteIterator(dataloader)
        #next(self.generated_feature_rehearsal_dataiter)
