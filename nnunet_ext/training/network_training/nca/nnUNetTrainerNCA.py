#########################################################################################################
#----------This class represents the nnUNet trainer for LWF training. Implementation--------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

# -- The implementation of this method is based on the following Source Code: -- #
# -- https://github.com/arunmallya/packnet/blob/master/src/lwf.py. -- #
# -- It represents the method proposed in the paper https://arxiv.org/pdf/1606.09282.pdf -- #
# -- The original implementation from https://github.com/lizhitwo/LearningWithoutForgetting -- #
# -- refers to the one that is used in this class, so when citing, cite both -- #

import numpy as np
import copy, torch
from time import time
from itertools import tee
from torch.cuda.amp import autocast
from nnunet_ext.network_architecture.Dummy_MultiHead_Module import Dummy_MultiHead_Module
from nnunet_ext.network_architecture.nca.OctreeNCA3D import OctreeNCA3D
from nnunet_ext.network_architecture.nca.OctreeNCA2D import OctreeNCA2D
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.utilities.helpful_functions import calculate_target_logits
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossLWF as LwFloss
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

class nnUNetTrainerNCA(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='nca', tasks_list_with_char=None,
                 mixed_precision=True, save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False,
                 transfer_heads=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, nca=False, network=None, use_param_split=False):
        r"""Constructor of LwF trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        assert nca, "This trainer only works with NCA networks!"
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads,
                         ViT_task_specific_ln, do_LSA, do_SPT, nca, network, use_param_split)

        self.initial_lr = 1e-3


    """
    def initialize_network(self):
        num_levels = len(self.net_num_pool_op_kernel_sizes)
        base_num_steps = int(3 * max(self.patch_size / 2**num_levels))

        num_steps = [5] * (num_levels-1) + [20]

        if self.threeD:
            num_steps = [6,7,8,9,10,20]
            #num_steps = [5] * (num_levels-1) + [base_num_steps]
            self.mh_network = Dummy_MultiHead_Module(OctreeNCA3D, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.network,
                                                  num_channels=16, 
                                 num_input_channels=self.num_input_channels,
                                 num_classes=self.num_classes,
                                 hidden_size=64,
                                 fire_rate=0.5,
                                 num_steps=num_steps,
                                 num_levels=num_levels,
                                 pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes)
        else:
            self.mh_network = Dummy_MultiHead_Module(OctreeNCA2D, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.network,
                                                  num_channels=16, 
                                 num_input_channels=self.num_input_channels,
                                 num_classes=self.num_classes,
                                 hidden_size=64,
                                 fire_rate=0.5,
                                 num_steps=num_steps,
                                 num_levels=num_levels)

        self.network = self.mh_network.model

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=0)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_num_epochs, eta_min=1e-6)

    def maybe_update_lr(self, epoch=None):
        self.lr_scheduler.step(epoch)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))


    def process_plans(self, plans):
        super().process_plans(plans)
        if self.threeD:
            self.batch_size = 2

    """