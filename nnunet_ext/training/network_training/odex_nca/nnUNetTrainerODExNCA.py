#########################################################################################################
#----------------------This class represents the nnUNet trainer for ODEx with NCA.-----------------------#
#########################################################################################################

# -- This implementation represents the method proposed in the-- #

import copy, torch
from torch.cuda.amp import autocast
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead


class nnUNetTrainerODExNCA(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='odex_nca', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, nca=False, network=None, use_param_split=False):
        r"""Constructor of Odex Trainer
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads,
                         ViT_task_specific_ln, do_LSA, do_SPT, nca, network, use_param_split)
    
        self.NQM_dict = dict()

    def run_training(self, task, output_folder):
        r"""Overwrite super class to adapt for ood detection
        """

        # -- Create a deepcopy of the previous, ie. currently set model if we do PLOP training -- #
        if task not in self.mh_network.heads:
            print("???? new task")

        # -- Run training using parent class -- #
        ret = super().run_training(task, output_folder)

        # compute NQM for task and save it in 
        self.NQM_dict[task] = 0.2 * (1 + len(self.NQM_dict)) # TODO: use real NQM

        # -- Return the result -- #
        return ret
