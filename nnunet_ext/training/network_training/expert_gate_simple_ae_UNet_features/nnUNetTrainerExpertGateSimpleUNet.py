#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.expert_gate2.nnUNetTrainerExpertGate2 import nnUNetTrainerExpertGate2
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

import numpy as np

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

class nnUNetTrainerExpertGateSimpleUNet(nnUNetTrainerExpertGate2):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='expert_gate_simple_ae_UNet_features', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, feature_extractor_path=None, network=None, use_param_split=False):
        r"""Constructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. --> Note that the only
            difference to the Multi-Head Trainer is the transfer_heads flag which should always be True for this Trainer!
        """
        assert feature_extractor_path != None
        assert network == '3d_fullres'
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split, feature_extractor_path)

        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                    deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                    tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, self.vit_type, version, split_gpu,
                    transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT, feature_extractor_path)
        
        
        