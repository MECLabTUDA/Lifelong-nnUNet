#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

from typing import Tuple
import numpy as np
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.rehearsal.nnUNetTrainerRehearsal import nnUNetTrainerRehearsal

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

class nnUNetTrainerRehearsalNoSkipsFrozen(nnUNetTrainerRehearsal):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='rehearsal_no_skips_frozen', tasks_list_with_char=None, 
                 
                 samples_in_perc=0.25,
                 seed=3299,

                 layer_name_for_feature_extraction: str="",
                 
                 mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. --> Note that the only
            difference to the Multi-Head Trainer is the transfer_heads flag which should always be True for this Trainer!
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, samples_in_perc, seed, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)
        
        self.layer_name_for_feature_extraction = layer_name_for_feature_extraction
        assert self.layer_name_for_feature_extraction.startswith(("conv_blocks_context","td", "tu", "conv_blocks_localization"))
        assert self.layer_name_for_feature_extraction.count('.') == 1, "layer_name must have exactly 1 dot"
        
    def freeze_network(self):
        self.print_to_log_file("freeze network!")
        assert self.layer_name_for_feature_extraction.startswith(("conv_blocks_context", "td", "tu", "conv_blocks_localization"))
        self.network.freeze_layers(self.layer_name_for_feature_extraction)
        self.initialize_optimizer_and_scheduler()

    def run_training(self, task, output_folder, build_folder=True):
        self.network.__class__ = Generic_UNet_no_skips
        if self.tasks_list_with_char[0][0] != task:
            ## freeze encoder
            self.freeze_network()
        return super().run_training(task, output_folder, build_folder)
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        if not isinstance(self.network, Generic_UNet_no_skips):
            self.network.__class__ = Generic_UNet_no_skips
        return super().run_iteration(data_generator, do_backprop, run_online_evaluation, detach, no_loss)
    
    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True, 
                                                         mirror_axes: Tuple[int] = None, use_sliding_window: bool = True, 
                                                         step_size: float = 0.5, use_gaussian: bool = True, pad_border_mode: str = 'constant', 
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False, verbose: bool = True, 
                                                         mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(self.network, Generic_UNet_no_skips):
            self.network.__class__ = Generic_UNet_no_skips
        return super().predict_preprocessed_data_return_seg_and_softmax(data, do_mirroring, mirror_axes, use_sliding_window, step_size, use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision)
    

    