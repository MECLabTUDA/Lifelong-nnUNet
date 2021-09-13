#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

class nnUNetTrainerSequential(nnUNetTrainerMultiHead):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='sequential', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True):
        r"""Constructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv)

    def run_training(self, task, output_folder):
        r"""Perform training using Sequential Trainer. Everything is the same as Multi Head, except when adding a new head,
            the state_dict from the last head will be used instead of the one from the initialization. This is the basic
            transfer learning.
            NOTE: If the task does not exist, a new head will be initialized using the last head, not the one from the
                  initialization of the class. This new head is saved under task and will then be trained.
        """
        # -- Run the training from parent class setting transfer to true --> this is the only trainer that does this -- #
        ret = super().run_training(task, output_folder, transfer=True)

        return ret  # Finished with training for the specific task