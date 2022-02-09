#########################################################################################################
#------------------This class represents the nnUNet trainer for frozen ViT training.-------------------#
#########################################################################################################

from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

class nnUNetTrainerFrozenUNet(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='multihead', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=True, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of frozen ViT trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)

        # -- Define a frozen argument indicating if the ViT module is already frozen or not -- #
        self.frozen = False
    
    def run_training(self, task, output_folder):
        r"""Perform training using mh trainer. After training on the first task, freeze the ViT module.
        """
        # -- Check if we trained on at least one task -- #
        if not self.frozen and len(self.mh_network.heads) == 1 and task not in self.mh_network.heads:
            # -- Freeze the whole ViT module, since it is trained on the first task -- #
            # -- Loop through the parameter names of the model -- #
            for name, param in self.network.named_parameters():
                # -- If the parameter name is in the list of the body_parameters, ie. param belongs to the body -- #
                if 'ViT' not in name:
                    # -- Set requires_grad accordingly -- #
                    param.requires_grad = False

            # -- Update mh module as well -- #
            for name, param in self.mh_network.model.named_parameters():
                # -- If the parameter name is in the list of the body_parameters, ie. param belongs to the body -- #
                if 'ViT' not in name:
                    # -- Set requires_grad accordingly -- #
                    param.requires_grad = False
            # -- Body -- #
            for name, param in self.mh_network.body.named_parameters():
                # -- If the parameter name is in the list of the body_parameters, ie. param belongs to the body -- #
                if 'ViT' not in name:
                    # -- Set requires_grad accordingly -- #
                    param.requires_grad = False
            # -- First Head -- #
            for name, param in self.mh_network.heads[list(self.mh_network.heads.keys())[0]].named_parameters():
                # -- If the parameter name is in the list of the body_parameters, ie. param belongs to the body -- #
                if 'ViT' not in name:
                    # -- Set requires_grad accordingly -- #
                    param.requires_grad = False

            # -- Set the flag so we don't have to do it again for the next task -- #
            self.frozen = True

        # -- Execute the training for the desired epochs -- #
        ret = super().run_training(task, output_folder)  # Execute training from parent class --> already_trained_on will be updated there

        return ret  # Finished with training for the specific task
        