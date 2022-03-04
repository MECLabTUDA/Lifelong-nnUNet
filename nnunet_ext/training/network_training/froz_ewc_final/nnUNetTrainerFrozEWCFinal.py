#########################################################################################################
#------------------This class represents the nnUNet trainer for frozen ViT training.-------------------#
#########################################################################################################

import math
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.ewc.nnUNetTrainerEWC import nnUNetTrainerEWC

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {'ewc_lambda': float}

class nnUNetTrainerFrozEWCFinal(nnUNetTrainerEWC):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='froz_ewc', ewc_lambda=0.4, tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=False,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of frozen EWC trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. This method uses the
            EWC on the whole network, whereas every second task, the ViT network is frozen and not thus updated nor regularized.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, ewc_lambda, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)
   
        # -- Update the path were the fisher and param values are stored to avoid conflicts -- #             
        self.ewc_data_path = join(self.trained_on_path, 'ewc_data_froz_ewc_final')

    def run_training(self, task, output_folder):
        r"""Perform training using EWC trainer. After training on the first task, freeze the ViT module.
            The ViT however is only frozen every second task and thus regularized when unfrozen.
        """
        if len(self.mh_network.heads) % 2 == 1:
            if task in self.mh_network.heads:        # Uneven number of heads, so no freezing
                self.print_to_log_file(f"Unfreeze the ViT for task {task}..")
                # -- Unfreeze the whole ViT module -- #
                self.freeze_ViT(False)
                # -- Reset the ewc value for the unfrozen_training -- #
                self.loss.ewc_lambda = self.ewc_lambda
                    
            else:   # Task not in heads so if added we're even, so freeze
                self.print_to_log_file(f"Freeze the ViT for task {task}..")
                # -- Freeze the whole ViT module -- #
                self.freeze_ViT(True)
                # -- Reset the ewc value for the frozen_training using our formula -- #
                self.loss.ewc_lambda = self.ewc_lambda * math.exp(-1/3) # --> reduce the weight of the EWC term so we can learn more when ViT is frozen

        else:
            if task in self.mh_network.heads:        # Even number of heads, so freezing
                self.print_to_log_file(f"Freeze the ViT for task {task}..")
                # -- Freeze the whole ViT module -- #
                self.freeze_ViT(True)
                # -- Reset the ewc value for the frozen_training using our formula -- #
                self.loss.ewc_lambda = self.ewc_lambda * math.exp(-1/3) # --> reduce the weight of the EWC term so we can learn more when ViT is frozen

            else:   # Task not in heads so if added we're uneven, so unfreeze
                # -- Unfreeze the whole ViT module -- #
                self.freeze_ViT(False)
                # -- Reset the ewc value for the unfrozen_training -- #
                self.loss.ewc_lambda = self.ewc_lambda

        # -- Execute the training for the desired epochs -- #
        ret = super().run_training(task, output_folder)  # Execute training from parent class --> already_trained_on will be updated there

        return ret  # Finished with training for the specific task


    def freeze_ViT(self, freeze):
        r"""Use this function to freeze all ViT components in the networks based on freeze.
        """
        # -- Loop through the parameter names of the model -- #
        for name, param in self.network.named_parameters():
            # -- If the parameter name is in the list of the body_parameters, ie. param belongs to the body -- #
            if 'ViT' in name:
                # -- Set requires_grad accordingly -- #
                param.requires_grad = not freeze

        # -- Update mh module as well -- #
        for name, param in self.mh_network.model.named_parameters():
            # -- If the parameter name is in the list of the body_parameters, ie. param belongs to the body -- #
            if 'ViT' in name:
                # -- Set requires_grad accordingly -- #
                param.requires_grad = not freeze
        # -- Body -- #
        for name, param in self.mh_network.body.named_parameters():
            # -- If the parameter name is in the list of the body_parameters, ie. param belongs to the body -- #
            if 'ViT' in name:
                # -- Set requires_grad accordingly -- #
                param.requires_grad = not freeze
        # -- First Head -- #
        for name, param in self.mh_network.heads[list(self.mh_network.heads.keys())[0]].named_parameters():
            # -- If the parameter name is in the list of the body_parameters, ie. param belongs to the body -- #
            if 'ViT' in name:
                # -- Set requires_grad accordingly -- #
                param.requires_grad = not freeze
        