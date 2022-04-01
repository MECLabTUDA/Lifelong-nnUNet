#########################################################################################################
#-------------------This class represents the nnUNet trainer for EWC ViT training.----------------------#
#########################################################################################################

import copy
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.training.network_training.ewc.nnUNetTrainerEWC import nnUNetTrainerEWC
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossEWC as EWCLoss
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {'ewc_lambda': float}

class nnUNetTrainerEWCLN(nnUNetTrainerEWC):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='ewc', ewc_lambda=0.4, tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=True, vit_type='base', version=1, split_gpu=False, transfer_heads=False,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of EWC ViT Trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, ewc_lambda, tasks_list_with_char,
                         mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads,
                         ViT_task_specific_ln, do_LSA, do_SPT, network, use_param_split)

        # -- Update the path were the fisher and param values are stored to avoid conflicts -- #             
        self.ewc_data_path = join(self.trained_on_path, 'ewc_data_ln')
                         
    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite the initialize function so the correct Loss function for the EWC method can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)
        
        # -- Reset self.loss from MultipleOutputLoss2 to DC_and_CE_loss so the EWC Loss can be initialized properly -- #
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (EWC) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters during each iteration -- #
        self.loss = EWCLoss(self.loss, self.ds_loss_weights,
                            self.ewc_lambda,
                            self.fisher,
                            self.params,
                            self.network.named_parameters(),
                            True, ['ViT', 'norm'], True)

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function needs to be changed for this EWC method, since we only want to
            use ViT related parameters in our EWC Loss.
        """
        # -- Run iteration as usual -- #
        l = super(nnUNetTrainerMultiHead, self).run_iteration(data_generator, do_backprop, run_online_evaluation)
         
        # -- Return the loss -- #
        if not no_loss:
            # -- After running one iteration and calculating the loss, update the parameters of the loss for the next iteration -- #
            # -- NOTE: The gradients DO exist even after the loss detaching of the super function, however the loss function -- #
            # --       does not need them, since they are only necessary for the Fisher values that are calculated once the -- #
            # --       training is done performing an epoch with no optimizer steps --> see after_train() for that -- #
            # -- Update the loss in such a way that only ViT parameters are considered and transmitted -- #
            parameters = [(name, param) for name, param in self.network.named_parameters() if 'ViT' in name]
            self.loss.update_network_params(parameters)
            
        return l

    def after_train(self):
        r"""This function needs to be executed once the training of the current task is finished.
            The function will use the same data to generate the gradients again and setting the
            models parameters.
        """
        # -- Extract all Fisher values -- #
        super().after_train()

        # -- Only keep the ones with the matching case in it to save time and space -- #
        for task in list(self.fisher.keys()):
            for key in list(self.fisher[task].keys()):
                if 'norm' not in key:
                    # -- Remove the entry -- #
                    del self.fisher[task][key]
                elif 'ViT' not in key:
                    # -- Remove the entry -- #
                    del self.fisher[task][key]

        for task in list(self.params.keys()):
            for key in list(self.params[task].keys()):
                if 'norm' not in key:
                    # -- Remove the entry -- #
                    del self.params[task][key]
                elif 'ViT' not in key:
                    # -- Remove the entry -- #
                    del self.params[task][key]

        print(self.fisher[list(self.fisher.keys())[0]].keys())
        # -- Storing and putting everything on CPU before is done in super class after this function is called -- #