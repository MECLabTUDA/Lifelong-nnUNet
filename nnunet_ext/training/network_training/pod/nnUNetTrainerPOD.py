#########################################################################################################
#-----------------------This class represents the nnUNet trainer for POD training.----------------------#
#########################################################################################################

# -- This implementation represents the method proposed in the paper https://arxiv.org/pdf/2011.11390.pdf -- #
# -- PODNet for further details: https://arxiv.org/pdf/2004.13513.pdf -- #
# -- This Trainer however does not use the pseudo-labeling approach presented in the paper, for this, we -- #
# -- have the PLOP Trainer -- #

import copy
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.training.network_training.plop.nnUNetTrainerPLOP import nnUNetTrainerPLOP
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossPOD as PODLoss

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {'pod_lambda': float, 'scales': int}

class nnUNetTrainerPOD(nnUNetTrainerPLOP):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='pod', pod_lambda=1e-2, scales=3, tasks_list_with_char=None,
                 mixed_precision=True, save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False,
                 transfer_heads=True, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of POD trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, pod_lambda, scales, tasks_list_with_char,
                         mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads,
                         ViT_task_specific_ln, do_LSA, do_SPT, network, use_param_split)
        
        # -- Remove placeholders from PLOP method that are not used here -- #
        del self.thresholds, self.max_entropy

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite the initialize function so the correct Loss function for the PLOP method can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)

        # -- Define a loss_base for the PODLoss so it can be initialized properly -- #
        loss_base = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (POD) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters during each iteration -- #
        self.loss_plop = PODLoss(loss_base,
                                 self.ds_loss_weights,
                                 self.pod_lambda,
                                 self.scales)

    def reinitialize(self, task):
        r"""This function is used to reinitialize the Trainer when a new task is trained for the POD Trainer.
        """

        # -- Execute the super function -- #
        if len(self.mh_network.heads) == 1 and task in self.mh_network.heads:
            super(nnUNetTrainerPLOP, self).reinitialize(task, True)
        else:
            super(nnUNetTrainerPLOP, self).reinitialize(task, False)
            # -- Print Loss update -- #
            self.print_to_log_file("I am using POD loss now")

    def run_training(self, task, output_folder):
        r"""Overwrite super class to adapt for POD training method.
        """
        # -- Create a deepcopy of the previous, ie. currently set model if we do PLOP training -- #
        if task not in self.mh_network.heads:
            if self.split_gpu and not self.use_vit:
                self.network.to('cpu')
            self.network_old = copy.deepcopy(self.network)
            
            if self.split_gpu and not self.use_vit:
                self.network.cuda(0)
                self.network_old.cuda(1)    # Put on second GPU
            
            # -- Register the hook here as well -- #
            self.register_forward_hooks(old=True)

        # -- Run training using grand parent class -- #
        ret = super(nnUNetTrainerPLOP, self).run_training(task, output_folder)

        # -- Return the result -- #
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function needs to be changed for the PLOP method, since intermediate results will be used within
            the Loss function to compute the Loss as proposed in the paper. --> For this, the pod flag is used.
        """
        # -- Use parent class for iteration -- #
        loss = super().run_iteration(data_generator, do_backprop, run_online_evaluation, detach, no_loss, pod=True)

        # -- Return the loss -- #
        return loss