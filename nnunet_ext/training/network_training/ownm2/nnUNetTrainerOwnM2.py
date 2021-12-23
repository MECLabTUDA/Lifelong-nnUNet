# EWC on whole ViT network, MiB KD loss; POD only on heads if desired

#########################################################################################################
#--------------------This class represents the nnUNet trainer for our own training.---------------------#
#########################################################################################################

# -- This implementation represents our own method -- #
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.training.network_training.ewc.nnUNetTrainerEWC import nnUNetTrainerEWC
from nnunet_ext.training.network_training.ownm1.nnUNetTrainerOwnM1 import nnUNetTrainerOwnM1
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossOwn1 as OwnLoss


class nnUNetTrainerOwnM2(nnUNetTrainerOwnM1):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='ownm2', ewc_lambda=0.4, mib_alpha=1., lkd=10, pod_lambda=1e-2,
                 scales=3, tasks_list_with_char=None, mixed_precision=True, save_csv=True, del_log=False, use_vit=True,
                 vit_type='base', version=1, split_gpu=False, transfer_heads=True, ViT_task_specific_ln=False, do_pod=True):
        r"""Constructor of our own trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, ewc_lambda, mib_alpha,
                         lkd, pod_lambda, scales, tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, vit_type,
                         version, split_gpu, transfer_heads, ViT_task_specific_ln, do_pod)
        
        # -- Define the path where the fisher and param values should be stored/restored -- #
        self.ewc_data_path = join(self.trained_on_path, 'ewc_data_ownm2')
        
    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite the initialize function so the correct Loss function for the EWC method can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)
        
        # -- Reset self.loss from MultipleOutputLoss2 to DC_and_CE_loss so the EWC Loss can be initialized properly -- #
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (Own Method) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters during each iteration -- #
        self.own_loss = OwnLoss(self.loss, self.ds_loss_weights, self.alpha, self.lkd, self.ewc_lambda,
                                self.fisher, self.params, self.network.named_parameters(), pod_lambda=self.pod_lambda,
                                scales=self.scales, do_pod=self.do_pod)

    def after_train(self):
        r"""This function needs to be executed once the training of the current task is finished.
            The function will use the same data to generate the gradients again and setting the
            models parameters.
        """
        # -- Execute the function from EWC Trainer -- #
        nnUNetTrainerEWC.after_train(self)