#########################################################################################################
#------------------This class represents the nnUNet trainer for frozen ViT training.-------------------#
#########################################################################################################

import math
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.ewc.nnUNetTrainerEWC import nnUNetTrainerEWC

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {'ewc_lambda': float}

class nnUNetTrainerFrozEWC(nnUNetTrainerEWC):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='froz_ewc', ewc_lambda=0.4, tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=False,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False, enhanced=False):
        r"""Constructor of frozen EWC trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. This method uses the
            EWC on the whole network, whereas every second task, the ViT network is frozen and not thus updated nor regularized.
            If enhanced is set, the EWC weight will be changed the following way for frozen runs: ewc_lambda*e^{-1/3}
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, ewc_lambda, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)
                    
        # -- Remove the old directory -- #
        try:
            # -- Only remove if empty -- #
            os.rmdir(self.trained_on_path)
        except:
            pass

        # -- Update the folder names including indicating if enhanced is used or not -- #
        fold_n = self.output_folder.split(os.path.sep)[-1]
        if enhanced and 'enhanced' not in self.output_folder:
            self.output_folder = join(os.path.sep, *self.output_folder.split(os.path.sep)[:-1], 'enhanced', fold_n)
        if enhanced and 'enhanced' not in self.trained_on_path:
            self.trained_on_path = join(self.trained_on_path, 'enhanced')
        if not enhanced and 'no_enhance' not in self.output_folder:
            self.output_folder = join(os.path.sep, *self.output_folder.split(os.path.sep)[:-1], 'no_enhance', fold_n)
        if not enhanced and 'no_enhance' not in self.trained_on_path:
            self.trained_on_path = join(self.trained_on_path, 'no_enhance')

        # -- Create the folder if necessary -- #
        maybe_mkdir_p(self.trained_on_path)

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                          ewc_lambda, tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, self.vit_type, version,
                          split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT, enhanced)

        # -- Store if the user wants to modify the EWC weight during frozen runs -- #
        self.enhanced = enhanced

        # -- Update the path were the fisher and param values are stored to avoid conflicts -- #   
        if self.enhanced: 
            self.ewc_data_path = join(self.trained_on_path, 'ewc_data_froz_ewc_enhanced')
        else:           
            self.ewc_data_path = join(self.trained_on_path, 'ewc_data_froz_ewc')

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite the initialize function so the paths can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)
        
        # -- Update self.trainer_path -- #
        if prev_trainer_path is not None and not call_for_eval:
            if self.enhanced:
                self.trainer_path = join(os.path.sep, *self.trainer_path.split(os.sep)[:-1], 'enhanced', "fold_%s" % str(self.fold))
            else:
                self.trainer_path = join(os.path.sep, *self.trainer_path.split(os.sep)[:-1], 'no_enhance', "fold_%s" % str(self.fold))
        else:   # If for eval, then this is a nnUNetTrainerV2 whereas the path is not build as implemented in _build_output_path
            self.trainer_path = prev_trainer_path


    def run_training(self, task, output_folder):
        r"""Perform training using EWC trainer. After training on the first task, freeze the ViT module.
            The ViT however is only frozen every second task and thus regularized when unfrozen.
        """
        # -- Execute the training for the desired epochs -- #
        if self.enhanced:
            output_folder = join(self._build_output_path(output_folder, False), 'enhanced', "fold_%s" % str(self.fold))
        else:
            output_folder = join(self._build_output_path(output_folder, False), 'no_enhance', "fold_%s" % str(self.fold))

        if len(self.mh_network.heads) % 2 == 1:
            if task in self.mh_network.heads:        # Uneven number of heads, so no freezing
                self.print_to_log_file(f"Unfreeze the ViT for task {task}..")
                # -- Unfreeze the whole ViT module -- #
                self.freeze_ViT(False)
                if self.enhanced:
                    # -- Reset the ewc value for the unfrozen_training -- #
                    self.loss.ewc_lambda = self.ewc_lambda
                    self.print_to_log_file(f"Using EWC loss weight: {self.ewc_lambda}..")

            else:   # Task not in heads so if added we're even, so freeze
                self.print_to_log_file(f"Freeze the ViT for task {task}..")
                # -- Freeze the whole ViT module -- #
                self.freeze_ViT(True)
                if self.enhanced:
                    # -- Reset the ewc value for the frozen_training using our formula -- #
                    self.loss.ewc_lambda = self.ewc_lambda * math.exp(-1/3) # --> reduce the weight of the EWC term so we can learn more when ViT is frozen
                    self.print_to_log_file(f"Reducing EWC loss weight from {self.ewc_lambda} to {self.loss.ewc_lambda}..")

        else:
            if task in self.mh_network.heads:        # Even number of heads, so freezing
                self.print_to_log_file(f"Freeze the ViT for task {task}..")
                # -- Freeze the whole ViT module -- #
                self.freeze_ViT(True)
                if self.enhanced:
                    # -- Reset the ewc value for the frozen_training using our formula -- #
                    self.loss.ewc_lambda = self.ewc_lambda * math.exp(-1/3) # --> reduce the weight of the EWC term so we can learn more when ViT is frozen
                    self.print_to_log_file(f"Reducing EWC loss weight from {self.ewc_lambda} to {self.loss.ewc_lambda}..")

            else:   # Task not in heads so if added we're uneven, so unfreeze
                self.print_to_log_file(f"Unfreeze the ViT for task {task}..")
                # -- Unfreeze the whole ViT module -- #
                self.freeze_ViT(False)
                if self.enhanced:
                    # -- Reset the ewc value for the unfrozen_training -- #
                    self.loss.ewc_lambda = self.ewc_lambda
                    self.print_to_log_file(f"Using EWC loss weight: {self.ewc_lambda}..")

        # -- Execute the training for the desired epochs -- #
        ret = super().run_training(task, output_folder, build_folder=False)  # Execute training from parent class --> already_trained_on will be updated there

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