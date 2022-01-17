#########################################################################################################
#----------------------This class represents the nnUNet trainer for MiB training.-----------------------#
#########################################################################################################

# -- This implementation represents the method proposed in the paper https://arxiv.org/pdf/2002.00718.pdf -- #

import copy, torch
from torch.cuda.amp import autocast
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossMiB as MiBLoss
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {'mib_alpha': float, 'lkd': float}

class nnUNetTrainerMiB(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='mib', mib_alpha=1., lkd=10, tasks_list_with_char=None,
                 mixed_precision=True, save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False,
                 transfer_heads=True, use_param_split=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None):
        r"""Constructor of MiB trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads, use_param_split,
                         ViT_task_specific_ln, do_LSA, do_SPT, network)
        
        # -- Set the alpha and kl variable for the MiB Loss calculation during training -- #
        self.alpha = mib_alpha
        self.lkd = lkd

        # -- Add flags in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        if already_trained_on is not None:
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                # -- Add the MiB temperature and checkpoint settings -- #
                self.already_trained_on[str(self.fold)]['used_alpha'] = self.alpha
                self.already_trained_on[str(self.fold)]['used_lkd'] = self.lkd
            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['used_alpha', 'used_lkd']
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            # -- Update settings in trained on file for restoring to be able to ensure that scales can not be changed during training -- #
            self.already_trained_on[str(self.fold)]['used_alpha'] = self.alpha
            self.already_trained_on[str(self.fold)]['used_lkd'] = self.lkd

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, already_trained_on, use_progress, identifier, extension,
                          mib_alpha, lkd, tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, self.vit_type,
                          version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT)

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite the initialize function so the correct Loss function for the MiB method can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)

        # -- Create a backup loss, so we can switch between original and LwF loss -- #
        self.loss_orig = copy.deepcopy(self.loss)

        # -- Choose the right loss function (MiB) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters during each iteration -- #
        self.loss_mib = MiBLoss(#self.num_classes-1, # Remove the background class since it has been added during initialization
                                self.alpha,
                                self.lkd,
                                self.ds_loss_weights)

    def reinitialize(self, task):
        r"""This function is used to reinitialize the Trainer when a new task is trained for the MiB Trainer.
            The most important thing here is that it sets the intermediate results accordingly in the loss.
            This should only be called when a new task is used --> by that time the new loss applies..
        """
        # -- Execute the super function -- #
        if len(self.mh_network.heads) == 1 and task in self.mh_network.heads:
            super().reinitialize(task, True)
        else:
            super().reinitialize(task, False)

            # -- Print Loss update -- #
            self.print_to_log_file("I am using MiB loss now")

    def run_training(self, task, output_folder):
        r"""Overwrite super class to adapt for MiB training method.
        """
        # -- Create a deepcopy of the previous, ie. currently set model if we do PLOP training -- #
        if task not in self.mh_network.heads:
            self.network_old = copy.deepcopy(self.network)
            if self.split_gpu and not self.use_vit:
                self.network_old.cuda(1)    # Put on second GPU

        # -- Run training using parent class -- #
        ret = super().run_training(task, output_folder)

        # -- Return the result -- #
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function needs to be changed for the MiB method, since the old modekls predictions
            will be used as proposed in the paper.
        """
        # -- Ensure that the first task is trained as usual and the validation without the plop loss as well -- #
        if self.task in self.mh_network.heads and len(self.mh_network.heads) == 1 or run_online_evaluation: # The very first task
            # -- Use the original loss for this -- #
            self.loss = self.loss_orig
            # -- Run iteration as usual using parent class -- #
            loss = super().run_iteration(data_generator, do_backprop, run_online_evaluation, detach, no_loss)
            # -- NOTE: If this is called during _perform_validation, run_online_evaluation is true --> Does not matter -- #
            # --       which loss is used, since we only calculate Dice and IoU and do not keep track of the loss -- #
        else:   # --> More than one head, ie. trained on more than one task  --> use PLOP
            # -- Switch to MiB loss -- #
            self.loss = self.loss_mib
            #------------------------------------------ Partially copied from original implementation ------------------------------------------#
            # -- Extract data -- #
            data_dict = next(data_generator)
            data = data_dict['data']
            target = data_dict['target']
            # -- Transform data to torch if necessary -- #
            data = maybe_to_torch(data)
            target = maybe_to_torch(target)
            # -- Put data on GPU -- #
            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)

            self.optimizer.zero_grad()

            if self.fp16:
                with autocast():
                    output = self.network(data) # --> self.interm_results is filled with intermediate result now!
                    # -- Extract the old results using the old network -- #
                    if self.split_gpu and not self.use_vit:
                        data = to_cuda(data, gpu_id=1)
                    output_o = self.network_old(data) # --> self.old_interm_results is filled with intermediate result now!
                    (x.detach for x in output_o)
                    if not no_loss:
                        loss = self.loss(output, output_o, target)

                if do_backprop:
                    self.amp_grad_scaler.scale(loss).backward()
                    self.amp_grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
            else:
                output = self.network(data)
                if self.split_gpu and not self.use_vit:
                    data = to_cuda(data, gpu_id=1)
                output_o = self.network_old(data)
                (x.detach for x in output_o)
                del data
                if not no_loss:
                    loss = self.loss(output, output_o, target)

                if do_backprop:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.optimizer.step()

            if run_online_evaluation:
                self.run_online_evaluation(output, target)

            del target
            #------------------------------------------ Partially copied from original implementation ------------------------------------------#
        
            # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
            if do_backprop:
                self.mh_network.update_after_iteration()

            # -- Detach the loss -- #
            if not no_loss and detach:
                loss = loss.detach().cpu().numpy()

        # -- Return the loss -- #
        if not no_loss:
            return loss