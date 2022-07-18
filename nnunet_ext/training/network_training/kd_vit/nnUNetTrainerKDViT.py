#########################################################################################################
#----------------------This class represents the nnUNet trainer for PLOP training.----------------------#
#########################################################################################################

# -- This implementation represents the method proposed in the paper https://arxiv.org/pdf/2011.11390.pdf -- #
# -- PODNet for further details: https://arxiv.org/pdf/2004.13513.pdf -- #

import copy, torch
from time import time
from tqdm import trange
from operator import attrgetter
from torch.cuda.amp import autocast
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.loss_functions.crossentropy import entropy
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossKDViT as KDLoss
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {'kd_temp': float, 'alpha': float, 'beta': float}

class nnUNetTrainerKDViT(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='kd_vit', kd_temp=10., alpha=2/3, beta=1/3, tasks_list_with_char=None,
                 mixed_precision=True, save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False,
                 transfer_heads=True, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, FeatScale=False, AttnScale=False,
                 filter_rate=0.35, filter_with=None, nth_filter=10, useFFT=False, f_map_type='none', conv_smooth=None,
                 ts_msa=True, cross_attn=False, cbam=False, network=None, use_param_split=False):
        r"""Constructor of KDViT trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads,
                         ViT_task_specific_ln, do_LSA, do_SPT, FeatScale, AttnScale,
                         filter_rate, filter_with, nth_filter, useFFT, f_map_type, conv_smooth, True, cross_attn, cbam, network, use_param_split)
        
        # -- Set the lambda scales variable for the PLOP Loss calculation during training -- #
        self.kd_temp = kd_temp
        self.alpha = alpha
        self.beta = beta
        self.ts_msa = True   # <-- Should always be True or this trainer makes no sense..

        # -- Add flags in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        if already_trained_on is not None:
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                # -- Add the PLOP temperature and checkpoint settings -- #
                self.already_trained_on[str(self.fold)]['used_temp'] = self.kd_temp
                self.already_trained_on[str(self.fold)]['used_alpha'] = self.alpha
                self.already_trained_on[str(self.fold)]['used_beta'] = self.beta
            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['used_temp', 'used_alpha', 'used_beta']
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            # -- Update settings in trained on file for restoring to be able to ensure that scales can not be changed during training -- #
            self.already_trained_on[str(self.fold)]['used_temp'] = self.kd_temp
            self.already_trained_on[str(self.fold)]['used_alpha'] = self.alpha
            self.already_trained_on[str(self.fold)]['used_beta'] = self.beta

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, already_trained_on, use_progress, identifier, extension,
                          kd_temp, alpha, beta, tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, self.vit_type,
                          version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT, FeatScale, AttnScale,
                          filter_rate, filter_with, nth_filter, useFFT, f_map_type, conv_smooth, ts_msa, cross_attn, cbam, network, use_param_split)

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite the initialize function so the correct Loss function for the PLOP method can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)

        # -- Choose the right loss function (KDLoss) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.loss = KDLoss(loss, self.ds_loss_weights, self.kd_temp, self.alpha, self.beta)

    def reinitialize(self, task):
        r"""This function is used to reinitialize the Trainer when a new task is trained for the PLOP Trainer.
            The most important thing here is that it sets the intermediate results accordingly in the loss.
            This should only be called when a new task is used --> by that time the new loss applies..
        """
        # -- Execute the super function -- #
        if len(self.mh_network.heads) == 1 and task in self.mh_network.heads:
            super().reinitialize(task, True)
        else:
            super().reinitialize(task, False)

            # -- Print Loss update -- #
            self.print_to_log_file("I am using KD_ViT loss now")

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function needs to be changed for the PLOP method, since intermediate results will be used within
            the Loss function to compute the Loss as proposed in the paper.
        """
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                if self.use_vit and self.filter_with is not None and self.iteration % self.nth_filter == 0:
                    output = self.network(data, fft_filter=self.filter_with, filter_rate=self.filter_rate)
                    if self.task in self.mh_network.heads and len(self.mh_network.heads) > 1:
                        # -- Build network with previous head -- #
                        # self.network = self.mh_network.assemble_model(list(self.network.ViT.blocks.msa_heads.items())[0][0])
                        x_o = self.network(data, fft_filter=self.filter_with, filter_rate=self.filter_rate, task_name=list(self.network.ViT.blocks.msa_heads.items())[0][0])
                        # self.network = self.mh_network.assemble_model(list(self.network.ViT.blocks.msa_heads.items())[1][0])
                else:
                    output = self.network(data)
                    if self.task in self.mh_network.heads and len(self.mh_network.heads) > 1:
                        # -- Build network with previous head -- #
                        # self.network = self.mh_network.assemble_model(list(self.network.ViT.blocks.msa_heads.items())[0][0])
                        x_o = self.network(data, task_name=list(self.network.ViT.blocks.msa_heads.items())[0][0])   # <-- Use previous head
                        # self.network = self.mh_network.assemble_model(list(self.network.ViT.blocks.msa_heads.items())[1][0])
                        # ----------- #
                        # Check if this actually does the work !!
                        # ----------- #
                del data
                if not no_loss:
                    if self.task in self.mh_network.heads and len(self.mh_network.heads) == 1:
                        l = self.loss(output, None, target)
                    else:
                        l = self.loss(output, x_o, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            if self.use_vit and self.filter_with is not None and self.iteration % self.nth_filter == 0:
                output = self.network(data, fft_filter=self.filter_with, filter_rate=self.filter_rate)
                if self.task in self.mh_network.heads and len(self.mh_network.heads) > 1:
                    # -- Build network with previous head -- #
                    # self.network = self.mh_network.assemble_model(list(self.network.ViT.blocks.msa_heads.items())[0][0])
                    x_o = self.network(data, fft_filter=self.filter_with, filter_rate=self.filter_rate, task_name=list(self.network.ViT.blocks.msa_heads.items())[0][0])
                    # self.network = self.mh_network.assemble_model(list(self.network.ViT.blocks.msa_heads.items())[1][0])
            else:
                output = self.network(data)
                if self.task in self.mh_network.heads and len(self.mh_network.heads) > 1:
                    # -- Build network with previous head -- #
                    # self.network = self.mh_network.assemble_model(list(self.network.ViT.blocks.msa_heads.items())[0][0])
                    x_o = self.network(data, task_name=list(self.network.ViT.blocks.msa_heads.items())[0][0])   # <-- Use previous head
                    # self.network = self.mh_network.assemble_model(list(self.network.ViT.blocks.msa_heads.items())[1][0])
                    
                    
            del data
            if not no_loss:
                if self.task in self.mh_network.heads and len(self.mh_network.heads) == 1:
                    l = self.loss(output, None, target)
                else:
                    l = self.loss(output, x_o, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        self.iteration += 1
        
        # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
        if do_backprop:
            self.mh_network.update_after_iteration()
        
        # -- Return the loss -- #
        if not no_loss:
            if detach:
                l = l.detach().cpu().numpy()
            return l