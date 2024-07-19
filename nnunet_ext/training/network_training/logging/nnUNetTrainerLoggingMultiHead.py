from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
from nnunet_ext.utilities.logger import _EmptybLogger, WandbLogger, try_get_id
import numpy as np
import os, copy, torch
from itertools import tee
from torch.cuda.amp import autocast
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from nnunet_ext.utilities.helpful_functions import *
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet_ext.training.model_restore import restore_model
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.network_architecture.MultiHead_Module import MultiHead_Module
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet_ext.paths import default_plans_identifier, evaluation_output_dir, preprocessing_output_dir, default_plans_identifier



class nnUNetTrainerLoggingMultiHead(nnUNetTrainerMultiHead):


    ### DO NOT CALL ON __init__ ###
    # otherwise, the logger will be initialized for inference as well
    def _maybe_init_logger(self, init_wandb=True):
        if not hasattr(self, 'wandb_logger'):
            if init_wandb:
                self.wandb_logger = WandbLogger({
                    'wandb_entity': None,
                    'wandb_project': "Lifelong nnUNet",
                    'wandb_run_name': f"{join_texts_with_char(self.tasks_list_with_char[0], self.tasks_list_with_char[1])}/{self.__class__.__name__}/{try_get_id()}",
                })
            else:
                self.wandb_logger = _EmptybLogger()

    def run_training(self, task, output_folder, build_folder=True):

        self._maybe_init_logger()
        return super().run_training(task, output_folder, build_folder)
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function runs an iteration based on the underlying model. It returns the detached or undetached loss.
            The undetached loss might be important for methods that have to extract gradients without always copying
            the run_iteration function.
            NOTE: The calling class needs to set self.network according to the desired task, this is not done in this
                  function but expected by the user.
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
                output = self.network(data)
                del data
                if not no_loss:
                    l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            if not no_loss:
                l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        
        # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
        if do_backprop:
            self.mh_network.update_after_iteration()
        
        # -- Return the loss -- #
        if not no_loss:
            if detach:
                l = l.detach().cpu().numpy()
                self.wandb_logger.update({'loss': l})
            return l
        
    def on_epoch_end(self):
        self.wandb_logger.increment_step()
        return super().on_epoch_end()
    
    def maybe_update_lr(self, epoch=None):
        self.wandb_logger.update({'lr': self.optimizer.param_groups[0]['lr']})
        super().maybe_update_lr(epoch)


        
    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file("Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class])
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")
        
        log_dict = {}
        for i, dc in enumerate(global_dc_per_class):
            log_dict[f'dice_class_{i}'] = dc
        self.wandb_logger.update(log_dict)

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []