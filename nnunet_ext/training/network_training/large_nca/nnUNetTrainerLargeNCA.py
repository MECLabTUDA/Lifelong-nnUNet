#########################################################################################################
#----------This class represents the nnUNet trainer for LWF training. Implementation--------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

# -- The implementation of this method is based on the following Source Code: -- #
# -- https://github.com/arunmallya/packnet/blob/master/src/lwf.py. -- #
# -- It represents the method proposed in the paper https://arxiv.org/pdf/1606.09282.pdf -- #
# -- The original implementation from https://github.com/lizhitwo/LearningWithoutForgetting -- #
# -- refers to the one that is used in this class, so when citing, cite both -- #

import numpy as np
import copy, torch
from time import time
from itertools import tee
from torch.cuda.amp import autocast
from nnunet_ext.network_architecture.Dummy_MultiHead_Module import Dummy_MultiHead_Module
from nnunet_ext.network_architecture.nca.OctreeNCA3D import OctreeNCA3D
from nnunet_ext.network_architecture.nca.OctreeNCA2D import OctreeNCA2D
from nnunet_ext.network_architecture.nca.LargeOctreeNCA2D import LargeOctreeNCA2D
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.utilities.helpful_functions import calculate_target_logits
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossLWF as LwFloss
from nnunet_ext.training.network_training.nca.nnUNetTrainerNCA import nnUNetTrainerNCA

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

class nnUNetTrainerLargeNCA(nnUNetTrainerNCA):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='large_nca', tasks_list_with_char=None,
                 mixed_precision=True, save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False,
                 transfer_heads=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, nca=False, train_nca_w_sigmoid=False, network=None, use_param_split=False):
        r"""Constructor of LwF trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        assert nca, "This trainer only works with NCA networks!"
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads,
                         ViT_task_specific_ln, do_LSA, do_SPT, nca, train_nca_w_sigmoid, network, use_param_split)
        

    def initialize_network(self):
        ret = super().initialize_network()

        for task in self.already_trained_on[str(self.fold)]['tasks_at_time_of_checkpoint'][1:]:
            self.reinitialize(task)


        self.network = self.mh_network.model
        return ret


    def reinitialize(self, task, print_loss_info=True):
        ret = super().reinitialize(task, print_loss_info)

        assert isinstance(self.mh_network.model, (OctreeNCA2D, LargeOctreeNCA2D))

        if isinstance(self.mh_network.model, OctreeNCA2D):
            self.mh_network.model = LargeOctreeNCA2D(self.mh_network.model,
                                                     num_channels=8,
                                                        num_new_classes = self.num_classes - 1,
                                                        hidden_size=32)
        elif isinstance(self.mh_network.model, LargeOctreeNCA2D):
            self.mh_network.model.add_new_nca(num_channels=8,
                                              num_new_classes = self.num_classes -1,
                                              hidden_size=32)
        else:
            raise NotImplementedError("Only OctreeNCA2D and LargeOctreeNCA2D supported in nnUNetTrainerLargeNCA")

        self.network = self.mh_network.model
        self.initialize_optimizer_and_scheduler()

        return ret

    def run_training(self, task, output_folder, build_folder=True):
        torch.cuda.reset_peak_memory_stats()
        ret = super().run_training(task, output_folder, build_folder)
        self.print_to_log_file(f"Peak memory usage during training: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB")
        return ret
    
    def on_epoch_end(self):
        """Overwrite this function, since we want to perform a validation after every nth epoch on all tasks
           from the head.
           NOTE: If the validation is done during run_iteration(), the validation will be performed for every batch
                 at every nth epoch which is not what we want. This will further lead into an error because too many
                 files will be then opened, thus we do it here.
        """
        # -- Include this so the frozen parts in the network are not updated -- #
        # self.initialize_optimizer_and_scheduler()

        # -- Perform everything the parent class makes -- #
        res = nnUNetTrainerV2.on_epoch_end(self) #bypass the evaluation implemented in nnUNetTrainerMultiHead

        # -- If the current epoch can be divided without a rest by self.save_every than its time for a validation -- #
        if self.epoch % self.save_every == self.save_every - 1:   # Same as checkpoint saving from nnU-Net (NOTE: this is because its 0 based)
            self._perform_validation(use_tasks=[self.task]) # <- evaluate only on current task
            self.save_checkpoint(join(self.output_folder, "training_" + str(self.epoch) +".model"), False)

        # -- Return the result from the parent class -- #
        return res