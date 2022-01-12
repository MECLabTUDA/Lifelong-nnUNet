#########################################################################################################
#----------------This class represents the ViT_U-Net Trainer. Implementation----------------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import os, torch
import torch.nn as nn
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet_ext.utilities.helpful_functions import get_ViT_LSA_SPT_folder_name

# -- Add this since default option file_descriptor has a limitation on the number of open files. -- #
# -- Default config might cause the runtime error: RuntimeError: received 0 items of ancdata -- #
torch.multiprocessing.set_sharing_strategy('file_system')

class nnViTUNetTrainer(nnUNetTrainerV2): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, use_progress=True, version=1,
                 vit_type='base', split_gpu=False, ViT_task_specific_ln=False, first_task_name=None, do_LSA=False,
                 do_SPT=False):
        r"""Constructor of ViT_U-Net Trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Set ViT task specific flags -- #
        self.ViT_task_specific_ln = ViT_task_specific_ln
        self.first_task_name = first_task_name

        # -- Set the desired network version -- #
        self.version = 'V' + str(version)

        # -- Create the variable indicating which ViT Architecture to use, base, large or huge -- #
        self.vit_type = vit_type.lower()

        # -- LSA and SPT flags -- #
        self.LSA, self.SPT = do_LSA, do_SPT

        # -- Update the output_folder accordingly -- #
        if self.version not in output_folder:
            output_folder = output_folder.replace(self.__class__.__name__, self.__class__.__name__+self.version)

        # -- Add the vit_type before the fold -- #
        if self.vit_type != output_folder.split(os.path.sep)[-1] and self.vit_type not in output_folder:
            output_folder = os.path.join(output_folder, self.vit_type)

        # -- Add the ViT_task_specific_ln before the fold -- #
        if self.ViT_task_specific_ln:
            if 'task_specific'!= output_folder.split(os.path.sep)[-1] and 'task_specific' not in output_folder:
                output_folder = os.path.join(output_folder, 'task_specific')
        else:
            if 'not_task_specific'!= output_folder.split(os.path.sep)[-1] and 'not_task_specific' not in output_folder:
                output_folder = os.path.join(output_folder, 'not_task_specific')

        # -- Add the LSA and SPT before the fold -- #
        folder_n = get_ViT_LSA_SPT_folder_name(self.LSA, self.SPT)
        # -- Add to the path -- #
        if folder_n != output_folder.split(os.path.sep)[-1] and folder_n not in output_folder:
            output_folder = os.path.join(output_folder, folder_n)

        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

        # -- Set save_every, so the super trainer class creates checkpoint individually and the validation metrics will be filtered accordingly -- #
        self.save_every = save_interval

        # -- Set use_prograss_bar if desired so a progress will be shown in the terminal -- #
        self.use_progress_bar = use_progress

        # -- Define if the model should be split onto multiple GPUs -- #
        self.split_gpu = split_gpu
        if self.split_gpu:
            assert torch.cuda.device_count() > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, use_progress, version, self.vit_type, split_gpu,
                          ViT_task_specific_ln, first_task_name, do_LSA, do_SPT)

    def process_plans(self, plans):
        r"""Modify the original function. This just reduces the batch_size by half.
        """# -- Initialize using parent class -- #
        super().process_plans(plans)

        # -- Reduce the batch_size by half after it has been set by super class -- #
        # -- Do this so it fits onto GPU --> if it still does not, model needs to be put onto multiple GPUs -- #
        self.batch_size = self.batch_size // 2

    def initialize(self, training=True, force_load_plans=False, num_epochs=500):
        r"""Overwrite parent function, since we want to be able to manually set the maximum number of epochs to train.
        """
        # -- Initialize using super class -- #
        super().initialize(training, force_load_plans) # --> This updates the corresponding variables automatically since we inherit this class

        # -- Set nr_epochs to provided number -- #
        self.max_num_epochs = num_epochs

    def initialize_network(self):
        r"""Modify the initialization by using the Generic_ViT_UNet instead of the conventional Generic_UNet.
        """
        #------------------------------------------ Copied from original implementation ------------------------------------------#
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        #------------------------------------------ Copied from original implementation ------------------------------------------#
        
        self.network = Generic_ViT_UNet(self.num_input_channels, self.base_num_features, self.num_classes, 
                                    len(self.net_num_pool_op_kernel_sizes), self.patch_size.tolist(),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    vit_version=self.version, vit_type=self.vit_type, split_gpu=self.split_gpu,
                                    ViT_task_specific_ln=self.ViT_task_specific_ln, first_task_name=self.first_task_name,
                                    do_LSA=self.LSA, do_SPT=self.SPT)
        
        # -- Set the task to use --> user can not register new task here since this is a simple one time Trainer, not a Sequential one or so -- #
        if self.ViT_task_specific_ln:
            self.network.ViT.use_task(self.first_task_name)
        
        #------------------------------------------ Modified from original implementation ------------------------------------------#
        if torch.cuda.is_available():
            self.network.cuda()
            # -- When the user wants to split the network, put everything defined in network.split_names onto second GPU -- #
            if self.split_gpu:
                for name in self.network.split_names:
                    getattr(self.network, name).cuda(1)
        self.network.inference_apply_nonlin = softmax_helper
        #------------------------------------------ Modified from original implementation ------------------------------------------#