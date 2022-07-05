#########################################################################################################
#-----------This class represents the Cascaded ViT_U-Net Trainer. Implementation------------------------#
#----------------inspired by original implementation (--> nnUNetTrainerV2CascadeFullRes).---------------#
#########################################################################################################

import os, torch
from nnunet_ext.utilities.helpful_functions import *
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes

# -- Add this since default option file_descriptor has a limitation on the number of open files. -- #
# -- Default config might cause the runtime error: RuntimeError: received 0 items of ancdata -- #
torch.multiprocessing.set_sharing_strategy('file_system')

class nnViTUNetTrainerCascadeFullRes(nnUNetTrainerV2CascadeFullRes): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, previous_trainer="nnViTUNetTrainer",
                 fp16=False, save_interval=5, use_progress=True, version=1, vit_type='base', split_gpu=False,
                 ViT_task_specific_ln=False, first_task_name=None, do_LSA=False, do_SPT=False,
                 FeatScale=False, AttnScale=False, useFFT=False, f_map_type='none', conv_smooth=None, ts_msa=False,
                 cross_attn=False, cbam=False):
        r"""Constructor of ViT_U-Net Trainer for full resolution cascaded nnU-Nets.
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
        
        # -- FFT flag to replace MSA -- #
        self.useFFT = useFFT
        self.f_map_type = f_map_type
        self.ts_msa = ts_msa
        self.cross_attn = cross_attn
        self.cbam = cbam
        self.conv_smooth = conv_smooth

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
        folder_n = get_ViT_LSA_SPT_scale_folder_name(self.LSA, self.SPT, self.featscale, self.attnscale, self.useFFT,\
                                                     self.f_map_type, self.conv_smooth, self.ts_msa, self.cross_attn, self.cbam)
        # -- Add to the path -- #
        if folder_n != output_folder.split(os.path.sep)[-1] and folder_n not in output_folder:
            output_folder = os.path.join(output_folder, folder_n)

        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic, previous_trainer, fp16)

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
                          deterministic, previous_trainer, fp16, save_interval, use_progress, version,
                          self.vit_type, split_gpu, ViT_task_specific_ln, first_task_name, do_LSA, do_SPT,
                          FeatScale, AttnScale, useFFT, f_map_type)

    def process_plans(self, plans):
        r"""Modify the original function. This just reduces the batch_size by half.
        """
        # -- Initialize using parent class -- #
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
        r"""Use the same function as from the nnViTUNetTrainer.
        """
        # -- This makes everything we want without duplicating the code -- #
        nnViTUNetTrainer.initialize_network(self)