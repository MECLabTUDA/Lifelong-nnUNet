#########################################################################################################
#------------------This class represents the nnUNet trainer for expert gate training.--------------------#
#########################################################################################################


#expert_gate_experiment = "expert_gate_monai"
#expert_gate_experiment = "expert_gate_monai_alex_features"
#expert_gate_experiment = "expert_gate_monai_UNet_features"
#expert_gate_experiment = "expert_gate_simple_ae"
#expert_gate_experiment = "expert_gate_simple_ae_alex_features"
#expert_gate_experiment = "expert_gate_simple_ae_UNet_features"
#expert_gate_experiment = "expert_gate_UNet"
#expert_gate_experiment = "expert_gate_UNet_alex_features"




import math
import traceback
from nnunet_ext.network_architecture.expert_gate_UNet import expert_gate_UNet
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

from nnunet_ext.network_architecture.expert_gate_autoencoder import expert_gate_autoencoder
from nnunet.dataset_conversion.utils import generate_dataset_json


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

from nnunet.network_architecture.initialization import InitWeights_He
from torch import Tensor, nn
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

from nnunet_ext.training.network_training.sequential import nnUNetTrainerSequential
from nnunet_ext.network_architecture.generic_UNet_features import genericUNet_features

#from monai.networks.nets import AutoEncoder
from nnunet_ext.network_architecture.expert_gate_monai_ae import ExpertGateMonaiAutoencoder
from nnunet_ext.network_architecture.expert_gate_autoencoder2 import VAE
from typing import Tuple

from  torch.nn.modules.upsampling import Upsample

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

class nnUNetTrainerExpertGate2(nnUNetTrainerMultiHead):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='expert_gate2', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False, feature_extractor_path = None):
        r"""Constructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. --> Note that the only
            difference to the Multi-Head Trainer is the transfer_heads flag which should always be True for this Trainer!
        """
        # -- Initialize using parent class -- #
        print("nnUNetTrainerExpertGate2")
        super(nnUNetTrainerExpertGate2, self).__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)
        self.online_eval_mse = []
        assert self.extension in [  "expert_gate_monai"                          ,
                                    "expert_gate_monai_alex_features"           ,
                                    "expert_gate_monai_UNet_features"           ,
                                    "expert_gate_simple_ae"                     ,
                                    "expert_gate_simple_ae_alex_features"       ,
                                    "expert_gate_simple_ae_UNet_features"       ,
                                    "expert_gate_UNet"                          ,
                                    "expert_gate_UNet_alex_features"            
                                ], self.extension
        self.feature_extractor_path = feature_extractor_path

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            self.print_to_log_file("WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            self.print_to_log_file("WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

    def initialize_network(self):
        r"""Extend Initialization of Network --> Load pre-trained model (specified to setup the network).
            Optimizer and lr initialization is still the same, since only the network is different.
        """
        if self.trainer_path is None:
            # -- Specify if 2d or 3d plans is necessary -- #
            _2d_plans = "_plans_2D.pkl" in self.plans_file
            # -- Load the correct plans file, ie. the one from the first task -- #
            self.process_plans(load_pickle(join(preprocessing_output_dir, 
                self.first_task if hasattr(self, 'first_task') else self.tasks_list_with_char[0][0], 
                self.identifier + "_plans_2D.pkl" if _2d_plans else self.identifier + "_plans_3D.pkl"))
            )
            # -- Backup the patch_size since we need to restore it again -- #
            patch_size = self.patch_size
            net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes

            # -- Create a Multi Head Generic_UNet from the current network using the provided split and first task name -- #
            # -- Do not rely on self.task for initialization, since the user might provide the wrong task (unintended), -- #
            # -- however for self.plans, the user needs to extract the correct plans_file path by himself using always the -- #
            # -- first task from a list of tasks since the network is build using the plans_file and thus the structure might vary -- #
            if self.use_vit:
                self.first_task_name = self.tasks_list_with_char[0][0]
                # -- Initialize from beginning and start training, since no model is provided using ViT architecture -- #
                nnViTUNetTrainer.initialize_network(self) # --> This updates the corresponding variables automatically since we inherit this class
                self.mh_network = MultiHead_Module(Generic_ViT_UNet, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.network,
                                                   input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                                   num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes),\
                                                   patch_size=patch_size.tolist(), vit_version=self.version, vit_type=self.vit_type,\
                                                   split_gpu=self.split_gpu, ViT_task_specific_ln=self.ViT_task_specific_ln,\
                                                   first_task_name=self.tasks_list_with_char[0][0], do_LSA=self.LSA, do_SPT=self.SPT)
            else:
                # -- Initialize from beginning and start training, since no model is provided -- #
                super(nnUNetTrainerV2, self).initialize_network() # --> This updates the corresponding variables automatically since we inherit this class
                self.mh_network = MultiHead_Module(Generic_UNet, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.network,
                                                   input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                                   num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes))
                
            # -- Load the correct plans file again, ie. the one from the current task -- #
            self.process_plans(load_pickle(self.plans_file))
            # -- Update the loss so there will be no error during forward function -- #
            self._update_loss_after_plans_change(net_num_pool_op_kernel_sizes, patch_size)  # patch_size etc. is restored in here
            del patch_size, net_num_pool_op_kernel_sizes

            # -- Add the split to the already_trained_on since it is simplified by now -- #
            self.already_trained_on[str(self.fold)]['used_split'] = self.mh_network.split
            # -- Save the updated dictionary as a json file -- #
            write_pickle(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.pkl'))
            # -- Update self.init_tasks so the storing works properly -- #
            self.update_init_args()
            return  # Done with initialization

        # -- Load the model and parameters and initialize it -- #
        # -- NOTE: self.trainer_model can be a nnUNetTrainerV2 or a Multi Head Network with a model, body and heads. -- #
        print("Loading trainer and setting the network for training")
        if 'fold_' in self.trainer_path:
            # -- Do not add the addition of fold_X to the path -- #
            checkpoint = join(self.trainer_path, "model_final_checkpoint.model")
        else:
            # -- fold_X is not in the path so add it to be in the correct location for the checkpoint loading -- #
            checkpoint = join(self.trainer_path, 'fold_'+str(self.fold), "model_final_checkpoint.model")
        pkl_file = checkpoint + ".pkl"
        use_extension = not 'nnUNetTrainerV2' in self.trainer_path
        self.trainer_model = restore_model(pkl_file, checkpoint, train=False, fp16=self.mixed_precision,\
                                           use_extension=use_extension, extension_type=self.extension,\
                                           param_search=self.param_split, network=self.network_name)
        self.trainer_model.initialize(True)

        # -- Delete the created log_file from the restored model and set it to None since we don't need it (eg. during eval) -- #
        if self.del_log:
            os.remove(self.trainer_model.log_file)
            self.trainer_model.log_file = None

        # -- Update the loss so there will be no error during forward function -- #
        self._update_loss_after_plans_change(self.trainer_model.net_num_pool_op_kernel_sizes, self.trainer_model.patch_size)

        # -- Some sanity checks and loads.. -- #
        # -- Check if the trainer contains plans.pkl file which it should have after sucessfull training -- #
        if 'fold_' in self.trainer_model.output_folder:
            # -- Remove the addition of fold_X from the output_folder, since the plans.pkl is outside of the fold_X directories -- #
            plans_dir = self.trainer_model.output_folder.replace('fold_', '')[:-1]
        else:
            # -- If no fold_ in output_folder, everything is fine -- #
            plans_dir = self.trainer_model.output_folder
        
        assert isfile(join(plans_dir, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file.."

        # -- Check that the trainer type is as expected -- #
        assert isinstance(self.trainer_model, (nnUNetTrainerV2, nnUNetTrainerMultiHead)), "The trainer needs to be nnUNetTrainerV2 or nnUNetTrainerMultiHead.."

        # -- Set mh_network -- #
        # -- Make it to Multi Head Network first and then update the heads if the model was of Multi Head type -- #
        # -- Use the first task in tasks_joined_name, since this represents the corresponding task name, whereas self.task -- #
        # -- is the task to train on, which is not equal to the one that will be initialized now using a pre-trained network -- #
        # -- (prev_trainer). -- #
        if self.trainer_model.__class__.__name__ == nnUNetTrainerV2.__name__:   # Important when doing evaluation, since nnUNetTrainerV2 has no mh_network
            self.mh_network = MultiHead_Module(Generic_UNet, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.trainer_model.network,
                                               input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                               num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes))
        elif self.trainer_model.__class__.__name__ == nnViTUNetTrainer.__name__:   # Important when doing evaluation, since nnViTUNetTrainer has no mh_network
            self.mh_network = MultiHead_Module(Generic_ViT_UNet, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.trainer_model.network,
                                               input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                               num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes),\
                                               patch_size=self.patch_size.tolist(), vit_version=self.version, vit_type=self.vit_type,\
                                               split_gpu=self.split_gpu, ViT_task_specific_ln=self.ViT_task_specific_ln,\
                                               first_task_name=self.tasks_list_with_char[0][0], do_LSA=self.LSA, do_SPT=self.SPT)
        else:
            self.mh_network = self.trainer_model.mh_network

        if not self.trainer_model.__class__.__name__ == nnUNetTrainerV2.__name__\
        and not self.trainer_model.__class__.__name__ == nnViTUNetTrainer.__name__: # Do not use isinstance, since nnUNetTrainerMultiHead is instance of nnUNetTrainerV2 
            # -- Ensure that the split that has been previously used and the current one are equal -- #
            # -- NOTE: Do this after initialization, since the splits might be different before but still lead to the same level after -- #
            # -- Split simplification. -- #
            prev_split = self.already_trained_on[str(self.fold)]['used_split']
            assert self.mh_network.split == prev_split,\
                "To continue training on the fold {} the same split, ie. \'{}\' needs to be provided, not \'{}\'.".format(self.fold, prev_split, self.mh_network.split)
            # -- Delete the prev_split --> not necessary anymore -- #
            del prev_split
            # -- Reset the mh_network using the base models -- #
            self.mh_network = self.trainer_model.mh_network
        
        self.network = None

    def run_training(self, task, output_folder, build_folder=True):
        r"""Perform training using Multi Head Trainer. Simply executes training method of parent class
            while updating trained_on.pkl file. It is important to provide the right path, in which the results
            for the desired task should be stored.
            NOTE: If the task does not exist, a new head will be initialized using the init_head from the initialization
                  of the class only if transfer is false. If transfer is set to true, the last head will be used instead
                  of the one from the initialization. This new head is saved under task and will then be trained.
        """
        # -- Update the self.output_folder, otherwise the data will always be in the same folder for every task -- #
        # -- and everything will be overwritten over and over again -- #
        # -- Do this after reinitialization since the function might change the path -- #
        if build_folder:
            self.output_folder = join(self._build_output_path(output_folder, False), "fold_%s" % str(self.fold))
        else:   # --> The output_folder is already built
            self.output_folder = output_folder

        # -- Make the directory so there will no problems when trying to save some files -- #
        maybe_mkdir_p(self.output_folder)
        # -- Create the dataloaders again, if they are still from the last task --> do this after building -- #
        # -- The output folder, otherwise the log file will be generated in the folder from the previous task -- #
        if self.task != task:
            # -- Recreate the dataloaders for training and validation -- #
            self.reinitialize(task)
            # -- Now reset self.task to the current task -- #
            self.task = task

        # -- Add the current task to the self.already_trained_on dict in case of restoring -- #
        self.update_save_trained_on_json(task, False)   # Add task to start_training

        # -- Register the task if it does not exist in one of the heads -- #
        if task not in self.mh_network.heads:
            self.mh_network.add_new_task(task, use_init=not self.transfer_heads)

        # -- Activate the model based on task --> self.mh_network.active_task is now set to task as well -- #
        #print("################## init network", self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        #self.network = expert_gate_autoencoder(self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        self.initialize_network2()


        #this works
        #self.was_initialized = False
        #self.initialize(num_epochs=self.max_num_epochs)

        self.dl_tr, self.dl_val = self.get_basic_generators()
        self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=np.arange(0, self.data_aug_params.get('num_threads')) if self.deterministic else None,
                    seeds_val=np.arange(self.data_aug_params.get('num_threads'), self.data_aug_params.get('num_threads') + self.data_aug_params.get('num_threads')// 2) if self.deterministic else None
                )


        self.initialize_optimizer_and_scheduler()
        
        #self._update_loss_after_plans_change(self.net_num_pool_op_kernel_sizes, self.patch_size_to_use)
        

        # -- Delete the trainer_model (used for restoring) -- #
        self.trainer_model = None
        
        self.iteration = 0
        # -- Run the training from parent class -- #
        ret = super(nnUNetTrainerV2, self).run_training()

        # -- Reset the val_metrics_exist flag since the training is finished and restoring will fail otherwise -- #
        self.already_trained_on[str(self.fold)]['val_metrics_should_exist'] = False

        # -- Add task to finished_training -- #
        self.update_save_trained_on_json(task, True)
        # -- Resave the final model pkl file so the already trained on is updated there as well -- #
        self.save_init_args(join(self.output_folder, "model_final_checkpoint.model"))

        # -- When model trained on second task and the self.new_trainer is still not updated, then update it -- #
        if self.new_trainer and len(self.already_trained_on) > 1:
            self.new_trainer = False

        # -- Before returning, reset the self.epoch variable, otherwise the following task will only be trained for the last epoch -- #
        self.epoch = 0

        # -- Empty the lists that are tracking losses etc., since this will lead to conflicts in additional tasks durig plotting -- #
        # -- Do not worry about it, the right data is stored during checkpoints and will be restored as well, but after -- #
        # -- a task is finished and before the next one starts, the data needs to be emptied otherwise its added to the lists. -- #
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []
        self.validation_results = dict()

        return ret  # Finished with training for the specific task

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function runs an iteration based on the underlying model. It returns the detached or undetached loss.
            The undetached loss might be important for methods that have to extract gradients without always copying
            the run_iteration function.
            NOTE: The calling class needs to set self.network according to the desired task, this is not done in this
                  function but expected by the user.
        """
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #
        
        data_dict = next(data_generator)
        """
        #print(data_dict.keys())
        #print(data_dict.items())
        #print(np.mean(data_dict['data']))
        print(torch.mean(data_dict['data']).item())
        print(data_dict['properties'][0]['list_of_data_files'])
        exit()
        """
        data = data_dict['data']
        #target = data_dict['target']

        data = maybe_to_torch(data)
        if self.extension in [ "expert_gate_monai_alex_features",
            "expert_gate_simple_ae_alex_features",
            "expert_gate_UNet_alex_features" ]:
            data = data.repeat(1,3,1,1)
            
        if self.extension in ["expert_gate_UNet_alex_features"]:
            data = data[:,:,:244,:244]

        if torch.cuda.is_available():
            data = to_cuda(data)
            
        #print(data.shape)
        #print(self.patch_size)
        #print(self.patch_size_to_use)
        #exit()

        if self.extension in ["expert_gate_monai_alex_features",
            "expert_gate_monai_UNet_features",
            "expert_gate_simple_ae_alex_features",
            "expert_gate_simple_ae_UNet_features",
            "expert_gate_UNet_alex_features"]:
            if np.any(data.shape[-2:] < self.patch_size[-2:]):
                factors = self.patch_size[-2:] / data.shape[-2:]
                factor = math.ceil(max(factors))
                m = Upsample(scale_factor=factor, mode='nearest')
                data = m(data)

            data = self.feature_extractor(data)
            if self.extension == "expert_gate_UNet_alex_features":
                if data.shape[-1] < 8 or data.shape[-2] < 12:    #TODO find better way to do this
                    x_to_go_to = data.shape[-1] // 2
                    x_to_go_to *= 2
                    y_to_go_to = data.shape[-2] // 2
                    y_to_go_to *= 2

                    data = data[:,:,:y_to_go_to, :x_to_go_to]


        target = data


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
            assert target.shape == output.shape
            assert torch.all(target == data)
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
            return l

    def _perform_validation(self, use_tasks=None, use_head=None, call_for_eval=False, param_search=False, use_all_data=False):
        r"""This function performs a full validation on all previous tasks and the current task.
            The Dice and IoU will be calculated and the results will be stored in 'val_metrics.json'.
            use_tasks can be a list of task_ids that should be used for the validation --> can be used when
            performing an external evaluation.
            :param use_tasks: If this function is used in an external call to perform an evaluation after training,
                              than a list of TaskIDs can be provided and they will be used instead of the heads that
                              are present in self.mh_network --> The tasks need to exist and preprocessed to be used
                              however the tasks do not have to be the tasks that were used for training, ie. corresponding
                              to the heads.
            :param use_head: Specify the head that should be used if the task to perform validation/evaluation on does
                             not exist --> should only be used when the function is called for evaluation purposes
            :param call_for_eval: Boolean indicating if this function is called from extern (True), ie. after training is
                                  is finished for evaluation purposes (so to say a misuse of this function). If it is set
                                  to False, the already_trained_on json file will be reset, so specify this if the function
                                  is used in the context of an evaluation. Further if set to False, the head for validation
                                  will be selected based on the task that the model gets validated on.
            :param param_search: Boolean indicating if this function is called during the parameter search method.
            NOTE: Have a look at nnunet_ext/run/run_evaluation.py to see how this function can be 'misused' for evaluation purposes.
        """
        # -- Assert if eval but not eval output dir in path -- #
        if call_for_eval and not param_search:
            assert join(*evaluation_output_dir.split(os.path.sep)[:-1]) in self.output_folder, "You want to perform an evaluation but the output folder does not represent the one for Evaluation results.."
        # -- Ensure that evaluation is performed per subject not per batch as usual -- #
        self.eval_batch = False

        # -- Extract the information of the current fold -- #
        trained_on_folds = self.already_trained_on[str(self.fold)]

        # -- Extract all tasks into a list to loop through -- #
        if use_tasks is None:
            tasks = list(self.mh_network.heads.keys())
        else:
            # -- Use the provided tasks -- #
            tasks = use_tasks[:]
        
        # -- NOTE: Since the head is an (ordered) ModuleDict, the current task is the last head, so there -- #
        # --       is nothing to restore at the end. -- #
        # -- For each previously trained task perform the validation on the full validation set -- #
        running_task_list = list()

        # -- For evaluation purposes, reduce the batch size by half to avoid cuda OOM -- #
        batch_backup = self.batch_size
        self.batch_size //= 2
        # -- Define the number of iterations during evaluation. If batch was even then times 2 otherwise times 3 -- #
        nr_batches = self.num_val_batches_per_epoch*2 if batch_backup % 2 == 0 else self.num_val_batches_per_epoch*3

        # -- Put current network into evaluation mode -- #
        self.network.eval()
        for idx, task in enumerate(tasks):
            # -- Update self.task so we can be sure in training that everything is as expected after the validation -- #
            self.task = task
            # -- Update running task list and create running task which are all (trained tasks and current task joined) for output folder name -- #
            running_task_list.append(task)
            running_task = join_texts_with_char(running_task_list, '_')

            # -- Get default configuration for nnunet/nnunet_ext model (finished training) -- #
            plans_file, _, self.dataset_directory, _, stage, \
            _ = get_default_configuration(self.network_name, task, running_task, trained_on_folds['prev_trainer'][idx],\
                                          self.tasks_joined_name, self.identifier, extension_type=self.extension)

            # -- Load the plans file -- #
            self.plans = load_pickle(plans_file)

            # -- Extract the folder with the preprocessed data in it -- #
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % stage)
                                                
            # -- Create the corresponding dataloaders for train and val (dataset loading and split performed in function) -- #
            self.dl_tr, self.dl_val = self.get_basic_generators(use_all_data)
                
            # -- Unpack the dataset if desired, since we might have to continue training so we have to unpack if desired -- #
            if self.unpack_data:
                unpack_dataset(self.folder_with_preprocessed_data)

            # -- Extract corresponding self.val_gen --> the used function is extern and does not change any values from self -- #
            if call_for_eval:   # --> Don't do any augmentation
                self.tr_gen, self.val_gen = get_no_augmentation(self.dl_tr, self.dl_val,
                                                                params=self.data_aug_params,
                                                                deep_supervision_scales=self.deep_supervision_scales,
                                                                pin_memory=self.pin_memory)
            else:
                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params['patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    pin_memory=self.pin_memory,
                                                                    use_nondetMultiThreadedAugmenter=False,
                                                                    seeds_train=np.arange(0, self.data_aug_params.get('num_threads')) if self.deterministic else None,
                                                                    seeds_val=np.arange(self.data_aug_params.get('num_threads'), self.data_aug_params.get('num_threads') + self.data_aug_params.get('num_threads')// 2) if self.deterministic else None
                    )

            # -- Update the log -- #
            self.print_to_log_file("Performing validation with validation data from task {}.".format(task))

            
            # -- ELSE: nn-UNet is used to perform evaluation, ie. external call, so there are -- #
            # --       no heads except one so omit it --> NOTE: The calling function needs to ensure -- #
            # --       that self.network is assembled correctly ! -- #

            # -- For evaluation, no gradients are necessary so do not use them -- #
            with torch.no_grad():
                # -- Put current network into evaluation mode -- #
                self.network.eval()

                if use_all_data:
                    for gen in [self.tr_gen, self.val_gen]:
                        # -- Run an iteration for each batch in validation generator -- #
                        gen_copy = tee(gen, 1)[0] # <-- Duplicate the generator so the names are extracted correctly during the loop
                        
                        # -- Loop through generator based on number of defined batches -- #
                        for _ in range(nr_batches):   # As we reduced batch_size by half, remember ?
                            # -- First, extract the subject names so we can map the predictions to the names -- #
                            data = next(gen_copy)
                            self.subject_names_raw.append(data['keys'])

                            # -- Run iteration without backprop but online_evaluation to be able to get TP, FP, FN for Dice and IoU -- #
                            if call_for_eval:
                                # -- Call only this run_iteration since only the one from MultiHead has no_loss flag -- #
                                _ = self.run_iteration(gen, False, True, no_loss=True)
                            else:
                                _ = self.run_iteration(gen, False, True)
                        del gen_copy 
                else:
                    # -- Run an iteration for each batch in validation generator -- #
                    val_gen_copy = tee(self.val_gen, 1)[0] # <-- Duplicate the generator so the names are extracted correctly during the loop
                    
                    # -- Loop through generator based on number of defined batches -- #
                    for _ in range(nr_batches):   # As we reduced batch_size by half, remember ?
                        # -- First, extract the subject names so we can map the predictions to the names -- #
                        data = next(val_gen_copy)
                        self.subject_names_raw.append(data['keys'])

                        # -- Run iteration without backprop but online_evaluation to be able to get TP, FP, FN for Dice and IoU -- #
                        if call_for_eval:
                            # -- Call only this run_iteration since only the one from MultiHead has no_loss flag -- #
                            _ = self.run_iteration(self.val_gen, False, True, no_loss=True)
                        else:
                            _ = self.run_iteration(self.val_gen, False, True)
                    del val_gen_copy 

            # -- Calculate Dice and IoU --> self.validation_results is already updated once the evaluation is done -- #
            self.finish_online_evaluation_extended(task)

        # -- After evaluation restore batch size -- #
        self.batch_size = batch_backup
        del batch_backup

        # -- Re-build dataloaders etc. with correct batch_size when this is done during training.. -- #
        if not call_for_eval:
            self.dl_tr, self.dl_val = self.get_basic_generators()
            self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                self.data_aug_params['patch_size_for_spatialtransform'],
                                                                self.data_aug_params,
                                                                deep_supervision_scales=self.deep_supervision_scales,
                                                                pin_memory=self.pin_memory,
                                                                use_nondetMultiThreadedAugmenter=False,
                                                                seeds_train=np.arange(0, self.data_aug_params.get('num_threads')) if self.deterministic else None,
                                                                seeds_val=np.arange(self.data_aug_params.get('num_threads'), self.data_aug_params.get('num_threads') + self.data_aug_params.get('num_threads')// 2) if self.deterministic else None
                                                                )
            # -- Everything else is set as we evaluated on the current task just now -- #


        # -- Put current network into train mode again -- #
        self.network.train()

        # -- Save the dictionary as json file in the corresponding output_folder -- #
        if call_for_eval:
            save_json(self.validation_results, join(self.output_folder, 'val_metrics_eval.json'), sort_keys=False)
        else:
            save_json(self.validation_results, join(self.output_folder, 'val_metrics.json'), sort_keys=False)

        # -- Save as csv if desired as well -- #
        if self.csv:
            # -- Transform the nested dict into a flat table -- #
            val_res = nestedDictToFlatTable(self.validation_results, ['Epoch', 'Task', 'subject_id', 'seg_mask', 'metric', 'value'])
            # -- Dump validation_results as csv file -- #
            if call_for_eval:
                dumpDataFrameToCsv(val_res, self.output_folder, 'val_metrics_eval.csv')
            else:
                dumpDataFrameToCsv(val_res, self.output_folder, 'val_metrics.csv')

        # -- Update already_trained_on if not already done before and only if this is not called during evaluation -- #
        if not call_for_eval and not self.already_trained_on[str(self.fold)]['val_metrics_should_exist']:
            # -- Set to True -- #
            self.already_trained_on[str(self.fold)]['val_metrics_should_exist'] = True
            # -- Save the updated dictionary as a json file -- #
            write_pickle(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.pkl'))
            # -- Update self.init_tasks so the storing works properly -- #
            self.update_init_args()
        
        # -- Ensure that evaluation is performed per batch from here on as usual -- #
        self.eval_batch = True

        # -- If this is executed during evaluation and metadata is stored in evaluation_folder then delete it -- #
        if call_for_eval and join(*evaluation_output_dir.split(os.path.sep)[:-1]) in self.trained_on_path:
            try:
                # -- Try to delete the metadata folder, since this one is empty -- #
                meta_path = join(os.path.sep, *self.trained_on_path.split(os.path.sep)[:self.trained_on_path.split(os.path.sep).index('metadata')+1])
                if os.path.exists(meta_path) and os.path.isdir(meta_path):
                    try:
                        delete_dir_con(meta_path)
                    except ValueError:
                        pass
            except ValueError:
                pass
        
        # -- Delete the already_trained_on file if it still exists (only during eval) -- #
        if call_for_eval and 'metadata' not in self.trained_on_path:    # --> somewhere else a new already_trained_on file is created
            try:
                if os.path.exists(join(self.trained_on_path, self.extension+'_trained_on.pkl')):
                    os.remove(join(self.trained_on_path, self.extension+'_trained_on.pkl'))
            except OSError:
                pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_checkpoint(self, fname, save_optimizer=True):
        r"""Overwrite the parent class, since we want to store the body and heads along with the current activated model
            and not only the current network we train on. If the class uses an old_model, we have to store this as well.
            The old model should be stored in self.network_old, always!
        """

        # -- Set the flag to True -- #
        self.already_trained_on[str(self.fold)]['checkpoint_should_exist'] = True
        # -- Add the current head keys for restoring (is in correct order due to OrderedDict type of heads) -- #
        self.already_trained_on[str(self.fold)]['tasks_at_time_of_checkpoint'] = list(self.mh_network.heads.keys())
        # -- Add the current active task for restoring -- #
        self.already_trained_on[str(self.fold)]['active_task_at_time_of_checkpoint'] = self.mh_network.active_task
        # -- Save the updated dictionary as a json file -- #
        write_pickle(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.pkl'))
        # -- Update self.init_tasks so the storing works properly -- #
        self.update_init_args()

        # -- Use parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
        super(nnUNetTrainerV2, self).save_checkpoint(fname, save_optimizer)

        try:
            if self.network_old is not None:    # --> If it does not exist, an error will be thrown, NOTE: "easier to ask for forgiveness than permission" (EAFP) rather than "look before you leap"
                # -- Set the network to the old network -- #
                self.network = self.network_old
                # -- Use parent class to save checkpoint -- #
                super(nnUNetTrainerV2, self).save_checkpoint(join(self.output_folder, "model_old.model"), save_optimizer)
        except AttributeError:
            # -- Noting to do -- #
            pass

    def load_checkpoint_ram(self, checkpoint, train=True, network_old=False, checkpoint_old=None):
        r"""Overwrite the parent function since the stored state_dict is for a Multi Head Trainer, however the
            load_checkpoint_ram funtion loads the state_dict into self.network which is the assembled model of 
            the  Multi Head Trainer and this would lead to an error because the expected state_dict structure
            and the saved one do not match. Set old network if the class uses the previous network during training.
            The old network is in self.network_old (always).
        """
        # -- For all tasks, create a corresponding head, otherwise the restoring would not work due to mismatching weights -- #
        self.mh_network.add_n_tasks_and_activate(self.already_trained_on[str(self.fold)]['tasks_at_time_of_checkpoint'],
                                                 self.already_trained_on[str(self.fold)]['active_task_at_time_of_checkpoint'])
        #self.network = expert_gate_autoencoder(self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        self.initialize_network2()
        self.initialize_optimizer_and_scheduler()
        # -- Use parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
        super(nnUNetTrainerV2, self).load_checkpoint_ram(checkpoint, train)

    def _update_loss_after_plans_change(self, net_num_pool_op_kernel_sizes, patch_size):
        # -- Reset the internal net_num_pool_op_kernel_sizes and patch_size -- #
        print("Updating the Loss based on the provided previous trainer")
        self.net_num_pool_op_kernel_sizes = net_num_pool_op_kernel_sizes
        self.patch_size = patch_size

        # -- Updating the loss accordingly so that the forward function will not fail due to plans changing and different task -- #
        #------------------------------------------ Partially copied from original implementation ------------------------------------------#
        ################# Here we wrap the loss for deep supervision ############
        # we need to know the number of outputs of the network
        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.ds_loss_weights = weights

        def internal_loss(x,y):
            reconst, mu, logvar = x
            return nn.L1Loss()(reconst, y) + (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))

        #TODO
        #self.loss= torch.nn.L1Loss()
        #self.loss = internal_loss
        self.loss = torch.nn.MSELoss()
        #self.loss = torch.nn.CrossEntropyLoss()
        
        ################# END ###################
        
    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            #mse =-1 * torch.nn.MSELoss()(output,target)
            mse = -1 * self.loss(output, target)
            self.online_eval_mse.append([mse.detach().cpu().numpy()])

    def finish_online_evaluation(self):
        online_eval_mse = np.mean(self.online_eval_mse, 0)
        self.all_val_eval_metrics.append(online_eval_mse)
        self.print_to_log_file("Average global negative mean squared error:", np.round(online_eval_mse, 4))
        self.print_to_log_file("bigger is better.(interpret this as an estimate for the mean squared error. This is not "
                               "exact.)")
        self.online_eval_mse = []

    def finish_online_evaluation_extended(self, task, unique_subject_names=None):
        # -- Stack the numpy arrays since they are stored differently depending on the run -- #
        self.subject_names_raw = np.array(self.subject_names_raw).flatten()
        
        # -- Sum the values for tp, fp and fn per subject to get exactly one value per subject in the end -- #
        # -- Extract the unique names -- #
        if unique_subject_names is None:
            # -- Extract them on your own based on predictions -- #
            subject_names = np.unique(self.subject_names_raw)
        else:
            subject_names = unique_subject_names
        

        # -- Build up a list (following the order of subject_names) that stores the indices for every subject since -- #
        # -- the self.subject_names_raw list matches every other list like tp, fp, fn -- #
        idxs = list()
        for subject in subject_names:
            # -- Get all indices of elements that match the current subject -- #
            idxs.append(np.where(self.subject_names_raw == subject))

        self.online_eval_mse = np.array(self.online_eval_mse)
        self.online_eval_mse.reshape(-1, self.online_eval_mse.shape[-1])

        global_mse_per_class_and_subject = list()
        for idx, mse in enumerate(self.online_eval_mse):
            # -- If possible, calculate the IoU and Dice per class label and per subject -- #
            if not np.isnan(mse).any():
                global_mse_per_class_and_subject.extend([mse])
            else:
                # -- Remove the subject from the list since some value(s) in tp are NaN -- #
                del subject_names[idx]

        
        store_dict = dict()
        for idx, subject in enumerate(subject_names):
            store_dict[subject] = dict()
            for class_label in range(len(global_mse_per_class_and_subject[idx])):
                store_dict[subject]['mask_'+str(class_label+1)] = { 
                                                                    'energy': np.float64(global_mse_per_class_and_subject[idx][class_label])
                                                                  }

        # -- Add the results to self.validation_results based on task, epoch, subject and class-- #
        if self.validation_results.get('epoch_'+str(self.epoch), None) is None:
            self.validation_results['epoch_'+str(self.epoch)] = { task: store_dict }
        else:   # Epoch entry does already exist in self.validation_results, so only add the task with the corresponding values
            self.validation_results['epoch_'+str(self.epoch)][task] = store_dict

        # -- Empty the variables for next iteration -- #
        self.online_eval_mse = []
        self.subject_names_raw = [] # <-- Subject names necessary to map IoU and Dice per subject

    def initialize_network2(self):
        net_numpool = len(self.net_num_pool_op_kernel_sizes)

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


        if self.extension in ["expert_gate_monai"]:
            self.network = ExpertGateMonaiAutoencoder(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(8,16,32,64),
                strides=(1,1,1,1),
                inter_channels=(64,),
                num_res_units=4
            )
        elif self.extension in ["expert_gate_monai_alex_features"]:
            self.network = ExpertGateMonaiAutoencoder(
                spatial_dims=2,
                in_channels=256,
                out_channels=256,
                channels=(128,64,32,64,256),
                strides=(1,1,1,1,1),
                inter_channels=(64,),
                num_res_units=4
            )
        elif self.extension in ["expert_gate_monai_UNet_features"]:
            self.network = ExpertGateMonaiAutoencoder(
                spatial_dims=3,
                in_channels=32,
                out_channels=32,
                channels=(16,8),
                strides=(1,1),
                inter_channels=(8,),
                num_res_units=3
            )        
        elif self.extension in ["expert_gate_simple_ae", 
            "expert_gate_simple_ae_alex_features", 
            "expert_gate_simple_ae_UNet_features"]:
            self.network = expert_gate_autoencoder(self.extension)            # the expert_gate_autoencoder architecture also depends on expert_gate_experiment
        elif self.extension in ["expert_gate_UNet"]:
            self.network = expert_gate_UNet(self.num_input_channels, self.base_num_features, self.num_input_channels,net_numpool,
                                        self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
            self.network.inference_apply_nonlin = lambda x : x
        elif self.extension in ["expert_gate_UNet_alex_features"]:
            self.network = expert_gate_UNet(256, self.base_num_features, 256,1, 
                                        self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
            self.network.inference_apply_nonlin = lambda x : x
        else:
            raise NotImplementedError("specified extension is not correct.")

        if self.extension in ["expert_gate_monai_UNet_features",
            "expert_gate_simple_ae_UNet_features"]:
            
            assert self.feature_extractor_path != None
            checkpoint = join(self.feature_extractor_path, "model_final_checkpoint.model")
            #checkpoint = join(self.output_folder.replace("2d", "3d_fullres").replace("nnUNetTrainerExpertGate2", "nnUNetTrainerSequential"),"model_final_checkpoint.model")
            featureExtractionTrainer: nnUNetTrainerSequential.nnUNetTrainerSequential = restore_model(checkpoint + ".pkl", checkpoint=checkpoint, 
            use_extension=True, 
            extension_type="sequential",
            network="3d_fullres")
            assert isinstance(featureExtractionTrainer, nnUNetTrainerSequential.nnUNetTrainerSequential)
            self.patch_size_to_use = featureExtractionTrainer.patch_size
            self.patch_size = self.patch_size_to_use
            
            print(self.patch_size)
            self.feature_extractor: Generic_UNet = featureExtractionTrainer.mh_network.assemble_model(self.task)
            self.feature_extractor.__class__ = genericUNet_features


        if self.extension in ["expert_gate_monai_alex_features",
            "expert_gate_simple_ae_alex_features",
            "expert_gate_UNet_alex_features"]:
            assert self.num_input_channels == 1, "alexNet features require a single input channel."
            alex_net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
            self.feature_extractor = alex_net.features
            assert(len(self.patch_size) == 2)   #assert we are doing 2d
            patch_size_too_small = self.patch_size < 256
            self.patch_size_to_use = self.patch_size
            self.patch_size_to_use[patch_size_too_small] = 256
            self.patch_size = self.patch_size_to_use
        

        if self.extension in ["expert_gate_monai_alex_features",
        "expert_gate_monai_UNet_features",
        "expert_gate_simple_ae_alex_features",
        "expert_gate_simple_ae_UNet_features",
        "expert_gate_UNet_alex_features"]:
            if torch.cuda.is_available():
                self.feature_extractor.cuda()
            self.feature_extractor.eval()



        self.network.train()
        if torch.cuda.is_available():
            self.network.cuda()

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"


        current_mode = self.network.training
        self.network.eval()
        #use_sliding_window = False
        do_mirroring = False
        ret = self.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)

        self.network.train(current_mode)
        return ret

    @property
    def tr_gen(self):
        return self._tr_gen

    @tr_gen.setter
    def tr_gen(self,value):
        if hasattr(self, 'patch_size'):
            print(self.patch_size)
        self._tr_gen = value

    @property
    def patch_size(self):
        if hasattr(self,'patch_size_to_use'):
            return self.patch_size_to_use
        return self._patch_size
        
    @patch_size.setter
    def patch_size(self,value):
        self._patch_size = value
