#########################################################################################################
#------------------This class represents the nnUNet trainer for frozen ViT training.-------------------#
#########################################################################################################

import os
from nnunet_ext.training.model_restore import restore_model
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.network_architecture.MultiHead_Module import MultiHead_Module
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
from nnunet_ext.paths import default_plans_identifier, preprocessing_output_dir, default_plans_identifier

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

class nnUNetTrainerFrozenUNet(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='multihead', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=True, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of frozen ViT trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)
    
    def initialize_network(self):
        r"""Extend Initialization of Network --> Load pre-trained model (specified to setup the network).
            Optimizer and lr initialization is still the same, since only the network is different.
        """
        if self.trainer_path is None:
            # -- Specify if 2d or 3d plans is necessary -- #
            _2d_plans = "_plans_2D.pkl" in self.plans_file
            # -- Load the correct plans file, ie. the one from the first task -- #
            self.process_plans(load_pickle(join(preprocessing_output_dir, self.tasks_list_with_char[0][0], self.identifier + "_plans_2D.pkl" if _2d_plans else self.identifier + "_plans_3D.pkl")))
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
                super().initialize_network() # --> This updates the corresponding variables automatically since we inherit this class
                self.reorder_UNet_components()
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
        
        # -- Set self.network to the model in mh_network --> otherwise the network is not initialized and not in right type -- #
        self.network = self.trainer_model.network    # Does not matter what the model is, will be updated in run_training anyway
   
    def run_training(self, task, output_folder, build_folder=True):
        r"""Perform training using mh trainer. After training on the first task, freeze the ViT module.
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

        # # -- Register the task in the ViT if task specific ViT is used -- #
        # if self.use_vit and self.ViT_task_specific_ln:
        #     if task not in self.network.ViT.norm:
        #         self.network.ViT.register_new_task(task)
        #         # -- Update self.mh_network.model as well since we now have a set of new LNs -- #
        #         self.mh_network.model = copy.deepcopy(self.network)
        #     # -- Set the correct task_name for training -- #
        #     self.network.ViT.use_task(task)
            
        # -- Check if we trained on at least one task -- #
        if len(self.mh_network.heads) > 1:
            print("Freeze body during training of task: ", task)
            # -- Freeze the whole body module, since it is trained on the first task -- #
            self.network = self.mh_network.assemble_model(task, freeze_body=True)
        else:
            # -- Activate the model based on task --> self.mh_network.active_task is now set to task as well -- #
            self.network = self.mh_network.assemble_model(task, freeze_body=False)
        # -- Put model into train mode -- #
        self.network.train()
        
        # -- Delete the trainer_model (used for restoring) -- #
        self.trainer_model = None
        
        # -- Run the training from parent class -- #
        ret = nnUNetTrainerV2.run_training(self)

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
        
    def reorder_UNet_components(self):
        r"""Use this to reorder the networks components.
        """
        # -- Create copies of the different parts and delete them all again -- #
        conv_blocks_localization = self.network.conv_blocks_localization
        conv_blocks_context = self.network.conv_blocks_context
        td = self.network.td
        tu = self.network.tu
        seg_outputs = self.network.seg_outputs
        del self.network.conv_blocks_localization, self.network.conv_blocks_context, self.network.td, self.network.tu, self.network.seg_outputs

        # -- Re-register all modules properly using backups to create a specific order -- #
        # -- NEW Order: Encoder -- Decoder -- Segmentation Head
        self.network.conv_blocks_context = conv_blocks_context  # Encoder part 1
        self.network.td = td  # Encoder part 2
        self.network.tu = tu   # Decoder part 1
        self.network.conv_blocks_localization = conv_blocks_localization   # Decoder part 2
        self.network.seg_outputs = seg_outputs  # Segmentation head