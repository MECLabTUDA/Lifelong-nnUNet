#########################################################################################################
#----------This class represents the nnUNet trainer for rehearsal training. Implementation--------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import random
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {'samples_in_perc': float}

class nnUNetTrainerRehearsal(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='rehearsal', tasks_list_with_char=None, samples_in_perc=0.25,
                 seed=3299, mixed_precision=True, save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1,
                 split_gpu=False, transfer_heads=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False,
                 network=None, use_param_split=False):
        r"""Constructor of Rehearsal trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)

        # -- Set samples based on samples_in_perc -- #
        self.samples = samples_in_perc
        assert self.samples > 0 and self.samples <= 1, "samples_in_perc should be between 0 and 1: (0, 1]."

        # -- Set the seed -- #
        self.seed = seed

        # -- Add seed in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        if already_trained_on is not None:
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                # -- Add the seed and sample size for prev_tasks -- #
                self.already_trained_on[str(self.fold)]['used_seed'] = self.seed
                self.already_trained_on[str(self.fold)]['used_sample_portion'] = self.samples

            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['used_seed', 'used_sample_portion']
                # -- Check that everything is provided as expected -- #
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            # -- Add the seed and used sample portion in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
            self.already_trained_on[str(self.fold)]['used_seed'] = self.seed
            self.already_trained_on[str(self.fold)]['used_sample_portion'] = self.samples

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                          tasks_list_with_char, samples_in_perc, seed, mixed_precision, save_csv, del_log, use_vit, self.vit_type,
                          version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT)

    def run_training(self, task, output_folder, build_folder=True):
        if self.tasks_list_with_char[0][0] == task:
            assert self.was_initialized
            self.save_checkpoint(join(self.output_folder, "before_training.model"), False)
        return super().run_training(task, output_folder, build_folder)

    #------------------------------------------ Partially copied from original implementation ------------------------------------------#
    def get_basic_generators(self, use_all_data=False):
        r"""Calculate the joined dataset for the rehearsal training task.
        """
        if use_all_data:    # In case of evaluation
            dl_tr, dl_val = super().get_basic_generators(use_all_data)
            return dl_tr, dl_val
        
        # -- Set the random seed based on self.seed -- #
        random.seed(self.seed)

        # -- Load the current dataset and perform the splitting -- #
        self.load_dataset()
        self.do_split()

        # -- Initialize the variables that contain the joined datasets for training and validation -- #
        # -- Without '[:]' for lists or '.copy()' for dicts both variables will change its values which is not desired -- #
        dataset_fused = self.dataset.copy()
        dataset_tr_fused = self.dataset_tr.copy()
        
        # -- Get the data regarding the current fold  -- #
        trained_on_folds = self.already_trained_on[str(self.fold)]

        # -- Extract the existing heads -- #
        try:
            tasks_in_head = list(self.mh_network.heads.keys())
        except: # Not even trained on one task, ie. self.mh_network does not exist yet
            tasks_in_head = list()

        # -- Check if the model already finished on some tasks before trying to load something -- #
        if len(tasks_in_head) != 0: # Exclude the task we're currently training on --> Note that heads is an ordered ModuleDict
            # -- Create backup for restoring the data of the current task after the previous data has been loaded etc. -- #
            dataset_val_backup = self.dataset_val.copy()
            dataset_directory_backup = self.dataset_directory   # Since this is a string, it won't update the reference as it would if it is a list

            # -- Extract init_length of dataset for updating the log -- #
            init_tr_len = len(self.dataset_tr)

            # -- Update log -- #
            self.print_to_log_file("Start building datasets for rehearsal training.")

            # -- Load random samples from the previous tasks -- #
            running_task_list = list()
            for idx, task in enumerate(tasks_in_head):
                # -- Update the log -- #
                self.print_to_log_file("Adding task \'{}\' to fused dataset for rehearsal training.".format(task))

                # -- Update running task list and create running task which are all (trained tasks and current task joined) for output folder name -- #
                running_task_list.append(task)
                running_task = join_texts_with_char(running_task_list, '_')

                # -- Get default configuration for nnunet/nnunet_ext model (finished training) -- #
                plans_file, _, self.dataset_directory, _, stage, \
                _ = get_default_configuration(self.network_name, task, running_task, trained_on_folds['prev_trainer'][idx],\
                                              self.tasks_joined_name, self.identifier, extension_type=self.extension)

                # -- Load the plans file -- #
                plans = load_pickle(plans_file)

                # -- Extract the folder with the preprocessed data in it -- #
                folder_with_preprocessed_data = join(self.dataset_directory, plans['data_identifier'] +
                                                     "_stage%d" % stage)
                                                    
                # -- Load the dataset for the task from the loop and perform the split on it -- #
                self.dataset = load_dataset(folder_with_preprocessed_data)
                self.do_split()

                # -- Extract random sample from train and validation set -- #
                sample_tr = random.sample(self.dataset_tr.items(), round(len(self.dataset_tr) * self.samples))
                
                # -- Extend the fused datasets -- #
                dataset_fused.update(self.dataset)
                dataset_tr_fused.update(sample_tr)

            # -- Restore the data from backup and delete unnecessary variables -- #
            # -- NOTE: Do not restore self.dataset, since it has to include all data that will be used -- #
            self.dataset = dataset_fused
            self.dataset_tr = dataset_tr_fused#dataset_tr_backup
            self.dataset_val = dataset_val_backup
            self.dataset_directory = dataset_directory_backup
            del dataset_val_backup, dataset_directory_backup

            # -- Update the log -- #
            self.print_to_log_file("Succesfully build dataset for rehearsal training, moving on with training."
                " Extended the train dataset from {} samples to {} samples.".format(init_tr_len, len(dataset_tr_fused)))

        # -- Create the dataloaders for training and validation -- #
        if self.threeD:
            dl_tr = DataLoader3D(dataset_tr_fused, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(dataset_tr_fused, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        
        # -- Remove all fused variables -- #
        del dataset_fused, dataset_tr_fused

        # -- Reset the seed for random to default -- #
        random.seed()

        # --- Return the dataloaders -- #
        return dl_tr, dl_val
    #------------------------------------------ Partially copied from original implementation ------------------------------------------#