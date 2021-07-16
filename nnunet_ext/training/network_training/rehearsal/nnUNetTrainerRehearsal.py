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


class nnUNetTrainerRehearsal(nnUNetTrainerMultiHead): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='rehearsal', tasks_list_with_char=None, samples_per_ds=0.25,
                 seed=3299, mixed_precision=True):
        r"""Constructor of Rehearsal trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision)

        # -- Set samples based on samples_per_ds -- #
        self.samples = samples_per_ds

        # -- Set the seed -- #
        self.seed = seed

        # -- Add seed in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        self.already_trained_on[str(self.fold)]['used_seed'] = self.seed

        # -- Add the used sample portion in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        self.already_trained_on[str(self.fold)]['used_sample_portion'] = self.samples

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, already_trained_on, use_progress, identifier, extension,
                          tasks_list_with_char, samples_per_ds, seed, mixed_precision)

        # -- Initialize self.splitted_dataset_val that holds for each task a sample validation set and the corresponding -- #
        # -- dataset_directory. For validation it is important to be able to distinguish between it since the -- #
        # -- corresponding paths need to be set the right way -- #
        #self.splitted_dataset_val = dict()

    #------------------------------------------ Partially copied from original implementation ------------------------------------------#
    def get_basic_generators(self):
        r"""Calculate the joined dataset for the rehearsal training task.
        """
        
        # -- Set the random seed based on self.seed -- #
        random.seed(self.seed)

        # -- Load the current dataset and perform the splitting -- #
        self.load_dataset()
        self.do_split()

        # -- Initialize the variables that contain the joined datasets for training and validation -- #
        # -- Without '[:]' for lists or '.copy()' for dicts both variables will change its values which is not desired -- #
        dataset_fused = self.dataset.copy()
        dataset_tr_fused = self.dataset_tr.copy()
        #dataset_val_fused = self.dataset_val.copy()

        # -- Get the data regarding the current fold  -- #
        trained_on_folds = self.already_trained_on[str(self.fold)]

        # -- Extract the existing heads -- #
        try:
            tasks_in_head = list(self.mh_network.heads.keys())[:-1]
        except: # Not even trained on one task, ie. self.mh_network does not exist yet
            tasks_in_head = list()

        # -- Check if the model already finished on some tasks before trying to load something -- #
        if len(tasks_in_head) != 0: # Exclude the task we're currently training on --> Note that heads is an ordered ModuleDict
            # -- Create backup for restoring the data of the current task after the previous data has been loaded etc. -- #
            dataset_tr_backup = self.dataset_tr.copy()
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
                #sample_val = random.sample(self.dataset_val.items(), round(len(self.dataset_val) * self.samples))

                # -- Add the extracted validation datasets to self.splitted_dataset_val since its needed for validation later on -- #
                #self.splitted_dataset_val[task] = [OrderedDict(sample_val), self.dataset_directory, plans_file]

                # -- Extend the fused datasets -- #
                dataset_fused.update(self.dataset)
                dataset_tr_fused.update(sample_tr)
                #dataset_val_fused.update(sample_val)

            # -- Restore the data from backup and delete unnecessary variables -- #
            # -- NOTE: Do not restore self.dataset, since it has to include all data that will be used -- #
            self.dataset = dataset_fused
            self.dataset_tr = dataset_tr_fused#dataset_tr_backup
            self.dataset_val = dataset_val_backup
            self.dataset_directory = dataset_directory_backup
            del dataset_tr_backup, dataset_val_backup, dataset_directory_backup

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
        del dataset_fused, dataset_tr_fused#, dataset_val_fused

        # -- Reset the seed for random to default -- #
        random.seed()

        # --- Return the dataloaders -- #
        return dl_tr, dl_val
    #------------------------------------------ Partially copied from original implementation ------------------------------------------#