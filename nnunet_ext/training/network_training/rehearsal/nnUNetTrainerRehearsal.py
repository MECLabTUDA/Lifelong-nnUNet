#########################################################################################################
#----------This class represents the nnUNet trainer for rehearsal training. Implementation--------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import random
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D


class nnUNetTrainerRehearsal(nnUNetTrainerSequential): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None,
                 identifier=default_plans_identifier, extension='rehearsal', samples_per_ds=0.25, seed=3299,
                 tasks_joined_name=None, trainer_class_name=None):
        r"""Constructor of Rehearsal trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16,
                         save_interval, already_trained_on, identifier, extension, trainer_class_name)

        # -- Set samples based on samples_per_ds -- #
        self.samples = samples_per_ds

        # -- Set the seed -- #
        self.seed = seed

        # -- Update seed in trained on file fore restoring to be able to ensure that seed can not be changed during training -- #
        self.already_trained_on[str(self.fold)]['used_seed'] = self.seed

        # -- Set tasks_joined_name for rehearsal dataset building -- #
        self.tasks_joined_name = tasks_joined_name

    #------------------------------------------ Partially copied by original implementation ------------------------------------------#
    def get_basic_generators(self):
        r"""Calculate the joined dataset for the rehearsal training task.
        """
        # -- Set the random seed based on self.seed -- #
        random.seed(self.seed)

        # -- Load the current dataset and perform the splitting -- #
        self.load_dataset()
        self.do_split()

        # -- Initialize the variables that contain the joined datasets for training and validation -- #
        self.dataset_tr_fused = self.dataset_tr
        self.dataset_val_fused = self.dataset_val

        # -- Get the data regarding the current fold  -- #
        trained_on_folds = self.already_trained_on[str(self.fold)]

        # -- Check if the model already finished on some tasks before trying to load something -- #
        if len(trained_on_folds['finished_training_on']) != 0:
            # -- Create backup for restoring the data of the current task after the previous data has been loaded etc. -- #
            dataset_backup = self.dataset
            dataset_tr_backup = self.dataset_tr
            dataset_val_backup = self.dataset_val

            # -- Extract init_length of dataset for updating the log -- #
            init_tr_len = len(self.dataset_tr)

            # -- Update log -- #
            self.print_to_log_file("Start building datasets for rehearsal training.")

            # -- Load random samples from the previous tasks -- #
            running_task_list = list()
            for idx, task in enumerate(trained_on_folds['finished_training_on']):
                # -- Update the log -- #
                self.print_to_log_file("Adding task \'{}\' to fused dataset for rehearsal training.".format(task))

                # -- Update running task list and create running task which are all (trained tasks and current task joined) for output folder name -- #
                running_task_list.append(task)
                running_task = join_texts_with_char(running_task_list, '_')

                # -- Get default configuration for nnunet/nnunet_ext model (finished training) -- #
                plans_file, _, self.dataset_directory, _, stage, \
                _ = get_default_configuration(self.trainer_class_name, task, running_task, trained_on_folds['prev_trainer'][idx],\
                                            self.tasks_joined_name, self.identifier, extension_type=self.extension)

                # -- Load the plans file -- #
                plans = load_pickle(plans_file)

                # -- Extract the folder with the preprocessed data in it -- #
                folder_with_preprocessed_data = join(self.dataset_directory, plans['data_identifier'] +
                                                    "_stage%d" % stage)
                                                    
                # -- Load the dataset for the task from the loop and perform the split on it -- #
                self.dataset = load_dataset(folder_with_preprocessed_data)
                self.do_spilt()

                # -- Extract random sample from train and validation set -- #
                sample_tr = random.sample(self.dataset_tr.items(), len(self.dataset_tr) * self.samples)
                sample_val = random.sample(self.dataset_val.items(), len(self.dataset_val) * self.samples)

                # -- Extend the fused datasets -- #
                self.dataset_tr_fused.update(sample_tr)
                self.dataset_val_fused.update(sample_val)

            # -- Restore the data from backup and delete unnecessary variables -- #
            self.dataset = dataset_backup
            self.dataset_tr = dataset_tr_backup
            self.dataset_val = dataset_val_backup
            del dataset_backup, dataset_tr_backup, dataset_val_backup

            # -- Update the log -- #
            self.print_to_log_file("Succesfully build dataset for rehearsal training, moving on with training."
                "Extended the train dataset from {} samples to {} samples.".format(init_tr_len), len(self.dataset_tr_fused))

        # -- Create the dataloaders for training and validation -- #
        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr_fused, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val_fused, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr_fused, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val_fused, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        
        # -- Reset the seed for random to default -- #
        random.seed()

        # --- Return the dataloaders -- #
        return dl_tr, dl_val
    #------------------------------------------ Partially copied by original implementation ------------------------------------------#
