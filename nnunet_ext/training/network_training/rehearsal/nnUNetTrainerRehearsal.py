#########################################################################################################
#----------This class represents the nnUNet trainer for rehearsal training. Implementation--------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import random
from collections import OrderedDict
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential


class nnUNetTrainerRehearsal(nnUNetTrainerSequential): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None,
                 identifier=default_plans_identifier, extension='rehearsal', tasks_joined_name=None, samples_per_ds=0.25,
                 seed=3299, trainer_class_name=None):
        r"""Constructor of Rehearsal trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16,
                         save_interval, already_trained_on, identifier, extension, tasks_joined_name, trainer_class_name)

        # -- Set samples based on samples_per_ds -- #
        self.samples = samples_per_ds

        # -- Set the seed -- #
        self.seed = seed

        # -- Add seed in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        self.already_trained_on[str(self.fold)]['used_seed'] = self.seed

        # -- Add the used sample portion in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        self.already_trained_on[str(self.fold)]['used_sample_portion'] = self.samples

        # -- Initialize self.splitted_dataset_val that holds for each task a sample validation set and the corresponding -- #
        # -- dataset_directory. For validation it is important to be able to distinguish between it since the -- #
        # -- corresponding paths need to be set the right way -- #
        #self.splitted_dataset_val = dict()

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
        # -- Without '[:]' for lists or '.copy()' for dicts both variables will change its values which is not desired -- #
        dataset_fused = self.dataset.copy()
        dataset_tr_fused = self.dataset_tr.copy()
        #dataset_val_fused = self.dataset_val.copy()

        # -- Get the data regarding the current fold  -- #
        trained_on_folds = self.already_trained_on[str(self.fold)]

        # -- Check if the model already finished on some tasks before trying to load something -- #
        if len(trained_on_folds['finished_training_on']) != 0:
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
            for idx, task in enumerate(trained_on_folds['finished_training_on']):
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
            self.dataset_tr = dataset_tr_backup
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
    #------------------------------------------ Partially copied by original implementation ------------------------------------------#

    """
    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        rThe Rehearsal Trainer needs its own validation, since the folder with preprocessed data changes through
            validation based from which task the data comes when performing the validation.
            NOTE: Unlike the Sequential Trainer, the data from all previous tasks will be used for validation
                  during training and during final validation -- both based on self.samples --.
        
        # -- Initialize the variable for all results from the validation -- #
        # -- A result is either None or an error --> in case this might be necessary -- #
        ret_joined = list()

        # -- If self.splitted_dataset_val is empty it has been trained on the first task so far --> no rehearsal used so far -- #
        if len(self.splitted_dataset_val) == 0:
            # -- Update the log -- #
            self.print_to_log_file("Performing validation with validation data from last trained task.")

            # -- Perform validation and return the results -- #
            # -- --> call grandparents class validation (nnUNetTrainerV2) since parent class validation (nnUNetTrainerSequential) is different -- #
            ret_joined.append(super(nnUNetTrainerSequential, self).validate(do_mirroring=do_mirroring,
                                                                            use_sliding_window=use_sliding_window,
                                                                            step_size=step_size,
                                                                            save_softmax=save_softmax, use_gaussian=use_gaussian,
                                                                            overwrite=overwrite, validation_folder_name=validation_folder_name,
                                                                            debug=debug, all_in_gpu=all_in_gpu,
                                                                            segmentation_export_kwargs=segmentation_export_kwargs,
                                                                            run_postprocessing_on_folds=run_postprocessing_on_folds))
            return ret_joined

        # -- If it reaches until there, the model has already trained on a previous task, so self.splitted_dataset_val is not empty -- #
        # -- Make a copy of the variables that will be updated in the upcoming loop -- #
        dataset_val_backup = self.dataset_val.copy()
        gt_niftis_folder_backup = self.gt_niftis_folder
        plans_backup = self.plans.copy()

        # -- For each previously trained task perform the validation -- #
        for task, dataset_val_and_folder_and_plans in self.splitted_dataset_val.items():
            # -- Change self.dataset_val to the validation dataset of the task from the loop -- #
            self.dataset_val = dataset_val_and_folder_and_plans[0]

            # -- Update self.dataset_directory for the split in validation -- #
            self.dataset_directory = dataset_val_and_folder_and_plans[1]

            # -- Update self.gt_niftis_folder that will be used in validation function so the files can be found -- #
            self.gt_niftis_folder = join(self.dataset_directory, "gt_segmentations")
            
            # -- Load plans file for validation -- #
            self.plans = load_pickle(dataset_val_and_folder_and_plans[2])

            print(self.dataset_directory, self.gt_niftis_folder)
            
            # -- Update the log -- #
            self.print_to_log_file("Performing validation with validation data from task {}.".format(task))
            
            # -- Call grandparents class validation (nnUNetTrainerV2) since parent class validation (nnUNetTrainerSequential) is different -- #
            ret_joined.append(super(nnUNetTrainerSequential, self).validate(do_mirroring=do_mirroring,
                                                                            use_sliding_window=use_sliding_window,
                                                                            step_size=step_size,
                                                                            save_softmax=save_softmax, use_gaussian=use_gaussian,
                                                                            overwrite=overwrite, validation_folder_name=validation_folder_name+task,
                                                                            debug=debug, all_in_gpu=all_in_gpu,
                                                                            segmentation_export_kwargs=segmentation_export_kwargs,
                                                                            run_postprocessing_on_folds=run_postprocessing_on_folds))
            
        # -- Restore variables to the corresponding validation set of the current task and remove backup variables -- #
        self.dataset_val = dataset_val_backup
        self.gt_niftis_folder = gt_niftis_folder_backup
        self.plans = plans_backup
        del dataset_val_backup, gt_niftis_folder_backup, plans_backup

        # -- Update the log -- #
        self.print_to_log_file("Performing validation with validation data from last trained task.")

        # -- Perform validation on the current task using grandparents class validation (nnUNetTrainerV2) -- #
        # -- since parent class validation (nnUNetTrainerSequential) is different -- #
        ret_joined.append(super(nnUNetTrainerSequential, self).validate(do_mirroring=do_mirroring,
                                                                        use_sliding_window=use_sliding_window,
                                                                        step_size=step_size,
                                                                        save_softmax=save_softmax, use_gaussian=use_gaussian,
                                                                        overwrite=overwrite, validation_folder_name=validation_folder_name,
                                                                        debug=debug, all_in_gpu=all_in_gpu,
                                                                        segmentation_export_kwargs=segmentation_export_kwargs,
                                                                        run_postprocessing_on_folds=run_postprocessing_on_folds))
        
        # -- Return the result which will be an list with None valuea and/or errors -- #
        return ret_joined
        """