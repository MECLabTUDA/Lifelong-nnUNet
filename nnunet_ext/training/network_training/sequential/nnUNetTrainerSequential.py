#########################################################################################################
#----------This class represents the nnUNet trainer for sequential training. Implementation-------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import os
import numpy as np
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import load_dataset
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTrainerSequential(nnUNetTrainerV2): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None,
                 identifier=default_plans_identifier, extension='sequential', tasks_joined_name=None, trainer_class_name=None):
        r"""Constructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

        # -- Set identifier to use for building the .json file that is used for restoring states -- #
        self.identifier = identifier

        # -- Initialize or set self.already_trained_on dictionary to keep track of the trained tasks so far for restoring -- #
        if already_trained_on is not None:
            self.already_trained_on = already_trained_on    # Use provided already_trained on
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                self.already_trained_on[str(self.fold)] = {'finished_training_on': list(), 'start_training_on': None, 'finished_validation_on': list(),
                                                           'used_identifier': self.identifier, 'prev_trainer': ['None']}  # Add current fold as new entry
        else:
            self.already_trained_on = {str(self.fold): {'finished_training_on': list(), 'start_training_on': None, 'finished_validation_on': list(),
                                                        'used_identifier': self.identifier, 'prev_trainer': ['None']}}
        
        # -- Set the path were the trained_on file will be stored: grand parent directory from output_folder, ie. were all tasks are stored -- #
        self.trained_on_path = os.path.dirname(os.path.dirname(os.path.realpath(output_folder)))

        # -- Set save_every, so the super trainer class creates checkpoint individually and the validation metrics will be filtered accordingly -- #
        self.save_every = save_interval
        
        # -- Store the fold for tracking and saving in the self.already_trained_on file -- #
        self.fold = fold

        # -- Extract network_name that might come in handy at a later stage -- #
        # -- For more details on how self.output_folder is built look at get_default_configuration -- #
        help_path = os.path.normpath(self.output_folder)    # Normalize path in order to avoid errors
        help_path = help_path.split(os.sep) # Split the path using '\' seperator
        self.network_name = help_path[-5]   # 5th element from back is the name of the used network

        # -- Set trainer_class_name -- #
        self.trainer_class_name = trainer_class_name

        # -- Set the extension for output file -- #
        self.extension = extension

        # -- Set tasks_joined_name for validation dataset building -- #
        self.tasks_joined_name = tasks_joined_name

        # -- Set variable that stores the IOU aka Jaccard Index
        self.all_val_iou_eval_metrics = list()

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer=None):
        r"""Overwrite parent initialize function, since we want to include a prev_trainer to enable sequence training and 
            we want to set the num_epochs as well.
        """
        # -- The Trainer embodies the actual model that will be used as foundation to continue training on -- #
        # -- It should be already initialized since the output_folder will be used. If it is None, the model will be initialized and trained. -- #
        # -- Further the trainer needs to be of class nnUNetTrainerV2 or nnUNetTrainerSequential for this method, nothing else. -- #
        # -- Set prev_trainer correctly as class instance and not a string -- #
        self.trainer = prev_trainer
        
        # -- Set nr_epochs to provided number -- #
        self.max_num_epochs = num_epochs

        # -- Initialize the trained_on_tasks and load trained_on_folds -- #
        trained_on_tasks = list()
        trained_on_folds = self.already_trained_on.get(str(self.fold), list())
        
        # -- Reset the trained_on_tasks if the trained_on_folds exist for the current fold -- #
        if isinstance(trained_on_folds, dict):
            trained_on_tasks = trained_on_folds.get('finished_training_on', list())

        # -- The new_trainer indicates if the model is a new sequential model, -- #
        # -- ie. if it has been trained on only one task so far (True) or on more than one (False) -- #
        if len(trained_on_tasks) > 1:
            self.new_trainer = False
        else:
            self.new_trainer = True

        super().initialize(training, force_load_plans) # --> This updates the corresponding variables automatically since we inherit this class

    def initialize_network(self):
        r"""Extend Initialization of Network --> Load pre-trained model (specified to setup the network).
        Optimizer and lr initialization is still the same, since only the network is different.
        :return:
        """
        if self.trainer is None:
            # -- Initialize from beginning and start training, since no model is provided -- #
            super().initialize_network() # --> This updates the corresponding variables automatically since we inherit this class
            return  # Done with initialization

        # -- Some sanity checks and loads.. -- #
        # -- Check if the trainer contains plans.pkl file which it should have after sucessfull training -- #
        if 'fold_' in self.trainer.output_folder:
            # -- Remove the addition of fold_X from the output_folder, since the plans.pkl is outside of the fold_X directories -- #
            plans_dir = self.trainer.output_folder.replace('fold_', '')[:-1]
        else:
            # -- If no fold_ in output_folder, everything is fine -- #
            plans_dir = self.trainer.output_folder
            
        assert isfile(join(plans_dir, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"

        # -- Check that the trainer type is as expected -- #
        assert isinstance(self.trainer, (nnUNetTrainerV2, nnUNetTrainerSequential)), "The trainer needs to be nnUNetTrainerV2 or nnUNetTrainerSequential"

        # -- If the trainer is already of Sequential type, there should also be a pkl file with the sets it has already been trained on ! -- #
        if isinstance(self.trainer, nnUNetTrainerSequential):   # If model was trained using nnUNetTrainerV2, the pickle file won't exist
            self.already_trained_on = load_json(join(self.trained_on_path, self.extension+'_trained_on.json'))
        
        # -- Load the model and parameters -- #
        print("Loading trainer and setting the network for training")
        self.trainer.load_final_checkpoint(train=True) #TRUE OR FALSE??!!    # Load state_dict of the final model

        # -- Set own network to trainer.network to use this pre-trained model -- #
        self.network = self.trainer.network

    def update_trainer(self, prev_trainer, output_folder):
        r"""This function updates the previous trainer in a class by reinitializing the network again and resetting the output_folder.
        """
        # -- Set prev_trainer correctly as class instance and not a string -- #
        self.trainer = prev_trainer

        # -- Update output_folder -- #
        self.output_folder = output_folder

        # -- Initialize the network again to update the actual trainer represented by self -- #
        self.initialize_network()

    def update_save_trained_on_json(self, task, finished=True):
        r"""This function updates the dictionary, if a model is trained for 3 different tasks, this list needs to be updated
            after each sucessful training of a task and stored accordingly! The 'finished' specifies if the task is finished training
            or just started for training.
            This function also saves the already_trained_on list as a pickle file under the path of the new model task (output_folder).
        """
        # -- Add the provided task at the end of the list, sort the list and dump it as pkl file -- #
        if finished:    # Task finished with training
            self.already_trained_on[str(self.fold)]['finished_training_on'].append(task)
            # -- Remove the task from start_training_on -- #
            self.already_trained_on[str(self.fold)]['start_training_on'] = None 
            # -- Update the prev_trainer -- #
            self.already_trained_on[str(self.fold)]['prev_trainer'].append(self.trainer_class_name)
        else:   # Task started to train
            # -- Add the current task -- #
            self.already_trained_on[str(self.fold)]['start_training_on'] = task
            # -- Update the prev_trainer -- #
            self.already_trained_on[str(self.fold)]['prev_trainer'][-1:] = [self.trainer.__class__.__name__]
        
        # -- Update the used_identifier -- #
        self.already_trained_on[str(self.fold)]['used_identifier'] = self.identifier

        # -- Save the updated dictionary as a json file -- #
        save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))

    def run_training(self, task):
        r"""Perform training using sequential trainer. Simply executes training method of parent class (nnUNetTrainerV2)
            while updating seq_trained_on.pkl file.
        """
        # -- Add the current task to the self.already_trained_on dict in case of restoring -- #
        self.update_save_trained_on_json(task, False)   # Add task to start_training
        ret = super().run_training()                    # Execute training from parent class
        self.update_save_trained_on_json(task, True)    # Add task to finished_training

        # -- When model trained on second task and the self.new_trainer is still not updated, then update it -- #
        if self.new_trainer and len(self.already_trained_on) > 1:
            self.new_trainer = False

        # -- Extract every self.save_everys validation metric (Disce + IOU score) -- #
        iou_results = self.all_val_iou_eval_metrics[::self.save_every]
        dice_results = self.all_val_eval_metrics[::self.save_every]

        # -- Transform this list into a dictionary and load the data into it so it can be saved it -- #
        validation = dict()
        for idx in range(len(dice_results)):
            validation[idx*self.save_every] = {'IOU': iou_results[idx], 'Dice': dice_results[idx] }

        # -- Save the dictionary as json file in the corresponding output_folder -- #
        save_json(validation, join(self.output_folder, 'val_metrics_during_training.json'))
        
        return ret  # Finished with training for the specific task

    #------------------------------------------ Partially copied by original implementation ------------------------------------------#
    def finish_online_evaluation(self):
        r"""Calculate the Dice Score and IOU (Intersection Over Union) on the validation dataset during training.
        """
        # -- Get current True-Positive, False-Positive and False-Negative -- #
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        # -- Claculate the IOU -- #
        global_iou_per_class = [i for i in [i / (i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]

        # -- Calculate the Dice -- #
        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]

        # -- Store IOU and Dice values. Ensure it is float64 so its JSON serializable -- #
        self.all_val_iou_eval_metrics.append(np.mean(global_iou_per_class, dtype="float64"))
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class, dtype="float64"))

        # -- Update the log file -- #
        self.print_to_log_file("Average global foreground IOU:", str(global_iou_per_class))
        self.print_to_log_file("(interpret this as an estimate for the IOU of the different classes. This is not "
                               "exact.)")
        self.print_to_log_file("Average global foreground Dice:", str(global_dc_per_class))
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
    #------------------------------------------ Partially copied by original implementation ------------------------------------------#

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        r"""The Sequential Trainer needs its own validation, since data from the previous tasks needs to be included in the
            validation as well. The validation data from previous tasks will be fully used for the final validation.
            NOTE: During training, only data from the same task will be used for validation. Only for the final validation, the
                  data from previous tasks will be included as well.
        """
        # -- Initialize the variable for all results from the validation -- #
        # -- A result is either None or an error --> in case this might be necessary -- #
        ret_joined = list()

        # -- Extract the information of the current fold -- #
        trained_on_folds = self.already_trained_on[str(self.fold)]

        # -- Extract the list of tasks the model has already finished training on -- #
        trained_on = trained_on_folds.get('finished_training_on', None)

        #  -- If the trained_on_folds raise an error, because at this point the model should have been trained on at least one task -- #
        assert trained_on is not None, "Before performing any validation, the model needs to be trained on at least one task."

        # -- If it reaches until there, the model has already trained on a previous task, so trained_on exists -- #
        # -- Make a copy of the variables that will be updated in the upcoming loop -- #
        # -- Without '[:]' for lists or '.copy()' for dicts both variables will change its values which is not desired -- #
        dataset_backup = self.dataset.copy()
        dataset_tr_backup = self.dataset_tr.copy()
        dataset_val_backup = self.dataset_val.copy()
        gt_niftis_folder_backup = self.gt_niftis_folder
        dataset_directory_backup = self.dataset_directory
        plans_backup = self.plans.copy()

        # -- For each previously trained task perform the validation on the full validation set -- #
        running_task_list = list()
        for idx, task in enumerate(trained_on):
            # -- Update running task list and create running task which are all (trained tasks and current task joined) for output folder name -- #
            running_task_list.append(task)
            running_task = join_texts_with_char(running_task_list, '_')

            # -- Get default configuration for nnunet/nnunet_ext model (finished training) -- #
            plans_file, _, self.dataset_directory, _, stage, \
            _ = get_default_configuration(self.network_name, task, running_task, trained_on_folds['prev_trainer'][idx],\
                                          self.tasks_joined_name, self.identifier, extension_type=self.extension)

            # -- Load the plans file -- #
            self.plans = load_pickle(plans_file)
            
            # -- Update self.gt_niftis_folder that will be used in validation function so the files can be found -- #
            self.gt_niftis_folder = join(self.dataset_directory, "gt_segmentations")

            # -- Extract the folder with the preprocessed data in it -- #
            folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                 "_stage%d" % stage)
                                                
            # -- Load the dataset for the task from the loop and perform the split on it -- #
            self.dataset = load_dataset(folder_with_preprocessed_data)
            self.do_split()

            # -- Update the log -- #
            self.print_to_log_file("Performing validation with validation data from task {}.".format(task))
            
            # -- Perform individual validations with updated self.gt_niftis_folder -- #
            ret_joined.append(super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                                               overwrite=overwrite, validation_folder_name=validation_folder_name+task, debug=debug,
                                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                                               run_postprocessing_on_folds=run_postprocessing_on_folds))

        # -- Restore variables to the corresponding validation set of the current task and remove backup variables -- #
        self.dataset = dataset_backup
        self.dataset_tr = dataset_tr_backup
        self.dataset_val = dataset_val_backup
        self.gt_niftis_folder = gt_niftis_folder_backup
        self.dataset_directory = dataset_directory_backup
        self.plans = plans_backup
        del dataset_backup, dataset_tr_backup, dataset_val_backup, gt_niftis_folder_backup, dataset_directory_backup, plans_backup 

        # -- Add to the already_trained_on that the validation is done for the task the model trained on previously -- #
        self.already_trained_on[str(self.fold)]['finished_validation_on'].append(trained_on[-1])

        # -- Save the updated dictionary as a json file -- #
        save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))
        
        # -- Return the result which will be an list with None valuea and/or errors -- #
        return ret_joined