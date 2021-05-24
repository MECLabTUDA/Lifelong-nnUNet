#########################################################################################################
#----------This class represents the nnUNet trainer for sequential training. Implementation-------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import os
import numpy as np
from nnunet_ext.paths import default_plans_identifier
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from batchgenerators.utilities.file_and_folder_operations import *


class nnUNetTrainerSequential(nnUNetTrainerV2): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None,
                 identifier=default_plans_identifier, extension='sequential', trainer_class_name=None):
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
                self.already_trained_on[str(self.fold)] = {'finished_training_on': list(), 'start_training_on': None,
                                                            'used_identifier': self.identifier, 'prev_trainer': list()}  # Add current fold as new entry
        else:
            self.already_trained_on = {str(self.fold): {'finished_training_on': list(), 'start_training_on': None,
                                                        'used_identifier': self.identifier, 'prev_trainer': list()}}
        
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

        # -- Set variable that stores the IOU aka Jaccard Index
        self.all_val_iou_eval_metrics = list()

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer=None):
        r"""Overwrite parent initialize function, since we want to include a prev_trainer to enable sequence training and 
            we want to set the num_epochs as well.
        """
        # -- The Trainer embodies the actual model that will be used as foundation to continue training on -- #
        # -- It should be already initialized since the output_folder will be used. If it is None, the model will be initialized and trained. -- #
        # -- Further the trainer needs to be of class nnUNetTrainerV2 or nnUNetTrainerSequential for this method, nothing else. -- #
        self.trainer = prev_trainer
        
        # -- Set nr_epochs to provided number -- #
        self.max_num_epochs = num_epochs

        # -- Initialize the trained_on_tasks and load trained_on_folds -- #
        trained_on_tasks = list()
        trained_on_folds = self.already_trained_on.get(str(self.fold), list())
        
        # -- Reset the trained_on_tasks if the trained_on_folds exist for the current fold -- #
        if isinstance(trained_on_folds, dict):
            trained_on_tasks = trained_on_folds.get('finished_training_on', list())

        # -- The new_trainer indicates if the model is a new sequential model, ie. if it has been trained on only one task so far (True) or on more than one (False) -- #
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
            # -- Set trainer in already_trained_on to the class name -- #
            #self.already_trained_on[str(self.fold)]['prev_trainer'] = self.__class__.__name__

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

        # -- Check that the trainer type is as expected: -- #
        assert isinstance(self.trainer, (nnUNetTrainerV2, nnUNetTrainerSequential)), "The trainer needs to be nnUNetTrainerV2 or nnUNetTrainerSequential"

        # -- If the trainer is already of Sequential type, there should also be a pkl file with the sets it has already been trained on ! -- #
        if isinstance(self.trainer, nnUNetTrainerSequential):   # If model was trained using nnUNetTrainerV2, the pickle file won't exist
            self.already_trained_on = load_json(join(self.trained_on_path, self.extension+'_trained_on.json'))
        
        # -- Set trainer in already_trained_on based on self.trainer (= prev_trainer) --> It was already done after finishing, but still -- #
        self.already_trained_on[str(self.fold)]['prev_trainer'][-1:] = [self.trainer.__class__.__name__]
        #self.already_trained_on[str(self.fold)]['prev_trainer'].append(self.trainer.__class__.__name__)

        # -- Load the model and parameters -- #
        print("Loading trainer and setting the network for training")
        self.trainer.load_final_checkpoint(train=False)    # Load state_dict of the final model

        # -- Set own network to trainer.network to use this pre-trained model -- #
        self.network = self.trainer.network

    def initialize_optimizer_and_scheduler(self):
        r"""This function initializes the optimizer and scheduler. If use same as parent class, delete the function.
            Only keep it if the optimizer and lr_scheduler of foundation model will be used."""
        assert self.network is not None, "self.initialize_network must be called first"
        
        # -- Use initial optimizer and lr_scheduler -- #
        super().initialize_optimizer_and_scheduler()

    def update_trainer(self, prev_trainer, output_folder):
        r"""This function updates the previous trainer in a class by reinitializing the network again and resetting the output_folder.
        """
        # -- Update intern trainer representing a pretrained (sequential or non-sequential) model -- #
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
            trained_on = self.already_trained_on[str(self.fold)]['finished_training_on']
            trained_on.append(task)
            self.already_trained_on[str(self.fold)]['finished_training_on'] = trained_on
            # -- Remove the task from start_training_on -- #
            self.already_trained_on[str(self.fold)]['start_training_on'] = None 
            # -- Update the prev_trainer -- #
            self.already_trained_on[str(self.fold)]['prev_trainer'].append(self.trainer_class_name)
        else:   # Task started to train
            self.already_trained_on[str(self.fold)]['start_training_on'] = task
            # -- Update the prev_trainer -- #
            self.already_trained_on[str(self.fold)]['prev_trainer'][-1:] = [self.trainer.__class__.__name__]
            #self.already_trained_on[str(self.fold)]['prev_trainer'].append(self.trainer.__class__.__name__)
        
        # -- Update the used_identifier -- #
        self.already_trained_on[str(self.fold)]['used_identifier'] = self.identifier

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
            validation['epoch_'+str(idx*self.save_every)+'_IOU'] = str(iou_results[idx])
            validation['epoch_'+str(idx*self.save_every)+'_Dice'] = str(dice_results[idx])

        # -- Save the dictionary as json file in the corresponding output_folder -- #
        save_json(validation, join(self.output_folder, 'val_metrics.json'))
        
        return ret  # Finished with training for the specific task

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

        # -- Store IOU and Dice values -- #
        self.all_val_iou_eval_metrics.append(np.mean(global_iou_per_class))
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

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