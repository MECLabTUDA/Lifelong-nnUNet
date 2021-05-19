#########################################################################################################
#----------This class represents the nnUNet trainer for sequential training. Implementation-------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import os
import torch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from batchgenerators.utilities.file_and_folder_operations import *


class nnUNetTrainerSequential(nnUNetTrainerV2): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5):
        r"""Instructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution U-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

        # -- Define self.already_trained_on dictionary to keep track of the trained tasks so far for restoring -- #
        self.already_trained_on = {self.fold: {'finished_training_on': list(), 'start_training_on': None}}
        
        # -- Set the path were the trained_on file will be stored: grand parent directory from output_folder, ie. were all tasks are stored -- #
        self.trained_on_path = os.path.dirname(os.path.dirname(os.path.realpath(output_folder)))

        # -- Store the save_interval that indicates at each epoch interval, the current validation metrices will be stored --- #
        self.save_interval = save_interval

        # -- Store the fold for tracking and saving in the self.already_trained_on file -- #
        self.fold = fold

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

        # -- The new_trainer indicates if the model is a new sequential model, ie. if it has been trained on only one task so far (True) or on more than one (False) -- #
        trained_on_tasks = self.already_trained_on.get(self.fold, list()).get('finished_training_on', list())
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
            # -- If no fold_ in output_folder, everything is finde -- #
            plans_dir = self.trainer.output_folder
        assert isfile(join(plans_dir, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"

        # -- Check that the trainer type is as expected: -- #
        assert isinstance(self.trainer, (nnUNetTrainerV2, nnUNetTrainerSequential)), "The trainer needs to be nnUNetTrainerV2 or nnUNetTrainerSequential"

        # -- If the trainer is already of Sequential type, there should also be a pkl file with the sets it has already been trained on ! -- #
        if isinstance(self.trainer, nnUNetTrainerSequential):   # If model was trained using nnUNetTrainerV2, the pickle file won't exist
            self.already_trained_on = load_json(join(self.already_trained_on, "seq_trained_on.json"))

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
        r"""This function updates the previous trainer in a class by reinitiaizes the network again and resetting the output_folder.
        """
        # -- Update intern trainer representing a pretrained (sequential or non-sequential) model -- #
        self.trainer = prev_trainer

        # -- Update output_folder --#
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
            trained_on = self.already_trained_on[self.fold]['finished_training_on']
            trained_on.append(task)
            self.already_trained_on[self.fold]['finished_training_on'] = trained_on
            # -- Remove the task from start_training_on -- #
            self.already_trained_on[self.fold]['start_training_on'] = None 
        else:   # Task started to train
            self.already_trained_on[self.fold]['start_training_on'] = task

        save_json(self.already_trained_on, join(self.trained_on_path, "seq_trained_on.json"))

    def run_training(self, task):
        r"""Perform training using sequential trainer. Simply executes training method of parent class (nnUNetTrainerV2)
            while updating seq_trained_on.pkl file.
        """
        # -- Add the current task to the self.already_trained_on dict in case of restoring -- #
        self.update_save_trained_on_json(task, False)   # Add task to start_training
        ret = super().run_training()    # Execute training from parent class
        self.update_save_trained_on_json(task, True)   # Add task to finished_training

        # -- When model trained on second task and the self.new_trainer is still not updated, then update it -- #
        if self.new_trainer and len(self.already_trained_on) > 1:
            self.new_trainer = False
        return ret