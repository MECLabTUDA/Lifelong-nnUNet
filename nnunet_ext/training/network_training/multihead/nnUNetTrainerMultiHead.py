#########################################################################################################
#----------This class represents the nnUNet Multiple Head Trainer. Implementation-----------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import os
import copy
import torch
import numpy as np
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import load_dataset
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet_ext.nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.network_architecture.MultiHead_Module import MultiHead_Module
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation

class nnUNetTrainerMultiHead(nnUNetTrainerV2): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='multi_head', tasks_list_with_char=None, trainer_class_name=None):
        r"""Constructor of Multi Head Trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

        # -- Set the provided split -- #
        self.split = split

        # -- Set the name of the head which is referred to as a task name -- #
        self.task = task

        # -- Set identifier to use for building the .json file that is used for restoring states -- #
        self.identifier = identifier

        # -- Store the fold for tracking and saving in the self.already_trained_on file -- #
        self.fold = fold

        # -- Initialize or set self.already_trained_on dictionary to keep track of the trained tasks so far for restoring -- #
        if already_trained_on is not None:
            self.already_trained_on = already_trained_on    # Use provided already_trained on
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                self.already_trained_on[str(self.fold)] = {'finished_training_on': list(), 'start_training_on': None, 'finished_validation_on': list(),
                                                           'used_identifier': self.identifier, 'prev_trainer': ['None'], 'val_metrics_should_exist': False,
                                                           'checkpoint_should_exist' : False, 'tasks_at_time_of_checkpoint': list(),
                                                           'active_task_at_time_of_checkpoint': None}  # Add current fold as new entry
            else: # It exists, then check if everything is in it
                pass
        else:
            self.already_trained_on = {str(self.fold): {'finished_training_on': list(), 'start_training_on': None, 'finished_validation_on': list(),
                                                        'used_identifier': self.identifier, 'prev_trainer': ['None'], 'val_metrics_should_exist': False,
                                                        'checkpoint_should_exist' : False, 'tasks_at_time_of_checkpoint': list(),
                                                        'active_task_at_time_of_checkpoint': None}}
        
        # -- Set the path were the trained_on file will be stored: grand parent directory from output_folder, ie. were all tasks are stored -- #
        self.trained_on_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(self.output_folder))))
        
        # -- Set save_every, so the super trainer class creates checkpoint individually and the validation metrics will be filtered accordingly -- #
        self.save_every = save_interval

        # -- Extract network_name that might come in handy at a later stage -- #
        # -- For more details on how self.output_folder is built look at get_default_configuration -- #
        help_path = os.path.normpath(self.output_folder)    # Normalize path in order to avoid errors
        help_path = help_path.split(os.sep) # Split the path using '\' seperator
        self.network_name = help_path[-5]   # 5th element from back is the name of the used network

        # -- Set trainer_class_name -- #
        self.trainer_class_name = self.__class__.__name__ #trainer_class_name

        # -- Set the extension for output file -- #
        self.extension = extension

        # -- Ensure that it is a tuple and that the first element is a list and second element a string -- #
        assert isinstance(tasks_list_with_char, tuple) and isinstance(tasks_list_with_char[0], list) and isinstance(tasks_list_with_char[1], str),\
             "tasks_list_with_char should be a tuple consisting of a list of tasks as the first and a string "+\
             "representing the character that is used to join the tasks as the second element.."
        
        # -- Store the tuple consisting of a list with tasks and the character that should be used to join the tasks -- #
        self.tasks_list_with_char = tasks_list_with_char
 
        # -- Set tasks_joined_name for validation dataset building -- #
        self.tasks_joined_name = join_texts_with_char(self.tasks_list_with_char[0], self.tasks_list_with_char[1])
        
        # -- Define a dictionary for the metrics for validation after every nth epoch -- #
        self.validation_results = dict()

        # -- If -c is used, the self.validation_results need to be restored as well -- #
        # -- Check if the val_metrics should exist -- #
        if self.already_trained_on[str(self.fold)]['val_metrics_should_exist']:
            try:
                # -- Try to load the file -- #
                self.validation_results = load_json(join(self.output_folder, 'val_metrics.json'))
            except: # File does not exist
                assert False, "The val_metrics.json file could not be loaded although it is expected to exist given the current state of the model."

        # -- Set use_prograss_bar if desired so a progress will be shown in the terminal -- #
        self.use_progress_bar = use_progress

        # -- Define the empty Multi Head Network which might be used before intialization, so there is no error thrown (rehearsal) -- #
        self.mh_network = None

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer=None):
        r"""Overwrite parent function, since we want to include a prev_trainer that is used as a base for the Multi Head Trainer.
            Further the num_epochs should be set by the user if desired.
        """
        # -- The Trainer embodies the actual model that will be used as foundation to continue training on -- #
        # -- It should be already initialized since the output_folder will be used. If it is None, the model will be initialized and trained. -- #
        # -- Further the trainer needs to be of class nnUNetTrainerV2 or nnUNetTrainerMultiHead for this method, nothing else. -- #
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

        # -- The new_trainer indicates if the model is a new multi head model, -- #
        # -- ie. if it has been trained on only one task so far (True) or on more than one (False) -- #
        if len(trained_on_tasks) > 1:
            self.new_trainer = False
        else:
            self.new_trainer = True
            
        super().initialize(training, force_load_plans) # --> This updates the corresponding variables automatically since we inherit this class

    def initialize_network(self):
        r"""Extend Initialization of Network --> Load pre-trained model (specified to setup the network).
            Optimizer and lr initialization is still the same, since only the network is different.
        """
        if self.trainer is None:
            # -- Initialize from beginning and start training, since no model is provided -- #
            super().initialize_network() # --> This updates the corresponding variables automatically since we inherit this class
            
            # -- Create a Multi Head Generic_UNet from the current network using the provided split and first task name -- #
            # -- Do not rely on self.task for initialization, since the user might provide the wrong task (unintended), -- #
            # -- however for self.plans, the user needs to extract the correct plans_file path by himself using always the -- #
            # -- first task from a list of tasks since the network is build using the plans_file and thus the structure might vary -- #
            self.mh_network = MultiHead_Module(Generic_UNet, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.network,
                                               input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                               num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes))
            # -- Add the split to the already_trained_on since it is simplified by now -- #
            self.already_trained_on[str(self.fold)]['used_split'] = self.mh_network.split
            # -- Save the updated dictionary as a json file -- #
            save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))
            return  # Done with initialization

        # -- Some sanity checks and loads.. -- #
        # -- Check if the trainer contains plans.pkl file which it should have after sucessfull training -- #
        if 'fold_' in self.trainer.output_folder:
            # -- Remove the addition of fold_X from the output_folder, since the plans.pkl is outside of the fold_X directories -- #
            plans_dir = self.trainer.output_folder.replace('fold_', '')[:-1]
        else:
            # -- If no fold_ in output_folder, everything is fine -- #
            plans_dir = self.trainer.output_folder
            
        assert isfile(join(plans_dir, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file.."

        # -- Check that the trainer type is as expected -- #
        assert isinstance(self.trainer, (nnUNetTrainerV2, nnUNetTrainerMultiHead)), "The trainer needs to be nnUNetTrainerV2 or nnUNetTrainerMultiHead.."

        # -- If the trainer is already of Multi Head type, there should also be a pkl file with the sets it has already been trained on ! -- #
        if isinstance(self.trainer, nnUNetTrainerMultiHead):   # If model was trained using nnUNetTrainerV2, the pickle file won't exist
            self.already_trained_on = load_json(join(self.trained_on_path, self.extension+'_trained_on.json'))
        
        # -- Load the model and parameters -- #
        # -- NOTE: self.trainer is a Multi Head Network, so it has a model, body and heads. -- #
        print("Loading trainer and setting the network for training")
        self.trainer.load_final_checkpoint(train=True) # Load state_dict of the final model

        # -- Set mh_network -- #
        # -- Make it to Multi Head network if it is not already -- #
        # -- Use the first task in tasks_joined_name, since this represents the corresponding task name, whereas self.task -- #
        # -- is the task to train on, which is not equal to the one that will be initialized now using a pre-trained network -- #
        # -- (prev_trainer). -- #
        if isinstance(self.trainer, nnUNetTrainerV2):
            self.mh_network = MultiHead_Module(Generic_UNet, self.split, self.tasks_list_with_char[0][0], prev_trainer=self.trainer.network,
                                               input_channels=self.num_input_channels, base_num_features=self.base_num_features,\
                                               num_classes=self.num_classes, num_pool=len(self.net_num_pool_op_kernel_sizes))
        else: # Already Multi Head type
            self.mh_network = self.trainer#.mh_network
            # -- Ensure that the split that has been previously used and the current one are equal -- #
            # -- NOTE: Do this after initialization, since the splits might be different before but still lead to the same level after -- #
            # --       simplification. -- #
            prev_split = self.already_trained_on[str(self.fold)]['used_split']
            assert self.mh_network.split == prev_split,\
                "To continue training on the fold {} the same split, ie. \'{}\' needs to be provided, not \'{}\'.".format(self.fold, self.mh_network.split, prev_split)
            # -- Delete the prev_split --> not necessary anymore -- #
            del prev_split
        
        # -- Set self.network to the model in mh_network --> otherwise the network is not initialized and not in right type -- #
        self.network = self.mh_network.model
    
    def run_training(self, task, output_folder):
        r"""Perform training using Multi Head Trainer. Simply executes training method of parent class
            while updating trained_on.pkl file. It is important to provide the right path, in which the results
            for the desired task should be stored.
            NOTE: If the task does not exist, a new head will be initialized using the init_head from the initialization
                  of the class. This new head is saved under task and will then be trained.
        """
        # -- Update the self.output_folder, otherwise the data will always be in the same folder for every task -- #
        # -- and everything will be overwritten over and over again -- #
        self.output_folder = join(output_folder, "fold_%s" % str(self.fold))

        # -- Make the directory so there will no problems when trying to save some files -- #
        maybe_mkdir_p(self.output_folder)

        # -- Add the current task to the self.already_trained_on dict in case of restoring -- #
        self.update_save_trained_on_json(task, False)   # Add task to start_training

        # -- Set self.trainer to None after this, since it will never be used afterwards. -- #
        # -- If a pre trained network is used to iitialize an extension network, this will only -- #
        # -- effect the first task, more or less at this point, the everything has been done with -- #
        # -- the trainer. So it was only needed here for the already_trained_on to set the correct -- #
        # -- previous_trainer in case of a restoring. From now on, the prev_trainer is always this -- #
        # -- current network from the extension and self.trainer has no use when it is trained using -- #
        # -- one of the extensions. Thus just set it to None, and the already_trained_on sets the prev-trainer -- #
        # -- correct and as expected as well. -- #
        self.trainer = None

        # -- Register the task if it does not exist in one of the heads -- #
        if task not in self.mh_network.heads.keys():
            # -- Add this task into heads -- #
            self.mh_network.add_new_task(task)

        # -- Activate the model based on task --> self.mh_network.active_task is now set to task as well -- #
        self.mh_network.assemble_model(task)
        
        # -- Run the training from parent class -- #
        ret = super().run_training()

        # -- Add task to finished_training -- #
        self.update_save_trained_on_json(task, True)

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

        return ret  # Finished with training for the specific task

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        r"""This function runs an iteration based on the underlying model.
            NOTE: The calling class needs to set self.network according to the desired task, this is not done in this
                  function but expected by the user.
        """
        # -- Set the network to the assembled model that is then used for training -- #
        # -- We can only set the network here to MH network, otherwise the parent class will use its own forward function. -- #
        # -- If we do not train on MH Network, the forward function will not be implemented resulting in an error -- #
        self.network = self.mh_network

        # -- Run iteration as usual -- #
        loss = super().run_iteration(data_generator, do_backprop, run_online_evaluation)

        # -- Reset the network so the model in Multi Head network is updated as well -- #
        self.mh_network = self.network
        self.network = self.mh_network.model
        
        # -- Return the loss -- #
        return loss

    def on_epoch_end(self):
        """Overwrite this function, since we want to perform a validation after every nth epoch on all tasks form
           from the head.
           NOTE: If the validation is done during run_iteration(), the validation will be performed for every batch
                 at every nth epoch which is not what we want. This will further lead into an error because too many
                 files will be then opened.
        """
        # -- Perform everything the parent class makes -- #
        res = super().on_epoch_end()

        # -- If the current epoch can be divided without a rest by self.save_every than its time for a validation -- #
        if self.epoch % self.save_every == (self.save_every - 1):   # Same as checkpoint saving from nnU-Net
            self._perform_validation()

        # -- Return the result from the parent class -- #
        return res

    def _perform_validation(self):
        r"""This function performs a full validation on all previous tasks and the current task.
            The Dice and IoU will be calculated and the results will be stored in 'val_metrics.json'.
        """
        # -- Extract the information of the current fold -- #
        trained_on_folds = self.already_trained_on[str(self.fold)]

        # -- Extract all tasks into a list to loop through -- #
        tasks = list(self.mh_network.heads.keys())

        # -- Add the current trainer_class name to prev_trainer, so the loop does not end in an error -- #
        # -- since this trainer is not yet a prev_trainer.. --> Remove the trainer again after the loop -- #
        # -- because this creates only a view and changes self.already_trained_on as well which we do not want to -- #
        trained_on_folds['prev_trainer'].append(self.trainer_class_name)
        
        # -- NOTE: Since the head is an (ordered) ModuleDict, the current task is the last head, so there -- #
        # --       is nothing to restore at the end. -- #
        # -- NOTE: Since the current task the model is training on is always added at the end of the list, -- #
        # --       After this loop everything is automatically set as before, so no restoring needs to be done -- #
        # -- For each previously trained task perform the validation on the full validation set -- #
        running_task_list = list()
        for idx, task in enumerate(tasks):
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
                                                
            # -- Create the corresponding dataloaders for train ind val (dataset loading and split performed in function) -- #
            # -- Since we do validation, there is no need to unpack the data -- #
            self.dl_tr, self.dl_val = self.get_basic_generators()

            # -- Load the dataset for the task from the loop and perform the split on it -- #
            #self.dataset = load_dataset(folder_with_preprocessed_data)
            #self.do_split()

            # -- Extract corresponding self.val_gen --> the used function is extern and does not change any values from self -- #
            self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,    # Changed due to do_split ;)
                                                                self.data_aug_params[
                                                                    'patch_size_for_spatialtransform'],
                                                                self.data_aug_params,
                                                                deep_supervision_scales=self.deep_supervision_scales,
                                                                pin_memory=self.pin_memory,
                                                                use_nondetMultiThreadedAugmenter=False)
            # -- Update the log -- #
            self.print_to_log_file("Performing validation with validation data from task {}.".format(task))

            # -- Activate the current task to train on the right model -- #
            # -- Set self.network, since the parent classes all use self.network to train -- #
            # -- NOTE: self.mh_network.model is also updated to task split ! -- #
            self.network = self.mh_network.assemble_model(task)
            
            # -- For evaluation, no gradients are necessary so do not use them -- #
            with torch.no_grad():
                # -- Put current network into evaluation mode -- #
                self.network.eval()
                # -- Run an iteration for each batch in validation generator -- #
                for _ in range(self.num_val_batches_per_epoch):
                    # -- Run iteration without backprop but online_evaluation to be able to get TP, FP, FN for Dice and IoU -- #
                    _ = self.run_iteration(self.val_gen, False, True)
            
            # -- Calculate Dice and IoU --> self.validation_results is already updated once the evaluation is done -- #
            self.finish_online_evaluation_extended(task)

        # -- Remove the trainer now from the list again  -- #
        trained_on_folds['prev_trainer'] = trained_on_folds['prev_trainer'][:-1]

        # -- Save the dictionary as json file in the corresponding output_folder -- #
        save_json(self.validation_results, join(self.output_folder, 'val_metrics.json'))

        # -- Update already_trained_on if not already done before -- #
        if not self.already_trained_on[str(self.fold)]['val_metrics_should_exist']:
            # -- Set to True -- #
            self.already_trained_on[str(self.fold)]['val_metrics_should_exist'] = True
            # -- Save the updated dictionary as a json file -- #
            save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))

    #------------------------------------------ Partially copied from original implementation ------------------------------------------#
    def finish_online_evaluation_extended(self, task):
        r"""Calculate the Dice Score and IoU (Intersection over Union) on the validation dataset during training.
            The metrics are calculated for every label in the masks, except for background.
            NOTE: The function name is different from the original one, since it is used in another context
                  than the original one, ie. it is only called in special cases why it needs to have a different name.
        """
        # -- Get current True-Positive, False-Positive and False-Negative -- #
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        # -- Calculate the IoU -- #
        global_iou_per_class = [i for i in [i / (i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]

        # -- Calculate the Dice -- #
        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]

        # -- Store IoU and Dice values. Ensure it is float64 so its JSON serializable -- #
        # -- Do not use self.all_val_eval_metrics since this is used for plotting and then the -- #
        # -- plots do not build correctly because based on self.save_every more dice values than -- #
        # -- expected (epochs) are in there --> see plot_progress function in network_trainer.py -- #
        iou = np.mean(global_iou_per_class, dtype="float64")
        dice = np.mean(global_dc_per_class, dtype="float64")

        # -- Update the log file -- #
        self.print_to_log_file("Average global foreground IoU for task {}: {}".format(task, str(global_iou_per_class)))
        self.print_to_log_file("(interpret this as an estimate for the IoU of the different classes. This is not "
                               "exact.)")
        self.print_to_log_file("Average global foreground Dice for task {}: {}".format(task, str(global_dc_per_class)))
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")

        # -- Add the results to self.validation_results based on task and epoch -- #
        if self.validation_results.get('epoch_'+str(self.epoch), None) is None:
            self.validation_results['epoch_'+str(self.epoch)] = { task: {
                                                                            'IoU': iou,
                                                                            'Dice': dice
                                                                        }
                                                                }
        else:   # Epoch entry does already exist in self.validation_results, so only add the task with the corresponding values
            self.validation_results['epoch_'+str(self.epoch)][task] =  { 'IoU': iou,
                                                                         'Dice': dice
                                                                       }
                                                       
        # -- Empty the variables for next iteration -- #
        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
    #------------------------------------------ Partially copied from original implementation ------------------------------------------#

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        r"""The Multi Head Trainer needs its own validation, since data from the previous tasks needs to be included in the
            validation as well. The validation data from previous tasks will be fully used for the final validation.
            NOTE: This function is different from _perform_validation since it uses the parent validate function that perfomrs
                  validations and saves the results (predicted segmentations) in corresponding folders.
        """
        # -- Initialize the variable for all results from the validation -- #
        # -- A result is either None or an error --> in case this might be necessary -- #
        ret_joined = list()

        # -- Extract the information of the current fold -- #
        trained_on_folds = self.already_trained_on[str(self.fold)]

        # -- Extract the list of tasks the model has already finished training on -- #
        #trained_on = trained_on_folds.get('finished_training_on', None)
        trained_on = list(self.mh_network.heads.keys())

        #  -- If the trained_on_folds raise an error, because at this point the model should have been trained on at least one task -- #
        #assert trained_on is not None, "Before performing any validation, the model needs to be trained on at least one task."
        assert len(trained_on) != 0, "Before performing any validation, the model needs to be trained on at least one task."

        # -- If it reaches until there, the model has already trained on a previous task, so trained_on exists -- #
        # -- Make a copy of the variables that will be updated in the upcoming loop -- #
        # -- Without '[:]' for lists or '.copy()' for dicts both variables will change its values which is not desired -- #
        #dataset_backup = self.dataset.copy()
        #dataset_tr_backup = self.dataset_tr.copy()
        #dataset_val_backup = self.dataset_val.copy()
        #gt_niftis_folder_backup = self.gt_niftis_folder
        #dataset_directory_backup = self.dataset_directory
        #plans_backup = self.plans.copy()

        # -- NOTE: Since the head is an (ordered) ModuleDict, the current task is the last head, so there -- #
        # --       is nothing to restore at the end. -- #
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
        
            # -- Activate the current task to train on the right model -- #
            # -- Set self.network, since the parent classes all use self.network to train -- #
            # -- NOTE: self.mh_network.model is also updated to task split ! -- #
            self.network = self.mh_network.assemble_model(task)

            # -- Before executing validate function,
            self.network.eval()

            #p = self.network.__dict__
            #print(p)
            #raise
            
            # -- Perform individual validations with updated self.gt_niftis_folder -- #
            ret_joined.append(super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                                               overwrite=overwrite, validation_folder_name=validation_folder_name+task, debug=debug,
                                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                                               run_postprocessing_on_folds=run_postprocessing_on_folds))

        # -- Restore variables to the corresponding validation set of the current task and remove backup variables -- #
        #self.dataset = dataset_backup
        #self.dataset_tr = dataset_tr_backup
        #self.dataset_val = dataset_val_backup
        #self.gt_niftis_folder = gt_niftis_folder_backup
        #self.dataset_directory = dataset_directory_backup
        #self.plans = plans_backup
        #del dataset_backup, dataset_tr_backup, dataset_val_backup, gt_niftis_folder_backup, dataset_directory_backup, plans_backup 

        # -- Restore the network the class is trained on using self.task -- #
        # -- Activate the current task to train on the right model -- #
        # -- Set self.network, since the parent classes all use self.network to train -- #
        # -- NOTE: self.mh_network.model is also updated to self.task split ! -- #
        #self.network = self.mh_network.assemble_model(self.task)

        # -- Add to the already_trained_on that the validation is done for the task the model trained on previously -- #
        self.already_trained_on[str(self.fold)]['finished_validation_on'].append(trained_on[-1])

        # -- Remove the additional prev_trainer currently existing in self.already_trained_on -- #
        self.already_trained_on[str(self.fold)]['prev_trainer'] = self.already_trained_on[str(self.fold)]['prev_trainer'][:-1]
        
        # -- Save the updated dictionary as a json file -- #
        save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))

        # -- At the end reset the log_file, so a new file is created for the next task given the updated output folder -- #
        self.log_file = None

        # -- Return the result which will be a list with None valuea and/or errors -- #
        return ret_joined
    
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
            if self.trainer is not None: # This is always the case when a pre trained network is used as initialization
                self.already_trained_on[str(self.fold)]['prev_trainer'][-1:] = [self.trainer.__class__.__name__]
            else: # When using directly the extension with no pre trained network or after first task train when self.trainer is set to None
                self.already_trained_on[str(self.fold)]['prev_trainer'][-1:] = [self.trainer_class_name]
        # -- Update the used_identifier -- #
        self.already_trained_on[str(self.fold)]['used_identifier'] = self.identifier

        # -- Save the updated dictionary as a json file -- #
        save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))

    def save_checkpoint(self, fname, save_optimizer=True):
        r"""Overwrite the parent class, since we want to store the body and heads along with the current activated model
            and not only the current network we train on.
        """
        # -- Set the network to the full MultiHead_Module network to save everything in the class not only the current model -- #
        self.network = self.mh_network

        # -- Use parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
        super().save_checkpoint(fname, save_optimizer)

        # -- Set the flag in already_trained_on -- #
        if not self.already_trained_on[str(self.fold)]['checkpoint_should_exist']:
            # -- Set the flag to True -- #
            self.already_trained_on[str(self.fold)]['checkpoint_should_exist'] = True
            # -- Add the current head keys for restoring (should be in correct order due to OrderedDict type of heads) -- #
            self.already_trained_on[str(self.fold)]['tasks_at_time_of_checkpoint'] = list(self.mh_network.heads.keys())
            # -- Add the current active task for restoring -- #
            self.already_trained_on[str(self.fold)]['active_task_at_time_of_checkpoint'] = self.mh_network.active_task
            # -- Save the updated dictionary as a json file -- #
            save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))

        # -- Reset network to the assembled model to continue training -- #
        self.network = self.mh_network.model

    def load_checkpoint_ram(self, checkpoint, train=True):
        r"""Overwrite the parent function since the stored state_dict is for a Multi Head Trainer, however the
            load_checkpoint_ram funtion loads the state_dict into self.network which is the assembled model of 
            the  Multi Head Trainer and this would lead to an error because the expected state_dict structure
            and the saved one do not match.
        """
        # -- For all tasks, create a corresponding head, otherwise the restoring would not work due to mismatching weights -- #
        self.mh_network.add_n_tasks_and_activate(self.already_trained_on[str(self.fold)]['tasks_at_time_of_checkpoint'],
                                                 self.already_trained_on[str(self.fold)]['active_task_at_time_of_checkpoint'])
        
        # -- Set the network to the full MultiHead_Module network to restore everything -- #
        self.network = self.mh_network
        
        # -- Use parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
        super().load_checkpoint_ram(checkpoint, train)

        # -- Reset network to the assembled model to continue training -- #
        self.network = self.mh_network.model