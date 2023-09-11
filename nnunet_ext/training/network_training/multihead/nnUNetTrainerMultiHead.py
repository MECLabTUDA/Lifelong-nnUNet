#########################################################################################################
#----------This class represents the nnUNet Multiple Head Trainer. Implementation-----------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import numpy as np
import os, copy, torch
from pandas import to_datetime
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
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet_ext.paths import default_plans_identifier, evaluation_output_dir, preprocessing_output_dir, default_plans_identifier

# -- Add this since default option file_descriptor has a limitation on the number of open files. -- #
# -- Default config might cause the runtime error: RuntimeError: received 0 items of ancdata -- #
torch.multiprocessing.set_sharing_strategy('file_system')

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

class nnUNetTrainerMultiHead(nnUNetTrainerV2): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='multihead', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=False,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of Multi Head Trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
            The transfer_heads flag is used when adding a new head, if True, the state_dict from the last head will be used
            instead of the one from the initialization. This is the basic transfer learning (difference between MH and SEQ folder structure).
            NOTE: If the task does not exist, a new head will be initialized using the last head, not the one from the
                  initialization of the class. This new head is saved under task and will then be trained.
            All vit related arguments like vit_type, version and split_gpu are only used if use_vit is True, in such a case,
            the Generic_ViT_UNet is used instead of the Generic_UNet.
        """
        # -- Create a backup of the original output folder that is provided -- #
        self.output_folder_orig = output_folder

        # -- Set ViT task specific flags -- #
        self.ViT_task_specific_ln = ViT_task_specific_ln

        # -- Check and set everything if the user wants to use the Generic_ViT_UNet -- #
        self.use_vit = use_vit
        # -- Create the variable indicating which ViT Architecture to use, base, large or huge -- #
        self.vit_type = vit_type.lower()
        # -- Set the desired network version -- #
        self.version = 'V' + str(version)
        # -- Set the flag for initializing the heads -- #
        self.transfer_heads = transfer_heads
        # -- LSA and SPT flags -- #
        self.LSA, self.SPT = do_LSA, do_SPT
        # -- Update the output_folder path accordingly -- #
        output_folder = self._build_output_path(output_folder, False)
        
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

        # -- Set the provided split -- #
        self.split = split

        # -- Set the name of the head which is referred to as a task name -- #
        self.task = task

        # -- Set identifier to use for building the .pkl file that is used for restoring states -- #
        self.identifier = identifier

        # -- Store the fold for tracking and saving in the self.already_trained_on file -- #
        self.fold = fold

        # -- Store the flag if it is desired to save the validation metrics at every nth epoch as a csv as well -- #
        self.csv = save_csv

        # -- Set trainer_class_name -- #
        self.trainer_class_name = self.__class__.__name__

        # -- Set if the log should be removed or not -- #
        self.del_log = del_log

        # -- Define if the model should be split onto multiple GPUs, only considered and done if ViT architecture is used -- #
        self.split_gpu = split_gpu
        if self.use_vit and self.split_gpu:
            assert torch.cuda.device_count() > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'

        # -- Initialize or set self.already_trained_on dictionary to keep track of the trained tasks so far for restoring -- #
        if already_trained_on is not None:
            self.already_trained_on = already_trained_on    # Use provided already_trained on
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                self.already_trained_on[str(self.fold)] = {'finished_training_on': list(), 'start_training_on': None, 'finished_validation_on': list(),
                                                           'used_identifier': self.identifier, 'prev_trainer': [self.trainer_class_name], 'val_metrics_should_exist': False,
                                                           'checkpoint_should_exist': False, 'tasks_at_time_of_checkpoint': list(),
                                                           'active_task_at_time_of_checkpoint': None}  # Add current fold as new entry
            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['finished_training_on', 'start_training_on', 'finished_validation_on', 'used_identifier', 'prev_trainer',\
                        'val_metrics_should_exist', 'checkpoint_should_exist','tasks_at_time_of_checkpoint',\
                        'active_task_at_time_of_checkpoint']
                # -- Check that everything is provided as expected -- #
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            self.already_trained_on = {str(self.fold): {'finished_training_on': list(), 'start_training_on': None, 'finished_validation_on': list(),
                                                        'used_identifier': self.identifier, 'prev_trainer': [self.trainer_class_name], 'val_metrics_should_exist': False,
                                                        'checkpoint_should_exist' : False, 'tasks_at_time_of_checkpoint': list(),
                                                        'active_task_at_time_of_checkpoint': None}}

        # -- Set the path were the trained_on file will be stored -- #
        self.trained_on_path = os.path.dirname(os.path.dirname(os.path.realpath(self.output_folder_orig)))
        self.trained_on_path = self._build_output_path(self.trained_on_path, True)
        
        # -- Create the folder if necessary -- #
        maybe_mkdir_p(self.trained_on_path)

        # -- Set save_every, so the super trainer class creates checkpoint individually and the validation metrics will be filtered accordingly -- #
        self.save_every = save_interval

        # -- Initialize subject_names list that is used to store the subject names for every nth evaluation -- #
        self.subject_names_raw = list() # Store the names as is, ie. not cleaned (removed duplicates etc.) --> For evaluation necessary

        # -- Extract network_name that might come in handy at a later stage -- #
        # -- For more details on how self.output_folder is built look at get_default_configuration -- #
        self.network_name = network
        assert self.network_name is not None, "Please provide the network setting that is used.."
        
        # -- Set the extension for output file -- #
        self.extension = extension

        # -- Set if the model should be compressed as floating point 16 -- #
        self.mixed_precision = mixed_precision

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

        # -- Define an empty trainer_model -- #
        self.trainer_model = None

        # -- Define flag for evaluation (per batch or per subject) -- #
        self.eval_batch = True

        # -- Set the flag if the param_split should be used instead of the general split -- #
        # -- Only set this to True if the parameter search method is used -- #
        self.param_split = use_param_split

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                          tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, self.vit_type, version, split_gpu,
                          transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT)

    def do_split(self):
        r"""Modify the original function. This enables the loading of the split
            for the parameter search method if the flag is set.
            When using the parameter search method, the split will be modified in such a way, that the
            training set of the original split will be split by 80:20. The original val_set will not be used
            during training or validation, ie. never when doing parameter search. However the split contains
            those under the test_set flag.
        """
        # -- Copied from original implementation and modified -- #
        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            # -- Set the splits file path -- #
            p_splits_file = join(self.dataset_directory, "splits_param_search.pkl")
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            if not isfile(p_splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split for parameter searching based on already existing split file {}...".format(splits_file))
                # -- Define new split list -- #
                new_splits = list()
                # -- Go through all defined folds and resplit the train set into 80:20 data split -- #
                for split in splits:
                    # -- Extract the names of the files in the train split -- #
                    train = split['train']
                    # -- Split the train randomly in train and val and add it to the new_splits -- #
                    train_, val_ = train_test_split(train, random_state=3299, test_size=0.2)
                    new_splits.append(OrderedDict())
                    new_splits[-1]['train'] = train_
                    new_splits[-1]['val'] = val_
                    new_splits[-1]['test'] = split['val']   # --> Only for storing, should be never used
                # -- Save the new splits file -- #
                save_pickle(new_splits, p_splits_file)

            if self.param_split:
                splits = load_pickle(p_splits_file)

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                    % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20/64:16:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]

                # -- Split the train randomly in train and val again if self.param_split is desired -- #
                if self.param_split:
                    tr_keys, val_keys = train_test_split(tr_keys, random_state=3299, test_size=0.2)
                # else:
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                    % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
        # -- Copied from original implementation and modified -- #

    def process_plans(self, plans):
        r"""Modify the original function. This just reduces the batch_size by half and manages the correct initialization of the network.
        """
        # -- Initialize using parent class -- #
        super().process_plans(plans)

        # -- Reduce the batch_size by half after it has been set by super class --> only if ViT is used -- #
        # -- Do this so it fits onto GPU --> if it still does not, model needs to be put onto multiple GPUs -- #
        if self.use_vit:
            self.batch_size = self.batch_size // 2

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite parent function, since we want to include a prev_trainer that is used as a base for the Multi Head Trainer.
            Further the num_epochs should be set by the user ifdesired.
        """
        # -- The Trainer embodies the actual model that will be used as foundation to continue training on -- #
        # -- It should be already initialized since the output_folder will be used. If it is None, the model will be initialized and trained. -- #
        # -- Further, the trainer needs to be of class nnUNetTrainerV2 or nnUNetTrainerMultiHead for this method, nothing else. -- #
        # -- Set prev_trainer_path correctly as a string and not any class instance -- #
        if prev_trainer_path is not None and not call_for_eval:
            self.trainer_path = join(self._build_output_path(prev_trainer_path, False), "fold_%s" % str(self.fold))
        else:   # If for eval, then this is a nnUNetTrainerV2 whereas the path is not build as implemented in _build_output_path
            self.trainer_path = prev_trainer_path
        
        # -- Initialize using super class -- #
        super().initialize(training, force_load_plans) # --> This updates the corresponding variables automatically since we inherit this class

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

            # -- Initialize from beginning and start training, since no model is provided -- #
            super().initialize_network() # --> This updates the corresponding variables automatically since we inherit this class
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
        else:
            self.mh_network = self.trainer_model.mh_network

        if not self.trainer_model.__class__.__name__ == nnUNetTrainerV2.__name__: # Do not use isinstance, since nnUNetTrainerMultiHead is instance of nnUNetTrainerV2 
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
    
    def reinitialize(self, task, print_loss_info=True):
        r"""This function is used to reinitialize the Multi Head Trainer when a new task is trained.
            Basically the dataloaders are created again with the new task data. This function will only
            be used when training before running the actual training.
        """
        # -- Empty the GPU cache -- #
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- Add the prev_trainer to the list for the current run -- #
        while len(self.already_trained_on[str(self.fold)]['prev_trainer']) < len(self.mh_network.heads)+1:
            self.already_trained_on[str(self.fold)]['prev_trainer'].append(self.trainer_class_name)

        # -- Create a new log file --> NOTE: Update output_folder before calling this function! -- #
        self.log_file = None
        # -- Update the log file -- #
        self.print_to_log_file("Updating the Dataloaders for new task \'{}\'.".format(task))

        # -- Extract the running task list based on the currently trained on tasks + the current task -- #
        running_task_list = self.already_trained_on[str(self.fold)]["finished_training_on"][:]
        running_task_list.append(task)
        running_task = join_texts_with_char(running_task_list, '_')
        
        # -- Get default configuration for nnunet/nnunet_ext model -- #
        plans_file, _, self.dataset_directory, _, stage, \
        _ = get_default_configuration(self.network_name, task, running_task, self.trainer_class_name,\
                                      self.tasks_joined_name, self.identifier, extension_type=self.extension,\
                                      print_loss_info=print_loss_info)

        # -- Load the plans file -- #
        self.plans = load_pickle(plans_file)

        # -- Extract the folder with the preprocessed data in it -- #
        self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                  "_stage%d" % stage)
                                            
        # -- Create the corresponding dataloaders for train and val (dataset loading and split performed in function) -- #
        # -- Since we do validation, there is no need to unpack the data -- #
        del self.dl_tr, self.dl_val # --> Avoid memory leak
        self.dl_tr, self.dl_val = self.get_basic_generators()
        
        # -- Unpack the dataset if this is desired -- #
        if self.unpack_data:
            unpack_dataset(self.folder_with_preprocessed_data)

        # -- Extract corresponding self.val_gen --> the used function is extern and does not change any values from self -- #
        del self.tr_gen, self.val_gen # --> Avoid memory leak
        self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                            #self.data_aug_params['patch_size_for_spatialtransform'],
                                                            self.patch_size,
                                                            self.data_aug_params,
                                                            deep_supervision_scales=self.deep_supervision_scales,
                                                            pin_memory=self.pin_memory,
                                                            use_nondetMultiThreadedAugmenter=False)

        #--------------------------------- Copied from original implementation ---------------------------------#
        self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                also_print_to_console=False)
        self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                also_print_to_console=False)
        #--------------------------------- Copied from original implementation ---------------------------------#

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

        # -- Register the task in the ViT if task specific ViT is used -- #
        if self.use_vit and self.ViT_task_specific_ln:
            if task not in self.network.ViT.norm:
                self.network.ViT.register_new_task(task)
                # -- Update self.mh_network.model as well since we now have a set of new LNs -- #
                self.mh_network.model = copy.deepcopy(self.network)
            # -- Set the correct task_name for training -- #
            self.network.ViT.use_task(task)

        # -- Activate the model based on task --> self.mh_network.active_task is now set to task as well -- #
        self.network = self.mh_network.assemble_model(task)
        
        # -- Delete the trainer_model (used for restoring) -- #
        self.trainer_model = None
        
        # -- Run the training from parent class -- #
        ret = super().run_training()

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
        data = data_dict['data']
        target = data_dict['target']

        # -- Track the subjects if desired for validation, the list gets emptied after validation is done -- #
        self.subject_names_raw.append(data_dict['keys'])

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

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

    def on_epoch_end(self):
        """Overwrite this function, since we want to perform a validation after every nth epoch on all tasks
           from the head.
           NOTE: If the validation is done during run_iteration(), the validation will be performed for every batch
                 at every nth epoch which is not what we want. This will further lead into an error because too many
                 files will be then opened, thus we do it here.
        """
        # -- Perform everything the parent class makes -- #
        res = super().on_epoch_end()

        # -- If the current epoch can be divided without a rest by self.save_every than its time for a validation -- #
        if self.epoch % self.save_every == self.save_every - 1:   # Same as checkpoint saving from nnU-Net (NOTE: this is because its 0 based)
            self._perform_validation()

        # -- Return the result from the parent class -- #
        return res

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
            :param use_all_data: Boolean indicating if the train and validation data should be used for validation.
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
                                                                    use_nondetMultiThreadedAugmenter=False)

            # -- Update the log -- #
            if use_all_data:
                self.print_to_log_file("Performing validation with validation and training data from task {}.".format(task))
            else:
                self.print_to_log_file("Performing validation with validation data from task {}.".format(task))

            # -- Activate the current task to train on the right model -- #
            # -- Set self.network, since the parent classes all use self.network to train -- #
            # -- NOTE: self.mh_network.model is also updated to task split ! -- #
            if task in self.mh_network.heads:
                self.network = self.mh_network.assemble_model(task)
                # -- Set the correct task_name for training -- #
                if self.use_vit and self.ViT_task_specific_ln:
                    self.network.ViT.use_task(task)
            else:
                assert use_head is not None, "The task to perform validation/evaluation on is not in the head and no head_name that should be used instead (use_head) is provided."
                self.network = self.mh_network.assemble_model(use_head)
                # -- Set the correct task_name for training -- #
                if self.use_vit and self.ViT_task_specific_ln:
                    self.network.ViT.use_task(use_head)
            # -- ELSE: nn-UNet is used to perform evaluation, ie. external call, so there are -- #
            # --       no heads except one so omit it --> NOTE: The calling function needs to ensure -- #
            # --       that self.network is assembled correctly ! -- #

            # -- Add validation subject names to list -- #
            unique_subject_names = self.dl_val.list_of_keys
            if use_all_data:    # --> Use train and val generator
                unique_subject_names.extend(self.dl_tr.list_of_keys)
                with torch.no_grad():
                     # -- Put current network into evaluation mode -- #
                    self.network.eval()
                    for gen in (self.tr_gen, self.val_gen):
                        # -- Loop through generator based on number of defined batches -- #
                        for _ in range(50):
                            # -- Run iteration without backprop but online_evaluation to be able to get TP, FP, FN for Dice and IoU -- #
                            if call_for_eval:
                                # -- Call only this run_iteration since only the one from MultiHead has no_loss flag -- #
                                _ = self.run_iteration(gen, False, True, no_loss=True)
                            else:
                                _ = self.run_iteration(gen, False, True)
            else:   # Eval only on val generator
                # -- For evaluation, no gradients are necessary so do not use them -- #
                with torch.no_grad():
                    # -- Put current network into evaluation mode -- #
                    self.network.eval()
                    # -- Loop through generator based on number of defined batches -- #
                    for _ in range(50):
                        self.num_batches_per_epoch
                        # -- Run iteration without backprop but online_evaluation to be able to get TP, FP, FN for Dice and IoU -- #
                        if call_for_eval:
                            # -- Call only this run_iteration since only the one from MultiHead has no_loss flag -- #
                            _ = self.run_iteration(self.val_gen, False, True, no_loss=True)
                        else:
                            _ = self.run_iteration(self.val_gen, False, True)
                    
            # -- Calculate Dice and IoU --> self.validation_results is already updated once the evaluation is done -- #
            self.finish_online_evaluation_extended(task, unique_subject_names)

        # -- Put current network into train mode again -- #
        self.network.train()

        # -- Save the dictionary as json file in the corresponding output_folder -- #
        if call_for_eval:
            save_json(self.validation_results, join(self.output_folder, 'tr_val_metrics_eval.json' if use_all_data else 'val_metrics_eval.json'), sort_keys=False)
        else:
            save_json(self.validation_results, join(self.output_folder, 'tr_val_metrics.json' if use_all_data else 'val_metrics.json'), sort_keys=False)

        # -- Save as csv if desired as well -- #
        if self.csv:
            # -- Transform the nested dict into a flat table -- #
            val_res = nestedDictToFlatTable(self.validation_results, ['Epoch', 'Task', 'subject_id', 'seg_mask', 'metric', 'value'])
            # -- Dump validation_results as csv file -- #
            if call_for_eval:
                dumpDataFrameToCsv(val_res, self.output_folder, 'tr_val_metrics_eval.csv' if use_all_data else 'val_metrics_eval.csv')
            else:
                dumpDataFrameToCsv(val_res, self.output_folder, 'tr_val_metrics.csv' if use_all_data else 'val_metrics.csv')

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

    #------------------------------------------ Partially copied from original implementation ------------------------------------------#
    def get_basic_generators(self, use_all_data=False):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size if not use_all_data else self.patch_size,
                                 self.patch_size, self.batch_size, False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size if not use_all_data else self.patch_size,
                                 self.patch_size, self.batch_size, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def run_online_evaluation(self, output, target):
        r"""Overwrite the existing one, since for evaluation at every nth epoch we want the tp, fp, fn
            per subject and not per batch over every subject.
            NOTE: Due to deep supervision the return value and the reference are now lists of tensors. We only need the full
                  resolution output because this is what we are interested in in the end.
        """
        # -- If calculation per batch is desired, change nothing -- #
        if self.eval_batch:
            super().run_online_evaluation(output, target)
        else:   # --> Do evaluation per subject not per batch
            # -- Look at the Note in the function description -- #
            output = output[0]
            target = target[0]
            #------------------------------------------ Copied from original implementation ------------------------------------------#
            with torch.no_grad():
                # -- Calculate tp, fp and fn as usual, for every element in the batch -- #
                num_classes = output.shape[1]
                output_softmax = softmax_helper(output)
                output_seg = output_softmax.argmax(1)
                target = target[:, 0]
                axes = tuple(range(1, len(target.shape)))
                tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                for c in range(1, num_classes):
                    tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                    fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                    fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)
            #------------------------------------------ Copied from original implementation ------------------------------------------#

                # -- Add the calculate tp, fp and fn to the lists --> If we sum them, then we get one value per batch -- #
                # -- Now we have one value per subject as in self.subject_names_raw -- #
                # -- NOTE: self.online_eval_XX are lists and conventionally the values are stored in lists. -- #
                # --       Here we store the values as numpys and transform self.online_eval_XX to a numpy later on -- #
                # --       for the calculation per subject, thus we need numpy since this makes it more sufficient. -- #
                self.online_eval_tp.append(tp_hard.detach().cpu().numpy())
                self.online_eval_fp.append(fp_hard.detach().cpu().numpy())
                self.online_eval_fn.append(fn_hard.detach().cpu().numpy())
    
    def finish_online_evaluation_extended(self, task, unique_subject_names=None):
        r"""Calculate the Dice Score and IoU (Intersection over Union) on the validation dataset during training.
            The metrics are calculated for every subject and for every label in the masks, except for background.
            NOTE: The function name is different from the original one, since it is used in another context
                  than the original one, ie. it is only called in special cases which is why it has a different name.
        """
        # -- Stack the numpy arrays since they are stored differently depending on the run -- #
        self.subject_names_raw = np.array(self.subject_names_raw).flatten()

        # -- Reshape the tp, fp, tn lists so the names are flat, but the different mask labels, ie. last dimension is still in tact -- #
        self.online_eval_tp = np.array(self.online_eval_tp).reshape(-1, np.array(self.online_eval_tp).shape[-1])
        self.online_eval_fp = np.array(self.online_eval_fp).reshape(-1, np.array(self.online_eval_fp).shape[-1])
        self.online_eval_fn = np.array(self.online_eval_fn).reshape(-1, np.array(self.online_eval_fn).shape[-1])
        
        # -- Sum the values for tp, fp and fn per subject to get exactly one value per subject in the end -- #
        # -- Extract the unique names -- #
        if unique_subject_names is None:
            # -- Extract them on your own based on predictions -- #
            subject_names = np.unique(self.subject_names_raw)
        else:
            subject_names = unique_subject_names
        
        # -- Sum the values per subject name based on the idxs -- #
        tp = list()
        fp = list()
        fn = list()

        # -- Build up a list (following the order of subject_names) that stores the indices for every subject since -- #
        # -- the self.subject_names_raw list matches every other list like tp, fp, fn -- #
        idxs = list()
        for subject in subject_names:
            # -- Get all indices of elements that match the current subject -- #
            idxs.append(np.where(self.subject_names_raw == subject))

        for subject_idxs in idxs:
            # -- Sum only those values that belong to the subject based on the subject_idxs -- #
            # -- NOTE: self.online_eval_XX dimensions: (nr_batches, subject_names (--> subject_idxs), nr_classes) -- #
            # --       The selection of rows returns all batch results per subject so we only sum them on axis 0 -- #
            # --       so we keep the results per nr_classes in tact on don't sum the array in a whole to a single value -- #
            tp.append(np.array(self.online_eval_tp)[subject_idxs].sum(axis=0))
            fp.append(np.array(self.online_eval_fp)[subject_idxs].sum(axis=0))
            fn.append(np.array(self.online_eval_fn)[subject_idxs].sum(axis=0))

        # -- Assign the correct values to corresponding lists and remove the three generated lists
        self.online_eval_tp, self.online_eval_fp, self.online_eval_fn = tp, fp, fn
        del tp, fp, fn

        # -- Calculate the IoU per class per subject --> use numpy since those operations do not work on conventional lists -- #
        global_iou_per_class_and_subject = list()
        global_dc_per_class_and_subject = list()
        for idx, (i, j, k) in enumerate(zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)):
            # -- If possible, calculate the IoU and Dice per class label and per subject -- #
            if not np.isnan(i).any():
                # -- IoU -- #
                global_iou_per_class_and_subject.extend([i / (i + j + k)])
                # -- Dice -- #
                global_dc_per_class_and_subject.extend([2 * i / (2 * i + j + k)])
            else:
                # -- Remove the subject from the list since some value(s) in tp are NaN -- #
                del subject_names[idx]

        # -- Store IoU and Dice values. Ensure it is float64 so its JSON serializable -- #
        # -- Do not use self.all_val_eval_metrics since this is used for plotting and then the -- #
        # -- plots do not build correctly because based on self.save_every more dice values than -- #
        # -- expected (epochs) are in there --> see plot_progress function in network_trainer.py -- #
        # -- Build IoU and Dice dict for storing per subject and class label -- #
        store_dict = dict()
        for idx, subject in enumerate(subject_names):
            store_dict[subject] = dict()
            for class_label in range(len(global_iou_per_class_and_subject[idx])):
                store_dict[subject]['mask_'+str(class_label+1)] = { 
                                                                    'IoU': np.float64(global_iou_per_class_and_subject[idx][class_label]),
                                                                    'Dice': np.float64(global_dc_per_class_and_subject[idx][class_label])
                                                                  }

        # -- Add the results to self.validation_results based on task, epoch, subject and class-- #
        if self.validation_results.get('epoch_'+str(self.epoch), None) is None:
            self.validation_results['epoch_'+str(self.epoch)] = { task: store_dict }
        else:   # Epoch entry does already exist in self.validation_results, so only add the task with the corresponding values
            self.validation_results['epoch_'+str(self.epoch)][task] = store_dict

        # -- Empty the variables for next iteration -- #
        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
        self.subject_names_raw = [] # <-- Subject names necessary to map IoU and dice per subject
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
        trained_on = list(self.mh_network.heads.keys())

        #  -- If the trained_on_folds raise an error, because at this point the model should have been trained on at least one task -- #
        assert len(trained_on) != 0, "Before performing any validation, the model needs to be trained on at least one task."

        # -- NOTE: Since the head is an (ordered) ModuleDict, the current task is the last head, so there -- #
        # --       is nothing to restore at the end. -- #
        # -- For each previously trained task perform the validation on the full validation set -- #
        running_task_list = list()
        # -- Before executing validate function, set network in eval mode -- #
        self.network.eval()
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
            # -- Set the correct task_name for training -- #
            if self.use_vit and self.ViT_task_specific_ln:
                self.network.ViT.use_task(task)
            # -- Perform individual validations with updated self.gt_niftis_folder -- #
            ret_joined.append(super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                                               overwrite=overwrite, validation_folder_name=validation_folder_name+task, debug=debug,
                                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                                               run_postprocessing_on_folds=run_postprocessing_on_folds))
        
        # -- Add to the already_trained_on that the validation is done for the task the model trained on previously -- #
        self.already_trained_on[str(self.fold)]['finished_validation_on'].append(trained_on[-1])

        # -- Save the updated dictionary as a json file -- #
        write_pickle(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.pkl'))
        # -- Update self.init_tasks so the storing works properly -- #
        self.update_init_args()
        # -- Resave the final model pkl file so the already trained on is updated there as well -- #
        self.save_init_args(join(self.output_folder, "model_final_checkpoint.model"))

        # -- At the end reset the log_file, so a new file is created for the next task given the updated output folder -- #
        self.log_file = None

        # -- Return the result which will be a list with None valuea and/or errors -- #
        return ret_joined
    
    def update_save_trained_on_json(self, task, finished=True):
        r"""This function updates the dictionary, if a model is trained for n different tasks, this list needs to be updated
            after each sucessful training of a task and stored accordingly! The 'finished' specifies if the task is finished training
            or just started for training.
            This function also saves the already_trained_on list as a pickle file under the path of the new model task (output_folder).
        """
        # -- Add the provided task at the end of the list, sort the list and dump it as pkl file -- #
        if finished:    # Task finished with training
            if task not in self.already_trained_on[str(self.fold)]['finished_training_on']:
                self.already_trained_on[str(self.fold)]['finished_training_on'].append(task)
            # -- Remove the task from start_training_on -- #
            self.already_trained_on[str(self.fold)]['start_training_on'] = None 
        else:   # Task started to train
            # -- Add the current task -- #
            self.already_trained_on[str(self.fold)]['start_training_on'] = task
            # -- Update the prev_trainer -- #
            if self.trainer_model is not None and len(self.already_trained_on[str(self.fold)]['prev_trainer']) == 1: # This is always the case when a pre-trained network is used as initialization
                self.already_trained_on[str(self.fold)]['prev_trainer'][-1:] = [self.trainer_model.__class__.__name__]  # --> The one from the used trainer
                self.already_trained_on[str(self.fold)]['prev_trainer'].append(self.trainer_class_name)                 # --> The current trainer we start training with
        # -- Update the used_identifier -- #
        self.already_trained_on[str(self.fold)]['used_identifier'] = self.identifier

        # -- Save the updated dictionary as a pkl file -- #
        write_pickle(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.pkl'))
        # -- Update self.init_tasks so the storing works properly -- #
        self.update_init_args()

    def save_checkpoint(self, fname, save_optimizer=True):
        r"""Overwrite the parent class, since we want to store the body and heads along with the current activated model
            and not only the current network we train on. If the class uses an old_model, we have to store this as well.
            The old model should be stored in self.network_old, always!
        """
        # -- Set the network to the full MultiHead_Module network to save everything in the class not only the current model -- #
        self.network = self.mh_network

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
        super().save_checkpoint(fname, save_optimizer)

        try:
            if self.network_old is not None:    # --> If it does not exist, an error will be thrown, NOTE: "easier to ask for forgiveness than permission" (EAFP) rather than "look before you leap"
                # -- Set the network to the old network -- #
                self.network = self.network_old
                # -- Use parent class to save checkpoint -- #
                super().save_checkpoint(join(self.output_folder, "model_old.model"), save_optimizer)
        except AttributeError:
            # -- Noting to do -- #
            pass

        # -- Reset network to the assembled model to continue training -- #
        self.network = self.mh_network.model

    def update_init_args(self):
        r"""This function is used to update the init_args variable that is saved during checkpoint storing.
            During this update, only the already_trained_on will be updated.
        """
        # -- Transform tuple to list -- #
        init = list(self.init_args)
        # -- Update already_trained_on (has position 12 --> if that changes, than change this here as well) -- #
        init[12] = self.already_trained_on
        # -- Transform list back to tuple -- #
        self.init_args = tuple(init)

    def save_init_args(self, fname):
        r"""This function needs to be executed after a finished training, if some arguments are changed once the final mode
            is stored. Those results need to be updated as well or there might be a problem when trying to restore later on.
        """
        #------------------------------------------ Copied from original implementation ------------------------------------------#
        # -- Save the results in case the script gets interrupted etc. for proper restoring -- #
        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans
        # -- Dump the file -- #
        write_pickle(info, fname + ".pkl")
        #------------------------------------------ Copied from original implementation ------------------------------------------#

    def load_best_checkpoint(self, train=True, network_old=False):
        r"""Update the parent class function to also loading the old network if there is one used by any inheriting class.
        """
        # -- If there is an old network, then load this as well -- #
        if network_old:
            # -- Change the checkpoint loading -- #
            if isfile(join(self.output_folder, "model_old.model")):
                if self.fold is None:
                    raise RuntimeError("Cannot load best checkpoint if self.fold is None")
                if isfile(join(self.output_folder, "model_best.model")):
                    self.load_checkpoint(join(self.output_folder, "model_best.model"), train, True, join(self.output_folder, "model_old.model"))
                else:
                    self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                           "back to load_latest_checkpoint")
                    self.load_latest_checkpoint(train)
        else:
            # -- use super class to load the checkpoint -- #
            super().load_best_checkpoint(train)

    def load_latest_checkpoint(self, train=True):
        r"""Update the parent class function to also loading the old network if there is one used by any inheriting class.
        """
        try:
            if self.network_old is not None:    # --> If it does not exist, an error will be thrown, NOTE: "easier to ask for forgiveness than permission" (EAFP) rather than "look before you leap"
                # -- Change the checkpoint loading -- #
                if isfile(join(self.output_folder, "model_old.model")):
                    if isfile(join(self.output_folder, "model_final_checkpoint.model")):
                        return self.load_checkpoint(join(self.output_folder, "model_final_checkpoint.model"), train, True, join(self.output_folder, "model_old.model"))
                    if isfile(join(self.output_folder, "model_latest.model")):
                        return self.load_checkpoint(join(self.output_folder, "model_latest.model"), train, True, join(self.output_folder, "model_old.model"))
                    if isfile(join(self.output_folder, "model_best.model")):
                        return self.load_best_checkpoint(train)
                raise RuntimeError("No checkpoint of the current or old model found")
        except AttributeError:
            # -- Use super function to do the work since there is no old model in this class -- #
            super().load_latest_checkpoint(train)

    def load_checkpoint(self, fname, train=True, network_old=False, fname_old=None):
        r"""Update the parent class function to also loading the old network if there is one used by any inheriting class.
        """
        # -- Copied from original code -- #
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        # -- Copied from original code -- #

        # -- Load the model with the old model if desired -- #
        saved_model_old = None
        if network_old:
            saved_model_old = torch.load(fname_old, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train, network_old, saved_model_old)

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

        # -- Set the network to the full MultiHead_Module network to restore everything -- #
        self.network = self.mh_network

        # -- Use parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
        super().load_checkpoint_ram(checkpoint, train)

        self.mh_network = copy.deepcopy(self.network)
        del self.network

        # -- Do the same for the old network if there is one -- #
        if network_old:
            # -- Ensure that checkpoint old is given as well -- #
            assert checkpoint_old is not None, "When loading a previous, ie. old network, please provide the checkpoint as well.."
            # -- Do not use the last head, but the second to last head, since this is actually the previous model -- #
            # -- whereas the last head is the current model we continue trianing on! --> relevant in case heads -- #
            # -- are different from each other. -- #
            self.network = self.mh_network.assemble_model(list(self.mh_network.heads.keys())[-2])
            # -- Set mode of old network always train = false since it should not be used to train but during training -- #
            super().load_checkpoint_ram(checkpoint_old, False)
            # -- Set the network_old model correctly or it does exist otherwise -- #
            self.network_old = copy.deepcopy(self.network)
            del self.network

        # -- Reset the running model to train on -- #
        self.network = self.mh_network.model

    def _build_output_path(self, output_folder, meta_data=False):
        r"""This function is used to build the output folder path during training when a new task is started.
            If the path is not adjusted given this method, the data files are scattered all over the place at
            different folders where they don't belong.
        """
        # -- First of all remove the fold_ from the path -- #
        if 'fold_' in output_folder.split(os.path.sep)[-1]:
            output_folder = os.path.join(*output_folder.split(os.path.sep)[:-1])
            

        # -- Generic_UNet will be used so update the path accordingly -- #
        if Generic_UNet.__name__ != output_folder.split(os.path.sep)[-1] and Generic_UNet.__name__ not in output_folder:
            if not meta_data:
                output_folder = os.path.join(output_folder, Generic_UNet.__name__)
            else:   # --> Path were meta data is stored, i.e. already_trained_on file
                output_folder = os.path.join(output_folder, 'metadata', Generic_UNet.__name__)

        # -- In every case, change the current folder in such a way that there is an indication if transfer_heads was true or false -- #
        if 'MH' not in output_folder and 'SEQ' not in output_folder:
            output_folder = os.path.join(output_folder, 'SEQ') if self.transfer_heads else os.path.join(output_folder, 'MH')
        
        # -- Return the folder -- #
        return output_folder

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
        # now wrap the loss
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}) # Redefine this since at this stage self.loss is a MultipleOutputLoss2
        self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
        ################# END ###################
        #------------------------------------------ Partially copied from original implementation ------------------------------------------#