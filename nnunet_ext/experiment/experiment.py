#########################################################################################################
#----------This class represents a Parameter Searcher that can be used to find suitable parameter-------#
#-----------values to train a network with to achieve good results based on the tested params.----------#
#########################################################################################################

from collections import OrderedDict
import nnunet_ext, random, itertools
from sklearn.model_selection import train_test_split
from nnunet_ext.utilities.helpful_functions import *
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import recursive_find_python_class
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet_ext.paths import param_search_output_dir, default_plans_identifier, preprocessing_output_dir

class Experiment():
    r"""Class that can be used to perform a specific Experiment using the nnU-Net extension. This was specifically
        developed for the Parameter Search method but it can also be used for other purposes.
    """
    def __init__(self, network, network_trainer, tasks_list_with_char, version=1, vit_type='base', eval_mode_for_lns='last_lns', fold=0,
                 plans_identifier=default_plans_identifier, mixed_precision=True, extension='multihead', save_csv=True, val_folder='validation_raw',
                 split_at=None, transfer_heads=False, use_vit=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, do_pod=False,
                 always_use_last_head=True, npz=False, output_base=None, output_exp=None, output_eval=None, settings_in_folder_name=True,
                 perform_validation=False, continue_training=False, unpack_data=True, deterministic=False, save_interval=5, num_epochs=100,
                 fp16=True, find_lr=False, valbest=False, disable_postprocessing_on_folds=False, split_gpu=False,
                 val_disable_overwrite=True, disable_next_stage_pred=False, exp_id=0, settings=dict()):
        r"""Constructor for Experiment.
        """
        # -- Define a empty dictionary that is used for backup purposes -- #
        self.backup_information = dict()

        # -- Set all nnU-Net related arguments -- #
        self.npz = npz
        self.fp16 = fp16
        self.find_lr = find_lr
        self.valbest = valbest
        self.network_ = network
        self.val_folder = val_folder
        self.unpack_data = unpack_data
        self.deterministic = deterministic
        self.network_trainer = network_trainer
        self.continue_training = continue_training
        self.perform_validation = perform_validation
        self.val_disable_overwrite = val_disable_overwrite
        self.disable_next_stage_pred = disable_next_stage_pred
        self.disable_postprocessing_on_folds = disable_postprocessing_on_folds

        # -- Set all the relevant attributes -- #
        self.fold = fold
        self.exp_id = exp_id
        self.do_pod = do_pod
        self.network = network
        self.split_at = split_at
        self.save_csv = save_csv
        self.settings = settings
        self.split_gpu = split_gpu
        self.extension = extension
        self.num_epochs = num_epochs
        self.save_interval = save_interval
        self.transfer_heads = transfer_heads
        self.network_trainer = network_trainer
        self.mixed_precision = mixed_precision
        self.plans_identifier = plans_identifier
        self.eval_mode_for_lns = eval_mode_for_lns
        self.always_use_last_head = always_use_last_head
        self.tasks_list_with_char = tasks_list_with_char
        self.settings_in_folder_name = settings_in_folder_name
        # -- Set tasks_joined_name for validation dataset building -- #
        self.tasks_joined_name = join_texts_with_char(self.tasks_list_with_char[0], self.tasks_list_with_char[1])
        
        # -- If ViT trainer, build the version correctly for finding the correct checkpoint later in restoring -- #
        # -- Create the variable indicating which ViT Architecture to use, base, large or huge and if to use it -- #
        self.LSA = do_LSA
        self.SPT = do_SPT
        self.use_vit = use_vit
        self.vit_type = vit_type.lower()
        self.version = 'V' + str(version)
        self.ViT_task_specific_ln = ViT_task_specific_ln

        # -- Path related attributes -- #
        # TODO: Do we need the output_base ??
        self.output_base = output_base
        self.output_exp = output_exp
        self.output_eval = output_eval
        assert self.output_base is not None and self.output_exp is not None and self.output_eval is not None,\
            'Please provide all three paths, being output_base, output_exp and output_eval..'

        # -- Load the Backup file if it exists -- #
        # TODO: Modify for backup of specific experiment, not overall parameter search!
        # backup_file = os.path.join(self.output_base, 'backup.pkl')
        # if os.path.isfile(backup_file):
        #     self.backup_information = load_pickle(backup_file)
        # else:
        #     # -- Assert if continue training is set, since no backup is stored -- #
        #     assert not self.continue_training, "There is no backup file, yet the user wants to continue with the parameter search.. Remove the -c flag."

        # -- Define dict with arguments for function calls -- # 
        basic_args = {'unpack_data': self.unpack_data, 'deterministic': self.deterministic, 'fp16': self.fp16}
        self.basic_exts = {'save_interval': self.save_interval, 'identifier': self.plans_identifier, 'extension': self.extension,
                           'tasks_list_with_char': copy.deepcopy(self.tasks_list_with_char), 'save_csv': self.save_csv,
                           'mixed_precision': self.mixed_precision, 'use_vit': self.use_vit, 'vit_type': self.vit_type, 'version': self.version,
                           'split_gpu': self.split_gpu, 'transfer_heads': self.transfer_heads, 'ViT_task_specific_ln': self.ViT_task_specific_ln,
                           'do_LSA': self.LSA, 'do_SPT': self.SPT, **basic_args}

        # -- Add the experiment to the backup file -- #
        # TODO
        

    def reset_experiment(self, exp_id, settings, settings_in_folder_name=True):
        r"""This function can be used to reset the Experiment this is not a one time Experiment.
        """
        # -- Reset the internal exp_id and settings -- #
        self.exp_id, self.settings, self.settings_in_folder_name = exp_id, settings, settings_in_folder_name

    def run_experiment(self):
        r"""This function is used to run a specific experiment based on the settings. This enables multiprocessing.
            settings should be a dictionary if structure: {param: value, param:value, ...}
        """
        # -- Define empty list holding all tasks we trained on so far -- #
        running_task_list = list()

        # -- Create the folder for a specific experiment -- #
        if self.settings_in_folder_name:    # Set this True for the parameter search method
            experiment_folder = os.path.join(self.output_exp, self.exp_id, '--'.join([f'{key}_{value}' for key, value in self.settings.items()]))
            evaluation_folder = os.path.join(self.output_eval, self.exp_id, '--'.join([f'{key}_{value}' for key, value in self.settings.items()]))
        else:
            experiment_folder = os.path.join(self.output_exp, self.exp_id)
            evaluation_folder = os.path.join(self.output_eval, self.exp_id)

        # -- Extract the trainer_class based on the extension -- #
        trainer_class_ref = recursive_find_python_class([join(nnunet_ext.__path__[0], "training", "network_training", self.extension)],
                                                        self.network_trainer, current_module='nnunet_ext.training.network_training.' + self.extension)
        
        # -- Build the arguments dict for the trainer -- #
        arguments = {**self.settings, **self.basic_exts}

        # TODO: paths for metafiles, maybe add extra flag to trainer for param so it will be build differently ?
        #       every experiments should have its own metadata folder maybe

        # TODO: Modify paths so every exp has one folder with settings in it: param_i_value—param_i+1_value-.../fold_X
        # TODO: Every exp has a summary file in the folder summarizing the experiment
        # self.output_experiment = os.path.join(self.output_exp, 'param_settings')

        # # -- Define the log file and output folder where it will be stored-- #
        # log_file = None # Create it in first call
        # output_folder = old_network_training_output_dir_base  # Store the log file were training is performed
        # log_file = print_to_log_file(log_file, output_folder, 'param_search_log', "The user wanted to perform the searching using multiple GPUs in parallel but does not provide more than one GPU..")

        # TODO: Convert hyperparameters to expected type, they are all floats in random mode!

        # TODO: Perform evaluation on every param exp --> everything should be in PARAM folder !!
        
        # -- Initilize variable that indicates if the trainer has been initialized -- #
        already_trained_on = None

        # -- Initialize the previous trainer path -- #
        prev_trainer_path = None

        # -- Loop through the tasks and train for each task the (finished) model -- #
        for idx, t in enumerate(self.tasks_list_with_char[0]):
            # -- Update running task list and create running task which are all (trained tasks and current task joined) for output folder name -- #
            if t not in running_task_list:
                running_task_list.append(t)
            running_task = join_texts_with_char(running_task_list, self.tasks_list_with_char[1])

            # -- Extract the configurations and check that trainer_class is not None -- #
            # -- NOTE: Each task will be saved as new folder using the running_task that are all previous and current task joined together. -- #
            # -- NOTE: Perform preprocessing and planning before ! -- #
            plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
            trainer_class = get_default_configuration(self.network, t, running_task, self.network_trainer, self.tasks_joined_name,\
                                                      self.plans_identifier, extension_type=self.extension)
            
            # TODO Modify output_folder_name!!
            
            if trainer_class is None:
                raise RuntimeError("Could not find trainer class in nnunet_ext.training.network_training")

            if idx == 0 and not self.continue_training:
                # -- At the first task, the base model can be a nnunet model, later, only current extension models are permitted -- #
                possible_trainers = set(trainer_map.values())   # set to avoid the double values, since every class is represented twice
                assert issubclass(trainer_class, tuple(possible_trainers)),\
                    "Network_trainer was found but is not derived from a provided extension nor nnUNetTrainerV2."\
                    " When using this function, it is only permitted to start with an nnUNetTrainerV2 or a provided extensions"\
                    " like nnUNetTrainerMultiHead or nnUNetTrainerRehearsal. Choose the right trainer or use the convential"\
                    " nnunet command to train."
            else:
                # -- Now, at a later stage, only trainer based on extension permitted! -- #
                assert issubclass(trainer_class, trainer_class_ref),\
                "Network_trainer was found but is not derived from {}."\
                " When using this function, only {} trainers are permitted."\
                " So choose {}"\
                " as a network_trainer corresponding to the network, or use the convential nnunet command to train.".format(trainer_class_ref, self.extension, trainer_class_ref)

            # -- Load trainer from last task and initialize new trainer if continue training is not set -- #
            if idx == 0:
                # -- To initialize a new trainer, always use the first task since this shapes the network structure. -- #
                # -- During training the tasks will be updated, so this should cause no problems -- #
                # -- Set the trainer with corresponding arguments --> can only be an extension from here on -- #
                trainer = trainer_class(self.split_at, self.tasks_list_with_char[0], plans_file, self.fold, output_folder=output_folder_name,\
                                        dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,\
                                        already_trained_on=already_trained_on, **arguments)
                trainer.initialize(True, num_epochs=self.num_epochs, prev_trainer_path=prev_trainer_path)
                
                # -- Get the desired network_trainer hyperparameters -- #
                try:
                    hyperparams = trainer.hyperparams   # --> dict {param_name: type}
                except AttributeError:
                    assert False, "The desired Trainer ({}) has no hyperparameters, so we can not perform a parameter search..".format(self.network_trainer)
                assert set(self.params_to_tune) == set(hyperparams),\
                    "The user provided a list of parameters to tune but they do not map with the one from the desired Trainer. The trainers parameters are: {}".format(','.join(list(hyperparams.keys())))

            # NOTE: Trainer has only weights and heads of first task at this point
            # --> Add the heads and load the state_dict from the latest model and not all_tasks[0]
            #     since this is only used for initialization

            # -- Update trained_on 'manually' if first task is done but finished_training_on is empty --> first task was used for initialization -- #
            if idx == 1 and len(trainer.already_trained_on[str(self.fold)]['finished_training_on']) == 0:
                trainer.update_save_trained_on_json(self.tasks_list_with_char[0][0], finished=True)

            # -- Find a matchting lr given the provided num_epochs -- #
            if self.find_lr:
                trainer.find_lr(num_iters=self.num_epochs)
            else:
                if self.continue_training:
                    # -- User wants to continue previous training while ignoring pretrained weights -- #
                    try: # --> There is only a checkpoint if it has passed save_every
                        trainer.load_latest_checkpoint()
                    except Exception as e:
                        # -- Print the Exception that has been thrown and continue -- #
                        print(e)
                        
                        # --> Found no checkpoint, so one task is finished but the current hasn't started yet or -- #
                        # -- did not reach save_every -- #
                        pass
                    # -- Set continue_training to false for possible upcoming tasks -- #
                    # -- --> otherwise an error might occur because there is no trainer to restore -- #
                    self.continue_training = False
                
                # -- Start to train the trainer --> if task is not registered, the trainer will do this automatically -- #
                trainer.run_training(task=t, output_folder=output_folder_name)
               
                if self.perform_validation:
                    if self.valbest:
                        trainer.load_best_checkpoint(train=False)
                    else:
                        trainer.load_final_checkpoint(train=False)

                    # -- Evaluate the trainers network -- #
                    trainer.network.eval()

                    # -- Perform validation using the trainer -- #
                    trainer.validate(save_softmax=self.npz, validation_folder_name=self.val_folder,
                                    run_postprocessing_on_folds=not self.disable_postprocessing_on_folds,
                                    overwrite=self.val_disable_overwrite)

            # -- Update prev_trainer and prev_trainer_path -- #
            prev_trainer_path = output_folder_name

            # TODO: Do evaluation at the end -- #

            # TODO: Update the summary and main summary of the experiments -- #