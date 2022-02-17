#########################################################################################################
#----------This class represents a Parameter Searcher that can be used to find suitable parameter-------#
#-----------values to train a network with to achieve good results based on the tested params.----------#
#########################################################################################################

import numpy as np
import nnunet_ext, glob
from nnunet_ext.utilities.helpful_functions import *
from nnunet_ext.evaluation.evaluator import Evaluator
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import recursive_find_python_class
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet_ext.training.model_restore import recursive_find_python_class_file

# -- Import all extensional trainers in a more generic way -- #
extension_keys = [x for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training")) if 'py' not in x]
trainer_keys = list()
for ext in extension_keys:
    trainer_name = [x[:-3] for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training", ext)) if '.py' in x]
    trainer_keys.extend(trainer_name)
# -- Sort based on the string but do this only on the lower keys  -- #
extension_keys.sort(key=lambda x: x.lower()), trainer_keys.sort(key=lambda x: x.lower())
sorted_pairs = zip(extension_keys, trainer_keys)
# NOTE: sorted_pairs does not include the nnViTUNetTrainer!

# -- Import the trainer classes and keep track of them -- #
TRAINER_MAP = dict()
for ext, tr in sorted_pairs:
    search_in = (nnunet_ext.__path__[0], "training", "network_training", ext)
    base_module = 'nnunet_ext.training.network_training.' + ext
    trainer_class = recursive_find_python_class([join(*search_in)], tr, current_module=base_module)
    # -- Track the classes based on their trainer strings and the extension as well -- #
    # -- Build mapping for extension to corresponding class -- #
    TRAINER_MAP[tr] = trainer_class
    TRAINER_MAP[ext] = trainer_class


class Experiment():
    r"""Class that can be used to perform a specific Experiment using the nnU-Net extension. This was specifically
        developed for the Parameter Search method but it can also be used for other purposes.
    """
    def __init__(self, network, network_trainer, tasks_list_with_char, version=1, vit_type='base', fold=0,
                 plans_identifier=default_plans_identifier, mixed_precision=True, extension='multihead', save_csv=True, val_folder='validation_raw',
                 split_at=None, transfer_heads=False, use_vit=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, do_pod=False,
                 always_use_last_head=True, npz=False, output_exp=None, output_eval=None, perform_validation=False, param_call=False,
                 unpack_data=True, deterministic=False, save_interval=5, num_epochs=100, fp16=True, find_lr=False, valbest=False, use_param_split=False,
                 disable_postprocessing_on_folds=False, split_gpu=False, val_disable_overwrite=True, disable_next_stage_pred=False, show_progress_tr_bar=True):
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
        self.perform_validation = perform_validation
        self.val_disable_overwrite = val_disable_overwrite
        self.disable_next_stage_pred = disable_next_stage_pred
        self.disable_postprocessing_on_folds = disable_postprocessing_on_folds

        # -- Set all the relevant attributes -- #
        self.fold = fold
        self.do_pod = do_pod
        self.network = network
        self.split_at = split_at
        self.save_csv = save_csv
        self.split_gpu = split_gpu
        self.extension = extension
        self.num_epochs = num_epochs
        self.param_call = param_call
        self.save_interval = save_interval
        self.param_split = use_param_split
        self.transfer_heads = transfer_heads
        self.network_trainer = network_trainer
        self.mixed_precision = mixed_precision
        self.plans_identifier = plans_identifier
        self.use_progress_bar = show_progress_tr_bar
        self.always_use_last_head = always_use_last_head
        self.tasks_list_with_char = tasks_list_with_char
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
        self.output_exp = output_exp
        self.output_eval = output_eval
        assert self.output_exp is not None and self.output_eval is not None,\
            'Please provide all three paths, being output_exp and output_eval..'

        # -- Define dict with arguments for function calls -- # 
        basic_args = {'unpack_data': self.unpack_data, 'deterministic': self.deterministic, 'fp16': self.fp16}
        self.basic_exts = {'save_interval': self.save_interval, 'identifier': self.plans_identifier, 'extension': self.extension,
                           'tasks_list_with_char': copy.deepcopy(self.tasks_list_with_char), 'save_csv': self.save_csv, 'use_progress': self.use_progress_bar,
                           'mixed_precision': self.mixed_precision, 'use_vit': self.use_vit, 'vit_type': self.vit_type, 'version': str(version),
                           'split_gpu': self.split_gpu, 'transfer_heads': self.transfer_heads, 'ViT_task_specific_ln': self.ViT_task_specific_ln,
                           'do_LSA': self.LSA, 'do_SPT': self.SPT, 'use_param_split': use_param_split, **basic_args}
        
        # -- Check that the hyperparameters match -- #
        trainer_file_to_import = recursive_find_python_class_file([join(nnunet_ext.__path__[0], "training", "network_training", self.extension)],
                                                                   self.network_trainer, current_module='nnunet_ext.training.network_training.' + self.extension)
        self.hyperparams = trainer_file_to_import.HYPERPARAMS   # --> dict {param_name: type}
        
        # -- Now create the Evaluator -- #
        self.basic_eval_args = {'network': self.network, 'network_trainer': self.network_trainer, 'tasks_list_with_char': self.tasks_list_with_char,
                                'version': self.version, 'vit_type': self.vit_type, 'plans_identifier': self.plans_identifier, 'mixed_precision': self.mixed_precision,
                                'extension': self.extension, 'save_csv': True, 'transfer_heads': self.transfer_heads, 'use_vit': self.use_vit,
                                'use_param_split': self.param_split, 'ViT_task_specific_ln': self.ViT_task_specific_ln, 'do_LSA': self.LSA, 'do_SPT': self.SPT}
        self.evaluator = Evaluator(model_list_with_char = (self.tasks_list_with_char[0][0], self.tasks_list_with_char[1]), **self.basic_eval_args)

    def run_experiment(self, exp_id, settings, settings_in_folder_name, gpu_ids, continue_tr=False):
        r"""This function is used to run a specific experiment based on the settings. This enables multiprocessing.
            settings should be a dictionary if structure: {param: value, param:value, ...}
        """
        # -- Create empty sumary file object -- #
        self.summary = None
        # -- Deepcopy the tasks since in case of -c they are changed which might cause trouble in restoring -- #
        all_tasks = copy.deepcopy(self.tasks_list_with_char[0]) # --> Will never be modified
        tasks = copy.deepcopy(self.tasks_list_with_char[0]) # --> Will be modified during -c process
            
        # -- Check that settings are of dict type -- #
        assert isinstance(settings, dict), 'The settings should be a dictionary looking like {{param: value, param:value, ...}}..'
        assert set(settings.keys()).issubset(set(self.hyperparams.keys())) and len(self.hyperparams.keys()) != 0,\
            "The user provided a list of parameters to tune but they do not map with the one from the desired Trainer. The trainers parameters are: {}".format(', '.join(list(self.hyperparams.keys())))

        # -- Set the GPUs for this experiment -- #
        # -- When running in parallel this has no effect on the parents environ, see https://stackoverflow.com/questions/57561116/does-process-in-python-have-its-own-os-environ-copy -- #
        assert isinstance(gpu_ids, (list, tuple)), 'Please provide the GPU ID(s) in form of a list or tuple..'
        cuda = join_texts_with_char(gpu_ids, ',')
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda   # --> When running in parallel this has no effect on the parents environ!

        # -- Bring settings into correct format -- #
        for k, v in tuple(settings.items()):
            settings[k] = self.hyperparams[k](v)    # --> self.hyperparams dict {param_name: type}

        # -- Define empty list holding all tasks we trained on so far -- #
        running_task_list = list()

        # -- Create the folder for a specific experiment -- #
        if settings_in_folder_name:    # Set this True for the parameter search method
            experiment_folder = os.path.join(self.output_exp, exp_id, '--'.join([f'{key}_{value}' for key, value in settings.items()]))
        else:
            experiment_folder = os.path.join(self.output_exp, exp_id)

        # -- Initilize variable that indicates if the trainer has been initialized -- #
        prev_trainer_path, already_trained_on, re_init, do_train, finished = None, None, False, True, False

        # -- Load the already trained on file etc. -- # 
        if continue_tr:
            print("Try to restore a state to continue with the training..")
            summaries = [os.path.join(experiment_folder, x) for x in os.listdir(experiment_folder) if '{}_summary_'.format(exp_id) in x]
            if len(summaries) != 1:
                assert True, "There are no or more than one summary file, how can this be when using -c? Remove -c or delete the additional log files that are not necessary."
            self.summary = summaries[0]
            # -- Load the already trained on pickle file -- #
            # -- NOTE: This was implemented once the already trained on file was changed to a pickle file -- #
            alr_tr_file = os.path.join(os.path.sep, *experiment_folder.split(os.path.sep)[:-1], self.extension+"_trained_on.pkl")
            assert os.path.isfile(alr_tr_file),\
                'There is no backup file for continuing with the experiment as it is expected: {}.'.format(alr_tr_file)
            already_trained_on = load_pickle(alr_tr_file)
            self.summary = print_to_log_file(self.summary, None, '', "Continuing with the experiment.. \n")

            # -- Do the whole restoring part to determine where to continue etc. -- #
            began_with = -1
            
            # -- Get the data regarding the current fold (if it exists, otherwise -1 is returned) -- #
            trained_on_folds = already_trained_on.get(str(self.fold), -1)
            if isinstance(trained_on_folds, dict):
                began_with = trained_on_folds.get('start_training_on', None)
                running_task_list = already_trained_on[str(self.fold)]['finished_training_on'][:] # Without '[:]' reference will change over time as well !
            
            # -- If began_with is None, a specific task training has not started --> start with the next task as -c would not have been set -- #
            if began_with is None:
                # -- Check if all tasks have been trained on so far, if so, this fold is finished with training, else it is not -- #
                run_tasks = running_task_list
                
                # -- If the lists are equal, continue with the evaluation, if not, specify the right task in the following steps -- #
                try:
                    if np.array(np.array(all_tasks) == np.array(run_tasks)).all():  # Use numpy because lists return true if at least one match in both lists!
                        # -- Update the user that the current fold is finished with training -- #
                        print("Fold {} has been trained on all tasks --> move on to the evaluation of the last task..".format(self.fold))
                        # -- Set the train flag to false so only the evaluation will be performed -- #
                        do_train, finished = False, True
                    # -- In this case the training stopped after a task was finished but not every task is trained -- #
                    else:
                        # -- Set began_with to None so it will be catched in the corresponding section to continue training -- #
                        began_with = None
                except ValueError: # --> The arrays do not match, ie. not finished on all tasks and validation is missing
                    # -- Set began_with to None so it will be catched in the corresponding section to continue training -- #
                    began_with = None
            
            # -- If this list is empty, the trainer did not train on any task --> Start directly with the first task as -c would not have been set -- #
            if began_with != -1: # At this point began_with is either a task or None but not -1
                if len(running_task_list) != 0:
                    # -- Substract the tasks from the tasks list --> Only use the tasks that are in tasks but not in finished_with -- #
                    remove_tasks = tasks[:]
                    for task in tasks:
                        # -- If the task has already been trained, remove the entry from the tasks dictionary -- #
                        if task in running_task_list:
                            prev_task = task    # Keep track to insert it at the end again
                            remove_tasks.remove(task)
                    # -- Reset the tasks so everything is as expected -- #
                    tasks = remove_tasks
                    del remove_tasks

                    # -- Insert the previous task to the beginning of the list to ensure that the model will be initialized the right way -- #
                    tasks.insert(0, prev_task)
                        
                    # -- Treat the last task as initialization, so set re_init to True by keeping continue_tr True  -- #
                    re_init = True
                   
                # -- ELSE -- #
                # -- If running_task_list is empty, the training failed at very first task, -- #
                # -- so nothing needs to be changed, simply continue with the training -- #
                # -- Set the prev_trainer and the init_identifier so the trainer will be build correctly -- #
                prev_trainer = TRAINER_MAP.get(already_trained_on[str(self.fold)]['prev_trainer'][-1], None)
                init_identifier = already_trained_on[str(self.fold)]['used_identifier']

            # -- began_with == -1 or no tasks to train --> nothing to restore -- #
            else:   # Start from beginning without continue_tr
                # -- Set continue_tr to False so there will be no error in the process of building the trainer -- #
                continue_tr = False
                
                # -- Set the prev_trainer and the init_identifier based on previous fold so the trainer will be build correctly -- #
                if already_trained_on.get(str(self.fold), None) is None:
                    prev_trainer = None
                    init_identifier = default_plans_identifier
                else:
                    prev_trainer = TRAINER_MAP.get(already_trained_on[str(self.fold)]['prev_trainer'][-1], None)
                    init_identifier = already_trained_on[str(self.fold)]['used_identifier']
                
        # -- Create a summary file for this experiment --> self.summary might be None, so provide all arguments -- #
        else:
            self.summary = print_to_log_file(self.summary, experiment_folder, '{}_summary'.format(exp_id), "Starting with the experiment.. \n")
        
            # -- Start with a general message describing the experiment -- #
            msg = ''
            for k, v in settings.items():
                msg += str(k) + ':' + str(v) + ', '
            self.summary = print_to_log_file(self.summary, None, '', 'Using trainer {} with the following settings: {}.'.format(self.network_trainer, msg[:-2]))
            self.summary = print_to_log_file(self.summary, None, '', 'The Trainer will be trained on the following tasks: {}.'.format(', '.join(self.tasks_list_with_char[0])))
        
        # -- Extract the trainer_class based on the extension -- #
        trainer_class_ref = recursive_find_python_class([join(nnunet_ext.__path__[0], "training", "network_training", self.extension)],
                                                        self.network_trainer, current_module='nnunet_ext.training.network_training.' + self.extension)
        
        # -- Build the arguments dict for the trainer -- #
        arguments = {**settings, **self.basic_exts}

        # -- Loop through the tasks and train for each task the (finished) model -- #
        for idx, t in enumerate(tasks):
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
            # -- Modify the output_folder_name -- #
            if not self.param_call:
                output_folder_name = os.path.join(experiment_folder, *output_folder_name.split(os.path.sep)[-2:])   # only add running_task, network_trainer + "__" + plans_identifier)
            else:
                output_folder_name = os.path.join(experiment_folder, output_folder_name.split(os.path.sep)[-2])     # only use experiment folder and add running_task

            if trainer_class is None:
                raise RuntimeError("Could not find trainer class in nnunet_ext.training.network_training")

            if idx == 0 and not continue_tr:
                # -- At the first task, the base model can be a nnunet model, later, only current extension models are permitted -- #
                possible_trainers = set(TRAINER_MAP.values())   # set to avoid the double values, since every class is represented twice
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

            # -- Load trainer from last task and initialize new trainer if continue_tr is not set -- #
            if idx == 0 and re_init:
                # -- Initialize the prev_trainer if it is not None. If it is None, the trainer will be initialized in the parent class -- #
                # -- Further check that all necessary information is provided, otherwise exit with error message -- #
                assert isinstance(t, str) and prev_trainer is not None and init_identifier is not None and isinstance(self.fold, int),\
                    "The informations for building the initial trainer to use for training are not fully provided, check the arguments.."
                
                # -- Get default configuration for nnunet/nnunet_ext model (finished training) -- #
                # -- --> Extension is used, always use the first task as this is the base and other tasks -- #
                # -- will result in other network structures (structure based on plans_file) -- #
                # -- Current task t might not be equal to all_tasks[0], since tasks might be changed in the -- #
                # -- preparation for -c. -- #
                plans_file, prev_trainer_path, dataset_directory, batch_dice, stage, \
                trainer_class = get_default_configuration(self.network, all_tasks[0], running_task, prev_trainer, self.tasks_joined_name,\
                                                          init_identifier, extension_type=self.extension)
                # -- Modify the prev_trainer_path -- #
                if not self.param_call:
                    prev_trainer_path = os.path.join(experiment_folder, *prev_trainer_path.split(os.path.sep)[-2:])   # only add running_task, network_trainer + "__" + plans_identifier)
                else:
                    prev_trainer_path = os.path.join(experiment_folder, prev_trainer_path.split(os.path.sep)[-2])     # only use experiment folder and add running_task

                # -- Ensure that trainer_class is not None -- #
                if trainer_class is None:
                    raise RuntimeError("Could not find trainer class in nnunet.training.network_training nor nnunet_ext.training.network_training")
                
                # -- Continue with next element, since the previous trainer is restored, otherwise it will be trained as well -- #
                do_train = False

            # -- Initialize new trainer -- #
            if idx == 0 or (idx == 1 and re_init):
                self.summary = print_to_log_file(self.summary, None, '', 'Initializing the Trainer..')
                # -- To initialize a new trainer, always use the first task since this shapes the network structure. -- #
                # -- During training the tasks will be updated, so this should cause no problems -- #
                # -- Set the trainer with corresponding arguments --> can only be an extension from here on -- #
                trainer = trainer_class(self.split_at, all_tasks[0], plans_file, self.fold, output_folder=output_folder_name,\
                                        dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,\
                                        already_trained_on=already_trained_on, network=self.network, **arguments)
                trainer.initialize(True, num_epochs=self.num_epochs, prev_trainer_path=prev_trainer_path)
                
                # NOTE: Trainer has only weights and heads of first task at this point
                # --> Add the heads and load the state_dict from the latest model and not all_tasks[0]
                #     since this is only used for initialization

            # -- Update trained_on 'manually' if first task is done but finished_training_on is empty --> first task was used for initialization -- #
            if idx == 1 and len(trainer.already_trained_on[str(self.fold)]['finished_training_on']) == 0:
                trainer.update_save_trained_on_json(all_tasks[0], finished=True)

            # -- Find a matchting lr given the provided num_epochs -- #
            if self.find_lr:
                trainer.find_lr(num_iters=self.num_epochs)
            else:
                if continue_tr:
                    # -- User wants to continue previous training -- #
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
                    continue_tr = False
                    
                # -- Start to train the trainer --> if task is not registered, the trainer will do this automatically -- #
                if do_train:
                    self.summary = print_to_log_file(self.summary, None, '', 'Start/Continue training on: {}.'.format(t))
                    trainer.run_training(task=t, output_folder=output_folder_name)
                    self.summary = print_to_log_file(self.summary, None, '', 'Finished training on: {}. So far trained on: {}.'.format(t, ', '.join(running_task_list)))
                
                    # -- Do validation if desired -- #
                    if self.perform_validation:
                        if self.valbest:
                            trainer.load_best_checkpoint(train=False)
                        else:
                            trainer.load_final_checkpoint(train=False)

                        # -- Evaluate the trainers network -- #
                        trainer.network.eval()

                        # -- Perform validation using the trainer -- #
                        self.summary = print_to_log_file(self.summary, None, '', 'Start with validation..')
                        trainer.validate(save_softmax=self.npz, validation_folder_name=self.val_folder,
                                        run_postprocessing_on_folds=not self.disable_postprocessing_on_folds,
                                        overwrite=self.val_disable_overwrite)
                        self.summary = print_to_log_file(self.summary, None, '', 'Finished with validation..')
                else:   # --> Load the checkpoint for eval
                    trainer.load_latest_checkpoint()
                    # -- Reset do_train again -- #
                    if idx == 0 and re_init and finished:   # Only keep not training if we already trained on all tasks but the evaluation is missing
                        do_train = False
                    else:
                        do_train = True

            # -- Always do the evaluation (even if it is already done) since it does not take long and we don't want to create an extra backup file for this -- #      
            # -- Reinitialize the Evaluator first -- #
            model_list_with_char = (running_task_list, self.tasks_list_with_char[1])    # --> Use the current trainer
            self.evaluator.reinitialize(model_list_with_char=model_list_with_char, **self.basic_eval_args)

            # -- Build the corresponding paths the evaluator should use -- #
            trainer_path = trainer.output_folder
            output_path = os.path.join(os.path.sep, *trainer_path.replace(self.output_exp, self.output_eval).split(os.path.sep)[:-1])    # Remove fold_X folder here

            # -- Remove the trainer if we're done after this evaluation, i.e. if do_train is false -- #
            ds_dir = trainer.dataset_directory
            if not do_train or set(all_tasks) == set(running_task_list):    # We can delete the trainer here to avoid CUDA OOM since evaluate_on restores the model
                del trainer
            
            # -- Do the actual evaluation on the current network -- #
            self.summary = print_to_log_file(self.summary, None, '', 'Doing evaluation for trainer {} (trained on {}) using the data from {}.'.format(self.network_trainer, ', '.join(running_task_list), ', '.join(running_task_list)))
            self.evaluator.evaluate_on([self.fold], self.tasks_list_with_char[0], None, self.always_use_last_head,
                                       self.do_pod, trainer_path, output_path)
            self.summary = print_to_log_file(self.summary, None, '', 'Finished with evaluation. The results can be found in the following folder: {}. \n'.format(join(output_path, 'fold_'+str(self.fold))))

            # -- Update the summary wrt to the used split -- #
            spts = ''
            if self.param_split:
                splits_file = join(ds_dir, "splits_param_search.pkl")
            else:
                splits_file = join(ds_dir, "splits_final.pkl")
            split = load_pickle(splits_file)
            for k, v in split[self.fold].items():
                spts += str(k) + ' := ' + ', '.join(str(v_) for v_ in v) + '\n'
            self.summary = print_to_log_file(self.summary, None, '', "For training, validation and evaluation of {} on fold {}, the used split can be found here: {}\nIt looks like the following:\n{}".format(t, self.fold, splits_file, spts))

        # -- Return the information of the experiment so we can map this in the higher level function, the parameter searcher -- #
        return exp_id, [y for x in os.walk(self.output_eval, exp_id) for y in glob.glob(os.path.join(x[0], 'summarized_val_metrics.csv'))]