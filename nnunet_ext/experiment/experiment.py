#########################################################################################################
#----------This class represents a Parameter Searcher that can be used to find suitable parameter-------#
#-----------values to train a network with to achieve good results based on the tested params.----------#
#########################################################################################################

import nnunet_ext, glob
from nnunet_ext.utilities.helpful_functions import *
from nnunet_ext.evaluation.evaluator import Evaluator
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import recursive_find_python_class
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet_ext.training.model_restore import recursive_find_python_class_file


# -- Import the trainer classes -- #
# TODO: Make it generic
from nnunet_ext.training.network_training.rw.nnUNetTrainerRW import nnUNetTrainerRW # Own implemented class
from nnunet_ext.training.network_training.ewc.nnUNetTrainerEWC import nnUNetTrainerEWC # Own implemented class
from nnunet_ext.training.network_training.lwf.nnUNetTrainerLWF import nnUNetTrainerLWF # Own implemented class
from nnunet_ext.training.network_training.mib.nnUNetTrainerMiB import nnUNetTrainerMiB # Own implemented class
from nnunet_ext.training.network_training.pod.nnUNetTrainerPOD import nnUNetTrainerPOD # Own implemented class
from nnunet_ext.training.network_training.plop.nnUNetTrainerPLOP import nnUNetTrainerPLOP # Own implemented class
from nnunet_ext.training.network_training.ownm1.nnUNetTrainerOwnM1 import nnUNetTrainerOwnM1 # Own implemented class
from nnunet_ext.training.network_training.ownm2.nnUNetTrainerOwnM2 import nnUNetTrainerOwnM2 # Own implemented class
from nnunet_ext.training.network_training.ownm3.nnUNetTrainerOwnM3 import nnUNetTrainerOwnM3 # Own implemented class
from nnunet_ext.training.network_training.ownm4.nnUNetTrainerOwnM4 import nnUNetTrainerOwnM4 # Own implemented class
from nnunet_ext.training.network_training.ewc_ln.nnUNetTrainerEWCLN import nnUNetTrainerEWCLN # Own implemented class
from nnunet_ext.training.network_training.ewc_vit.nnUNetTrainerEWCViT import nnUNetTrainerEWCViT # Own implemented class
from nnunet_ext.training.network_training.ewc_unet.nnUNetTrainerEWCUNet import nnUNetTrainerEWCUNet # Own implemented class
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead # Own implemented class
from nnunet_ext.training.network_training.rehearsal.nnUNetTrainerRehearsal import nnUNetTrainerRehearsal # Own implemented class
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential # Own implemented class
from nnunet_ext.training.network_training.freezed_vit.nnUNetTrainerFreezedViT import nnUNetTrainerFreezedViT # Own implemented class
from nnunet_ext.training.network_training.freezed_unet.nnUNetTrainerFreezedUNet import nnUNetTrainerFreezedUNet # Own implemented class
from nnunet_ext.training.network_training.freezed_nonln.nnUNetTrainerFreezedNonLN import nnUNetTrainerFreezedNonLN # Own implemented class


class Experiment():
    r"""Class that can be used to perform a specific Experiment using the nnU-Net extension. This was specifically
        developed for the Parameter Search method but it can also be used for other purposes.
    """
    def __init__(self, network, network_trainer, tasks_list_with_char, version=1, vit_type='base', eval_mode_for_lns='last_lns', fold=0,
                 plans_identifier=default_plans_identifier, mixed_precision=True, extension='multihead', save_csv=True, val_folder='validation_raw',
                 split_at=None, transfer_heads=False, use_vit=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, do_pod=False,
                 always_use_last_head=True, npz=False, output_exp=None, output_eval=None, perform_validation=False, continue_training=False,
                 unpack_data=True, deterministic=False, save_interval=5, num_epochs=100, fp16=True, find_lr=False, valbest=False, use_param_split=False,
                 disable_postprocessing_on_folds=False, split_gpu=False, val_disable_overwrite=True, disable_next_stage_pred=False, params_to_tune=list()):
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
        self.do_pod = do_pod
        self.network = network
        self.split_at = split_at
        self.save_csv = save_csv
        self.split_gpu = split_gpu
        self.extension = extension
        self.num_epochs = num_epochs
        self.save_interval = save_interval
        self.param_split = use_param_split
        self.transfer_heads = transfer_heads
        self.params_to_tune = params_to_tune
        self.network_trainer = network_trainer
        self.mixed_precision = mixed_precision
        self.plans_identifier = plans_identifier
        self.eval_mode_for_lns = eval_mode_for_lns
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
                           'tasks_list_with_char': copy.deepcopy(self.tasks_list_with_char), 'save_csv': self.save_csv,
                           'mixed_precision': self.mixed_precision, 'use_vit': self.use_vit, 'vit_type': self.vit_type, 'version': self.version,
                           'split_gpu': self.split_gpu, 'transfer_heads': self.transfer_heads, 'ViT_task_specific_ln': self.ViT_task_specific_ln,
                           'do_LSA': self.LSA, 'do_SPT': self.SPT, 'use_param_split': use_param_split, **basic_args}
        
        # -- Check that the hyperparameters match -- #
        trainer_file_to_import = recursive_find_python_class_file([join(nnunet_ext.__path__[0], "training", "network_training", self.extension)],
                                                                   self.network_trainer, current_module='nnunet_ext.training.network_training.' + self.extension)
        self.hyperparams = trainer_file_to_import.HYPERPARAMS   # --> dict {param_name: type}
        assert set(self.params_to_tune) == set(self.hyperparams) and len(self.hyperparams.keys()) != 0,\
            "The user provided a list of parameters to tune but they do not map with the one from the desired Trainer. The trainers parameters are: {}".format(','.join(list(self.hyperparams.keys())))

        # -- Add the trainer mapping dict -- #
        # TODO: Make it generic
        self.trainer_map = {'rw': nnUNetTrainerRW, 'nnUNetTrainerRW': nnUNetTrainerRW,
                            'ewc': nnUNetTrainerEWC, 'nnUNetTrainerEWC': nnUNetTrainerEWC,
                            'lwf': nnUNetTrainerLWF, 'nnUNetTrainerLWF': nnUNetTrainerLWF,
                            'mib': nnUNetTrainerMiB, 'nnUNetTrainerMiB': nnUNetTrainerMiB,
                            'pod': nnUNetTrainerPOD, 'nnUNetTrainerPOD': nnUNetTrainerPOD,
                            'plop': nnUNetTrainerPLOP, 'nnUNetTrainerPLOP': nnUNetTrainerPLOP,
                            'ownm1': nnUNetTrainerOwnM1, 'nnUNetTrainerOwnM1': nnUNetTrainerOwnM1,
                            'ownm2': nnUNetTrainerOwnM2, 'nnUNetTrainerOwnM2': nnUNetTrainerOwnM2,
                            'ownm3': nnUNetTrainerOwnM3, 'nnUNetTrainerOwnM3': nnUNetTrainerOwnM3,
                            'ownm4': nnUNetTrainerOwnM4, 'nnUNetTrainerOwnM4': nnUNetTrainerOwnM4,
                            'ewc_ln': nnUNetTrainerEWCLN, 'nnUNetTrainerEWCLN': nnUNetTrainerEWCLN,
                            'ewc_vit': nnUNetTrainerEWCViT, 'nnUNetTrainerEWCViT': nnUNetTrainerEWCViT,
                            'ewc_unet': nnUNetTrainerEWCUNet, 'nnUNetTrainerEWCUNet': nnUNetTrainerEWCUNet,
                            'rehearsal': nnUNetTrainerRehearsal, 'nnUNetTrainerRehearsal': nnUNetTrainerRehearsal,
                            'multihead': nnUNetTrainerMultiHead, 'nnUNetTrainerMultiHead': nnUNetTrainerMultiHead,
                            'sequential': nnUNetTrainerSequential, 'nnUNetTrainerSequential': nnUNetTrainerSequential,
                            'freezed_vit': nnUNetTrainerFreezedViT, 'nnUNetTrainerFreezedViT': nnUNetTrainerFreezedViT,
                            'freezed_unet': nnUNetTrainerFreezedUNet, 'nnUNetTrainerFreezedUNet': nnUNetTrainerFreezedUNet,
                            'freezed_nonln': nnUNetTrainerFreezedNonLN, 'nnUNetTrainerFreezedNonLN': nnUNetTrainerFreezedNonLN}

        # -- Now create the Evaluator -- #
        self.basic_eval_args = {'network': self.network, 'network_trainer': self.network_trainer, 'tasks_list_with_char': self.tasks_list_with_char,
                                'version': self.version, 'vit_type': self.vit_type, 'plans_identifier': self.plans_identifier, 'mixed_precision': self.mixed_precision,
                                'extension': self.extension, 'save_csv': True, 'transfer_heads': self.transfer_heads, 'use_vit': self.use_vit,
                                'use_param_split': self.param_split, 'ViT_task_specific_ln': self.ViT_task_specific_ln, 'do_LSA': self.LSA, 'do_SPT': self.SPT}
        self.evaluator = Evaluator(model_list_with_char = (self.tasks_list_with_char[0][0], self.tasks_list_with_char[1]), **self.basic_eval_args)

    def run_experiment(self, exp_id, settings, settings_in_folder_name, gpu_ids):
        r"""This function is used to run a specific experiment based on the settings. This enables multiprocessing.
            settings should be a dictionary if structure: {param: value, param:value, ...}
        """
        # TODO: Do restoring etc.

        # -- Create empty sumary file object -- #
        self.summary = None
            
        # -- Check that settings are of dict type -- #
        assert isinstance(settings, dict), 'The settings should be a dictionary looking like {{param: value, param:value, ...}}..'
        
        # -- Set the GPUs for this experiment -- #
        assert isinstance(gpu_ids, (list, tuple)), 'Please provide the GPU ID(s) in form of a list or tuple..'
        cuda = join_texts_with_char(gpu_ids, ',')
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda   # --> When running in parallel this has no effect on the parents environ!

        # TODO: When running in parallel this has no effect on the parents environ! TRUE ????

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

        if self.continue_training:
            # TODO: If -c load the existing summary --> Check if the matching case works!!
            self.summary = glob.glob(os.path.join(experiment_folder, '{}_summary*.txt'.format(exp_id)))
            assert len(self.summary) == 1, "There are no or more than one summary file, how can this be when using -c?"
            self.summary = self.summary[0]

        # -- Extract the trainer_class based on the extension -- #
        trainer_class_ref = recursive_find_python_class([join(nnunet_ext.__path__[0], "training", "network_training", self.extension)],
                                                        self.network_trainer, current_module='nnunet_ext.training.network_training.' + self.extension)
        
        # -- Build the arguments dict for the trainer -- #
        arguments = {**settings, **self.basic_exts}

        # -- Create a summary file for this experiment --> self.summary might be None, so provide all arguments -- #
        self.summary = print_to_log_file(self.summary, experiment_folder, '{}_summary'.format(exp_id), "Starting with the experiment.. \n \n")
        
        # -- Start with a general message describing the experiment -- #
        msg = ''
        for k, v in settings.items():
            msg += str(k) + ':' + str(v) + ', '
        self.summary = print_to_log_file(self.summary, 'Using trainer {} with the following settings: {}.'.format(self.network_trainer, msg[:-2]))
        self.summary = print_to_log_file(self.summary, 'The Trainer will be trained on the following tasks: {}.'.format(', '.join(self.tasks_list_with_char[0])))
        
        # -- Initilize variable that indicates if the trainer has been initialized -- #
        already_trained_on = None

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
            
            print(output_folder_name)
            
            # -- Modify the output_folder_name -- #
            output_folder_name = os.path.join(experiment_folder, *output_folder_name.split(os.path.sep)[-3:])   # only add running_task, network_trainer + "__" + plans_identifier)
            
            print(output_folder_name)
            # raise

            
            if trainer_class is None:
                raise RuntimeError("Could not find trainer class in nnunet_ext.training.network_training")

            if idx == 0 and not self.continue_training:
                # -- At the first task, the base model can be a nnunet model, later, only current extension models are permitted -- #
                possible_trainers = set(self.trainer_map.values())   # set to avoid the double values, since every class is represented twice
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

            # -- Initialize new trainer -- #
            if idx == 0:
                self.summary = print_to_log_file(self.summary, 'Initializing the Trainer..')
                # -- To initialize a new trainer, always use the first task since this shapes the network structure. -- #
                # -- During training the tasks will be updated, so this should cause no problems -- #
                # -- Set the trainer with corresponding arguments --> can only be an extension from here on -- #
                trainer = trainer_class(self.split_at, self.tasks_list_with_char[0], plans_file, self.fold, output_folder=output_folder_name,\
                                        dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,\
                                        already_trained_on=already_trained_on, **arguments)
                trainer.initialize(True, num_epochs=self.num_epochs, prev_trainer_path=None)
                
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
                    self.continue_training = False
                
                # -- Start to train the trainer --> if task is not registered, the trainer will do this automatically -- #
                self.summary = print_to_log_file(self.summary, 'Start/Continue training on: {}.'.format(t))
                trainer.run_training(task=t, output_folder=output_folder_name)
                self.summary = print_to_log_file(self.summary, 'Finished training on: {}. So far trained on: {}.'.format(t, ', '.join(running_task_list)))
                
                # -- Do validation if desired -- #
                if self.perform_validation:
                    if self.valbest:
                        trainer.load_best_checkpoint(train=False)
                    else:
                        trainer.load_final_checkpoint(train=False)

                    # -- Evaluate the trainers network -- #
                    trainer.network.eval()

                    # -- Perform validation using the trainer -- #
                    self.summary = print_to_log_file(self.summary, 'Start with validation..')
                    trainer.validate(save_softmax=self.npz, validation_folder_name=self.val_folder,
                                     run_postprocessing_on_folds=not self.disable_postprocessing_on_folds,
                                     overwrite=self.val_disable_overwrite)
                    self.summary = print_to_log_file(self.summary, 'Finished with validation..')
                    
            # -- Reinitialize the Evaluator first -- #
            model_list_with_char = (running_task_list, self.tasks_list_with_char[1])    # --> Use the current trainer
            self.evaluator.reinitialize(model_list_with_char, **self.basic_eval_args)

            # -- Build the corresponding paths the evaluator should use -- #
            trainer_path = trainer.output_folder
            output_path = trainer_path.replace(self.output_exp, self.output_eval)

            print(trainer_path)
            print(output_path)
            # raise

            # -- Do the actual evaluation on the current network -- #
            self.summary = print_to_log_file(self.summary, 'Doing evaluation for trainer {} (trained on {}) using the data from {}.'.format(self.network_trainer, ', '.join(running_task_list), ', '.join(running_task_list)))
            self.evaluator.evaluate_on(self.fold, self.tasks_list_with_char[0], None, self.always_use_last_head,
                                       self.do_pod, self.eval_mode_for_lns, trainer_path, output_path)
            self.summary = print_to_log_file(self.summary, 'Finished with evaluation. The results can be found in the following folder: {}. \n'.format(join(output_path, 'fold_'+str(self.fold))))
            
            # -- Update the summary wrt to the used split -- #
            spts = ''
            if self.param_split:
                splits_file = join(trainer.dataset_directory, "splits_param_search.pkl")
            else:
                splits_file = join(trainer.dataset_directory, "splits_final.pkl")
            split = load_pickle(splits_file)
            for k, v in split[self.fold]:
                spts += str(k) + ' := ' + ', '.join(str(v_) for v_ in v) + '\n'
            self.summary = print_to_log_file(self.summary, "For training, validation and evaluation of {} on fold {}, the used split can be found here: {}\nIt looks like the following:\n{}.".format(t, self.fold, p_splits_file, spts))

        # -- Return the information of the experiment so we can map this in the higher level function, the parameter searcher -- #
        return exp_id, glob.glob(join(output_path, 'fold_'+str(self.fold), '*val*.csv'))