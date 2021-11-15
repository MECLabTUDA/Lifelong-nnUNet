#########################################################################################################
#--------------This class represents an Evaluator that can be used to evaluate a specific---------------#
#--------------trained model such as an extension but also a conventional nn-UNet Trainer.--------------#
#########################################################################################################

import numpy as np
import pandas as pd
import os, itertools
from time import time
import torch.nn as nn
from collections import OrderedDict
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.training.model_restore import restore_model
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet_ext.paths import network_training_output_dir, evaluation_output_dir
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunet.paths import network_training_output_dir as network_training_output_dir_orig
from nnunet_ext.utilities.helpful_functions import join_texts_with_char, dumpDataFrameToCsv
from nnunet.run.default_configuration import get_default_configuration as get_default_configuration_orig

# -- Import the trainer classes -- #
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential

class Evaluator():  # Do not inherit the one from the nnunet implementation since ours is different
    r"""Class that can be used to perform an Evaluation on any nnUNet related Trainer.
    """
    def __init__(self, network, network_trainer, tasks_list_with_char, model_list_with_char, version=1, vit_type='base',
                 plans_identifier=default_plans_identifier, mixed_precision=True, extension='multihead', save_csv=True):
        r"""Constructor for evaluator.
        """
        # -- Set all the relevant attributes -- #
        self.network = network
        self.network_trainer = network_trainer
        self.tasks_list_with_char = tasks_list_with_char
        self.model_list_with_char = model_list_with_char
        self.model_joined_name = join_texts_with_char(self.model_list_with_char[0], self.model_list_with_char[1])
        self.plans_identifier = plans_identifier
        self.mixed_precision = mixed_precision
        self.extension = extension
        self.save_csv = save_csv
        # -- Set tasks_joined_name for validation dataset building -- #
        self.tasks_joined_name = join_texts_with_char(self.tasks_list_with_char[0], self.tasks_list_with_char[1])
        # -- If ViT trainer, build the version correctly for finding the correct checkpoint later in restoring -- #
        if nnViTUNetTrainer.__name__ in self.network_trainer:
            # -- Set the desired network version -- #
            self.version = 'V' + str(version)
        
        # -- Create the variable indicating which ViT Architecture to use, base, large or huge -- #
        self.vit_type = vit_type.lower()

    def reinitialize(self, network, network_trainer, tasks_list_with_char, model_list_with_char,
                     plans_identifier=default_plans_identifier, mixed_precision=True, extension='multihead', save_csv=True):
        r"""This function changes the network and trainer so no new Evaluator needs to be created.
        """
        # -- Just perform initialization again -- #
        self.__init__(network, network_trainer, tasks_list_with_char, model_list_with_char, plans_identifier,
                      mixed_precision, extension, save_csv)

    def evaluate_on(self, folds, tasks, use_head=None):
        r"""This function performs the actual evaluation given the transmitted tasks.
            :param folds: List of integer values specifying the folds on which the evaluation should be performed.
            :param tasks: List with tasks following the Task_XXX structure/name for direct loading.
            :param use_head: A task specifying which head to use --> if it is set to None, the last trained head will be used if necessary.
        """
        # ---------------------------------------------
        # Evaluate for each task and all provided folds
        # ---------------------------------------------
        # -- Loop through folds so each fold will be trained in full before the next one will be started -- #
        for t_fold in folds:
            # -- Build the trainer_path first -- #
            if 'nnUNetTrainerV2' in self.network_trainer:
                trainer_path = join(network_training_output_dir_orig, self.network, self.tasks_joined_name, self.network_trainer+'__'+self.plans_identifier, 'fold_'+str(t_fold))
                output_path = join(evaluation_output_dir, self.network, self.tasks_joined_name, self.network_trainer+'__'+self.plans_identifier)
                output_path = output_path.replace('nnUNet_ext', 'nnUNet')
            elif nnViTUNetTrainer.__name__ in self.network_trainer:
                trainer_path = join(network_training_output_dir, self.network, self.tasks_joined_name, self.network_trainer+'__'+self.plans_identifier, self.vit_type, 'fold_'+str(t_fold))
                output_path = join(evaluation_output_dir, self.network, self.tasks_joined_name, self.network_trainer+'__'+self.plans_identifier, self.vit_type)
                trainer_path = trainer_path.replace(nnViTUNetTrainer.__name__, nnViTUNetTrainer.__name__+self.version)
                output_path = output_path.replace(nnViTUNetTrainer.__name__, nnViTUNetTrainer.__name__+self.version)
            else:   # Any other extension like CL extension for example (using MH Architecture)
                trainer_path = join(network_training_output_dir, self.network, self.tasks_joined_name, self.model_joined_name, self.network_trainer+'__'+self.plans_identifier, 'fold_'+str(t_fold))
                output_path = join(evaluation_output_dir, self.network, self.tasks_joined_name, self.model_joined_name, self.network_trainer+'__'+self.plans_identifier)

            # -- Create the directory if it does not exist -- #
            maybe_mkdir_p(output_path)

            # -- Load the trainer for evaluation -- #
            print("Loading trainer and setting the network for evaluation")
            # -- Do not add the addition of fold_X to the path -- #
            checkpoint = join(trainer_path, "model_final_checkpoint.model")
            pkl_file = checkpoint + ".pkl"
            use_extension = not 'nnUNetTrainerV2' in trainer_path
            trainer = restore_model(pkl_file, checkpoint, train=False, fp16=self.mixed_precision,\
                                    use_extension=use_extension, extension_type=self.extension, del_log=True)
            trainer.initialize(False)
            
            # -- Delete the created log_file from the training folder and set it to None -- #
            os.remove(trainer.log_file)
            trainer.log_file = None

            # -- If this is a conventional nn-Unet Trainer, then make a MultiHead Trainer out of it, so we can use the _perform_validation function -- #
            if not use_extension or nnViTUNetTrainer.__name__ in trainer_path:
                # -- Ensure that use_model only contains one task for the conventional Trainer -- #
                assert len(self.tasks_list_with_char[0]) == 1, "When trained with {}, only one task could have been used for training, not {} since this is no extension.".format(self.network_trainer, len(self.use_model))
                # -- Store the epoch of the trainer to set it correct after initialization of the MultiHead Trainer -- #
                epoch = trainer.epoch
                # -- Extract the necessary information of the current Trainer to build a MultiHead Trainer -- #
                # -- NOTE: We can use the get_default_configuration from nnunet and not nnunet_ext since the information -- #
                # --       we need can be extracted from there as well without as much 'knowledge' since we do not need -- #
                # --       everything for a MultiHead Trainer -- #
                if 'nnUNetTrainerV2' in trainer_path:
                    plans_file, prev_trainer_path, dataset_directory, batch_dice, stage, \
                    _ = get_default_configuration_orig(self.network, self.tasks_list_with_char[0][0], self.network_trainer, self.plans_identifier)
                else:   # ViT_U-Net
                    plans_file, prev_trainer_path, dataset_directory, batch_dice, stage, \
                    _ = get_default_configuration(self.network, self.tasks_list_with_char[0][0], None, self.network_trainer, None,
                                                  self.plans_identifier, extension_type=None)
                    # -- Modify prev_trainer_path based on desired version and ViT type -- #
                    if not nnViTUNetTrainer.__name__+'V' in prev_trainer_path:
                        prev_trainer_path = prev_trainer_path.replace(nnViTUNetTrainer.__name__, nnViTUNetTrainer.__name__+self.version)
                    if self.vit_type != prev_trainer_path.split(os.path.sep)[-1] and self.vit_type not in prev_trainer_path:
                        prev_trainer_path = os.path.join(prev_trainer_path, self.vit_type)

                # -- Build a simple MultiHead Trainer so we can use the perform validation function without re-coding it -- #
                trainer = nnUNetTrainerMultiHead('seg_outputs', self.tasks_list_with_char[0][0], plans_file, t_fold, output_folder=output_path,\
                                                dataset_directory=dataset_directory, tasks_list_with_char=(self.tasks_list_with_char[0], self.tasks_list_with_char[1]),\
                                                batch_dice=batch_dice, stage=stage, already_trained_on=None)
                trainer.initialize(False, num_epochs=0, prev_trainer_path=prev_trainer_path)
                # -- Reset the epoch -- #
                trainer.epoch = epoch

            # -- Set trainer output path to evaluation folder and set csv attribute as desired -- #
            trainer.output_folder = join(output_path, 'fold_'+str(t_fold))
            trainer.csv = self.save_csv
                
            # -- Adapt the already_trained_on with only the prev_trainer part since this is necessary for the validation part -- #
            trainer.already_trained_on[str(t_fold)]['prev_trainer'] = [nnUNetTrainerMultiHead.__name__]*len(tasks)
                
            # -- Set the head based on the users input -- #
            if use_head is None:
                use_head = list(trainer.mh_network.heads.keys())[-1]
            
            # -- Create a new log_file in the evaluation folder based on changed output_folder -- #
            trainer.print_to_log_file("The {} model trained on {} will be used for this evaluation with the {} head.".format(self.network_trainer, ', '.join(self.tasks_list_with_char[0]), use_head))
            trainer.print_to_log_file("The used checkpoint can be found at {}.".format(join(trainer_path, "model_final_checkpoint.model")))
            trainer.print_to_log_file("Start performing evaluation on fold {} for the following tasks: {}.\n".format(t_fold, ', '.join(tasks)))
            start_time = time()

            # -- Delete all heads except the last one if it is a Sequential Trainer, since then always the last head should be used -- #
            if nnUNetTrainerSequential.__name__ in self.network_trainer:
                # -- Create new heads dict that only contains the last head -- #
                last_name = list(trainer.mh_network.heads.keys())[-1]
                last_head = trainer.mh_network.heads[last_name]
                trainer.mh_network.heads = nn.ModuleDict()
                trainer.mh_network.heads[last_name] = last_head

            # -- Run validation of the trainer while updating the head of the model based on the task/use_head -- #
            trainer._perform_validation(use_tasks=tasks, use_head=use_head, call_for_eval=True)

            # -- Update the log file -- #
            trainer.print_to_log_file("Finished with the evaluation on fold {}. The results can be found at: {} or {}.\n".format(t_fold, join(trainer.output_folder, 'val_metrics.csv'), join(trainer.output_folder, 'val_metrics.json')))
            
            # -- Update the log file -- #
            trainer.print_to_log_file("Summarizing the results (calculate mean and std)..")
            # -- Load the validation_metrics -- #
            data = pd.read_csv(join(trainer.output_folder, 'val_metrics.csv'), sep = '\t')
            # -- Calculate the mean and std values for all tasks per masks and metrics over all subjects -- #
            # -- Extract all relevant information like tasks, metrics and seg_masks -- #
            eval_tasks = data['Task'].unique()
            eval_metrics = data['metric'].unique()
            eval_masks = data['seg_mask'].unique()
            # -- Define dataframe to summarize the resultss -- #
            summary = pd.DataFrame([], columns = ['trainer', 'network', 'overall train order', 'trained on', 'trained on fold', 'used head for eval', 'eval on task', 'metric', 'seg mask', 'mean +/- std', 'mean +/- std [in %]', 'checkpoint'])
            row = OrderedDict()
            # -- Add values -- #
            row['trainer'] = self.network_trainer
            row['network'] = self.network
            row['overall train order'] = ' -- '.join(self.tasks_list_with_char[0])
            row['trained on'] = ' -- '.join(self.model_list_with_char[0])
            row['trained on fold'] = t_fold
            row['used head for eval'] = None
            row['eval on task'] = None
            row['metric'] = None
            row['seg mask'] = None
            row['mean +/- std'] = None
            row['mean +/- std [in %]'] = None
            row['checkpoint'] = join(trainer_path, "model_final_checkpoint.model")
            # -- Define the path for the summary file -- #
            output_file = join(trainer.output_folder, 'summarized_val_metrics.txt')
            # -- Loop through the data and calculate the mean and std values -- #
            with open(output_file, 'w') as out:
                out.write('Evaluation performed after Epoch {}, trained on fold {}.\n\n'.format(trainer.epoch, t_fold))
                out.write("The {} model trained on {} has been used for this evaluation with the {} head. ".format(self.network_trainer, ', '.join(self.tasks_list_with_char[0]), use_head))
                out.write("The used checkpoint can be found at {}.\n\n".format(join(trainer_path, "model_final_checkpoint.model")))
                # -- Calculate mean and std values -- #
                for combi in itertools.product(eval_tasks, eval_metrics, eval_masks): # --> There will never be duplicate combinations
                    mean = np.mean(data.loc[(data['Task'] == combi[0]) & (data['metric'] == combi[1]) & (data['seg_mask'] == combi[2])]['value'])
                    std = np.std(data.loc[(data['Task'] == combi[0]) & (data['metric'] == combi[1]) & (data['seg_mask'] == combi[2])]['value'])
                    # -- Write the results into the file -- #
                    out.write("Evaluation performed for fold {}, task {} using segmentation mask {} and {} as metric:\n".format(t_fold, combi[0], combi[2].split('_')[-1], combi[1]))
                    out.write("mean (+/- std):\t {} +/- {}\n\n".format(mean, std))
                    # -- Update the row values -- #
                    if combi[0] in trainer.mh_network.heads:
                        row['used head for eval'] = combi[0]
                    else:
                        row['used head for eval'] = use_head
                    row['eval on task'] = combi[0]
                    row['metric'] = combi[1]
                    row['seg mask'] = combi[2]
                    row['mean +/- std'] = '{:0.4f} +/- {:0.4f}'.format(mean, std)
                    row['mean +/- std [in %]'] = '{:0.2f}% +/- {:0.2f}%'.format(100*mean, 100*std)
                    row_series = pd.Series(list(row.values()), index = summary.columns)
                    summary = summary.append(row_series, ignore_index=True)

            # -- Store the summarized .csv file -- #
            dumpDataFrameToCsv(summary, trainer.output_folder, 'summarized_val_metrics.csv')

            # -- Update the log file -- #
            trainer.print_to_log_file("The summarized results of the evaluation on fold {} can be found at: {} or {}.\n\n".format(t_fold, output_file, join(trainer.output_folder, 'summarized_val_metrics.csv')))
            trainer.print_to_log_file("The Evaluation took %.2f seconds." % (time() - start_time))