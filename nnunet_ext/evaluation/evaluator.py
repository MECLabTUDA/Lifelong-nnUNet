#########################################################################################################
#--------------This class represents an Evaluator that can be used to evaluate a specific---------------#
#--------------trained model such as an extension but also a conventional nn-UNet Trainer.--------------#
#########################################################################################################

import numpy as np
import pandas as pd
import torch.nn as nn
import os, itertools, time
from collections import OrderedDict
from nnunet_ext.utilities.helpful_functions import *
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.training.model_restore import restore_model
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet_ext.paths import network_training_output_dir, evaluation_output_dir
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunet.paths import network_training_output_dir as network_training_output_dir_orig
from nnunet.run.default_configuration import get_default_configuration as get_default_configuration_orig

# -- Import the trainer classes -- #
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

class Evaluator():  # Do not inherit the one from the nnunet implementation since ours is different
    r"""Class that can be used to perform an Evaluation on any nnUNet related Trainer.
    """
    def __init__(self, network, network_trainer, tasks_list_with_char, model_list_with_char, version=1, vit_type='base',
                 plans_identifier=default_plans_identifier, mixed_precision=True, extension='multihead', save_csv=True,
                 transfer_heads=False, use_vit=False, use_param_split=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False):
        r"""Constructor for evaluator.
        """
        # -- Set all the relevant attributes -- #
        self.network = network
        self.save_csv = save_csv
        self.extension = extension
        self.param_split = use_param_split
        self.transfer_heads = transfer_heads
        self.network_trainer = network_trainer
        self.mixed_precision = mixed_precision
        self.plans_identifier = plans_identifier
        self.tasks_list_with_char = tasks_list_with_char
        self.model_list_with_char = model_list_with_char
        self.model_joined_name = join_texts_with_char(self.model_list_with_char[0], self.model_list_with_char[1])
        # -- Set tasks_joined_name for validation dataset building -- #
        self.tasks_joined_name = join_texts_with_char(self.tasks_list_with_char[0], self.tasks_list_with_char[1])
        
        # -- If ViT trainer, build the version correctly for finding the correct checkpoint later in restoring -- #
        # -- Set the desired network version -- #
        self.version = 'V' + str(version)
        
        # -- Create the variable indicating which ViT Architecture to use, base, large or huge and if to use it -- #
        self.LSA = do_LSA
        self.SPT = do_SPT
        self.use_vit = use_vit
        self.vit_type = vit_type.lower()
        self.ViT_task_specific_ln = ViT_task_specific_ln

    def reinitialize(self, network, network_trainer, tasks_list_with_char, model_list_with_char, version=1, vit_type='base',
                     plans_identifier=default_plans_identifier, mixed_precision=True, extension='multihead', save_csv=True,
                     transfer_heads=False, use_vit=False, use_param_split=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False):
        r"""This function changes the network and trainer so no new Evaluator needs to be created.
        """
        # -- Just perform initialization again -- #
        self.__init__(network, network_trainer, tasks_list_with_char, model_list_with_char, version, vit_type, plans_identifier,
                      mixed_precision, extension, save_csv, transfer_heads, use_vit, use_param_split, ViT_task_specific_ln,
                      do_LSA, do_SPT)

    def evaluate_on(self, folds, tasks, use_head=None, always_use_last_head=False, do_pod=True, enhanced=False,
                    trainer_path=None, output_path=None, use_all_data=False):
        r"""This function performs the actual evaluation given the transmitted tasks.
            :param folds: List of integer values specifying the folds on which the evaluation should be performed.
            :param tasks: List with tasks following the Task_XXX structure/name for direct loading.
            :param use_head: A task specifying which head to use --> if it is set to None, the last trained head will be used if necessary.
            :param always_use_last_head: Specifies if only the last head is used for the evaluation.
            :param do_pod: Specifies the POD embedding is used or not --> Only works for our own methods.
            :param enhanced: Specifies if the enhanced FrozEWC is used or not --> Only works for our nnUNetTrainerFrozEWC trainer.
            :param eval_mode_for_lns: Specifies how the evaluation is performed when using task specific LNs wrt to the LNs (last_lns or corr_lns).
            :param trainer_path: Specifies part to the trainer network including the fold_X being the last folder of the path (only used for parameter search method).
            :param output_path: Specifies part where the eval results are stored excluding the fold_X (only used for parameter search method).
            :param use_all_data: Specifies if the evaluation is also done on the training data.
        """
        # ---------------------------------------------
        # Evaluate for each task and all provided folds
        # ---------------------------------------------
        # -- Loop through folds so each fold will be trained in full before the next one will be started -- #
        for t_fold in folds:
            # -- Build the paths if they are not provided --> Only provide the paths if you know what you're doing .. -- #
            if trainer_path is None or output_path is None:
                # -- Build the trainer_path first -- #
                if 'nnUNetTrainerV2' in self.network_trainer:
                    trainer_path = join(network_training_output_dir_orig, self.network, self.tasks_joined_name, self.network_trainer+'__'+self.plans_identifier, 'fold_'+str(t_fold))
                    output_path = join(evaluation_output_dir, self.network, self.tasks_joined_name, self.network_trainer+'__'+self.plans_identifier)
                    output_path = output_path.replace('nnUNet_ext', 'nnUNet')
                else:
                    raise Exception

                # -- Re-Modify trainer path for own methods if necessary -- #
                if 'OwnM' in self.network_trainer:
                    trainer_path = join(os.path.sep, *trainer_path.split(os.path.sep)[:-1], 'pod' if do_pod else 'no_pod', 'fold_'+str(t_fold))
                    output_path = join(os.path.sep, *output_path.split(os.path.sep)[:-1], 'pod' if do_pod else 'no_pod', 'last_head' if always_use_last_head else 'corresponding_head')

            # -- Load the trainer for evaluation -- #
            print("Loading trainer and setting the network for evaluation")
            # -- Do not add the addition of fold_X to the path -- #
            checkpoint = join(trainer_path, "model_final_checkpoint.model")
            pkl_file = checkpoint + ".pkl"
            use_extension = not 'nnUNetTrainerV2' in trainer_path
            trainer = restore_model(pkl_file, checkpoint, train=False, fp16=self.mixed_precision,\
                                    use_extension=use_extension, extension_type=self.extension, del_log=True,\
                                    param_search=self.param_split, network=self.network)

            # -- If this is a conventional nn-Unet Trainer, then make a MultiHead Trainer out of it, so we can use the _perform_validation function -- #
            if not use_extension:
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
                
                # -- Build a simple MultiHead Trainer so we can use the perform validation function without re-coding it -- #
                trainer = nnUNetTrainerMultiHead('seg_outputs', self.tasks_list_with_char[0][0], plans_file, t_fold, output_folder=output_path,\
                                                 dataset_directory=dataset_directory, tasks_list_with_char=(self.tasks_list_with_char[0], self.tasks_list_with_char[1]),\
                                                 batch_dice=batch_dice, stage=stage, already_trained_on=None, use_param_split=self.param_split, network=self.network)
                # -- Remove the trained_on_path if it is empty -- #
                try:
                    os.rmdir(trainer.trained_on_path)
                except:
                    pass
                
                # -- Remove the Generic_UNet/MH part from the ouptput folder -- #
                if 'nnUNetTrainerV2' in trainer.output_folder:
                    fold_ = trainer.output_folder.split(os.path.sep)[-1]
                    trainer.output_folder = join(os.path.sep, *trainer.output_folder.split(os.path.sep)[:-3], fold_)
                
                trainer.initialize(False, num_epochs=0, prev_trainer_path=prev_trainer_path, call_for_eval=True)
                
                # -- Reset the epoch -- #
                trainer.epoch = epoch

            # -- Delete the current trainers log file since its in the wrong path -- #
            try:
                os.remove(trainer.log_file)
            except: # Either it does not exist or trainer.log_file is None
                pass
            trainer.log_file = None
            
            # -- Create the directory if it does not exist -- #
            maybe_mkdir_p(output_path)

            # -- Summarize those informations -- #
            model_sum = pd.DataFrame([], columns = ['network', 'total nr. of parameters', 'trainable parameters', 'model size [in MB]'])
            row = OrderedDict()

            # -- Get Trainer specifics like nr. of params, nr. of trainable params, size of model -- #
            # -- For one (ViT_)U-Net network -- #
            row['network'] = self.network_trainer
            row['total nr. of parameters'], row['trainable parameters'] = get_nr_parameters(trainer.network)
            row['model size [in MB]'] = round(get_model_size(trainer.network), 3)
            row_series = pd.Series(list(row.values()), index = model_sum.columns)
            model_sum = model_sum.append(row_series, ignore_index=True)

            # -- For the MH network -- #
            row['network'] = 'Multi Head Network with task specific heads'
            row['total nr. of parameters'], row['trainable parameters'] = get_nr_parameters(trainer.mh_network)
            row['model size [in MB]'] = round(get_model_size(trainer.mh_network), 3)
            row_series = pd.Series(list(row.values()), index = model_sum.columns)
            model_sum = model_sum.append(row_series, ignore_index=True)
            
            # -- Set trainer output path and set csv attribute as desired -- #
            if output_path not in trainer.output_folder:
                trainer.output_folder = join(output_path, 'fold_'+str(t_fold))
            trainer.csv = self.save_csv
            os.makedirs(trainer.output_folder, exist_ok=True)
                
            # -- Adapt the already_trained_on with only the prev_trainer part since this is necessary for the validation part -- #
            trainer.already_trained_on[str(t_fold)]['prev_trainer'] = [nnUNetTrainerMultiHead.__name__]*len(tasks)

            # -- Set the head based on the users input -- #
            if use_head is None:
                use_head = list(trainer.mh_network.heads.keys())[-1]
            
            # -- Create a new log_file in the evaluation folder based on changed output_folder -- #
            trainer.print_to_log_file("The {} model trained on {} will be used for this evaluation with the {} head.".format(self.network_trainer, ', '.join(self.tasks_list_with_char[0]), use_head))
            if use_all_data:
                trainer.print_to_log_file("Be aware that the training data is also used during validation as the flag has been set..")
            trainer.print_to_log_file("The used checkpoint can be found at {}.".format(join(trainer_path, "model_final_checkpoint.model")))
            trainer.print_to_log_file("Start performing evaluation on fold {} for the following tasks: {}.\n".format(t_fold, ', '.join(tasks)))
            start_time = time.time()

            # -- Delete all heads except the last one if it is a Sequential Trainer, since then always the last head should be used -- #
            if always_use_last_head:
                # -- Create new heads dict that only contains the last head -- #
                last_name = list(trainer.mh_network.heads.keys())[-1]
                last_head = trainer.mh_network.heads[last_name]
                trainer.mh_network.heads = nn.ModuleDict()
                trainer.mh_network.heads[last_name] = last_head

            # -- Run validation of the trainer while updating the head of the model based on the task/use_head -- #
            trainer._perform_validation(use_tasks=tasks, use_head=use_head, call_for_eval=True, param_search=self.param_split, use_all_data=use_all_data)

            # -- Update the log file -- #
            trainer.print_to_log_file("Finished with the evaluation on fold {}. The results can be found at: {} or {}.\n".format(t_fold, join(trainer.output_folder, 'tr_val_metrics.csv' if use_all_data else 'val_metrics.csv'),\
                                                                                                                                         join(trainer.output_folder, 'tr_val_metrics_eval.json' if use_all_data else 'val_metrics_eval.json')))

            # -- Update the log file -- #
            trainer.print_to_log_file("Summarizing the results (calculate mean and std)..")
            # -- Load the validation_metrics -- #
            data = pd.read_csv(join(trainer.output_folder, 'tr_val_metrics_eval.csv' if use_all_data else 'val_metrics_eval.csv'), sep = '\t')
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
            output_file = join(trainer.output_folder, 'summarized_tr_val_metrics.txt' if use_all_data else 'summarized_val_metrics.txt')
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

            # -- Store the summarized .csv files -- #
            dumpDataFrameToCsv(summary, trainer.output_folder, 'summarized_tr_val_metrics.csv' if use_all_data else 'summarized_val_metrics.csv')
            dumpDataFrameToCsv(model_sum, trainer.output_folder, 'model_summary.csv')

            # -- Update the log file -- #
            trainer.print_to_log_file("The summarized results of the evaluation on fold {} can be found at: {} or {}.\n\n".format(t_fold, output_file, join(trainer.output_folder, 'summarized_tr_val_metrics.csv' if use_all_data else 'summarized_val_metrics.csv')))
            trainer.print_to_log_file("The Evaluation took %.2f seconds." % (time.time() - start_time))