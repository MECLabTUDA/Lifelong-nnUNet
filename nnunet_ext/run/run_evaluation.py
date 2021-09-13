#########################################################################################################
#----------This class represents the Evaluation of networks using the extended nnUNet trainer-----------#
#----------                                     version.                                     -----------#
#########################################################################################################

import itertools
import numpy as np
import os, argparse
import pandas as pd
from time import time
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.training.model_restore import restore_model
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_ext.paths import network_training_output_dir, evaluation_output_dir
from nnunet.paths import network_training_output_dir as network_training_output_dir_orig

# -- Import the trainer classes -- #
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead # Own implemented class


#------------------------------------------- Inspired by original implementation -------------------------------------------#
def run_evaluation():
    # -- First of all check that evaluation_output_dir is set otherwise we do not perform an evaluation -- #
    assert evaluation_output_dir is not None, "Before running any evaluation, please specify the Evaluation folder (EVALUATION_FOLDER) as described in the paths.md."

    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")  # Can only be a multi head, sequential, rehearsal, ewc or lwf
    
    # -- nnUNet arguments untouched --> Should not intervene with sequential code, everything should work -- #
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    
    # -- Additional arguments specific for multi head training -- #
    parser.add_argument("-f", "--folds", nargs="+", help="Specify on which folds to train on. Use a fold between 0, 1, ..., 5 or \'all\'", required=True)
    parser.add_argument("-trained_on", nargs="+", help="Specify a list of task ids the network has trained with to specify the correct path to the networks. "
                                                       "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                                                       "data folder", required=True)
    parser.add_argument("-use_model", "--use", nargs="+", help="Specify a list of task ids that specify the exact network that should be used for evaluation. "
                                                       "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                                                       "data folder", required=True)
    parser.add_argument("--fp32_used", required=False, default=False, action="store_true",
                        help="Specify is mixed precision has been used during training or not")
    parser.add_argument("-evaluate_on", nargs="+", help="Specify a list of task ids the network will be evaluated on. "
                                                        "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                                                        "data folder", required=True)
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=[0],
                        help='Try to train the model on the GPU device with <DEVICE> ID. '+
                            ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.'+
                            ' Default: Only GPU device with ID 0 will be used.')
    parser.add_argument('-store_csv', required=False, default=False, action="store_true",
                        help='Set this flag if the validation data and any other data if applicable should be stored'
                            ' as a .csv file as well. Default: .csv are not created.')

    # -- Build mapping for network_trainer to corresponding extension name -- #
    ext_map = {'nnUNetTrainerMultiHead': 'multihead', 'nnUNetTrainerSequential': 'sequential',
               'nnUNetTrainerRehearsal': 'rehearsal', 'nnUNetTrainerEWC': 'ewc', 'nnUNetTrainerLWF': 'lwf',
               'nnUNetTrainerV2': 'standard'}


    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet) arguments -- #
    args = parser.parse_args()
    network = args.network
    network_trainer = args.network_trainer
    plans_identifier = args.p
    
    # -- Extract the arguments specific for all trainers from argument parser -- #
    trained_on = args.trained_on    # List of the tasks that helps to navigate to the correct folder, eg. A B C
    use_model = args.use            # List of the tasks representing the network to use, e. use A B from folder A B C
    evaluate_on = args.evaluate_on  # List of the tasks that should be used to evaluate the model
    
    # -- Extract further arguments -- #
    save_csv = args.store_csv
    fold = args.folds
    cuda = args.device
    mixed_precision = not args.fp32_used

    # -- Assert if device value is ot of predefined range and create string to set cuda devices -- #
    for idx, c in enumerate(cuda):
        assert c > -1 and c < 8, 'GPU device ID out of range (0, ..., 7).'
        cuda[idx] = str(c)  # Change type from int to str otherwise join_texts_with_char will throw an error
    cuda = join_texts_with_char(cuda, ',')
    
    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    
    # -------------------------------
    # Transform tasks to task names
    # -------------------------------
    # -- Transform fold to list if it is set to 'all'
    if fold[0] == 'all':
        fold = list(range(6))
    else: # change each fold type from str to int
        fold = list(map(int, fold))

    # -- Assert if fold is not a number or a list as desired, meaning anything else, like Tuple or whatever -- #
    assert isinstance(fold, (int, list)), "To Evaluate multiple tasks with {} trainer, only one or multiple folds specified as integers are allowed..".format(network_trainer)

    # -- Build all necessary task lists -- #
    tasks_for_folder = list()
    use_model_w_tasks = list()
    evaluate_on_tasks = list()
    for idx, t in enumerate(trained_on):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        tasks_for_folder.append(t)
    for idx, t in enumerate(use_model):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        use_model_w_tasks.append(t)
    for idx, t in enumerate(evaluate_on):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        evaluate_on_tasks.append(t)


    # ----------------------------------------------
    # Define dict with arguments for function calls
    # ----------------------------------------------
    # -- Join all task names together with a '_' in between them -- #
    char_to_join_tasks = '_'
    tasks_joined_name = join_texts_with_char(tasks_for_folder, char_to_join_tasks)
    model_joined_name = join_texts_with_char(use_model_w_tasks, char_to_join_tasks)

    
    # ---------------------------------------------
    # Evaluate for each task and all provided folds
    # ---------------------------------------------
    # -- Loop through folds so each fold will be trained in full before the next one will be started -- #
    for t_fold in fold:
        # -- Build the trainer_path first -- #
        if 'nnUNetTrainerV2' in network_trainer:
            trainer_path = join(network_training_output_dir_orig, network, tasks_joined_name, network_trainer+'__'+plans_identifier, 'fold_'+str(t_fold))
            output_path = join(evaluation_output_dir, network, tasks_joined_name, network_trainer+'__'+plans_identifier)
            output_path = output_path.replace('nnUNet_ext', 'nnUNet')
        else:
            trainer_path = join(network_training_output_dir, network, tasks_joined_name, model_joined_name, network_trainer+'__'+plans_identifier, 'fold_'+str(t_fold))
            output_path = join(evaluation_output_dir, network, tasks_joined_name, model_joined_name, network_trainer+'__'+plans_identifier)

        # -- Create the directory if it does not exist -- #
        maybe_mkdir_p(output_path)

        # -- Load the trainer for evaluation -- #
        print("Loading trainer and setting the network for evaluation")
        # -- Do not add the addition of fold_X to the path -- #
        checkpoint = join(trainer_path, "model_final_checkpoint.model")
        pkl_file = checkpoint + ".pkl"
        use_extension = not 'nnUNetTrainerV2' in trainer_path
        trainer = restore_model(pkl_file, checkpoint, train=False, fp16=mixed_precision,\
                                use_extension=use_extension, extension_type=ext_map[network_trainer]) 
        trainer.initialize(False)
        
        # -- Delete the created log_file from the training folder and set it to None -- #
        os.remove(trainer.log_file)
        trainer.log_file = None

        # -- If this is a conventional nn-Unet Trainer, than make a MultiHead Trainer out of it, so we can use the _perform_validation function -- #
        if 'nnUNetTrainerV2' in trainer_path:
            # -- Ensure that use_model only contains one task for the conventional Trainer -- #
            assert len(tasks_for_folder) == 1, "When trained with {}, only one task could have been used for training, not {} since this is no extension.".format(network_trainer, len(use_model))
            # -- Store the epoch of the trainer to set it correct after initialization of the MultiHead Trainer -- #
            epoch = trainer.epoch
            # -- Extract the necessary information of the current Trainer to build a MultiHead Trainer -- #
            # -- NOTE: We can use the get_default_configuration from nnunet and not nnunet_ext since the information -- #
            # --       we need can be extracted from there as well without as much 'knowledge' since we do not need -- #
            # --       everything for a MultiHead Trainer -- #
            plans_file, prev_trainer_path, dataset_directory, batch_dice, stage, \
            _ = get_default_configuration(network, tasks_for_folder[0], network_trainer, plans_identifier)
            # -- Build a simple MultiHead Trainer so we can use the perform validation function without re-coding it -- #
            trainer = nnUNetTrainerMultiHead('seg_outputs', tasks_for_folder[0], plans_file, t_fold, output_folder=output_path,\
                                             dataset_directory=dataset_directory, tasks_list_with_char=(tasks_for_folder, char_to_join_tasks),\
                                             batch_dice=batch_dice, stage=stage, already_trained_on=None)
            trainer.initialize(False, num_epochs=0, prev_trainer_path=prev_trainer_path)
            # -- Adapt the already_trained_on with only the prev_trainer part since this is necessary for the validatio part -- #
            trainer.already_trained_on[str(t_fold)]['prev_trainer'] = [nnUNetTrainerMultiHead.__name__]*len(evaluate_on_tasks)
            # -- Reset the epoch -- #
            trainer.epoch = epoch

        # -- Set trainer output path to evaluation folder and set csv attribute as desired -- #
        trainer.output_folder = join(output_path, 'fold_'+str(t_fold))
        trainer.csv = save_csv

        # -- Assemble the last task from the head since this is the last trained model which is the one specified through use_model!!!
        trainer.network = trainer.mh_network.assemble_model(list(trainer.mh_network.heads.keys())[-1])
        
        # -- Create a new log_file in the evaluation folder based on changed output_folder -- #
        trainer.print_to_log_file("The {} model trained on {} will be used for this evaluation.".format(network_trainer, ', '.join(tasks_for_folder)))
        trainer.print_to_log_file("The used checkpoint can be found at {}.".format(join(trainer_path, "model_final_checkpoint.model")))
        trainer.print_to_log_file("Start performing evaluation on fold {} for the following tasks: {}.\n".format(t_fold, ', '.join(evaluate_on_tasks)))
        start_time = time()

        # -- Run validation of the trainer using evaluate_on_tasks if the trainer is one of the extensions -- #
        # -- use_extension indicates if this model is an extension or not, so is the same as multihead in the function call -- #
        trainer._perform_validation(use_tasks=evaluate_on_tasks, call_for_eval=True, multihead=use_extension)

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
        # -- Define the path for the summary file -- #
        output_file = join(trainer.output_folder, 'summarized_val_metrics.txt')
        # -- Loop through the data and calculate the mean and std values -- #
        with open(output_file, 'w') as out:
            out.write('Evaluation performed after Epoch {}, trained on fold {}.\n'.format(trainer.epoch, t_fold))
            out.write("The {} model trained on {} has been used for this evaluation. ".format(network_trainer, ', '.join(tasks_for_folder)))
            out.write("The used checkpoint can be found at {}.\n".format(join(trainer_path, "model_final_checkpoint.model")))
            # -- Calculate mean and std values -- #
            for combi in itertools.product(eval_tasks, eval_metrics, eval_masks): # --> There will never be duplicate combinations
                mean = np.mean(data.loc[(data['Task'] == combi[0]) & (data['metric'] == combi[1]) & (data['seg_mask'] == combi[2])]['value'])
                std = np.std(data.loc[(data['Task'] == combi[0]) & (data['metric'] == combi[1]) & (data['seg_mask'] == combi[2])]['value'])
                # -- Write the results into the file -- #
                out.write("Evaluation performed for fold {}, task {} using segmentation mask {} and {} as metric:\n".format(t_fold, combi[0], combi[2].split('_')[-1], combi[1]))
                out.write("mean (+/- std):\t {} +/- {}\n\n".format(mean, std))

        # -- Update the log file -- #
        trainer.print_to_log_file("The summarized results of the evaluation on fold {} can be found at: {}.\n\n".format(t_fold, join(trainer.output_folder, 'summarized_val_metrics.txt')))
        trainer.print_to_log_file("The Evaluation took %.2f seconds." % (time() - start_time))

    # -- Reset cuda device as environment variable, otherwise other GPUs might not be available ! -- #
    del os.environ["CUDA_VISIBLE_DEVICES"]
#------------------------------------------- Inspired by original implementation -------------------------------------------#


if __name__ == "__main__":
    run_evaluation()