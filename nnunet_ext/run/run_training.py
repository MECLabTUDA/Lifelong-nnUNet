#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet sequential ---------#
#----------training version. Implementation inspired by original implementation.------------------------#
#########################################################################################################

import copy
import numpy as np
import os, argparse
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.paths import network_training_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_ext.utilities.helpful_functions import delete_dir_con, join_texts_with_char, move_dir

# -- Import the trainer classes -- #
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.training.network_training.ewc.nnUNetTrainerEWC import nnUNetTrainerEWC # Own implemented class
from nnunet_ext.training.network_training.lwf.nnUNetTrainerLWF import nnUNetTrainerLWF # Own implemented class
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead # Own implemented class
from nnunet_ext.training.network_training.rehearsal.nnUNetTrainerRehearsal import nnUNetTrainerRehearsal # Own implemented class
#from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential # Own implemented class


#------------------------------------------- Inspired by original implementation -------------------------------------------#
def run_training(extension='multihead'):
    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")  # Can only be a multi head, sequential, rehearsal, ewc or lwf
    
    # -- nnUNet arguments untouched --> Should not intervene with sequential code, everything should work -- #
    parser.add_argument("-val", "--validation_only", help="Use this if you want to only run the validation. This will validate each model "
                                                          "of the sequential pipeline, so they should have been saved and not deleted.",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="Use this if you want to continue a training. The program will determine "
                                                          "by itself where to continue with the learning, so provide all tasks.",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    # -- Additional arguments specific for multi head training -- #
    parser.add_argument("-t", "--task_ids", nargs="+", help="Specify a list of task ids to train on (ids or names). Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder", required=True)
    parser.add_argument("-f", "--folds", nargs="+", help="Specify on which folds to train on. Use a fold between 0, 1, ..., 5 or \'all\'", required=True)
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=[0],
                        help='Try to train the model on the GPU device with <DEVICE> ID. '+
                            ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.'+
                            ' Default: Only GPU device with ID 0 will be used.')
    parser.add_argument("-s", "--split_at", action='store', type=str, nargs=1, required=True,
                        help='Specify the path in the network in which the split will be performed. '+
                            ' Use a single string for it, specify between layers using a dot \'.\' notation.'+
                            ' This is a required field and no default will be set. Use the same names as'+
                            ' present in the desired network architecture.')
    parser.add_argument('-num_epochs', action='store', type=int, nargs=1, required=False, default=500,
                        help='Specify the number of epochs to train the model.'
                            ' Default: Train for 500 epochs.')
    parser.add_argument('-save_interval', action='store', type=int, nargs=1, required=False, default=25,
                        help='Specify after which epoch interval to update the saved data.'
                            ' Default: If disable_saving False, the result will be updated every 25th epoch.')
    parser.add_argument('-store_csv', required=False, default=False, action="store_true",
                        help='Set this flag if the validation data and any other data if applicable should be stored'
                            ' as a .csv file as well. Default: .csv are not created.')
    parser.add_argument('--init_seq', action='store_true', default=False,
                        help='Specify if the first task from -t is already trained and represents '
                             ' an init network_trainer to do (extensional) training on or not. If so, -initialize_with_network_trainer '
                             ' needs to be provided as well.'
                             ' Default: False')
    parser.add_argument('-initialize_with_network_trainer', type=str, required=False, default=None,
                        help='Specify the network_trainer that should be used as a foundation to start training sequentially.'
                            ' The network_trainer of the first provided task needs to be finished with training and either a (extensional) network_trainer'
                            ' or a standard nnUNetTrainer. Default: None.')
    parser.add_argument('-used_identifier_in_init_network_trainer', type=str, required=False, default=None,
                        help='Specify the identifier that should be used for the network_trainer that is used as a foundation to start training sequentially.'
                            ' Default: default_plans_identifier from paths.py (nnUNetPlansv2.1).')
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space. Further for sequential tasks "
                             "the intermediate model won't be saved then, remeber that.")
    
    # -- Add arguments for rehearsal method -- #
    if extension == 'rehearsal':
        parser.add_argument('-seed', action='store', type=int, nargs=1, required=False, default=3299,
                            help='Specify the seed with which the samples will be selected for building the dataset.'
                                ' Default: seed will be set to 3299. --> If -c is set, this will be omitted,'
                                ' since the seed should not be changed between training.')
        parser.add_argument('-samples_in_perc', action='store', type=float, nargs=1, required=False, default=0.25,
                            help='Specify how much of the previous tasks should be considered during training.'
                                ' The number should be between 0 and 1 specifying the percentage that will be considered.'
                                ' This percentage is used for each previous task individually.'
                                ' Default: 0.25, ie. 25% of each previous task will be considered.')
    
    # -- Add arguments for ewc method -- #
    if extension == 'ewc':
        parser.add_argument('-ewc_lambda', action='store', type=float, nargs=1, required=False, default=0.4,
                            help='Specify the importance of the previous tasks for the EWC method.'
                                ' This number represents the lambda value in the loss function calculation as proposed in the paper.'
                                ' Default: ewc_lambda = 0.4')
    
    # -- Add arguments for ewc method -- #
    if extension == 'lwf':
        parser.add_argument('-lwf_temperature', action='store', type=float, nargs=1, required=False, default=2.0,
                            help='Specify the temperature variable for the LWF method.'
                                ' Default: lwf_temperature = 2.0')

    # -- Build mapping for extension to corresponding class -- #
    ext_map = {'standard': nnUNetTrainerV2, 'nnUNetTrainerV2': nnUNetTrainerV2,
               'multihead': nnUNetTrainerMultiHead, 'nnUNetTrainerMultiHead': nnUNetTrainerMultiHead,
               #'sequential': nnUNetTrainerSequential, 'nnUNetTrainerSequential': nnUNetTrainerSequential,
               'rehearsal': nnUNetTrainerRehearsal, 'nnUNetTrainerRehearsal': nnUNetTrainerRehearsal,
               'ewc': nnUNetTrainerEWC, 'nnUNetTrainerEWC': nnUNetTrainerEWC,
               'lwf': nnUNetTrainerLWF, 'nnUNetTrainerLWF': nnUNetTrainerLWF}


    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet) arguments -- #
    args = parser.parse_args()
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds
    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data
    deterministic = args.deterministic
    valbest = args.valbest
    fp32 = args.fp32
    run_mixed_precision = not fp32
    val_folder = args.val_folder
    continue_training = args.continue_training

    # -- Extract the arguments specific for all trainers from argument parser -- #
    task = args.task_ids    # List of the tasks
    fold = args.folds       # List of the folds
    split = args.split_at   # String that specifies the path to the layer where the split needs to be done
    
    if isinstance(split, list):    # When the split get returned as a list, extract the path to avoid later appearing errors
        split = split[0].strip()
    
    # -- Check that split is string and not empty -- #
    assert len(split) > 0,\
        "When providing a split, ensure that it is not empty, otheriwse no split can be performed."
    # -- Check that the number of tasks is greater than 1, else a conventional nnUNetTrainerV2 should be used -- #
    assert len(task) > 1,\
        "When training on only one task, the conventional training of the nnU-Net should be used, not the extension."
    
    num_epochs = args.num_epochs    # The number of epochs to train for each task
    if isinstance(num_epochs, list):    # When the num_epochs get returned as a list, extract the number to avoid later appearing errors
        num_epochs = num_epochs[0]
    init_seq = args.init_seq    # Indicates if the first task is an initialization. If True, the training start from 0 and will not be based on a provided model
    
    # -- Extract necessary information for initilization -- #
    prev_trainer = args.initialize_with_network_trainer # Trainer for first task with first fold
    if init_seq:
        # -- Check that prev_trainer is provided -- #
        assert prev_trainer is not None,\
            "When using the first provided task as a base for training with an extension, the network_trainer needs to be specified as well."
    
    # -- Set init_identifier if not provided -- #
    init_identifier = args.used_identifier_in_init_network_trainer  # Extract init_identifyer
    if init_identifier is None:
        init_identifier = default_plans_identifier  # Set to init_identifier
    disable_saving = args.disable_saving
    save_interval = args.save_interval
    save_csv = args.store_csv
    if isinstance(save_interval, list):    # When the save_interval gets returned as a list, extract the number to avoid later appearing errors
        save_interval = save_interval[0]
    cuda = args.device

    # -- Assert if device value is ot of predefined range and create string to set cuda devices -- #
    for idx, c in enumerate(cuda):
        assert c > -1 and c < 8, 'GPU device ID out of range (0, ..., 7).'
        cuda[idx] = str(c)  # Change type from int to str otherwise join_texts_with_char will throw an error
    cuda = join_texts_with_char(cuda, ',')
    
    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    # -- Check that given the extension the right network has been provided -- #
    assert network_trainer == str(ext_map[extension]).split('.')[-1][:-2],\
        "When training {}, the network is called \'{}\' and should be used, nothing else.".format(extension, str(ext_map[extension]).split('.')[-1][:-2])

    # -- Extract rehearsal arguments -- #
    seed, samples = None, None  # --> So the dictionary arguments can be build without an error even if not rehearsal desired ..
    if extension == 'rehearsal':
        # -- Extract the seed and samples_in_perc -- #
        seed = args.seed
        samples = args.samples_in_perc

        # -- If seed and samples are lists extract the desired element to prevent possible errors -- #
        if isinstance(seed, list):
            seed = seed[0]
        if isinstance(samples, list):
            samples = samples[0]

        # -- Check that samples is between 0 and 1 and really greater than 0 -- #
        assert samples != 0, "Instead of setting samples_in_perc to 0 use the provided Multi Head (single task) Trainer."
        assert samples > 0 and samples <= 1, "Your provided samples_in_perc is not in the specified range of (0, 1]."

        # -- Notify the user that the seed should not have been changed if -c is activated -- #
        if continue_training:
            print("Note: It will be continued with previous training, be sure that the provided seed has not "
                  "changed from previous one, because this will change the datasets on which the model will be trained..")

    # -- Extract ewc arguments -- #
    ewc_lambda = None  # --> So the dictionary arguments can be build without an error even if not ewc desired ..
    if extension == 'ewc':
        # -- Extract ewc_lambda -- #
        ewc_lambda = args.ewc_lambda

        # -- Notify the user that the ewc_lambda should not have been changed if -c is activated -- #
        if continue_training:
            print("Note: It will be continued with previous training, be sure that the provided ewc_lambda has not "
                  "changed from previous one..")

    # -- Extract lwf arguments -- #
    lwf_temperature = None  # --> So the dictionary arguments can be build without an error even if not lwf desired ..
    if extension == 'lwf':
        # -- Extract ewc_lambda -- #
        lwf_temperature = args.lwf_temperature

        # -- Notify the user that the ewc_lambda should not have been changed if -c is activated -- #
        if continue_training:
            print("Note: It will be continued with previous training, be sure that the provided lwf_temperature has not "
                  "changed from previous one..")

    
    # -------------------------------
    # Transform tasks to task names
    # -------------------------------
    # -- Transform fold to list if it is set to 'all'
    if fold[0] == 'all':
        fold = list(range(6))
    else: # change each fold type from str to int
        fold = list(map(int, fold))

    # -- Assert if fold is not a number or a list as desired, meaning anything else, like Tuple or whatever -- #
    assert isinstance(fold, (int, list)), "Training multiple tasks in {} mode only uses one or multiple folds specified as integers..".format(extension)

    # -- Calculate the folder name based on all task names the final model is trained on -- #
    tasks_for_folds = list()
    for idx, t in enumerate(task):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Map the fold to corresponding task in dictoinary -- #
        tasks_for_folds.append(t)

    # ----------------------------------------------
    # Define dict with arguments for function calls
    # ----------------------------------------------
    # -- Join all task names together with a '_' in between them -- #
    char_to_join_tasks = '_'
    tasks_list_with_char = (tasks_for_folds, char_to_join_tasks)
    tasks_joined_name = join_texts_with_char(tasks_for_folds, char_to_join_tasks)
    
    # -- Create all argument dictionaries that are used for function calls to make it more generic -- #
    basic_args = {'unpack_data': decompress_data, 'deterministic': deterministic, 'fp16': run_mixed_precision }          
    basic_exts = {'save_interval': save_interval, 'identifier': init_identifier, 'extension': extension,
                  'tasks_list_with_char': copy.deepcopy(tasks_list_with_char), 'save_csv': save_csv,
                  'mixed_precision': run_mixed_precision, **basic_args}
    reh_args = {'samples_per_ds': samples, 'seed': seed, **basic_exts}
    ewc_args = {'ewc_lambda': ewc_lambda, **basic_exts}
    lwf_args = {'lwf_temperature': lwf_temperature, **basic_exts}
    
    # -- Join the dictionaries into a dictionary with the corresponding class name -- #
    args_f = {'nnUNetTrainerV2': basic_args, 'nnUNetTrainerMultiHead': basic_exts,
              #'nnUNetTrainerSequential': basic_exts,
              'nnUNetTrainerRehearsal': reh_args,
              'nnUNetTrainerEWC': ewc_args, 'nnUNetTrainerLWF': lwf_args}

    
    # ---------------------------------------------
    # Train for each task for all provided folds
    # ---------------------------------------------
    # -- Initilize variable that indicates if the trainer has been initialized -- #
    already_trained_on = None

    # -- Loop through folds so each fold will be trained in full before the next one will be started -- #
    for t_fold in fold:
        # -- Initialize running_task_list that includes all tasks that are performed for this fold -- #
        running_task_list = list()

        # -- Initialize the previous trainer path -- #
        prev_trainer_path = None

        # -- Reset the tasks for each fold so it will be trained on every task -- #
        tasks = tasks_for_folds

        # -- Deepcopy the tasks since in case of -c they are changed which might cause trouble in restoring -- #
        all_tasks = copy.deepcopy(tasks)

        # -- Set began_with -- #
        began_with = tasks[0]
        
        # -- When continual_learning flag used there is a special case: The algorithm needs to know where to continue with its training -- #
        # -- For this the already_trained_on file will be loaded and the tasks based on the content prepared for training. -- #
        # -- The started but yet not finished task will be continued and then the remaining task(s) will be trained the normal way. -- #
        if continue_training:
            print("Try to restore a state to continue with the training..")

            # -- Load already trained on file from ../network_training_output_dir/network/tasks_joined_name -- #
            already_trained_on = load_json(join(network_training_output_dir, network, tasks_joined_name, extension+"_trained_on.json"))
            
            # -- Initialize began_with and running_task_list for continuing training -- #
            began_with = -1
            running_task_list = list()

            # -- Get the data regarding the current fold (if it exists, otherwise -1 is returned) -- #
            trained_on_folds = already_trained_on.get(str(t_fold), -1)

            # -- If something is retrieved, ie. trained_on_folds is a dict, then set began_with and running_task_list -- #
            if isinstance(trained_on_folds, dict):
                began_with = trained_on_folds.get('start_training_on', None)
                running_task_list = already_trained_on[str(t_fold)]['finished_training_on'][:] # Without '[:]' reference will change over time as well !
            
            # -- If began_with is None, the fold has not started training with --> start with the first task as -c would not have been set -- #
            if began_with is None:
                # -- Check if all tasks have been trained on so far, if so, this fold is finished with training, else it is not -- #
                run_tasks = running_task_list

                # -- If the lists are equal, continue with the next fold, if not, specify the right task in the following steps -- #
                if np.array(np.array(all_tasks) == np.array(run_tasks)).all()\
                   and np.array(np.array(all_tasks) == trained_on_folds.get('finished_validation_on', np.array(list()))).all():  # Use numpy because lists return true if at least one match in both lists!
                    # -- Update the user that the current fold is finished with training -- #
                    print("Fold {} has been trained on all tasks --> move on to the next fold..".format(t_fold))
                    # -- Treat the last fold as initialization, so set init_seq to True -- #
                    init_seq = True
                    continue    # Continue with next fold

                # -- If the validation is not done yet, do only the validation of the last task -- #
                if np.array(np.array(all_tasks) == np.array(run_tasks)).all():
                    # -- Update the user that the current fold is finished with training -- #
                    print("Fold {} has been trained on all tasks however the validation is still missing..".format(t_fold))
                    # -- Set validation_only to True, so the trainer will be build in the following and only validated -- #
                    validation_only = True
                    # -- Remove the last task from the running_task_list because we want to do validation on this task -- #
                    running_task_list = running_task_list[:-1]
                
            # -- If we began with training but nothing is finished yet, then continue from where wee left -- #
            #if began_with != -1:

            # -- If this list is empty, the trainer did not train on any task --> Start directly with the first task as -c would not have been set -- #
            if began_with != -1: #and len(running_task_list) != 0: # At this point began_with is either a task or -1 but not None
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

                    # -- Only when we want to train change tasks and running_task_list -- #
                    if not validation_only:
                        # -- Insert the previous task to the beginning of the list to ensure that the model will be initialized the right way -- #
                        tasks.insert(0, prev_task)
                        # -- Remove the prev_task in running_task_list, since this is now the first in tasks --> otherwise this is redundant and raises error -- #
                        running_task_list.remove(prev_task)
                    
                    # -- Treat the last fold as initialization, so set init_seq to True by keeping continue_learning to True  -- #
                    init_seq = True
                
                # -- ELSE -- #
                # -- If running_task_list is empty, the training failed at very first task, -- #
                # -- so nothing needs to be changed, simply continue with the training -- #
                
                # -- Set the prev_trainer and the init_identifier so the trainer will be build correctly -- #
                prev_trainer = ext_map.get(already_trained_on[str(t_fold)]['prev_trainer'][-1], None)
                init_identifier = already_trained_on[str(t_fold)]['used_identifier']

                # -- Set began_with to first task since at this point it is either a task or it can be None if previous fold was not trained in full -- #
                began_with = tasks[0]

                # -- Ensure that seed and sample portion is not changed when using rehearsal method --- #
                if extension == 'rehearsal':
                    assert seed == int(trained_on_folds['used_seed']),\
                        "To continue training on the fold {} the same seed, ie. \'{}\' needs to be provided, not \'{}\'.".format(t_fold, trained_on_folds['used_seed'], seed)
                    assert samples == float(trained_on_folds['used_sample_portion']),\
                        "To continue training on the fold {} the same portion of samples for previous tasks should be used, ie. \'{}\' needs to be provided, "\
                        "not \'{}\'.".format(t_fold, trained_on_folds['used_sample_portion'], samples)
            
                # -- Ensure that ewc_lambda is not changed when using EWC method --- #
                if extension == 'ewc':
                    assert ewc_lambda == float(trained_on_folds['used_ewc_lambda']),\
                        "To continue training on the fold {} the same ewc_lambda, ie. \'{}\' needs to be provided, not \'{}\'.".format(t_fold, trained_on_folds['used_ewc_lambda'], ewc_lambda)
            
                # -- Ensure that lwf_temperature is not changed when using LWF method --- #
                if extension == 'lwf':
                    assert lwf_temperature == float(trained_on_folds['used_lwf_temperature']),\
                        "To continue training on the fold {} the same lwf_temperature, ie. \'{}\' needs to be provided, not \'{}\'.".format(t_fold, trained_on_folds['used_lwf_temperature'], lwf_temperature)
            
                # -- Update the user that the fold for training has been found -- #
                print("Fold {} has not been trained on all tasks --> continue the training with task {}..".format(t_fold, began_with))
            
            # -- began_with == -1 or no tasks to train --> nothing to restore -- #
            else:   # Start with new fold, use init_seq that is provided from argument parser
                # -- Treat the last fold as initialization as provided by user -- #
                init_seq = args.init_seq
                # -- Set continue_learning to False so there will be no error in the process of building the trainer -- #
                continue_training = False
                
                # -- Set the prev_trainer and the init_identifier based on previous fold so the trainer will be build correctly -- #
                if already_trained_on.get(str(t_fold), None) is None:
                    prev_trainer = None
                    init_identifier = default_plans_identifier
                else:
                    prev_trainer = ext_map.get(already_trained_on[str(t_fold)]['prev_trainer'][-1], None)
                    init_identifier = already_trained_on[str(t_fold)]['used_identifier']
                
                # -- Set began_with to first task since at this point it is either a task or it can be None if previous fold was not trained in full -- #
                began_with = tasks[0]
    
        # -- Loop through the tasks and train for each task the (finished) model -- #
        for idx, t in enumerate(tasks):
            # -- Check if the first task is the same as began_with so there is no misunderstanding -- #
            if idx == 0:
                assert t == began_with, "Training should be continued, however the wrong task is used --> user has changed order of the tasks.."

            # -- Update running task list and create running task which are all (trained tasks and current task joined) for output folder name -- #
            running_task_list.append(t)
            running_task = join_texts_with_char(running_task_list, char_to_join_tasks)
            
            # -- Extract the configurations and check that trainer_class is not None -- #
            # -- NOTE: Each task will be saved as new folder using the running_task that are all previous and current task joined together. -- #
            # -- NOTE: Perform preprocessing and planning before ! -- #
            plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
            trainer_class = get_default_configuration(network, t, running_task, network_trainer, tasks_joined_name,\
                                                      plans_identifier, extension_type=extension)
            
            if trainer_class is None:
                raise RuntimeError("Could not find trainer class in nnunet_ext.training.network_training")

            # -- Check that network_trainer is of the right type -- #
            if idx == 0 and not continue_training:
                # -- At the first task, the base model can be a nnunet model, later, only current extension models are permitted -- #
                possible_trainers = set(ext_map.values())   # set to avoid the double values, since every class is represented twice
                assert issubclass(trainer_class, tuple(possible_trainers)),\
                    "Network_trainer was found but is not derived from a provided extension nor nnUNetTrainerV2."\
                    " When using this function, it is only permitted to start with an nnUNetTrainerV2 or a provided extensions"\
                    " like nnUNetTrainerMultiHead or nnUNetTrainerRehearsal. Choose the right trainer or use the convential"\
                    " nnunet command to train."
            else:
                # -- Now, at a later stage, only trainer based on extension permitted! -- #
                assert issubclass(trainer_class, ext_map[extension]),\
                    "Network_trainer was found but is not derived from {}."\
                    " When using this function, only {} trainers are permitted."\
                    " So choose {}"\
                    " as a network_trainer corresponding to the network, or use the convential nnunet command to train.".format(ext_map[extension], extension, ext_map[extension])
            
            # -- Load trainer from last task and initialize new trainer if continue training is not set -- #
            if idx == 0 and init_seq:# and not continue_training:
                # -- Initialize the prev_trainer if it is not None. If it is None, the trainer will be initialized in the parent class -- #
                # -- Further check that all necessary information is provided, otherwise exit with error message -- #
                assert isinstance(t, str) and prev_trainer is not None and init_identifier is not None and isinstance(t_fold, int),\
                    "The informations for building the initial trainer to use for training are not fully provided, check the arguments.."
                
                # -- Get default configuration for nnunet/nnunet_ext model (finished training) -- #
                # -- --> Extension is used, always use the first task as this is the base and other tasks -- #
                # -- will result in other network structures (structure based on plans_file) -- #
                # -- Current task t might not be equal to all_tasks[0], since tasks might be changed in the -- #
                # -- preparation for -c. -- #
                #plans_file, init_output_folder, dataset_directory, batch_dice, stage, \
                plans_file, prev_trainer_path, dataset_directory, batch_dice, stage, \
                trainer_class = get_default_configuration(network, all_tasks[0], running_task, prev_trainer, tasks_joined_name,\
                                                          init_identifier, extension_type=extension)
                
                # -- Ensure that trainer_class is not None -- #
                if trainer_class is None:
                    raise RuntimeError("Could not find trainer class in nnunet.training.network_training nor nnunet_ext.training.network_training")

                """
                # -- Load (pre-trained) prev_trainer for intialization --> will be initialized in next loop, always use init_output_folder -- #
                if trainer_class.__name__ == nnUNetTrainerV2.__name__:  # --> Initialization with nnUNetTrainerV2
                    prev_trainer = trainer_class(plans_file, t_fold, output_folder=init_output_folder, dataset_directory=dataset_directory,\
                                                 batch_dice=batch_dice, stage=stage, **(args_f[trainer_class.__name__]))
                else:
                    prev_trainer = trainer_class(split, t, plans_file, t_fold, output_folder=init_output_folder, dataset_directory=dataset_directory,\
                                                 batch_dice=batch_dice, stage=stage,\
                                                 already_trained_on=already_trained_on, **(args_f[trainer_class.__name__]))
                
                # -- Create already trained on for this task we are going to skip -- #
                if already_trained_on is None:
                    already_trained_on = { str(t_fold): {'finished_training_on': [t], 'prev_trainer': [prev_trainer.__class__.__name__],
                                                         'finished_validation_on': [t], 'start_training_on': None,
                                                         'used_identifier': init_identifier, 'val_metrics_should_exist': False,
                                                         'checkpoint_should_exist' : False, 'tasks_at_time_of_checkpoint': list(), # --> Can only be the case when init_seq
                                                         'active_task_at_time_of_checkpoint': None}
                                         }  # --> Do not set the split, since it needs to be simplified first and then checked if it matches

                # -- Ensure that prev_trainer and already trained on is not none, since it should be everything from last loop -- #
                assert prev_trainer is not None and already_trained_on is not None and len(already_trained_on) != 0,\
                    "The information that is necessary for training on the previous model are not provided as it should be.."
                    
                # -- Initialize prev_trainer -- #
                prev_trainer.initialize(training=False)
                """
                # -- Only if we do not want to do a validation we continue with the next element in the loop -- #
                if not validation_only:
                    # -- Continue with next element, since the previous trainer is restored, otherwise it will be trained as well -- #
                    continue
            
            # -- Restore previous trained trainer as prev_trainer for the task that will be trained in this loop -- #
            #elif idx == 1 and init_seq:# and not continue_training:
                # -- Ensure that prev_trainer and already trained on is not none, since it should be everything from last loop -- #
            #    assert prev_trainer is not None and already_trained_on is not None and len(already_trained_on) != 0,\
            #        "The information that is necessary for training on the previous model is not provided as it should be.."
                
                # -- Initialize prev_trainer -- #
            #    prev_trainer.initialize(training=False)

            # -- Set the correct trainer, but only for very first task (with or without prev_trainer) -- #
            if idx == 0 or idx == 1 and init_seq:
                # -- To initialize a new trainer, always use the first task since this shapes the network structure. -- #
                # -- During training the tasks will be updated, so this should cause no problems -- #
                # -- Set the trainer with corresponding arguments --> can only be an extension from here on -- #
                trainer = trainer_class(split, all_tasks[0], plans_file, t_fold, output_folder=output_folder_name, dataset_directory=dataset_directory,\
                                        batch_dice=batch_dice, stage=stage,\
                                        already_trained_on=already_trained_on, **(args_f[trainer_class.__name__]))
                
                # -- Initialize the trainer with the dataset, task, fold, optimizer, etc. -- #
                trainer.initialize(not validation_only, num_epochs=num_epochs, prev_trainer_path=prev_trainer_path)
            
            # -- Nothing needs to be done with the trainer, since it makes everything by itself, for instance -- #
            # -- if the task we train on does not exist at the later point, it simply initializes it as a new head -- #

            # -- disable the saving of checkpoints if desired -- #
            if disable_saving:
                trainer.save_final_checkpoint = False # Whether or not to save the final checkpoint
                trainer.save_best_checkpoint = False  # Whether or not to save the best checkpoint according to
                trainer.save_intermediate_checkpoints = True  # Whether or not to save checkpoint_latest. We need that in case
                trainer.save_latest_only = True  # If false it will not store/overwrite _latest but separate files each

            # -- Initialize the trainer since it is newly created every time because of the new dataset, task, fold, optimizer, etc. -- #
            #trainer.initialize(not validation_only, num_epochs=num_epochs, prev_trainer=prev_trainer)

            # -- Update trained_on 'manually' if first task is done but finished_training_on is empty --> first task was used for initialization -- #
            if idx == 1 and len(trainer.already_trained_on[str(t_fold)]['finished_training_on']) == 0:
                trainer.update_save_trained_on_json(tasks[0], finished=True)

            # -- Find a matchting lr given the provided num_epochs -- #
            if find_lr:
                trainer.find_lr(num_iters=num_epochs)
            else:      
                if not validation_only:
                    if continue_training:
                        # -- User wants to continue previous training while ignoring pretrained weights -- #
                        trainer.load_latest_checkpoint()
                        # -- Set continue_training to false for possible upcoming tasks -- #
                        # -- --> otherwise an error might occur because there is no trainer to restore -- #
                        continue_training = False
                    elif (not continue_training) and (args.pretrained_weights is not None):
                        # -- Start a new training and use pretrained_weights if they are set -- #
                        load_pretrained_weights(trainer.network, args.pretrained_weights)
                    else:
                        # -- Start new training without setting pretrained_weights -- #
                        pass

                    # -- Start to train the trainer --> if task is not registered, the trainer will do this automatically -- #
                    trainer.run_training(task=t, output_folder=output_folder_name)
                else:
                    if valbest:
                        trainer.load_best_checkpoint(train=False)
                    else:
                        trainer.load_final_checkpoint(train=False)

                # -- Evaluate the trainers network -- #
                trainer.network.eval()

                # -- Perform validation using the trainer -- #
                trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                                 run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                                 overwrite=args.val_disable_overwrite)

                # -- If disable saving, and init_seq, remove the initial trainer -- #
                if disable_saving and idx == 0 and init_seq:
                    # -- At this stage, the initial trainer has been used as a base, and the trainer folder will be removed from the directory -- #
                    del_folder = join(network_training_output_dir, network, tasks_joined_name, t, prev_trainer.__class__.__name__ + "__" + init_identifier)
                    delete_dir_con(del_folder)
                    # -- Now, in the folder ../network_training_output_dir/network/tasks_joined_name are only identifier and results for extension training -- #
            
            # -- If the models for each sequence should not be stored, delete the last model and only keep the current finished ones -- #
            if disable_saving and prev_trainer_path is not None:
                # -- Delete content of the folder -- #
                delete_dir_con(prev_trainer_path)

            # -- Update prev_trainer and prev_trainer_path -- #
            #prev_trainer = trainer
            prev_trainer_path = output_folder_name
            
            # -- Update the already_trained_on file, so it is up to date and can be used for next iteration -- #
            #already_trained_on = trainer.already_trained_on

            # -- Reset validation_only in case it has been set during -c in the beginning -- #
            validation_only = args.validation_only

        
        # -- If the models for each sequence should not be stored, move the content of the final saved model to the parent folder -- #
        if disable_saving:
            p_folder_path = os.path.dirname(os.path.realpath(prev_trainer_path))
            # -- Delete content of the folder -- #
            move_dir(prev_trainer_path, p_folder_path)
        
        # -- Reset init_seq and prev_trainer -- #
        init_seq = args.init_seq
        prev_trainer = args.initialize_with_network_trainer

    # -- Reset cuda device as environment variable, otherwise other GPUs might not be available ! -- #
    del os.environ["CUDA_VISIBLE_DEVICES"]
#------------------------------------------- Inspired by original implementation -------------------------------------------#