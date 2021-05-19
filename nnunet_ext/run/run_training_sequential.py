#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet sequential ---------#
#----------training version. Implementation inspired by original implementation.------------------------#
#########################################################################################################

import os, argparse
from nnunet_ext.utilities.helpful_functions import delete_dir_con, join_texts_with_char, move_dir
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.paths import network_training_output_dir
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential # Own implemented class
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


#------------------------------------------- Inspired by original implementation -------------------------------------------#
def main():
    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")  # Can only be a sequential one
    
    # -- nnUNet arguments untouched --> Should not intervene with sequential code, everything should work -- #
    parser.add_argument("-val", "--validation_only", help="Use this if you want to only run the validation. This will validate each model "
                                                          "of the sequential pipeline, so they should have been saved and not deleted.",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="Use this if you want to continue a training. The program will determine "
                                                          "by itself where to continue with the learning, so provide all tasks.",
                        action="store_true")    # TODO: Determine where it failed using the already_trained_on file!!!!!
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

    # -- Additional arguments specific for sequential training -- #
    parser.add_argument("-t", "--task_ids", nargs="+", help="Specify a list of task ids to train on (ids or names). Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder", required=True)
    parser.add_argument("-f", "--folds", nargs="+", help="Specify on which folds to train on. Use a fold between 0, 1, ..., 5 or \'all\'", required=True)
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=0,
                        help='Try to train the model on the GPU device with <DEVICE> ID. '+
                            ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.'+
                            ' Default: Only GPU device with ID 0 will be used.')
    parser.add_argument('-num_epochs', action='store', type=int, nargs=1, required=False, default=500,
                        help='Specify the number of epochs to train the model.'
                            ' Default: Train for 500 epochs.')
    parser.add_argument('-save_interval', action='store', type=int, nargs=1, required=False, default=5,
                        help='Specify after which epoch interval to update the saved data.'
                            ' Default: If disable_saving False, the result will be updated every 5th epoch.')
    parser.add_argument('--init_seq', action='store_true', default=False,
                        help='Specify if the first task from -t is already trained and represents '
                             ' an init network_trainer to do sequential training on or not. If so, -initialize_with_network_trainer '
                             ' needs to be provided as well.'
                             ' Default: False')
    parser.add_argument('-initialize_with_network_trainer', type=str, required=False, default=None,
                        help='Specify the network_trainer that should be used as a foundation to start training sequentially.'
                            ' The network_trainer of the first provided task needs to be finished with training and either a sequential network_trainer'
                            ' or a standard nnUNetTrainer. Default: None.')
    parser.add_argument('-used_identifier_in_init_network_trainer', type=str, required=False, default=None,
                        help='Specify the identifier that should be used for the network_trainer that is used as a foundation to start training sequentially.'
                            ' Default: default_plans_identifier from paths.py (nnUNetPlansv2.1).')
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space. Further for sequential tasks "
                             "the intermediate model won't be saved then, remeber that.")


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

    # -- Extract the arguments specific for sequential training from argument parser -- #
    task = args.task_ids    # List of the tasks
    fold = args.folds       # List of the folds
    num_epochs = args.num_epochs    # The number of epochs to train for each task
    if isinstance(num_epochs, list):    # When the num_epochs get returned as a list, extract the number to avoid later appearing errors
        num_epochs = num_epochs[0]
    init_seq = args.init_seq    # Indicates if the first task is an initialization. If True, the training start from 0 and will not be based on a provided model
    if init_seq:
        # -- Extract necessary information for initilization -- #
        init_identifier = args.used_identifier_in_init_network_trainer  # Extract init_identifyer
        prev_trainer = args.initialize_with_network_trainer # Trainer for first task with first fold
        # -- Check that prev_trainer is provided -- #
        assert prev_trainer is not None, "When using the first provided task as a base for sequential training, the network_trainer needs to be specified as well."
        # -- Set init_identifier if not provided -- #
        if init_identifier is None:
            init_identifier = default_plans_identifier  # Set to init_identifier
    disable_saving = args.disable_saving
    save_interval = args.save_interval
    cuda = args.device

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
    if fold[0] == '\'all\'':
        fold = list(range(6))
    else: # change each fold type from str to int
        fold = list(map(int, fold))

    # -- Assert if fold is not a number or a list as desired, meaning anything else, like Tuple or whatever -- #
    assert isinstance(fold, (int, list)), "Training multiple tasks sequentially only uses one or multiple folds specified as integers.."

    # -- Calculate the folder name based on all task names the final model is trained on -- #
    tasks_for_folds = list()
    for idx, t in enumerate(task):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Map the fold to corresponding task in dictoinary -- #
        tasks_for_folds.append(t)

    # -- Join all task names together with a '_' in between them
    tasks_joined_name = join_texts_with_char(tasks_for_folds, '_')


    # ----------------------------------------------------------
    # Train sequentially for each task for all provided folds
    # ----------------------------------------------------------
    # -- Initilize variable that indicates if the trainer has been initialized -- #
    trainer_initialized = False
    
    # -- Loop through folds so each fold will be trained in full before the next one will be started -- #
    for t_fold in fold:
        # -- Initialize running_task_list that includes all tasks that are performed for this fold -- #
        prev_trainer_path = None
        running_task_list = list()

        # -- Reset the tasks for each fold so it will be trained on every task -- #
        tasks = tasks_for_folds

        # -- When continual_learning flag used there is a special case: The algorithm needs to know where to continue with its training -- #
        # -- For this the already_trained_on file will be loaded and the tasks based on the content prepared for training. -- #
        # -- The started but yet not finished task will be continued and then the remaining task(s) will be trained the normal way. -- #
        if continue_training:
            # -- Load already trained on file from ../network_training_output_dir/network/tasks_joined_name -- #
            already_trained_on = load_json(join(network_training_output_dir, network, tasks_joined_name))

            # -- Get the task that the model started to train as well as all tasks that are finished with training -- #
            began_with = already_trained_on.get(t_fold, -1).get('start_training_on', None)
            running_task_list = already_trained_on[t_fold]['finished_training_on']

            # -- If began_with is None, the fold has not started training with --> Start with the first task as -c would not have been set -- #
            if began_with is None:
                # -- Check if all tasks have been trained on so far, if so, this fold is finished with training, else it is not -- #
                all_tasks = tasks
                run_tasks = running_task_list

                # -- If the sorted lists are equal, continue with the next fold, if not, specify the right task in the following steps -- #
                if all_tasks.sort() == run_tasks.sort():
                    # -- Update the user that the current fold is finished with training -- #
                    print("Fold {} has been trained on all tasks --> move on to the next fold..".format(t_fold))
                    continue

            # -- If this list is empty, the trainer did not train on any task --> Start directly with the first task as -c would not have been set -- #
            if began_with != -1 and len(running_task_list) != 0: # At this point began_with is either a task or -1 but not None
                # -- Substract the tasks from the tasks list --> Only use the tasks that are in tasks but not in finished_with -- #
                for task in tasks:
                    # -- If the task has already been trained, remove the entry from the tasks dictionary -- #
                    if task in running_task_list:
                        tasks.remove(task)
                # -- Update the user that the fold for training has been found -- #
                print("Fold {} has not been trained on all tasks --> continue the training with task {}..".format(t_fold, began_with))
    
        # -- Loop through the tasks and train for each task the (finished) model -- #
        for idx, t in enumerate(tasks):
            # -- Check if the first task is the same as began_with if continue_learning is selected -- #
            if continue_training:
                assert t == began_with, "Training should be continued, however the wrong task is used --> user has changed order of the tasks.."

            # -- Update running task list and create running task which are all (trained tasks and current task joined for output folder name -- #
            running_task_list.append(t)
            running_task = join_texts_with_char(running_task_list, '_')

            # -- Extract the configurations and check that trainer_class is not None -- #
            # -- NOTE: Each task will be saved as new folder using the running_task that are all previous and current ask joined together. -- #
            # -- NOTE: Perform preprocessing and planning for sequential before ! --> Load always a sequential trainer at this point ! -- #
            plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
            trainer_class = get_default_configuration(network, t, running_task, network_trainer, tasks_joined_name,\
                                                      plans_identifier, extension_type='sequential', base_model_seq=True)

            if trainer_class is None:
                raise RuntimeError("Could not find trainer class in nnunet_ext.training.network_training")

            # -- Check that network_trainer is of sequential type -- #
            if idx == 0 and not continue_training:
                # -- At the first task, the base model can be a nnunet model, later, only sequential models are permiited -- #
                assert issubclass(trainer_class, (nnUNetTrainerSequential, nnUNetTrainerV2)),\
                    "Network_trainer was found but is not derived from nnUNetTrainerSequential nor nnUNetTrainerV2."\
                    " When using this function, it is only permitted to start with an Sequential trainer or nnUNetTrainerV2."\
                    " Choose the right trainer or use the convential nnunet command to train."
            else:
                # -- Now, at a later stage, only sequential trainer permitted! -- #
                assert issubclass(trainer_class, nnUNetTrainerSequential),\
                    "Network_trainer was found but is not derived from nnUNetTrainerSequential."\
                    " When using this function, only sequential trainers are permitted."\
                    " So choose nnUNetTrainerSequential"\
                    " as a network_trainer corresponding to the network, or use the convential nnunet command to train."

            # -- Load trainer from last task and initialize new trainer if continue training is wrong -- #
            if idx == 0 and init_seq and not continue_training:
                # -- Initialize the prev_trainer if it is not None. If it is None, the trainer will be initialized in the parent class -- #
                # -- Further check that all necessary information is provided, otheriwse exit with error message -- #
                assert isinstance(t, str) and prev_trainer is not None and init_identifier is not None and isinstance(t_fold, int),\
                    "The informations for building the initial trainer to use for sequential training is not fully provided, check the arguments.."
                
                # -- If the trainer is not from sequential origin, change the search_in and base_module parameter for the get_default_configuration
                if isinstance(prev_trainer, nnUNetTrainerSequential):
                    base_model_seq = True
                else:
                    base_model_seq = False
                
                # -- Get default configuration for nnunet/nnunet_ext model (finished training) -- #
                plans_file, init_output_folder, dataset_directory, batch_dice, stage, \
                trainer_class = get_default_configuration(network, t, running_task, prev_trainer, tasks_joined_name,\
                                                          init_identifier, extension_type='sequential', base_model_seq=base_model_seq)
                
                # -- Ensure that trainer_class is not None -- #
                if trainer_class is None:
                    raise RuntimeError("Could not find trainer class in nnunet.training.network_training or nnunet_ext.training.network_training")
                
                # -- Load trainer and initialize it -- #
                prev_trainer = trainer_class(plans_file, t_fold, output_folder=init_output_folder, dataset_directory=dataset_directory,
                                            batch_dice=batch_dice, stage=stage)
                prev_trainer.initialize(training=False)

                # -- Continue with next element, since the previous trainer is initialized, otherwise it will be trained as well -- #
                continue

            # -- Extract trainer and set saving indicating variables to False if desired -- #
            trainer = trainer_class(plans_file, t_fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                                    batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                    deterministic=deterministic, fp16=run_mixed_precision, save_interval=save_interval)
            
            if disable_saving:
                trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
                trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
                trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
                trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

            # -- Update trained_on 'manually' if first task is done but finished_training_on is empty --> first task was used for initialization -- #
            if idx == 1 and len(trainer.already_trained_on[t_fold]['finished_training_on']) == 0:
                trainer.update_save_trained_on_json(t, finished=True)

            # -- Initialize the trainer if trainer_initialized is False, else just update the prev_trainer and output_folder -- #
            # -- If prev_trainer is not provided, ie. None, the first task embodies the initialization of prev_trainer through training -- #
            if not trainer_initialized:
                trainer.initialize(not validation_only, num_epochs=num_epochs, prev_trainer=prev_trainer)
                trainer_initialized = True
            else:
                trainer.update_trainer(prev_trainer = prev_trainer, output_folder = output_folder_name)

            # -- Find a matchting lr given the provided num_epochs -- #
            if find_lr:
                trainer.find_lr(num_iters=num_epochs)
            else:
                if not validation_only:
                    if continue_training:
                        # -- User wants to continue previous training while ignoring pretrained weights -- #
                        trainer.load_latest_checkpoint()
                        # -- Set continue_training to false for possible upcoming tasks -- #
                        # -- --> otherwise an error might occur because there is no trainer to reinitialize -- #
                        continue_training = False
                    elif (not continue_training) and (args.pretrained_weights is not None):
                        # -- Start a new training and use pretrained_weights if they are set -- #
                        load_pretrained_weights(trainer.network, args.pretrained_weights)
                    else:
                        # -- Start new training without setting pretrained_weights -- #
                        pass

                    # -- Start to train the trainer -- #
                    trainer.run_training(task=t)
                else:
                    # -- Load best or finale checkpoint based on valbest -- #
                    if valbest:
                        trainer.load_best_checkpoint(train=False)
                    else:
                        trainer.load_final_checkpoint(train=False)

                # -- Evaluate the trainers network -- #
                trainer.network.eval()

                # -- Predict the trainers validation -- #
                trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                                run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                                overwrite=args.val_disable_overwrite)

                if network == '3d_lowres' and not args.disable_next_stage_pred:
                    print("predicting segmentations for the next stage of the cascade")
                    predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))

                # -- If disable saving, and init_seq, remove the initial trainer -- #
                if disable_saving and idx == 0 and init_seq:
                    # -- At this stage, the initial trainer has been used as a base, and the trainer folder will be removed from the directory -- #
                    del_folder = join(network_training_output_dir, network, tasks_joined_name, t, prev_trainer.__class__.__name__ + "__" + init_identifier)
                    delete_dir_con(del_folder)
                    # -- Now, in the folder ../network_training_output_dir/network/tasks_joined_name are only identifier and results for sequential training -- #
            
            # -- Update prev_trainer and prev_trainer_path -- #
            prev_trainer = trainer
            prev_trainer_path = output_folder_name

            # -- If the models for each sequence should not be stored, delete the last model and only keep the current finished ones -- #
            if disable_saving and prev_trainer_path is not None:
                # -- Delete content of the folder -- #
                delete_dir_con(prev_trainer_path)
        
        # -- If the models for each sequence should not be stored, move the content of the final saved model to the parent folder -- #
        if disable_saving:
            p_folder_path = os.path.dirname(os.path.realpath(prev_trainer_path))
            # -- Delete content of the folder -- #
            move_dir(prev_trainer_path, p_folder_path)

if __name__ == "__main__":
    main()
#------------------------------------------- Inspired by original implementation -------------------------------------------#