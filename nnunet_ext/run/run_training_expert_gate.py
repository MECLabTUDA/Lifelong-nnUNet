#########################################################################################################
#---------------This class represents the Training of networks using the extended nnUNet ---------------#
#----------training versions. Implementation inspired by original implementation.-----------------------#
#########################################################################################################

from nnunet_ext.training.network_training.expert_gate.nnUNetTrainerExpertGate import nnUNetTrainerExpertGate
import numpy as np
import os, argparse, copy, warnings, nnunet_ext
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet_ext.paths import default_plans_identifier, network_training_output_dir, preprocessing_output_dir
from nnunet_ext.utilities.helpful_functions import delete_dir_con, join_texts_with_char, move_dir
from nnunet.experiment_planning.summarize_plans import summarize_plans

#------------------------------------------- Inspired by original implementation -------------------------------------------#
def run_training(extension='multihead'):

    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")

    # -- nnUNet arguments untouched --> Should not intervene with extension code, everything should work -- #
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
    parser.add_argument("-f", "--folds", nargs="+", help="Specify on which folds to train on. Use a fold between 0, 1, ..., 4 or \'all\'", required=True)
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=[0],
                        help='Try to train the model on the GPU device with <DEVICE> ID. '+
                            ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.'+
                            ' Default: Only GPU device with ID 0 will be used.')
    parser.add_argument('--reduce_threads', action='store_true', default=False,
                        help='If the network uses too much CPU Threads, set this flag and it will be reduced to about 20 to 30 Threads.')
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
                            ' Default: If disable_saving is False, the result will be updated every 25th epoch.')
    parser.add_argument('--store_csv', required=False, default=False, action="store_true",
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
    parser.add_argument("--continue_from_previous_dir", required=False, action='store_true',
                        help='Instead of continuing with the experiment with a certain task order, continue from a dedicated'
                            'experiment with previous tasks.')
    parser.add_argument('-used_identifier_in_init_network_trainer', type=str, required=False, default=None,
                        help='Specify the identifier that should be used for the network_trainer that is used as a foundation to start training sequentially.'
                            ' Default: default_plans_identifier from paths.py (nnUNetPlansv2.1).')
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space. Further for sequential tasks "
                             "the intermediate model won't be saved then, remeber that.")
    parser.add_argument('--use_vit', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will be used instead of the Generic_UNet. '+
                             'Note that then the flags -v, -v_type, --task_specific_ln and --use_mult_gpus should be set accordingly.')
    parser.add_argument('--task_specific_ln', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will have task specific Layer Norms.')
    parser.add_argument('--use_mult_gpus', action='store_true', default=False,
                        help='If this is set, the ViT model will be placed onto a second GPU. '+
                             'When this is set, more than one GPU needs to be provided when using -d.')
    parser.add_argument("-v", "--version", action='store', type=int, nargs=1, default=[1], choices=[1, 2, 3, 4],
                        help='Select the ViT input building version. Currently there are four'+
                            ' possibilities: 1, 2, 3 or 4.'+
                            ' Default: version one will be used. For more references wrt, to the versions, see the docs.')
    parser.add_argument("-v_type", "--vit_type", action='store', type=str, nargs=1, default='base', choices=['base', 'large', 'huge'],
                        help='Specify the ViT architecture. Currently there are only three'+
                            ' possibilities: base, large or huge.'+
                            ' Default: The smallest ViT architecture, i.e. base will be used.')
    parser.add_argument('--do_LSA', action='store_true', default=False,
                        help='Set this flag if Locality Self-Attention should be used for the ViT.')
    parser.add_argument('--do_SPT', action='store_true', default=False,
                        help='Set this flag if Shifted Patch Tokenization should be used for the ViT.')
    parser.add_argument('--no_transfer_heads', required=False, default=False, action="store_true",
                        help='Set this flag if a new head should not be initialized using the last head'
                            ' during training, ie. the very first head from the initialization of the class is used.'
                            ' Default: The previously trained head is used as initialization of the new head.')
    
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
    
    # -- Add arguments for ewc methods -- #
    if extension in ['ewc', 'ewc_vit', 'ewc_unet', 'ewc_ln', 'froz_ewc', 'froz_ewc_final', 'ownm1', 'ownm2', 'ownm3', 'ownm4']:
        parser.add_argument('-ewc_lambda', action='store', type=float, nargs=1, required=False, default=0.4,
                            help='Specify the importance of the previous tasks for the EWC method.'
                                ' This number represents the lambda value in the loss function calculation as proposed in the paper.'
                                ' Default: ewc_lambda = 0.4')

    # -- Add arguments for RW method -- #
    if extension == 'rw':
        parser.add_argument('-rw_alpha', action='store', type=float, nargs=1, required=False, default=0.9,
                            help='Specify the rw_alpha parameter that is used to calculate the Fisher values --> should be [0, 1].'
                                ' Default: rw_alpha = 0.9')
        parser.add_argument('-rw_lambda', action='store', type=float, nargs=1, required=False, default=0.4,
                            help='Specify the importance of the previous tasks for the RW method using the EWC regularization.'
                                ' Default: rw_lambda = 0.4')
        parser.add_argument('-fisher_update_after', action='store', type=int, nargs=1, required=False, default=10,
                            help='Specify after which iteration (batch iteration, not epoch) the fisher values are updated/calculated.'
                                ' Default: The result will be updated every 10th epoch.')

    # -- Add arguments for lwf method -- #
    if extension == 'lwf':
        parser.add_argument('-lwf_temperature', action='store', type=float, nargs=1, required=False, default=2.0,
                            help='Specify the temperature variable for the LwF method.'
                                ' Default: lwf_temperature = 2.0')

    # -- Add arguments for PLOP method -- #
    if extension in ['plop', 'pod', 'ownm1', 'ownm2', 'ownm3', 'ownm4']:
        parser.add_argument('-pod_lambda', action='store', type=float, nargs=1, required=False, default=1e-2,
                            help='Specify the lambda weighting for the distillation loss.'
                                ' Default: pod_lambda = 0.01')
        parser.add_argument('-pod_scales', action='store', type=int, nargs=1, required=False, default=3,
                            help='Specify the number of scales for the PLOP method.'
                                ' Default: pod_scales = 3')
    
    # -- Add arguments for MiB method -- #
    if extension in ['mib', 'ownm1', 'ownm2', 'ownm3']:
        parser.add_argument('-mib_alpha', action='store', type=float, nargs=1, required=False, default=0.9,
                            help='Specify the mib_alpha parameter to hard-ify the soft-labels.'
                                ' Default: mib_alpha = 0.9')
        parser.add_argument('-mib_lkd', action='store', type=float, nargs=1, required=False, default=1,
                            help='Specify the weighting of the KL loss.'
                                ' Default: mib_lkd = 1')

    # -- Add arguments for own method -- #
    if extension in ['ownm4']:
        parser.add_argument('-pseudo_alpha', action='store', type=float, nargs=1, required=False, default=3.0,
                            help='Specify the pseudo_alpha parameter to be used during pseudo-labeling.'
                                ' Default: pseudo_alpha = 3.0')

    # -- Add arguments for own method -- #
    if extension in ['ownm1', 'ownm2', 'ownm3', 'ownm4']:
        parser.add_argument('--no_pod', required=False, default=False, action="store_true",
                            help='Set this flag if the POD embedding should not be included in the loss calculation.'
                                ' Default: POD embedding will be included.')

    if extension in ['froz_ewc']:
        parser.add_argument('--adaptive', required=False, default=False, action="store_true",
                            help='Set this flag if the EWC loss should be changed during the frozen training process (ewc_lambda*e^{-1/3}). '
                                 ' Default: The EWC loss will not be altered.')

    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet) arguments -- #
    args = parser.parse_args()
    network = args.network
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
    reduce_threads = args.reduce_threads
    transfer_heads = not args.no_transfer_heads
    
    if reduce_threads:
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'

    if isinstance(split, list):    # When the split get returned as a list, extract the path to avoid later appearing errors
        split = split[0].strip()
    
    # -- Check that split is string and not empty -- #
    assert len(split) > 0,\
        "When providing a split, ensure that it is not empty, otheriwse no split can be performed."
    # -- Check that the number of tasks is greater than 1, else a conventional nnUNetTrainerV2 should be used -- #
    if len(task) == 1:
        warnings.warn("When training on only one task, the conventional training of the nnU-Net can be used.")

    # -- Extract the vit_type structure and check it is one from the existing ones -- #
    vit_type = args.vit_type
    if isinstance(vit_type, list):    # When the vit_type gets returned as a list, extract the type to avoid later appearing errors
        vit_type = vit_type[0].lower()
    # assert vit_type in ['base', 'large', 'huge'], 'Please provide one of the following three existing ViT types: base, large or huge..'
    
    # -- Extract the desired version -- #
    version = args.version
    if isinstance(version, list):    # When the version gets returned as a list, extract the number to avoid later appearing errors
        version = version[0]
    # assert version in range(1, 5), 'We only provide three versions, namely 1, 2, 3 or 4 but not {}..'.format(version)

    # -- Extract ViT specific flags to as well -- #
    use_vit = args.use_vit
    ViT_task_specific_ln = args.task_specific_ln
    
    # -- LSA and SPT flags -- #
    do_LSA = args.do_LSA
    do_SPT = args.do_SPT
    
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
    
    # -- Check if the user wants to split the network onto multiple GPUs -- #
    split_gpu = args.use_mult_gpus
    if split_gpu:
        assert len(cuda) > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'
    
    cuda = join_texts_with_char(cuda, ',')
    
    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    # -------------------------------
    # Transform tasks to task names
    # -------------------------------
    # -- Transform fold to list if it is set to 'all'
    if fold[0] == 'all':
        fold = list(range(5))
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



    # -- Loop through folds so each fold will be trained in full before the next one will be started -- #
    for t_fold in fold:
    
        # -- Loop through the tasks and train for each task the (finished) model -- #
        for idx, t in enumerate(tasks_for_folds):

            plans_file: str = join(preprocessing_output_dir, t, plans_identifier + "_plans_3D.pkl")
            dataset_directory: str = join(preprocessing_output_dir, t)

            plans = load_pickle(plans_file)
            possible_stages = list(plans['plans_per_stage'].keys())
            stage = possible_stages[-1]
            
            summarize_plans(plans_file)

            output_folder_name = join(network_training_output_dir, "expert_gate", t, 
            "nnUNetTrainerExpertGate" + "__" + plans_identifier)


            folder_with_preprocessed_data = join(dataset_directory, plans['data_identifier'] +
                                        "_stage%d" % stage)
            
            #trainer = nnUNetTrainerExpertGate(split, t, plans_file, t_fold, output_folder=output_folder_name, dataset_directory=dataset_directory,\
            #                            batch_dice=batch_dice, stage=stage)
            trainer = nnUNetTrainerExpertGate(plans_file,t_fold,output_folder_name, 
                dataset_directory, stage=stage, deterministic=deterministic
            )
            trainer.initialize(True, num_epochs=num_epochs)


            # -- disable the saving of checkpoints if desired -- #
            if disable_saving:
                trainer.save_final_checkpoint = False # Whether or not to save the final checkpoint
                trainer.save_best_checkpoint = False  # Whether or not to save the best checkpoint according to
                trainer.save_intermediate_checkpoints = True  # Whether or not to save checkpoint_latest. We need that in case
                trainer.save_latest_only = True  # If false it will not store/overwrite _latest but separate files each

            # -- Find a matchting lr given the provided num_epochs -- #
            if find_lr:
                trainer.find_lr(num_iters=num_epochs)

            # -- Start to train the trainer
            trainer.run_training()


#------------------------------------------- Inspired by original implementation -------------------------------------------#

# -- Add the main function calls for the setup file -- #
# -- Main function for setup execution of expert gate method -- #
def main_expert_gate():
    r"""Run training for expert gate Trainer
    """
    run_training(extension='expert_gate')