#########################################################################################################
#---------------This class represents the Training of networks using the extended nnUNet ---------------#
#----------training versions. Implementation inspired by original implementation.-----------------------#
#########################################################################################################

import copy

from pandas.core import base
import numpy as np
import os, argparse
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet_ext.paths import default_plans_identifier, network_training_output_dir
from nnunet_ext.utilities.helpful_functions import delete_dir_con, join_texts_with_char, move_dir

# -- Import the trainer classes -- #
from nnunet_ext.training.network_training.rw.nnUNetTrainerRW import nnUNetTrainerRW # Own implemented class
from nnunet_ext.training.network_training.ewc.nnUNetTrainerEWC import nnUNetTrainerEWC # Own implemented class
from nnunet_ext.training.network_training.lwf.nnUNetTrainerLWF import nnUNetTrainerLWF # Own implemented class
from nnunet_ext.training.network_training.mib.nnUNetTrainerMiB import nnUNetTrainerMiB # Own implemented class
from nnunet_ext.training.network_training.pod.nnUNetTrainerPOD import nnUNetTrainerPOD # Own implemented class
from nnunet_ext.training.network_training.plop.nnUNetTrainerPLOP import nnUNetTrainerPLOP # Own implemented class
from nnunet_ext.training.network_training.ownm1.nnUNetTrainerOwnM1 import nnUNetTrainerOwnM1 # Own implemented class
from nnunet_ext.training.network_training.ownm2.nnUNetTrainerOwnM2 import nnUNetTrainerOwnM2 # Own implemented class
from nnunet_ext.training.network_training.ewc_ln.nnUNetTrainerEWCLN import nnUNetTrainerEWCLN # Own implemented class
from nnunet_ext.training.network_training.ewc_vit.nnUNetTrainerEWCViT import nnUNetTrainerEWCViT # Own implemented class
from nnunet_ext.training.network_training.ewc_unet.nnUNetTrainerEWCUNet import nnUNetTrainerEWCUNet # Own implemented class
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead # Own implemented class
from nnunet_ext.training.network_training.rehearsal.nnUNetTrainerRehearsal import nnUNetTrainerRehearsal # Own implemented class
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential # Own implemented class
from nnunet_ext.training.network_training.freezed_vit.nnUNetTrainerFreezedViT import nnUNetTrainerFreezedViT # Own implemented class
from nnunet_ext.training.network_training.freezed_unet.nnUNetTrainerFreezedUNet import nnUNetTrainerFreezedUNet # Own implemented class
from nnunet_ext.training.network_training.freezed_nonln.nnUNetTrainerFreezedNonLN import nnUNetTrainerFreezedNonLN # Own implemented class

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
    parser.add_argument("-v", "--version", action='store', type=int, nargs=1, default=[1],
                        help='Select the ViT input building version. Currently there are four'+
                            ' possibilities: 1, 2, 3 or 4.'+
                            ' Default: version one will be used. For more references wrt, to the versions, see the docs.')
    parser.add_argument("-v_type", "--vit_type", action='store', type=str, nargs=1, default='base',
                        help='Specify the ViT architecture. Currently there are only three'+
                            ' possibilities: base, large or huge.'+
                            ' Default: The smallest ViT architecture, i.e. base will be used.')
    parser.add_argument('--transfer_heads', required=False, default=False, action="store_true",
                        help='Set this flag if a new head will be initialized using the last head'
                            ' during training. Default: The very first head from the initialization of the class is used.')
    
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
    if extension in ['ewc', 'ewc_vit', 'ownm1', 'ownm2']:
        parser.add_argument('-ewc_lambda', action='store', type=float, nargs=1, required=False, default=0.4,
                            help='Specify the importance of the previous tasks for the EWC method.'
                                ' This number represents the lambda value in the loss function calculation as proposed in the paper.'
                                ' Default: ewc_lambda = 0.4')

    # -- Add arguments for RW method -- #
    if extension == 'rw':
        parser.add_argument('-rw_alpha', action='store', type=float, nargs=1, required=False, default=0.9,
                            help='Specify the mib_alpha parameter that is used to calculate the Fisher values --> should be [0, 1].'
                                ' Default: mib_alpha = 0.9')
        parser.add_argument('-rw_lambda', action='store', type=float, nargs=1, required=False, default=0.4,
                            help='Specify the importance of the previous tasks for the RW method using the EWC regularization.'
                                ' Default: rw_lambda = 0.4')
        parser.add_argument('-update_after', action='store', type=int, nargs=1, required=False, default=10,
                            help='Specify after which iteration (batch iteration, not epoch) the fisher values are updated/calculated.'
                                ' Default: The result will be updated every 10th epoch.')

    # -- Add arguments for lwf method -- #
    if extension == 'lwf':
        parser.add_argument('-lwf_temperature', action='store', type=float, nargs=1, required=False, default=2.0,
                            help='Specify the temperature variable for the LwF method.'
                                ' Default: lwf_temperature = 2.0')

    # -- Add arguments for PLOP method -- #
    if extension in ['plop', 'pod', 'ownm1', 'ownm2']:
        parser.add_argument('-pod_lambda', action='store', type=float, nargs=1, required=False, default=1e-2,
                            help='Specify the lambda weighting for the distillation loss.'
                                ' Default: pod_lambda = 0.01')
        parser.add_argument('-plop_scales', action='store', type=int, nargs=1, required=False, default=3,
                            help='Specify the number of scales for the PLOP method.'
                                ' Default: plop_scales = 3')

    # -- Add arguments for MiB method -- #
    if extension in ['mib', 'ownm1', 'ownm2']:
        parser.add_argument('-mib_alpha', action='store', type=float, nargs=1, required=False, default=1.0,
                            help='Specify the mib_alpha parameter to hard-ify the soft-labels.'
                                ' Default: mib_alpha = 1.0')
        parser.add_argument('-lkd', action='store', type=float, nargs=1, required=False, default=10,
                            help='Specify the weighting of the KL loss.'
                                ' Default: lkd = 10')

    # -- Add arguments for own method -- #
    if extension in ['ownm2']:
        parser.add_argument('-pseudo_alpha', action='store', type=float, nargs=1, required=False, default=3.0,
                            help='Specify the pseudo_alpha parameter to be used during pseudo-labeling.'
                                ' Default: pseudo_alpha = 3.0')

    # -- Build mapping for extension to corresponding class -- #
    trainer_map = {'rw': nnUNetTrainerRW, 'nnUNetTrainerRW': nnUNetTrainerRW,
                   'ewc': nnUNetTrainerEWC, 'nnUNetTrainerEWC': nnUNetTrainerEWC,
                   'lwf': nnUNetTrainerLWF, 'nnUNetTrainerLWF': nnUNetTrainerLWF,
                   'mib': nnUNetTrainerMiB, 'nnUNetTrainerMiB': nnUNetTrainerMiB,
                   'pod': nnUNetTrainerPOD, 'nnUNetTrainerPOD': nnUNetTrainerPOD,
                   'plop': nnUNetTrainerPLOP, 'nnUNetTrainerPLOP': nnUNetTrainerPLOP,
                   'ownm1': nnUNetTrainerOwnM1, 'nnUNetTrainerOwnM1': nnUNetTrainerOwnM1,
                   'ownm2': nnUNetTrainerOwnM2, 'nnUNetTrainerOwnM2': nnUNetTrainerOwnM2,
                   'ewc_ln': nnUNetTrainerEWCLN, 'nnUNetTrainerEWCLN': nnUNetTrainerEWCLN,
                   'ewc_vit': nnUNetTrainerEWCViT, 'nnUNetTrainerEWCViT': nnUNetTrainerEWCViT,
                   'ewc_unet': nnUNetTrainerEWCUNet, 'nnUNetTrainerEWCUNet': nnUNetTrainerEWCUNet,
                   'rehearsal': nnUNetTrainerRehearsal, 'nnUNetTrainerRehearsal': nnUNetTrainerRehearsal,
                   'multihead': nnUNetTrainerMultiHead, 'nnUNetTrainerMultiHead': nnUNetTrainerMultiHead,
                   'sequential': nnUNetTrainerSequential, 'nnUNetTrainerSequential': nnUNetTrainerSequential,
                   'freezed_vit': nnUNetTrainerFreezedViT, 'nnUNetTrainerFreezedViT': nnUNetTrainerFreezedViT,
                   'freezed_unet': nnUNetTrainerFreezedUNet, 'nnUNetTrainerFreezedUNet': nnUNetTrainerFreezedUNet,
                   'freezed_nonln': nnUNetTrainerFreezedNonLN, 'nnUNetTrainerFreezedNonLN': nnUNetTrainerFreezedNonLN}


    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet) arguments -- #
    args = parser.parse_args()
    network = args.network
    network_trainer = str(trainer_map[extension]).split('.')[-1][:-2]
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
    transfer_heads = args.transfer_heads

    if isinstance(split, list):    # When the split get returned as a list, extract the path to avoid later appearing errors
        split = split[0].strip()
    
    # -- Check that split is string and not empty -- #
    assert len(split) > 0,\
        "When providing a split, ensure that it is not empty, otheriwse no split can be performed."
    # -- Check that the number of tasks is greater than 1, else a conventional nnUNetTrainerV2 should be used -- #
    assert len(task) > 1,\
        "When training on only one task, the conventional training of the nnU-Net should be used, not the extension."

    # -- Extract the vit_type structure and check it is one from the existing ones -- #
    vit_type = args.vit_type
    if isinstance(vit_type, list):    # When the vit_type gets returned as a list, extract the type to avoid later appearing errors
        vit_type = vit_type[0].lower()
    assert vit_type in ['base', 'large', 'huge'], 'Please provide one of the following three existing ViT types: base, large or huge..'
    
    # -- Extract the desired version -- #
    version = args.version
    if isinstance(version, list):    # When the version gets returned as a list, extract the number to avoid later appearing errors
        version = version[0]
    assert version in range(1, 5), 'We only provide three versions, namely 1, 2, 3 or 4 but not {}..'.format(version)

    # -- Extract ViT specific flags to as well -- #
    use_vit = args.use_vit
    ViT_task_specific_ln = args.task_specific_ln
    
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

    # -- Reset transfer heads if the Trainer is of sequential type -- #
    if extension in ['sequential', 'plop', 'freezed_nonln', 'freezed_unet', 'freezed_vit']:
        # -- Transfer heads is always True here even if the user did not set it -- #
        transfer_heads = True

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
    if extension in ['ewc', 'ewc_vit', 'ewc_ln', 'ewc_unet', 'ownm1', 'ownm2']:
        # -- Extract ewc_lambda -- #
        ewc_lambda = args.ewc_lambda
        if isinstance(ewc_lambda, list):
            ewc_lambda = ewc_lambda[0]

        # -- Notify the user that the ewc_lambda should not have been changed if -c is activated -- #
        if continue_training:
            print("Note: It will be continued with previous training, be sure that the provided ewc_lambda has not "
                  "changed from previous one..")

        # -- Ensure that init_seq only works for EWC trainers since no other trainer stores the fisher and param values -- #
        if init_seq and prev_trainer != str(trainer_map[extension]).split('.')[-1][:-2]:
            assert False, "You can only use a pre-trained EWC model as a base and no other model, since the corresponding"+\
                          " parameters that are necessary for every task are not stored in any other trainer."

    # -- Extract rw arguments -- #
    rw_alpha, rw_lambda, update_fisher_every = None, None, None  # --> So the dictionary arguments can be build without an error even if not ewc desired ..
    if extension in ['rw']:
        # -- Extract rw_alpha -- #
        rw_alpha = args.rw_alpha
        if isinstance(rw_alpha, list):
            rw_alpha = rw_alpha[0]
        # -- Extract rw_lambda -- #
        rw_lambda = args.rw_lambda
        if isinstance(rw_lambda, list):
            rw_lambda = rw_lambda[0]
        # -- Extract update_fisher_every -- #
        update_fisher_every = args.update_after
        if isinstance(update_fisher_every, list):
            update_fisher_every = update_fisher_every[0]

        # -- Notify the user that the ewc_lambda should not have been changed if -c is activated -- #
        if continue_training:
            print("Note: It will be continued with previous training, be sure that the provided RW related settings have not "
                  "changed from previous one..")

        # -- Ensure that init_seq only works for EWC trainers since no other trainer stores the fisher and param values -- #
        if init_seq and prev_trainer != str(trainer_map[extension]).split('.')[-1][:-2]:
            assert False, "You can only use a pre-trained RW model as a base and no other model, since the corresponding"+\
                          " parameters that are necessary for every task are not stored in any other trainer."

    # -- Extract lwf arguments -- #
    lwf_temperature = None  # --> So the dictionary arguments can be build without an error even if not lwf desired ..
    if extension == 'lwf':
        # -- Extract lwf_temperature for dist_loss -- #
        lwf_temperature = args.lwf_temperature
        if isinstance(lwf_temperature, list):
            lwf_temperature = lwf_temperature[0]

        # -- Notify the user that the lwf_temperature should not have been changed if -c is activated -- #
        if continue_training:
            print("Note: It will be continued with previous training, be sure that the provided lwf_temperature has not "
                  "changed from previous one..")

    # -- Extract PLOP arguments -- #
    pod_lambda, plop_scales = None, None  # --> So the dictionary arguments can be build without an error even if not plop desired ..
    if extension in ['plop', 'pod', 'ownm1', 'ownm2']:
        # -- Extract pos lambda for dist_loss -- #
        pod_lambda = args.pod_lambda
        if isinstance(pod_lambda, list):
            pod_lambda = pod_lambda[0]
        # -- Extract plop scale for dist_loss -- #
        plop_scales = args.plop_scales
        if isinstance(plop_scales, list):
            plop_scales = plop_scales[0]

        # -- Notify the user that the plop_scales should not have been changed if -c is activated -- #
        if continue_training:
            print("Note: It will be continued with previous training, be sure that the provided pod_lambda and plop_scales have not "
                  "changed from previous one..")

    # -- Extract MiB arguments -- #
    mib_alpha, lkd = None, None  # --> So the dictionary arguments can be build without an error even if not plop desired ..
    if extension in ['mib', 'ownm1', 'ownm2']:
        # -- Extract mib lambda for dist_loss -- #
        mib_alpha = args.mib_alpha
        if isinstance(mib_alpha, list):
            mib_alpha = mib_alpha[0]
        # -- Extract plop scale for dist_loss -- #
        lkd = args.lkd
        if isinstance(lkd, list):
            lkd = lkd[0]

        # -- Notify the user that the mib_alpha and lkd should not have been changed if -c is activated -- #
        if continue_training:
            print("Note: It will be continued with previous training, be sure that the provided mib_alpha and lkd have not "
                  "changed from previous one..")

    # -- Extract own arguments -- #
    pseudo_alpha = None  # --> So the dictionary arguments can be build without an error even if not plop desired ..
    if extension in ['ownm2']:
        # -- Extract pseudo lambda for dist_loss -- #
        pseudo_alpha = args.pseudo_alpha
        if isinstance(pseudo_alpha, list):
            pseudo_alpha = pseudo_alpha[0]

        # -- Notify the user that the mib_alpha and lkd should not have been changed if -c is activated -- #
        if continue_training:
            print("Note: It will be continued with previous training, be sure that the provided pseudo_alpha has not "
                  "changed from previous one..")

    
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

    # ----------------------------------------------
    # Define dict with arguments for function calls
    # ----------------------------------------------
    # -- Join all task names together with a '_' in between them -- #
    char_to_join_tasks = '_'
    tasks_list_with_char = (tasks_for_folds, char_to_join_tasks)
    tasks_joined_name = join_texts_with_char(tasks_for_folds, char_to_join_tasks)
    
    # -- Create all argument dictionaries that are used for function calls to make it more generic -- #
    basic_args = {'unpack_data': decompress_data, 'deterministic': deterministic, 'fp16': run_mixed_precision}  
    basic_vit =  {'vit_type': vit_type, 'version': version, 'split_gpu': split_gpu, **basic_args}
    basic_exts = {'save_interval': save_interval, 'identifier': init_identifier, 'extension': extension,
                  'tasks_list_with_char': copy.deepcopy(tasks_list_with_char), 'save_csv': save_csv,
                  'mixed_precision': run_mixed_precision, 'use_vit': use_vit, 'vit_type': vit_type, 'version': version,
                  'split_gpu': split_gpu, 'transfer_heads': transfer_heads, 'ViT_task_specific_ln': ViT_task_specific_ln,
                  **basic_args}
    ewc_args = {'ewc_lambda': ewc_lambda, **basic_exts}
    mib_args = {'lkd': lkd, 'mib_alpha': mib_alpha, **basic_exts}
    lwf_args = {'lwf_temperature': lwf_temperature, **basic_exts}
    reh_args = {'samples_per_ds': samples, 'seed': seed, **basic_exts}
    plop_args = {'pod_lambda': pod_lambda, 'scales': plop_scales, **basic_exts}
    rw_args = {'rw_lambda': rw_lambda, 'rw_alpha': rw_alpha, 'fisher_update_after': update_fisher_every, **basic_exts}
    ownm1_args = {'ewc_lambda': ewc_lambda, 'pod_lambda': pod_lambda, 'scales': plop_scales, 'lkd': lkd, 'mib_alpha': mib_alpha, **basic_exts}
    ownm2_args = {'ewc_lambda': ewc_lambda, 'pod_lambda': pod_lambda, 'scales': plop_scales, 'lkd': lkd, 'mib_alpha': mib_alpha, **basic_exts}
    # ownm2_args = {'ewc_lambda': ewc_lambda, 'pod_lambda': pod_lambda, 'scales': plop_scales, 'pseudo_alpha': pseudo_alpha, **basic_exts}
    
    # -- Join the dictionaries into a dictionary with the corresponding class name -- #
    args_f = {'nnUNetTrainerRW': rw_args, 'nnUNetTrainerMultiHead': basic_exts,
              'nnUNetTrainerFreezedViT': basic_exts, 'nnUNetTrainerEWCViT': ewc_args,
              'nnUNetTrainerFreezedNonLN': basic_exts, 'nnUNetTrainerEWCLN': ewc_args,
              'nnUNetTrainerFreezedUNet': basic_exts, 'nnUNetTrainerEWCUNet': ewc_args,
              'nnUNetTrainerSequential': basic_exts, 'nnUNetTrainerRehearsal': reh_args,
              'nnUNetTrainerMiB': mib_args, 'nnUNetTrainerEWC': ewc_args, 'nnUNetTrainerLWF': lwf_args,
              'nnUNetTrainerPLOP': plop_args, 'nnUNetTrainerV2': basic_args, 'nnViTUNetTrainer': basic_vit,
              'nnUNetTrainerPOD': plop_args, 'nnUNetTrainerOwnM1': ownm1_args, 'nnUNetTrainerOwnM2': ownm2_args}

    
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
            if use_vit:
                base_path = join(network_training_output_dir, network, tasks_joined_name, 'metadata', Generic_ViT_UNet.__name__+'V'+str(version), vit_type.lower())
            else:
                base_path = join(network_training_output_dir, network, tasks_joined_name, 'metadata', Generic_UNet.__name__)
            if ViT_task_specific_ln:
                base_path = join(base_path, 'task_specific')
            else:
                base_path = join(base_path, 'not_task_specific')
            if transfer_heads:
                already_trained_on = load_json(join(base_path, 'SEQ', extension+"_trained_on.json"))
            else:
                already_trained_on = load_json(join(base_path, 'MH', extension+"_trained_on.json"))
            
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
                try:
                    if np.array(np.array(all_tasks) == np.array(run_tasks)).all()\
                        and np.array(np.array(all_tasks) == trained_on_folds.get('finished_validation_on', np.array(list()))).all():  # Use numpy because lists return true if at least one match in both lists!
                        # -- Update the user that the current fold is finished with training -- #
                        print("Fold {} has been trained on all tasks --> move on to the next fold..".format(t_fold))
                        # -- Treat the last fold as initialization, so set init_seq to True -- #
                        init_seq = True
                        continue    # Continue with next fold
                    # -- In this case the training stopped after a task was finished but not every task is trained -- #
                    else:
                        # -- Set began_with to None so it will be catched in the corresponding section to continue training -- #
                        began_with = None
                except ValueError: # --> The arrays do not match, ie. not finished on all tasks and validation is missing
                    # -- Set began_with to None so it will be catched in the corresponding section to continue training -- #
                    began_with = None

            # -- If this list is empty, the trainer did not train on any task --> Start directly with the first task as -c would not have been set -- #
            if began_with != -1: # At this point began_with is either a task or None but not -1
                if began_with is None:  # --> Only the case when training is finished but validation on last task is missing
                    # -- Update the user that the current fold is finished with training -- #
                    print("Fold {} has been trained on all tasks however the validation is still missing..".format(t_fold))
                    # -- Set validation_only to True, so the trainer will be build in the following and only validated -- #
                    validation_only = True
                    # -- Add only the first and last task since the validation of the last task is missing -- #
                    tasks = tasks[-1:]
                    # -- Remove the last task from the running_task_list because we want to do validation on this task -- #
                    running_task_list = running_task_list[:-1]
                    # -- Set everything for the upcoming loop and break the current one -- #
                    began_with = tasks[0]
                    init_seq = True
                    # -- Set the prev_trainer and the init_identifier so the trainer will be build correctly -- #
                    prev_trainer = trainer_map.get(already_trained_on[str(t_fold)]['prev_trainer'][-1], None)
                    init_identifier = already_trained_on[str(t_fold)]['used_identifier']

                else:
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
                            
                        # -- Treat the last fold as initialization, so set init_seq to True by keeping continue_learning to True  -- #
                        init_seq = True
                    
                    # -- ELSE -- #
                    # -- If running_task_list is empty, the training failed at very first task, -- #
                    # -- so nothing needs to be changed, simply continue with the training -- #
                    # -- Set the prev_trainer and the init_identifier so the trainer will be build correctly -- #
                    prev_trainer = trainer_map.get(already_trained_on[str(t_fold)]['prev_trainer'][-1], None)
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
                    print("Fold {} has not been trained on all tasks --> continue the training with restoring task {}..".format(t_fold, began_with))
            
            # -- began_with == -1 or no tasks to train --> nothing to restore -- #
            else:   # Start with new fold, use init_seq that is provided from argument parser
                # -- Treat the last fold as initialization as specified by user -- #
                init_seq = args.init_seq
                # -- Set continue_learning to False so there will be no error in the process of building the trainer -- #
                continue_training = False
                
                # -- Set the prev_trainer and the init_identifier based on previous fold so the trainer will be build correctly -- #
                if already_trained_on.get(str(t_fold), None) is None:
                    prev_trainer = None
                    init_identifier = default_plans_identifier
                else:
                    prev_trainer = trainer_map.get(already_trained_on[str(t_fold)]['prev_trainer'][-1], None)
                    init_identifier = already_trained_on[str(t_fold)]['used_identifier']
                
                # -- Set began_with to first task since at this point it is either a task or it can be None if previous fold was not trained in full -- #
                began_with = tasks[0]
    
        # -- Loop through the tasks and train for each task the (finished) model -- #
        for idx, t in enumerate(tasks):
            # -- Check if the first task is the same as began_with so there is no misunderstanding -- #
            if idx == 0:
                assert t == began_with, "Training should be continued, however the wrong task is used --> user has changed order of the tasks.."

            # -- Update running task list and create running task which are all (trained tasks and current task joined) for output folder name -- #
            if t not in running_task_list:
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
                possible_trainers = set(trainer_map.values())   # set to avoid the double values, since every class is represented twice
                assert issubclass(trainer_class, tuple(possible_trainers)),\
                    "Network_trainer was found but is not derived from a provided extension nor nnUNetTrainerV2."\
                    " When using this function, it is only permitted to start with an nnUNetTrainerV2 or a provided extensions"\
                    " like nnUNetTrainerMultiHead or nnUNetTrainerRehearsal. Choose the right trainer or use the convential"\
                    " nnunet command to train."
            else:
                # -- Now, at a later stage, only trainer based on extension permitted! -- #
                assert issubclass(trainer_class, trainer_map[extension]),\
                    "Network_trainer was found but is not derived from {}."\
                    " When using this function, only {} trainers are permitted."\
                    " So choose {}"\
                    " as a network_trainer corresponding to the network, or use the convential nnunet command to train.".format(trainer_map[extension], extension, trainer_map[extension])
            
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
                plans_file, prev_trainer_path, dataset_directory, batch_dice, stage, \
                trainer_class = get_default_configuration(network, all_tasks[0], running_task, prev_trainer, tasks_joined_name,\
                                                          init_identifier, extension_type=extension)
                
                # -- Ensure that trainer_class is not None -- #
                if trainer_class is None:
                    raise RuntimeError("Could not find trainer class in nnunet.training.network_training nor nnunet_ext.training.network_training")

                # -- Only if we do not want to do a validation we continue with the next element in the loop -- #
                if not validation_only:
                    # -- Continue with next element, since the previous trainer is restored, otherwise it will be trained as well -- #
                    continue
            
            # -- Set the correct trainer, but only for very first task (with or without prev_trainer) -- #
            if idx == 0 or idx == 1 and init_seq:
                # -- To initialize a new trainer, always use the first task since this shapes the network structure. -- #
                # -- During training the tasks will be updated, so this should cause no problems -- #
                # -- Set the trainer with corresponding arguments --> can only be an extension from here on -- #
                trainer = trainer_class(split, all_tasks[0], plans_file, t_fold, output_folder=output_folder_name, dataset_directory=dataset_directory,\
                                        batch_dice=batch_dice, stage=stage,\
                                        already_trained_on=already_trained_on, **(args_f[trainer_class.__name__]))
                trainer.initialize(not validation_only, num_epochs=num_epochs, prev_trainer_path=prev_trainer_path)

                # NOTE: Trainer has only weights and heads of first task at this point
                # --> Add the heads and load the state_dict from the latest model and not all_tasks[0]
                #     since this is only used for initialization
                
            # -- Nothing needs to be done with the trainer, since it makes everything by itself, for instance -- #
            # -- if the task we train on does not exist at the later point, it simply initializes it as a new head -- #

            # -- disable the saving of checkpoints if desired -- #
            if disable_saving:
                trainer.save_final_checkpoint = False # Whether or not to save the final checkpoint
                trainer.save_best_checkpoint = False  # Whether or not to save the best checkpoint according to
                trainer.save_intermediate_checkpoints = True  # Whether or not to save checkpoint_latest. We need that in case
                trainer.save_latest_only = True  # If false it will not store/overwrite _latest but separate files each

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

            # -- If the models for each sequence should not be stored, delete the last model and only keep the current finished one -- #
            # -- NOTE: If the previous trainer was a nnU-Net, i.e. not an extension, then do not remove it -- #
            if disable_saving and prev_trainer_path is not None:
                # -- Delete content of the folder -- #
                if network_training_output_dir in prev_trainer_path:   # Only if located in nnunet_ext, else don't remove it
                    delete_dir_con(prev_trainer_path)

            # -- Update prev_trainer and prev_trainer_path -- #
            prev_trainer_path = output_folder_name
            
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

#------------------------------------------- Inspired by original implementation -------------------------------------------#