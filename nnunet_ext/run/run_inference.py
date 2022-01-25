#########################################################################################################
#----------------------------This class allows the user to make predictions-----------------------------#
#########################################################################################################

import os, argparse
from nnunet_ext.evaluation.evaluator import Evaluator
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.paths import evaluation_output_dir, default_plans_identifier
from nnunet_ext.inference.predict import predict_from_folder
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_ext.paths import network_training_output_dir
from nnunet.paths import network_training_output_dir as network_training_output_dir_orig
from nnunet_ext.utilities.helpful_functions import get_ViT_LSA_SPT_folder_name
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet

import sys

def run_inference():
    # -- First of all check that evaluation_output_dir is set otherwise we do not perform an evaluation -- #
    assert evaluation_output_dir is not None, "Before running any evaluation, please specify the Evaluation folder (EVALUATION_FOLDER) as described in the paths.md."

    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")  # Can only be a multi head, sequential, rehearsal, ewc or lwf

    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=False)
    parser.add_argument('-o', "--output_folder", required=False, default=None, help="folder for saving predictions")
    parser.add_argument('-chk',
                        help='checkpoint name, model_final_checkpoint' or 'model_best',
                        required=False,
                        default='model_final_checkpoint')
    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")
    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("-f", "--folds",  action='store', type=str, nargs="+",
                        help="Specify on which fold training happened. Use a fold between 0, 1, ..., 4 or \'all\'", required=True)
    parser.add_argument("-trained_on", action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network has trained with to specify the correct path to the networks. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument("-use_model", "--use", action='store', type=str, nargs="+",
                        help="Specify a list of task ids that specify the exact network that should be used for evaluation. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument('-use_head', action='store', type=str, nargs=1, required=False, default=None,
                        help='Specify which head to use for the evaluation of tasks the network is not trained on. When using a non nn-UNet extension, that' +
                              'is not necessary. If this is not set, always the latest trained head will be used.')
    parser.add_argument("--fp32_used", required=False, default=False, action="store_true",
                        help="Specify if mixed precision has been used during training or not")
    parser.add_argument("-evaluate_on",  action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network will be evaluated on. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=False, default=None)
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=[0],
                        help='Try to train the model on the GPU device with <DEVICE> ID. '+
                            ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.'+
                            ' Default: Only GPU device with ID 0 will be used.')
    parser.add_argument("-v", "--version", action='store', type=int, nargs=1, default=[1], choices=[1, 2, 3, 4],
                        help='Select the ViT input building version. Currently there are only four'+
                            ' possibilities: 1, 2, 3 or 4.'+
                            ' Default: version one will be used. For more references wrt, to the versions, see the docs.')
    parser.add_argument("-v_type", "--vit_type", action='store', type=str, nargs=1, default='base', choices=['base', 'large', 'huge'],
                        help='Specify the ViT architecture. Currently there are only three'+
                            ' possibilities: base, large or huge.'+
                            ' Default: The smallest ViT architecture, i.e. base will be used.')
    parser.add_argument('--use_vit', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will be used instead of the Generic_UNet. '+
                             'Note that then the flags -v, -v_type and --use_mult_gpus should be set accordingly.')
    parser.add_argument('--task_specific_ln', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will have task specific Layer Norms.')


    parser.add_argument('--no_transfer_heads', required=False, default=False, action="store_true",
                        help='Set this flag if a new head should not be initialized using the last head'
                            ' during training, ie. the very first head from the initialization of the class is used.'
                            ' Default: The previously trained head is used as initialization of the new head.')


    parser.add_argument('--use_mult_gpus', action='store_true', default=False,
                        help='If this is set, the ViT model will be placed onto a second GPU. '+
                             'When this is set, more than one GPU needs to be provided when using -d.')


    parser.add_argument('--always_use_last_head', action='store_true', default=False,
                        help='If this is set, during the evaluation, always the last head will be used, '+
                             'for every dataset the evaluation is performed on. When an extension network was trained with '+
                             'the -transfer_heads flag then this should be set, i.e. nnUNetTrainerSequential or nnUNetTrainerFreezedViT.')


    parser.add_argument('--no_pod', action='store_true', default=False,
                        help='This will only be considered if our own trainers are used. If set, this flag indicates that the POD '+
                             'embedding has not been used.')
    parser.add_argument('--do_LSA', action='store_true', default=False,
                        help='Set this flag if Locality Self-Attention should be used for the ViT.')
    parser.add_argument('--do_SPT', action='store_true', default=False,
                        help='Set this flag if Shifted Patch Tokenization should be used for the ViT.')

    # -- Build mapping for network_trainer to corresponding extension name -- #
    ext_map = {'nnViTUNetTrainer': None, 'nnViTUNetTrainerCascadeFullRes': None,
               'nnUNetTrainerFreezedViT': 'freezed_vit', 'nnUNetTrainerEWCViT': 'ewc_vit',
               'nnUNetTrainerFreezedNonLN': 'freezed_nonln', 'nnUNetTrainerEWCLN': 'ewc_ln',
               'nnUNetTrainerMultiHead': 'multihead', 'nnUNetTrainerSequential': 'sequential',
               'nnUNetTrainerFreezedUNet': 'freezed_unet', 'nnUNetTrainerEWCUNet': 'ewc_unet',
               'nnUNetTrainerMiB': 'mib', 'nnUNetTrainerPLOP': 'plop', 'nnUNetTrainerV2': 'standard',
               'nnUNetTrainerOwnM1': 'ownm1', 'nnUNetTrainerOwnM2': 'ownm2', 'nnUNetTrainerPOD': 'pod',
               'nnUNetTrainerOwnM3': 'ownm3', 'nnUNetTrainerOwnM4': 'ownm4', 'nnUNetTrainerRW': 'rw',
               'nnUNetTrainerRehearsal': 'rehearsal', 'nnUNetTrainerEWC': 'ewc', 'nnUNetTrainerLWF': 'lwf'}


    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet) arguments -- #
    args = parser.parse_args()
    network = args.network
    network_trainer = args.network_trainer
    plans_identifier = args.p
    always_use_last_head = args.always_use_last_head
    
    # -- Extract the arguments specific for all trainers from argument parser -- #
    trained_on = args.trained_on    # List of the tasks that helps to navigate to the correct folder, eg. A B C
    use_model = args.use            # List of the tasks representing the network to use, e. use A B from folder A B C
    evaluate_on = args.evaluate_on  # List of the tasks that should be used to evaluate the model
    use_head = args.use_head        # One task specifying which head should be used
    if isinstance(use_head, list):
        use_head = use_head[0]
        
    # -- Arguments specific of prediction extraction -- #
    input_folder = args.input_folder
    output_folder = args.output_folder
    if input_folder is None:
        assert evaluate_on is not None

    # -- Extract further arguments -- #
    fold = args.folds
    cuda = args.device
    mixed_precision = not args.fp32_used
    transfer_heads = not args.no_transfer_heads
    do_pod = not args.no_pod

    # -- Extract ViT specific flags to as well -- #
    use_vit = args.use_vit
    ViT_task_specific_ln = args.task_specific_ln
    
    # -- Extract the vit_type structure and check it is one from the existing ones -- #s
    vit_type = args.vit_type.lower()
    if isinstance(vit_type, list):    # When the vit_type gets returned as a list, extract the type to avoid later appearing errors
        vit_type = vit_type[0].lower()
    # assert vit_type in ['base', 'large', 'huge'], 'Please provide one of the following three existing ViT types: base, large or huge..'

    # -- LSA and SPT flags -- #
    do_LSA = args.do_LSA
    do_SPT = args.do_SPT
    
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

    # -- Extract the desired version, only considered in case of ViT Trainer -- #
    version = args.version
    if isinstance(version, list):    # When the version gets returned as a list, extract the number to avoid later appearing errors
        version = version[0]
    # assert version in range(1, 5), 'We only provide three versions, namely 1, 2, 3 or 4, but not {}..'.format(version)
    
    
    # -------------------------------
    # Transform tasks to task names
    # -------------------------------
    # -- Transform fold to list if it is set to 'all'
    if fold[0] == 'all':
        fold = list(range(5))
    else: # change each fold type from str to int
        fold = list(map(int, fold))

    # -- Assert if fold is not a number or a list as desired, meaning anything else, like Tuple or whatever -- #
    assert isinstance(fold, (int, list)), "To Evaluate multiple tasks with {} trainer, only one or multiple folds specified as integers are allowed..".format(network_trainer)

    # -- Build all necessary task lists -- #
    tasks_for_folder = list()
    use_model_w_tasks = list()
    evaluate_on_tasks = list()
    if use_head is not None:
        use_head = convert_id_to_task_name(int(use_head)) if not use_head.startswith("Task") else use_head
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

    char_to_join_tasks = '_'
    
    
    # ---------------------------------------------
    # Get model_folder_name
    # ---------------------------------------------

    # -- Extract the folder name in case we have a ViT -- #
    folder_n = get_ViT_LSA_SPT_folder_name(do_LSA, do_SPT)
    version = 'V' + str(version)

    tasks_list_with_char = (tasks_for_folder, char_to_join_tasks)
    model_list_with_char = (use_model_w_tasks, char_to_join_tasks)
    tasks_joined_name = join_texts_with_char(tasks_list_with_char[0], tasks_list_with_char[1])
    model_joined_name = join_texts_with_char(model_list_with_char[0], model_list_with_char[1])
    
    # -- Build the trainer_path first -- #
    if 'nnUNetTrainerV2' in network_trainer:   # always_last_head makes no sense here, there is only one head
        trainer_path = join(network_training_output_dir_orig, network, tasks_joined_name, network_trainer+'__'+plans_identifier)
        output_path = join(evaluation_output_dir, network, tasks_joined_name, network_trainer+'__'+plans_identifier)
        output_path = output_path.replace('nnUNet_ext', 'nnUNet')
    elif nnViTUNetTrainer.__name__ in network_trainer: # always_last_head makes no sense here, there is only one head
        trainer_path = join(network_training_output_dir, network, tasks_joined_name, network_trainer+'__'+plans_identifier, vit_type,\
                            'task_specific' if ViT_task_specific_ln else 'not_task_specific', folder_n)
        output_path = join(evaluation_output_dir, network, tasks_joined_name, network_trainer+'__'+plans_identifier, vit_type,\
                        'task_specific' if ViT_task_specific_ln else 'not_task_specific', folder_n)
        trainer_path = trainer_path.replace(nnViTUNetTrainer.__name__, nnViTUNetTrainer.__name__+version)
        output_path = output_path.replace(nnViTUNetTrainer.__name__, nnViTUNetTrainer.__name__+version)
    else:   # Any other extension like CL extension for example (using MH Architecture)
        if use_vit:
            trainer_path = join(network_training_output_dir, network, tasks_joined_name, model_joined_name,\
                                network_trainer+'__'+plans_identifier, Generic_ViT_UNet.__name__+version, vit_type,\
                                'task_specific' if ViT_task_specific_ln else 'not_task_specific', folder_n, 'SEQ' if transfer_heads else 'MH')
            output_path = join(evaluation_output_dir, network, tasks_joined_name, model_joined_name,\
                                network_trainer+'__'+plans_identifier, Generic_ViT_UNet.__name__+version, vit_type,\
                                'task_specific' if ViT_task_specific_ln else 'not_task_specific', folder_n, 'SEQ' if transfer_heads else 'MH',\
                                'last_head' if always_use_last_head else 'corresponding_head')
        else:
            trainer_path = join(network_training_output_dir, network, tasks_joined_name, model_joined_name,\
                                network_trainer+'__'+plans_identifier, Generic_UNet.__name__, 'SEQ' if transfer_heads else 'MH')
            output_path = join(evaluation_output_dir, network, tasks_joined_name, model_joined_name,\
                                network_trainer+'__'+plans_identifier, Generic_UNet.__name__, 'SEQ' if transfer_heads else 'MH',\
                                'last_head' if always_use_last_head else 'corresponding_head')

    # -- Re-Modify trainer path for own methods if necessary -- #
    if 'OwnM' in network_trainer:
        trainer_path = join(os.path.sep, *trainer_path.split(os.path.sep)[:-1], 'pod' if do_pod else 'no_pod')
        output_path = join(os.path.sep, *output_path.split(os.path.sep)[:-1], 'pod' if do_pod else 'no_pod', 'last_head' if always_use_last_head else 'corresponding_head')

    output_path = join(output_path, 'predictions')

    # Note that unlike the trainer_path from run_evaluation, this does not include the fold because plans.pkl is one level above 

    # --------------------------------------------------------------------------
    # Make predictions (based on nnUNet/nnunet/inference/predict_simple.py 
    # --------------------------------------------------------------------------
    
    if output_folder is None:
        output_folder = output_path
    chk = args.chk
    overwrite_existing = args.overwrite_existing
    disable_tta = args.disable_tta
    lowres_segmentations = None
    save_npz = False
    num_threads_preprocessing = 1
    all_in_gpu = None
    num_parts = 1
    part_id = 0

    num_threads_nifti_save = 2
    step_size = 0.5


    # Additional parameters needed for restoring extension trainers/models
    params_ext = {
        'use_head': use_head,
        'always_use_last_head': always_use_last_head,
        'extension': ext_map[network_trainer],
        'param_split': False,
        'network': network,
        'network_trainer': network_trainer,
        'use_model': use_model,
        'tasks_list_with_char': tasks_list_with_char,
        'plans_identifier': plans_identifier,
        'vit_type': vit_type,
        'version': version
    }

    if input_folder is None:
        input_folders = [os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task_name, 'imagesTr') for task_name in evaluate_on_tasks]

    else:
        input_folders = [input_folder]

    for input_folder in input_folders:
        predict_from_folder(params_ext, trainer_path, input_folder, output_folder, fold, save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode="normal", overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=mixed_precision,
                            step_size=step_size, checkpoint_name=chk)

# -- Main function for setup execution -- #
def main():
    run_inference()

if __name__ == "__main__":
    run_inference()