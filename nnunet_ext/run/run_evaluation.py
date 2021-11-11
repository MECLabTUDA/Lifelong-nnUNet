#########################################################################################################
#----------This class represents the Evaluation of networks using the extended nnUNet trainer-----------#
#----------                                     version.                                     -----------#
#########################################################################################################

import os, argparse
from nnunet_ext.paths import evaluation_output_dir
from nnunet_ext.evaluation.evaluator import Evaluator
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

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
    parser.add_argument("-f", "--folds",  action='store', type=int, nargs="+",
                        help="Specify on which folds to train on. Use a fold between 0, 1, ..., 5 or \'all\'", required=True)
    parser.add_argument("-trained_on", action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network has trained with to specify the correct path to the networks. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument("-use_model", "--use", action='store', type=str, nargs="+",
                        help="Specify a list of task ids that specify the exact network that should be used for evaluation. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument('-use_head', action='store', type=str, nargs=1, required=False, default=None,
                        help='Specify which head to use for the evaluation. When using a non nn-UNet extension, that' +
                              'is not necessary. If this is not set, always the latest trained head will be used.')
    parser.add_argument("--fp32_used", required=False, default=False, action="store_true",
                        help="Specify is mixed precision has been used during training or not")
    parser.add_argument("-evaluate_on",  action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network will be evaluated on. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=[0],
                        help='Try to train the model on the GPU device with <DEVICE> ID. '+
                            ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.'+
                            ' Default: Only GPU device with ID 0 will be used.')
    parser.add_argument('-store_csv', required=False, default=False, action="store_true",
                        help='Set this flag if the validation data and any other data if applicable should be stored'
                            ' as a .csv file as well. Default: .csv are not created.')
    parser.add_argument("-v", "--version", action='store', type=int, nargs=1, default=[1],
                        help='Select the ViT input building version. Currently there are only three'+
                            ' possibilities: 1, 2 or 3.'+
                            ' Default: version one will be used. For more references wrt, to the versions, see the docs.')
    parser.add_argument("-v_type", "--vit_type", action='store', type=str, nargs=1, default='base',
                        help='Specify the ViT architecture. Currently there are only three'+
                            ' possibilities: base, large or huge.'+
                            ' Default: The smallest ViT architecture, i.e. base will be used.')

    # -- Build mapping for network_trainer to corresponding extension name -- #
    ext_map = {'nnUNetTrainerMultiHead': 'multihead', 'nnUNetTrainerSequential': 'sequential',
               'nnUNetTrainerRehearsal': 'rehearsal', 'nnUNetTrainerEWC': 'ewc', 'nnUNetTrainerLWF': 'lwf',
               'nnUNetTrainerV2': 'standard', 'nnViTUNetTrainer': None, 'nnViTUNetTrainerCascadeFullRes': None}


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
    use_head = args.use_head        # One task specifying which head should be used
    if isinstance(use_head, list):
        use_head = use_head[0]
        
    # -- Extract further arguments -- #
    save_csv = args.store_csv
    fold = args.folds
    cuda = args.device
    mixed_precision = not args.fp32_used
    
    # -- Extract the vit_type structure and check it is one from the existing ones -- #s
    vit_type = args.vit_type
    if isinstance(vit_type, list):    # When the vit_type gets returned as a list, extract the type to avoid later appearing errors
        vit_type = vit_type[0].lower()
    assert vit_type in ['base', 'large', 'huge'], 'Please provide one of the following three existing ViT types: base, large or huge..'

    # -- Assert if device value is ot of predefined range and create string to set cuda devices -- #
    for idx, c in enumerate(cuda):
        assert c > -1 and c < 8, 'GPU device ID out of range (0, ..., 7).'
        cuda[idx] = str(c)  # Change type from int to str otherwise join_texts_with_char will throw an error
    cuda = join_texts_with_char(cuda, ',')
    
    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    # -- Extract the desired version, only considered in case of ViT Trainer -- #
    version = args.version
    if isinstance(version, list):    # When the version gets returned as a list, extract the number to avoid later appearing errors
        version = version[0]
    assert version in [1, 2, 3], 'We only provide three versions, namely 1, 2 or 3, but not {}..'.format(version)
    
    
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


    # ----------------------------------------------
    # Define dict with arguments for function calls
    # ----------------------------------------------
    # -- Join all task names together with a '_' in between them -- #
    char_to_join_tasks = '_'
    
    
    # ---------------------------------------------
    # Evaluate for each task and all provided folds
    # ---------------------------------------------
    evaluator = Evaluator(network, network_trainer, (tasks_for_folder, char_to_join_tasks), (use_model_w_tasks, char_to_join_tasks), 
                          version, vit_type, plans_identifier, mixed_precision, ext_map[network_trainer], save_csv)
    evaluator.evaluate_on(fold, evaluate_on_tasks, use_head)


if __name__ == "__main__":
    run_evaluation()