#########################################################################################################
#-----------This class represents the Parameter Search of networks using the extended nnUNet------------#
#-----------                            trainer version.                                    ------------#
#########################################################################################################

import os, argparse, nnunet_ext
from nnunet_ext.utilities.helpful_functions import *
from nnunet_ext.parameter_search.param_searcher import ParamSearcher
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_ext.paths import param_search_output_dir, default_plans_identifier


# -- Extract all extensional trainers in a more generic way -- #
extension_keys = [x for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training")) if 'py' not in x]
trainer_keys = list()
for ext in extension_keys:
    trainer_name = [x[:-3] for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training", ext)) if '.py' in x]
    trainer_keys.extend(trainer_name)
# -- Sort based on the sum of ordinal number per lowered string -- #
extension_keys.sort(key=lambda x: sum([ord(y) for y in x.lower()])), trainer_keys.sort(key=lambda x: sum([ord(y) for y in x.lower()]))

# -- Build mapping for network_trainer to corresponding extension name -- #
EXT_MAP = dict(zip(trainer_keys, extension_keys))
# NOTE: sorted_pairs does not include the nnViTUNetTrainer!


def run_param_search():
    # -- First of all check that evaluation_output_dir is set otherwise we do not perform an evaluation -- #
    assert param_search_output_dir is not None, "Before running any parameter search, please specify the Parameter Search folder (PARAM_SEARCH_FOLDER) as described in the paths.md."

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
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic", required=False, default=False, action="store_true",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.")
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
    
    # -- Additional arguments specificly for training -- #
    parser.add_argument("-t", "--task_ids", nargs="+", help="Specify a list of task ids to train on (ids or names). Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder", required=True)
    parser.add_argument("-f", "--fold",  action='store', type=int, nargs=1, choices=[0, 1, 2, 3, 4],
                        help="Specify on which fold to perform parameter search on. Use a fold between 0, 1, ..., 4 but not \'all\'", required=True)
    parser.add_argument("-s", "--split_at", action='store', type=str, nargs=1, required=True,
                        help='Specify the path in the network in which the split will be performed. '+
                            ' Use a single string for it, specify between layers using a dot \'.\' notation.'+
                            ' This is a required field and no default will be set. Use the same names as'+
                            ' present in the desired network architecture.')
    parser.add_argument('-save_interval', action='store', type=int, nargs=1, required=False, default=15,
                        help='Specify after which epoch interval to update the saved data. Default: every 5th epoch.')
    parser.add_argument('-num_epochs', action='store', type=int, nargs=1, required=False, default=100,
                        help='Specify the number of epochs to train for every experiment.'
                            ' Default: Train for 100 epochs.')
    parser.add_argument('-use_head', action='store', type=str, nargs=1, required=False, default=None,
                        help='Specify which head to use for the evaluation of tasks the network is not trained on. When using a non nn-UNet extension, that' +
                              'is not necessary. If this is not set, always the latest trained head will be used.')
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
    parser.add_argument('--always_use_last_head', action='store_true', default=False,
                        help='If this is set, during the evaluation, always the last head of the network will be used, '+
                             'for every dataset the evaluation is performed on.')
    
    # -- Additional arguments specificly for parameter search -- #
    parser.add_argument('--do_val', action='store_true', default=False,
                        help='Set this if the validation should be performed after every experiment. Default: no validation.')
    parser.add_argument("-fixate", action='store', type=str, nargs='+', required=False, default=None,
                        help="Specify for which parameters values should be fixed. "+
                             "They should have the following structure: param_name:value (using "+
                             "no blank space since one can provide more than one fixed value)."+
                             "If a parameter is not mentioned in fixate, it will be "+
                             "automatically fixated using the default value.")
    parser.add_argument("-random_range", action='store', type=str, nargs='+', required=False, default=None,
                        help="Specify the range of every parameter the method should do the search with. "+
                             "They should have the following structure: param_name:[range_low,range_high] (using "+
                             "no blank space since one can provide more than one random range).")
    parser.add_argument("-random_picks", action='store', type=int, nargs=1, required=False, default=None,
                        help="Specify the number of picks when doing random search. Note that there will be "+
                             "a permutation of the picks ie. hyperparameter settings. ")
    parser.add_argument("-random_seed", action='store', type=int, nargs=1, required=False, default=None,
                        help="Specify if a seed should be used for the random parameter value picks. "+
                             "If not, the seed will be set by the random package using the systems clock as base.")         
    parser.add_argument("-grid_vals", action='store', type=str, nargs='+', required=False, default=None,
                        help="Specify the range of every parameter the method should do the search with. "+
                             "They should have the following structure: param_name:[val1,val2,...] (using "+
                             "no blank space since one can provide more than one grid value list). "+
                             "Note that we permute all possible combinations, ie. the nr of experiments "+
                             "will be the product between all provided values along the hyperparameters.")
    parser.add_argument("-mode",  action='store', type=str, nargs=1, required=True, choices=['grid', 'random'],
                        help="Specify the parameter search method. This can either be \'random\' or \'grid\'.")
    parser.add_argument("--in_parallel", action="store_true", default=False,
                        help="Use this if you want to perform the experiments in parallel on different GPUs. "+
                             "Note that then multiple GPU IDs have to be provided, otherwise only one GPU will be "+
                             "used and this flag has no relevance. Consider this as well when a network needs "+
                             "multiple GPUs for one experiment. In any case, the programm will automatically determine "+
                             "which experiment to run on which GPU.")

    # -- Some additional flags for training etc. -- #
    parser.add_argument("-c", "--continue_training", action="store_true",
                        help="Use this if you want to continue with the parameter search (after killed eg.). The program will determine "
                             "by itself where to continue with the searching, so provide the same arguments as before.")
    parser.add_argument('--use_vit', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will be used instead of the Generic_UNet. '+
                             'Note that then the flags -v, -v_type and --use_mult_gpus should be set accordingly.')
    parser.add_argument('--task_specific_ln', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will have task specific Layer Norms.')
    parser.add_argument('--use_mult_gpus', action='store_true', default=False,
                        help='If this is set, the ViT model will be placed onto a second GPU. '+
                             'When this is set, more than one GPU needs to be provided when using -d.')
    parser.add_argument('--no_transfer_heads', required=False, default=False, action="store_true",
                        help='Set this flag if a new head should not be initialized using the last head'
                            ' during training, ie. the very first head from the initialization of the class is used.'
                            ' Default: The previously trained head is used as initialization of the new head.')
    parser.add_argument('--no_pod', action='store_true', default=False,
                        help='This will only be considered if our own trainers are used. If set, this flag indicates that the POD '+
                             'embedding should not been used.')
    parser.add_argument('--do_LSA', action='store_true', default=False,
                        help='Set this flag if Locality Self-Attention should be used for the ViT.')
    parser.add_argument('--do_SPT', action='store_true', default=False,
                        help='Set this flag if Shifted Patch Tokenization should be used for the ViT.')
    parser.add_argument('--enhanced', required=False, default=False, action="store_true",
                        help='Set this flag if the EWC loss should be changed during the frozen training process (ewc_lambda*e^{-1/3}). '
                             ' Default: The EWC loss will not be altered. --> Makes only sense with our nnUNetTrainerFrozEWC trainer.')


    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet + ViT) arguments -- #
    args = parser.parse_args()
    npz = args.npz
    fp32 = args.fp32
    network = args.network
    find_lr = args.find_lr
    valbest = args.valbest
    use_vit = args.use_vit
    do_pod = not args.no_pod
    enhanced = args.enhanced
    plans_identifier = args.p
    mixed_precision = not fp32
    val_folder = args.val_folder
    deterministic = args.deterministic
    network_trainer = args.network_trainer
    continue_training = args.continue_training
    transfer_heads = not args.no_transfer_heads
    ViT_task_specific_ln = args.task_specific_ln
    decompress_data = not args.use_compressed_data
    always_use_last_head = args.always_use_last_head
    val_disable_overwrite = args.val_disable_overwrite
    disable_next_stage_pred = args.disable_next_stage_pred
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds
    
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
    
    save_interval = args.save_interval
    if isinstance(save_interval, list):    # When the save_interval gets returned as a list, extract the number to avoid later appearing errors
        save_interval = save_interval[0]

    # -- LSA and SPT flags -- #
    do_LSA = args.do_LSA
    do_SPT = args.do_SPT

    # -- Extract the arguments specific for all trainers from argument parser -- #
    fold = args.fold[0] if isinstance(args.fold, list) else args.fold
    tasks = args.task_ids
    split = args.split_at           # String that specifies the path to the layer where the split needs to be done
    use_head = args.use_head        # One task specifying which head should be used
    
    if isinstance(split, list):     # When the split get returned as a list, extract the path to avoid later appearing errors
        split = split[0].strip()
        
    # -- Check that split is string and not empty -- #
    assert len(split) > 0,\
        "When providing a split, ensure that it is not empty, otheriwse no split can be performed."
    # -- Check that the number of tasks is greater than 1, else a conventional nnUNetTrainerV2 should be used -- #
    assert len(tasks) > 1,\
        "When training on only one task, the conventional training of the nnU-Net should be used, not the extension."

    if isinstance(use_head, list):
        use_head = use_head[0]
    
    num_epochs = args.num_epochs        # The number of epochs to train a task
    if isinstance(num_epochs, list):    # When the num_epochs gets returned as a list, extract the number to avoid later appearing errors
        num_epochs = num_epochs[0]

    cuda = args.device

    # -- Assert if device value is out of predefined range and create string to set cuda devices -- #
    for idx, c in enumerate(cuda):
        assert c > -1 and c < 8, 'GPU device ID out of range (0, ..., 7).'
        cuda[idx] = str(c)  # Change type from int to str otherwise join_texts_with_char will throw an error
    cuda = join_texts_with_char(cuda, ',')
    
    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    # -- Check if the user wants to split the network onto multiple GPUs -- #
    split_gpu = args.use_mult_gpus
    if split_gpu:
        assert len(cuda) > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'
    
    # -- Parameter Search specific arguments -- #
    search_mode = args.mode[0] if isinstance(args.mode, list) else args.mode
    perform_validation = args.do_val
    # vals_per_param = args.vals_per_hp
    run_in_parallel = args.in_parallel
    
    # -- Which parameters to fixate and at which values -- #
    fixate_at = args.fixate # Structure: parameter_name:value

    # -- Transform the fixate_at -- #
    fixate_params = dict()
    if fixate_at is not None:
        for x in fixate_at:
            parameter, value = x.split(':') # Structure: parameter_name:value
            fixate_params[parameter] = float(value)

    grid_picks = dict()
    rand_range = dict()
    rand_pick = None
    rand_seed = None
    if search_mode == 'random':
        # -- Extract the random range to be allowed to pick -- #
        rand_rs = args.random_range  # structure: param_name:[range_low,range_high]
        # -- Process the input -- #
        for x in rand_rs:
            param_name, range = (x_.replace(' ', '') for x_ in x.split(':'))
            assert param_name not in fixate_params, "How should the parameter {} be tuned if you fixated it..".format(param_name)
            range = range.replace(' ', '')  # in case the user did not listen to the instructions..
            rand_range[param_name] = (float(range[1:-1].split(',')[0]), float(range[1:-1].split(',')[1]))

        # -- Extract the amount of allowed picks -- #
        rand_pick = args.random_picks[0] if isinstance(args.random_picks, list) else args.random_picks
        rand_seed = args.random_seed[0] if isinstance(args.random_seed, list) else args.random_seed
    else:
        # -- Extract the values for the grid search -- #
        grid_vals = args.grid_vals  # structure: param_name:[val1,val2,...]
        # -- Process the input -- #
        for x in grid_vals: # This will fail if the user provides the arguments with blank spaces within the range values..
            param_name, vals = (x_.replace(' ', '') for x_ in x.split(':'))
            assert param_name not in fixate_params, "How should the parameter {} be tuned if you fixated it..".format(param_name)
            vals = vals.replace(' ', '')  # in case the user did not listen to the instructions..
            grid_picks[param_name] = [float(x_) for x_ in vals[1:-1].split(',')]

    # -- Do some sanity checks before proceeding -- #
    if run_in_parallel and len(args.device) == 1:
        print("The user wanted to perform the searching using multiple GPUs in parallel but only provides one GPU..")
    if search_mode == 'random':
        assert rand_range is not None and rand_pick is not None,\
            "The user selected a random search method but does not provide the value ranges (and/or) nr of allowed value picks for the search.."
    else:
        assert grid_vals is not None, "The user selected a grid search method but does not provide the value list for the search.."
    
    
    # -------------------------------
    # Transform tasks to task names
    # -------------------------------
    # -- Assert if fold is not a number as desired, meaning anything else, like Tuple or whatever -- #
    assert isinstance(fold, int), "To Evaluate multiple tasks with {} trainer, only one or multiple folds specified as integers are allowed..".format(network_trainer)

    # -- Build all necessary task lists -- #
    tasks_for_folder = list()
    if use_head is not None:
        use_head = convert_id_to_task_name(int(use_head)) if not use_head.startswith("Task") else use_head
    for idx, t in enumerate(tasks):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        tasks_for_folder.append(t)

    char_to_join_tasks = '_'
    tasks_list_with_char = (tasks_for_folder, char_to_join_tasks)


    # ----------------------------------------------
    # Perform parameter search based on users input
    # ----------------------------------------------
    # -- Summarize all arguments -- #
    unet_args =  {'network': network, 'network_trainer': network_trainer, 'unpack_data': decompress_data, 'npz': npz,
                  'deterministic': deterministic, 'fp16': mixed_precision, 'find_lr': find_lr, 'valbest': valbest,
                  'plans_identifier': plans_identifier, 'val_folder': val_folder, 'disable_postprocessing_on_folds': disable_postprocessing_on_folds,
                  'val_disable_overwrite': val_disable_overwrite, 'disable_next_stage_pred': disable_next_stage_pred}
    param_args = {'save_interval': save_interval, 'extension': EXT_MAP[network_trainer], 'split_at': split,
                  'tasks_list_with_char': copy.deepcopy(tasks_list_with_char), 'continue_training': continue_training, 'do_pod': do_pod,
                  'mixed_precision': mixed_precision, 'use_vit': use_vit, 'vit_type': vit_type, 'version': version, 'num_epochs': num_epochs,
                  'split_gpu': split_gpu, 'transfer_heads': transfer_heads, 'ViT_task_specific_ln': ViT_task_specific_ln, 'fold': fold,
                  'do_LSA': do_LSA, 'do_SPT': do_SPT, 'do_pod': do_pod, 'search_mode': search_mode, 'grid_picks': grid_picks, 'rand_range': rand_range,
                  'rand_pick': rand_pick, 'rand_seed': rand_seed, 'always_use_last_head': always_use_last_head, 'enhanced': enhanced,
                  'perform_validation': perform_validation, 'fixate_params': fixate_params, 'run_in_parallel': run_in_parallel, **unet_args}
    
    # -- Create the ParamSearcher -- #
    searcher = ParamSearcher(**param_args)

    # -- Do the parameter search as usual -- #
    searcher.start_searching()

# -- Main function for setup execution -- #
def main():
    run_param_search()

if __name__ == "__main__":
    run_param_search()