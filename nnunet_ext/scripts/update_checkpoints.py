import os, argparse
from glob import glob
from collections import OrderedDict
from nnunet_ext.utilities.helpful_functions import *
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet_ext.paths import network_training_output_dir, default_plans_identifier
from nnunet.paths import network_training_output_dir as network_training_output_dir_orig

def modify_checkpoints():
    r"""This function can be used to modify a checkpoint, after it has been moved to a different location or
        if a pre-trained network provided by someone should be used. This is important since the paths of a trainer
        are saved within the checkpoint and high likely lead to non existing locations or locations where a user
        has no permission. By using this function, the checkpoint paths will be modfied so one can use the network without
        problems regarding the paths that were stored during the training of the network. For more information see
        update_networks.md
    """
    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add one argument -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("-trained_on", nargs="+", help="Specify a list of task ids for which the data should be removed. Each of these "
                                                            "ids must have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder")
    parser.add_argument("-use", "--use_model", action='store', type=str, nargs="+", default=None,
                        help="Specify a list of task ids that specify the exact network to change the checkpoint. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder.",
                             required=False)
    parser.add_argument("-f", "--folds", nargs="+", help="Specify on which folds to modify. Use a fold between 0, 1, ..., 4 or \'all\'", required=False)
    parser.add_argument("-rw", "--replace_with", action='store', type=str, nargs=2,
                        help="First specify the part to replace and the path it will be replaced with.", required=True)
    parser.add_argument('--use_vit', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will be used instead of the Generic_UNet. '+
                             'Note that then the flags -v, -v_type, --task_specific_ln and --use_mult_gpus should be set accordingly.')
    parser.add_argument('--task_specific_ln', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet has task specific Layer Norms.')
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
    parser.add_argument('--no_pod', required=False, default=False, action="store_true",
                        help='Set this flag if the POD embedding has been included in the loss calculation (only for our own methods).'
                            ' Default: POD embedding has been included.')
    parser.add_argument('-r', action='store_true', default=False,
                        help='Use this if all subfolders should be scanned for checkpoints. If this is set, '+
                             'all other flags for specifying one simple model will not be considered then.')
    parser.add_argument('-rm', action='store_true', default=False,
                        help='Use this if all subfolders should be scanned for checkpoints. If this is set, '+
                             'then only the checkpoints wrt to the network_trainer are used.')
    
    # -- Extract parser arguments -- #
    args = parser.parse_args()
    fold = args.folds        # List of the folds
    recursive = args.r       # If set, everything in tasks will be considered
    recursive_m = args.rm    # If set, all checkpoints relevant for the network_trainer will be used
    model = args.use_model   # List of the model to use
    network = args.network
    tasks = args.trained_on  # List of the tasks
    plans_identifier = args.p
    trainer = args.network_trainer
    old_part, new_part = args.replace_with
    assert old_part != new_part, 'Why do you want to replace a specific part with the same part ? --> Check your -rw/--replace_with arguments..'

    # -- ViT related arguments -- #
    use_vit = args.use_vit
    vit_type = args.vit_type
    if isinstance(vit_type, list):    # When the vit_type gets returned as a list, extract the type to avoid later appearing errors
        vit_type = vit_type[0].lower()
        
    # -- Extract the desired version -- #
    version = args.version
    if isinstance(version, list):    # When the version gets returned as a list, extract the number to avoid later appearing errors
        version = version[0]

    ViT_task_specific_ln = args.task_specific_ln
    transfer_heads = not args.no_transfer_heads

    # -- LSA and SPT flags -- #
    do_pod = not args.no_pod
    do_LSA = args.do_LSA
    do_SPT = args.do_SPT
    
    # -- If the tasks are None, then set them as an empty list so the loop will not fail -- #
    assert tasks is not None, "Please set the -trained_on flag.."

    # -- When no task is provided, print a Note -- #
    if len(tasks) == 0:
        print('Note: No tasks are provided, so nothing can be changed, be aware of that.')
    
    # -- Check that -r is set if no model is provided -- #
    if model is None:
        assert recursive or recursive_m, "If the model flag is not used than at least set -r or -rm."
    
    # -- Build the model folder based on task ids -- #
    tasks_for_folder = list()
    for t in tasks:
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task to the list -- #
        tasks_for_folder.append(t)

    tasks_joined_name = join_texts_with_char(tasks_for_folder, '_')
    if len(tasks_for_folder) == 1 and not use_vit:  # --> This can only be a nnUNet 
        model_base_folder = os.path.join(network_training_output_dir_orig, network, tasks_joined_name)
    else:
        model_base_folder = os.path.join(network_training_output_dir, network, tasks_joined_name)

    assert os.path.isdir(model_base_folder), 'The folder {} based on the users input does not exist..'.format(model_base_folder)

    # -- If -r than walk through whole directory while keeping track of files with .model.pkl in it at the end -- #
    # -- Here, all trainers will be considered in the folders --> This may take a while depending on the amount -- #
    # -- of trained networks -- #
    if recursive:   # use os walk to go through whole directory
        print("This will modify all checkpoints and plans in the directory: {}.".format(model_base_folder))
        chkpts = [y for x in os.walk(model_base_folder) for y in glob(os.path.join(x[0], '*.model.pkl'))]
        plans = [y for x in os.walk(model_base_folder) for y in glob(os.path.join(x[0], 'plans.pkl'))]
        chkpts.extend(plans)
    
    elif recursive_m:
        assert trainer is not None, 'When using the -rm flag please set network_trainer to specify the folder..'
        print("This will modify all checkpoints and plans in the directory {} belonging to network_trainer {}.".format(model_base_folder, trainer))
        chkpts = [y for x in os.walk(model_base_folder) for y in glob(os.path.join(x[0], '*.model.pkl')) if trainer in y]
        plans = [y for x in os.walk(model_base_folder) for y in glob(os.path.join(x[0], 'plans.pkl'))]
        chkpts.extend(plans)

    else:   # --> User wants a very specific folder, so we obey and do not modify the plans file..
        # -- Transform fold to list if it is set to 'all'
        if fold[0] == 'all':
            fold = list(range(5))
        else: # change each fold type from str to int
            fold = list(map(int, fold))

        # -- Assert if fold is not a number or a list as desired, meaning anything else, like Tuple or whatever -- #
        assert isinstance(fold, (int, list)), "PLease, only use one or multiple folds specified as integers.."

        tasks_for_model = list() 
        for t in model:
            # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
            if not t.startswith("Task"):
                task_id = int(t)
                t = convert_id_to_task_name(task_id)
            # -- Add corresponding task to the list -- #
            tasks_for_model.append(t)
        model_joined_name = join_texts_with_char(tasks_for_model, '_')

        # -- Define the list for the checkpoint paths -- #
        chkpts = list()
        for f in fold:
            if network_training_output_dir in model_base_folder:    # --> If extension than treat it like it
                # -- Specify the correct folder -- #
                if use_vit:
                    # -- Extract the folder name in case we have a ViT -- #
                    folder_n = ''
                    if do_SPT:
                        folder_n += 'SPT'
                    if do_LSA:
                        folder_n += 'LSA' if len(folder_n) == 0 else '_LSA'
                    if len(folder_n) == 0:
                        folder_n = 'traditional'
                    # -- Build folder -- #
                    folder = join(model_base_folder, model_joined_name, trainer+'__'+plans_identifier, Generic_ViT_UNet.__name__+'V'+str(version), vit_type.lower())
                    folder = join(folder, 'task_specific', folder_n) if ViT_task_specific_ln else join(folder, 'not_task_specific', folder_n)
                else:
                    folder = join(model_base_folder, model_joined_name, trainer+'__'+plans_identifier, Generic_UNet.__name__)
                
                folder = join(folder, 'SEQ') if transfer_heads else join(folder, 'MH')
                folder = join(folder, 'fold_'+str(f))

                # -- Re-Modify trainer path for own methods if necessary -- #
                if 'OwnM' in trainer:
                    folder = join(os.path.sep, *folder.split(os.path.sep)[:-1], 'pod' if do_pod else 'no_pod', 'fold_'+str(f))
            else:   # --> nnUNet
                folder = join(model_base_folder, trainer+'__'+plans_identifier)

            assert os.path.isdir(folder), 'The folder {} based on the users input does not exist..'.format(folder)

            # -- Extract the checkpoint paths -- #
            chkpts.extend([y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.model.pkl'))])

    # -- Loop through chkpts, load them and replace everything in them based on old_part, new_part -- #
    for checkpoint in chkpts:
        # -- Load the content -- #
        content = load_pickle(checkpoint)
        # -- Do replacements in all possibile areas -- #
        content = _do_replacement(content, old_part, new_part)
        # -- Dump the checkpoint back so the changes are valid -- #
        write_pickle(content, checkpoint)

def _do_replacement(content, old, new):
    r"""This function goes iterative through the checkpoint and replaces every content based on the users input.
    """
    # -- Loop through the content and do replacements -- #
    if isinstance(content, (int, bool, float)):
        # -- Just return the content, nothing to replace here -- #
        return content
    elif isinstance(content, str):
        # -- D replacement and return then -- #
        content.replace(old, new)
    elif isinstance(content, dict):   # non-primitive types
        content = __dict_replace_value(content, old, new, ordered=True)
    elif isinstance(content, list):   # non-primitive types
        content = __list_replace_value(content, old, new)
    # -- Return the content -- #
    return content

# -- Copied from https://stackoverflow.com/questions/55704719/python-replace-values-in-nested-dictionary -- #
def __dict_replace_value(d, old, new, ordered=False):
    x = {} if not ordered else OrderedDict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = __dict_replace_value(v, old, new)
        elif isinstance(v, (list, tuple)):
            v = __list_replace_value(v, old, new)
        elif isinstance(v, str):
            v = v.replace(old, new)
        x[k] = v
    return x

def __list_replace_value(l, old, new):
    typ = 'tuple' if isinstance(l, tuple) else 'list'
    x = []   # --> Tuple or list
    for e in l:
        if isinstance(e, (list, tuple)):
            e = __list_replace_value(e, old, new)
        elif isinstance(e, dict):
            e = __dict_replace_value(e, old, new)
        elif isinstance(e, str):
            e = e.replace(old, new)
        x.append(e)
    return x if typ == 'list' else tuple(x)
# -- Copied from https://stackoverflow.com/questions/55704719/python-replace-values-in-nested-dictionary -- #

def main():
    modify_checkpoints()

if __name__ == '__main__':
    modify_checkpoints()