#########################################################################################################
#----------This class represents the default model configurations for the nnUNet extensions.------------#
#----------Implementation inspired by original implementation, same code is marked as such.-------------#
#########################################################################################################

import nnunet, nnunet_ext
from nnunet_ext.utilities.helpful_functions import copy_dir
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.experiment_planning.summarize_plans import summarize_plans
from nnunet_ext.nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.paths import network_training_output_dir as orig_network_training_output_dir
from nnunet_ext.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier

#------------------------------------------- Partially copied from original implementation -------------------------------------------#
def get_default_configuration(network, task, running_task, network_trainer, tasks_joined_name, plans_identifier=default_plans_identifier,
                              search_in=None, base_module=None, extension_type='sequential'):
    r"""This function extracts paths to the plans_file, specifies the output_folder_name, dataset_directory, batch_dice, stage, and trainer_class.
        The extension type specifies which nnUNet extension will be used (sequential, rehearsal, etc.).
    """
    # -- If network_trainer is actual trainer than transform it into a string -- #
    if not isinstance(network_trainer, str):
        network_trainer = str(network_trainer).split('.')[-1][:-2].split(' ')[0]    # May be sth like "TrainerXY at ..."
        
    # -- Extract network_trainer type -- #
    is_classic_trainer = network_trainer == str(nnUNetTrainerV2).split('.')[-1][:-2]
    
    # -- If search_in not provided set it with base_module -- #
    if search_in is None:
        search_in = (nnunet_ext.__path__[0], "training", "network_training", extension_type)
        base_module = 'nnunet_ext.training.network_training.' + extension_type

    # -- If the trainer to extract is of not one from the provided extension, then search in nnU-Net module -- #
    if is_classic_trainer:
        search_in = (nnunet.__path__[0], "training", "network_training")
        base_module = 'nnunet.training.network_training'

    # -- Update possible network list -- #
    assert network in ['2d', '3d_lowres', '3d_fullres'], \
        "The network for the nnU-Net CL extension can only be one of the following: \'2d\', \'3d_lowres\', \'3d_fullres\'"
    dataset_directory = join(preprocessing_output_dir, task)

    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")

    plans = load_pickle(plans_file)
    possible_stages = list(plans['plans_per_stage'].keys())

    if network == '2d' or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    trainer_class = recursive_find_python_class([join(*search_in)], network_trainer,
                                                current_module=base_module)
                                                
    output_folder_name = join(network_training_output_dir, network, tasks_joined_name, running_task, network_trainer + "__" + plans_identifier)

    # -- Copy the model to nnunet_ext folder if it is not of sequential origin -- #
    if is_classic_trainer:
        source = join(orig_network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)
        dest = output_folder_name
        # -- NOTE: If dest exists, it will be emptied, since the folder should be empty at this point, --#
        # --       because the copied model is the base of the sequential training -- #
        copy_dir(source, dest)

    print("###############################################")
    print("I am running the following nnUNet: %s" % network)
    print("My trainer class is: ", trainer_class)
    print("For that I will be using the following configuration:")
    summarize_plans(plans_file)
    print("I am using stage %d from these plans" % stage)

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("I am using batch dice + CE loss")
    else:
        batch_dice = False
        print("I am using sample dice + CE loss")

    print("\nI am using data from this folder: ", join(dataset_directory, plans['data_identifier']))
    print("###############################################")
    
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class
#------------------------------------------- Partially copied from original implementation -------------------------------------------#