###################################################################################################
#------This module contains useful functions for trainers from the nnUNet_extensions project.-----#
###################################################################################################

import torch
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

def get_prev_trainers(previous_task_names=list(), network_name=None, tasks_joined_name=None,
                      already_trained_on=dict(), fold=None, extension=None, prev_trainer=None):
    r"""This function is used to build a ModuleList with all models given a certain list of previous tasks.
        This can be used for EWC, LWF and other methods.
        :param previous_task_names: List of task_names for which a model has finished with training
        :param network_name: Name of the network the models have been trained on --> should be the same used by the calling function
        :param tasks_joined_name: Joined name of all tasks the model trains on --> Navigator to the right folder in which the models are stored
        :param already_trained_on: Dictionary with information of the training --> should be the same used by the calling function
        :param fold: An integer indicating to load the models trained on which fold --> should be the same used by the calling function
        :param extension: Name of the extension that is used --> should be the same used by the calling function
        :param prev_trainer: None, the name or class of the previous trainer. Only if the calling function has an initialization using another trainer like nnUNetTrainerV2.
        :return: ModuleList with initialized trainers
    """
    # -- Sanity checks -- #
    assert len(previous_task_names) != 0, "To use this function, the previous_task_names list should contain at least one element."
    assert network_name is not None, "Before a trained model can be loaded and initialized, the network needs to be known."
    assert tasks_joined_name is not None, "To navigate to the right folder with the trained models the (joined) name of all tasks should be provided."
    assert len(already_trained_on) is not None, "Provide a valid already_trained_on that includes important informations for the restoring of models."
    assert isinstance(fold, int) and fold >= 0 and fold < 6, "Provide a valid fold to load the right models. The fold needs to be an integer number between 0 and 5."
    
    # -- Initialize a ModleList in which all previous tasks models will be stored -- #
    trainers = torch.nn.ModuleList()

    # -- Loop through previous tasks and initialize its models -- #
    running_task_list = list()
    for task in previous_task_names:
        # -- Update running task list and create running task which are all (trained tasks and current task joined) for output folder name -- #
        running_task_list.append(task)
        running_task = join_texts_with_char(running_task_list, '_')

        # -- Get default configuration for nnunet/nnunet_ext model (finished training) -- #
        plans_file, output_folder, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(network_name, task, running_task, prev_trainer, tasks_joined_name,\
                                                  already_trained_on[str(fold)]['used_identifier'],\
                                                  extension_type=extension)
                                                  
        # -- Ensure that trainer_class is not None -- #
        if trainer_class is None:
            raise RuntimeError("Could not find trainer class in nnunet.training.network_training nor nnunet_ext.training.network_training")

        # -- Load the corresponding trainer based on trainer_class --> Special arguments are not necessary to set, since -- #
        # -- the trainers are not used to train but they will be restored for predictions instead -- #
        if trainer_class.__name__ == str(nnUNetTrainerV2).split('.')[-1][:-2]:   # nnUNetTrainerV2
            prev_trainer = trainer_class(plans_file, fold, output_folder=output_folder, dataset_directory=dataset_directory,
                                         batch_dice=batch_dice, stage=stage)
        else:
            prev_trainer = trainer_class(plans_file, fold, output_folder=output_folder, dataset_directory=dataset_directory,
                                            batch_dice=batch_dice, stage=stage, already_trained_on=already_trained_on,
                                            identifier=already_trained_on[str(fold)]['used_identifier'], extension=extension,
                                            tasks_joined_name=tasks_joined_name, trainer_class_name=trainer_class.__name__)

        # -- Initialize prev_trainer -- #
        prev_trainer.initialize(training=False)

        # -- Set the trainer to evaluation so it can be properly used for inference -- #
        #prev_trainer.network.eval()

        #for name, param in prev_trainer.network.named_parameters():
        #    print(param.grad)

        # -- Add the model to the trainers -- #
        trainers.append(prev_trainer.network)

    # -- Return the list of initialized trainers -- #
    return trainers