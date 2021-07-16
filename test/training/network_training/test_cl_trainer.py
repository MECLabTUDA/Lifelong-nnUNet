#####################################################################################$############
# -- Test suite to test the Multi Head CL Trainers provided with this extension of the nnUNet -- #
##################################################################################################

import os, sys, copy
from importlib.machinery import SourceFileLoader
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.scripts.delete_specified_task import main as delete_specified_task
from nnunet_ext.training.network_training.ewc.nnUNetTrainerEWC import nnUNetTrainerEWC # Own implemented class
from nnunet_ext.training.network_training.lwf.nnUNetTrainerLWF import nnUNetTrainerLWF # Own implemented class
from nnunet_ext.paths import network_training_output_dir_base as old_network_training_output_dir_base
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead# Own implemented class
from nnunet_ext.training.network_training.rehearsal.nnUNetTrainerRehearsal import nnUNetTrainerRehearsal# Own implemented class

# -- Import the test function that tests Multi Head Trainer -- #
file = 'test/training/network_training/test_multi_head_trainer.py' # --> Update this if sth changes
test_multi_head_trainer = SourceFileLoader('test_multi_head_trainer', file).load_module() # --> Update this if sth changes


if __name__ == "__main__":
    # -- Block all prints that are done during testing which are no errors but done in calling functions -- #
    sys.stdout = open(os.devnull, 'w')

    # -- Create all the mappings and arguments for the testing function -- #
    ext_map = {'multihead': nnUNetTrainerMultiHead, 
               'rehearsal': nnUNetTrainerRehearsal,
               'ewc': nnUNetTrainerEWC, 'lwf': nnUNetTrainerLWF}

    # -- Define the tasks to train on -- #
    tasks = ['Task-11_Hippocampus', 'Task-22_Heart', 'Task-33_Prostate']

    # -- Join all task names together with a '_' in between them -- #
    char_to_join_tasks = '_'
    tasks_list_with_char = (tasks, char_to_join_tasks)

    # -- Build the arguments dictionary -- #
    basic_exts = {'unpack_data': True, 'deterministic': False, 'fp16': True, # Every trainer needs those args
                  'save_interval': 1, 'identifier': default_plans_identifier,
                  'tasks_list_with_char': copy.deepcopy(tasks_list_with_char),
                  'use_progress': False}
    reh_args = {'samples_per_ds': 0.15, 'seed': 3299, **basic_exts}
    ewc_args = {'ewc_lambda': 0.4, **basic_exts}
    lwf_args = {'lwf_temperature': 2.0, **basic_exts}
    args_f = {'nnUNetTrainerMultiHead': basic_exts, 'nnUNetTrainerRehearsal': reh_args,
              'nnUNetTrainerEWC': ewc_args, 'nnUNetTrainerLWF': lwf_args}

    # -- Run the test suite and catch any error, delete the folder and raise this error -- #
    try:
        # -- Run the test suite -- #
        test_multi_head_trainer.test_multi_head_trainer(ext_map, args_f)
    except Exception as e:  # Error occured or test failed
        # -- Delete the generated data -- #
        delete_specified_task(False, test_data=True, task_ids=list())
        # -- Enable the prints again -- #
        sys.stdout = sys.__stdout__
        # -- Raise the error -- #
        print("An Error occured or a test failed, for more information see the log file at \'{}\'.".format(old_network_training_output_dir_base))
        raise e

    # -- Ensure that the data is really removed -- #
    delete_specified_task(False, test_data=True, task_ids=list())
    