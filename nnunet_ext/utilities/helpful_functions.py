##########################################################################################################
#------This module contains useful functions that are used throughout the nnUNet_extensions project.-----#
##########################################################################################################

from types import ModuleType
import sys, os, shutil, importlib
from batchgenerators.utilities.file_and_folder_operations import join
from nnunet_ext.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir

def delete_dir_con(path):
    r"""This function deletes the whole content in the folder specified by the path and then deletes the empty folder
        if it still exists.
        :param path: Full path to the folder for which the content needs to be deleted
    """
    # -- Delete content of the folder -- #
    shutil.rmtree(path)

    # -- If the folder still exists, remove it because it is just empty -- #
    if os.path.exists(path):
        os.rmdir(path)

def copy_dir(source, dest):
    r"""This function copies the content of the source folder into dest.
        :param source: Full path to the folder containing the desired content, e.g. ../data/XY
        :param dest: Full path to which source will be copied, e.g. ../copied_data/XY
        This function will then copy everything from ../data/XY/.. into ../copied_data/XY/.
    """
    # -- Check if dest exists, if yes, than empty the folder since, otherwise create the folder -- #
    if os.path.exists(dest):
        # -- Delete directory -- #
        delete_dir_con(dest)

    # -- Copy content from source to dest -- #
    shutil.copytree(source, dest)

def move_dir(source, dest):
    r"""This function moves the content of the source folder into dest.
        :param source: Full path to the folder containing the desired content, e.g. ../data/XY
        :param dest: Full path to which source will be moved, e.g. ../moved_data/XY
        This function will then move everything from ../data/XY/.. into ../moved_data/XY/.
    """
    # -- Check if dest exists, if yes, than empty the folder since, otherwise create the folder -- #
    if not os.path.exists(dest):
        os.makedirs(dest)

    # -- Move content from source to dest -- #
    shutil.move(source, dest, copy_function=shutil.copytree)

    # -- Delete empty folder from which everything has been moved -- #
    try:
        delete_dir_con(source)
    except: # Folder does not exist anymore
        pass

def join_texts_with_char(texts, combine_with):
    r"""This function takes a list of strings and joins them together with the combine_with between each text from texts.
        :param texts: List of strings
        :param combine_with: A char or string with which each element in the list will be connected with
        :return: A String consisting of every string from the list combined using combine_with
    """
    assert isinstance(combine_with, str), "The character to combine the string series from the list with needs to be from type <string>."
    # -- Combine the string series from the list with the combine_with -- #
    return combine_with.join(texts)

def delete_task(task_id):
    r"""This function can be used to sucessfully delete a task in all nnU-Net directories.
        If it does not exists there is nothign to delete.
        :param task_id: A string representing the ID of the task to delete, e.g. '002' for the Decathlong Heart Dataset.
    """
    # -- Ensure that the task_id is of type string since an integer with a beginning 0 is not accepted as integer -- #
    # -- Further ensure that the provided task_id is three digits long so we can be sure it is unique and only one task is removed -- #
    assert isinstance(task_id, str) and len(task_id) == 3, "The provided task_id should be a string and exactly three digits long."
    
    # -- Extract all existing tasks as long as the desired id is in there -- #
    task = [x for x in os.listdir(nnUNet_raw_data) if task_id in x] # Only the one with the mapping id
    
    # -- Check that the list only includes one element, if not raise an error since we do not know what to delete now -- #
    assert len(task) == 1, "There are multiple tasks with the same task_id: {}.".format(join_texts_with_char(task, ' '))

    # -- Delete the task in raw_data, cropped_data and preprocessed_data -- #
    # -- For each deletion make a seperate try, since some script might crashed in between there and some parts are missing/deleted! -- #
    try:
        delete_dir_con(join(nnUNet_raw_data, task[0]))
    except Exception as e:
        print(e)
    try:
        delete_dir_con(join(nnUNet_cropped_data, task[0]))
    except Exception as e:
        print(e)
    try:
        delete_dir_con(join(preprocessing_output_dir, task[0]))
    except Exception as e:
        print(e)

def refresh_mod_imports(mod, reload=False):
    r"""This function can be used especially during generic testing, when a specific import
        needs to be refreshed, ie. resetted or reloaded.
        :param mod: String specifying module name or distinct string (stem like 'nnunet') that is
                    included in a bunch of modules that should all be refreshed/removed from sys modules.
                    If reload is True than mod needs to be a module otherwise the reload fails.
        :param reload: Boolean specifying if using importlib to reload a module.
        Example: All imported modules during a test that are connnected to nnUNet or Lifelong-nnUNet
                 need to be refreshed, then simply provide mod = 'nnunet'.
                 If a specific module needs to be reloaded like MultiHead_Module, then mod = MultiHead_Module
                 and reload = True. Note that mod needs to be of type Module for this to work.
    """
    # -- When the user wants to reload a Module, then use importlib -- #
    if reload == True:
        # -- Check that mod is a Module, otherwise importlib will throw an error -- #
        assert isinstance(mod, ModuleType), "When trying to reload a module, then please provide a module."
        # -- Reload/Reimport mod -- #
        importlib.reload(mod)
    # -- User does not want to use importlib, ie. the modules that contain mod in their name are removed from the sys modules -- #
    else:
        # -- Check that mod is of type string in this case -- #
        assert isinstance(mod, str), "When removing all modules based on a mapping string, than mod should be a string."
        # -- loop through all present modules in sys -- #
        for key in list(sys.modules.keys()):
            # -- If mod is part of the key name in any shape or form -- #
            if mod in key:
                # -- Remove it from the sys modules -- #
                del sys.modules[key]