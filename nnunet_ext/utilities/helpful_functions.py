##########################################################################################################
#------This module contains useful functions that are used throughout the nnUNet_extensions project.-----#
##########################################################################################################

import os, shutil

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
    delete_dir_con(source)

def join_texts_with_char(texts, combine_with):
    r"""This function takes a list of strings and joins them together with the combine_with between each text from texts.
        :param texts: List of strings
        :param combine_with: A char or string with which each element in the list will be connected with
        :return: A String consisting of every string from the list combined using combine_with
    """
    assert isinstance(combine_with, str), "The character to combine the string series from the list with needs to be from type <string>."
    # -- Combine the string series from the list with the combine_with -- #
    return combine_with.join(texts)