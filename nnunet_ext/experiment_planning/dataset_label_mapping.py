#########################################################################################################
#----------This class represents the Transformation of Decathlon(-like) datasets by using a mapping-----#
#----------to change the labels masks. This is based on the dataset.json of the provided dataset.-------#
#########################################################################################################

import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.utils import split_4d
from nnunet_ext.utilities.helpful_functions import delete_dir_con
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.paths import preprocessing_output_dir, nnUNet_raw_data, nnUNet_cropped_data
from nnunet.experiment_planning.nnUNet_convert_decathlon_task import crawl_and_remove_hidden_from_decathlon

def _extract_desired_channels(path, task_name, channels):
    r"""This function transforms all nifti scans specified in the path in such a way that only the desired
        channels are extracted. In this process a new folder is generated that needs to be deleted after 
        the processing of the data (beyond this function) is finished.
        :param path: Path to the corresponding dataset, like '/.../Task05_Prostate'
        :param task_name: The full name of the task, like 'Task05_Prostate'
        :param channels: List of integers representing the channels that should be selected (should be 0-based),
                         like [0, 2]
        :return: Either None or the name of the path to the new folder if it exists
        NOTE: This function expects Decathlon-Like datastructure whereas the path leads to all the folders,
              namely 'imagesTr', 'imagesTs' and 'labelsTr'. It copies the file to a new path, whereas the name
              is the same as transmitted, extended with '_mod'."""
    # -- Ensure that channels is not all, if so just return -- #
    if channels == 'all':
        return
    
    # -- Ensure that the path exists and is a directory -- #
    assert os.path.isdir(path), "The provided path does not point to a directory: {}.".format(path)

    # -- Define the folder names that might exists and contain data with multiple channels -- #
    pos_folders = ['imagesTr', 'imagesTs']

    # -- Define the new task_name with respect to return -- #
    new_task = task_name+'_mod'

    # -- Create a new directory for the 'new' task, which contains the modified images -- #
    new_path = join(path.replace(task_name, new_task))
    assert not os.path.exists(new_path), "The generated task at {} name for temporary copies already exists although it should not..".format(new_path)

    # -- Create all necessary directories -- #
    os.makedirs(new_path)

    # -- Copy the labels folder (since they always have one channel) and all other files except imagesTr, imagesTs -- #
    shutil.copytree(join(path, "labelsTr"), join(new_path, "labelsTr"))
    shutil.copy(join(path, "dataset.json"), new_path)

    # -- Go into each folder (if it exists) and then extract the desired channels -- #
    for folder in pos_folders:
        # -- Create a new path and folder-- #
        base = join(path, folder)
        # -- Only proceed if the directory exists and if it contains files -- #
        if os.path.isdir(base) and len(os.listdir(base)) > 0:
            # -- Create new directory in the new_tasks path -- #
            os.makedirs(join(new_path, folder))
            # -- Loop through the files -- #
            for file in os.listdir(base):
                # -- Load the file if it is in nifti format and a non binary file -- #
                if 'nii.gz' in file and '._' not in file:
                    # -- Load the file in form of a (SimpleITK -->) array -- #
                    img = sitk.GetArrayFromImage(sitk.ReadImage(join(base, file)))
                    # -- Based on channels extract the right slices -- #
                    if len(img.shape) == 4:
                        try:
                            img = img[channels, :, :, :]
                            if len(channels) == 1: # Only one channel, so go from 4D to 3D
                                img = img[0]
                        except:
                            assert False, "You provided the wrong channel(s): {}. The image has \'{}\' channel(s)..".format(channels, img.shape[0])
                    # -- Save the image again at new location -- #
                    sitk.WriteImage(sitk.GetImageFromArray(img), join(new_path, folder, file))

    return new_path  # Return the path to the new task

def _perform_transformation_on_mask_using_mapping(mask, mapping):
    r"""This function changes the labeling in the mask given a specified mapping. The mask should be a SimpleITK image."""
    # -- Ensure that mapping is not empty -- #
    assert mapping is not None and len(mapping) != 0, "The mapping dict is empty --> can not change labels according to empty mapping."
    
    # -- Load the mask using SimpleITK and convert it to a numpy array for transformation -- #
    img_array = sitk.GetArrayFromImage(mask).astype(np.float32)

    # -- Now, loop through the mappings again and start to transform the mask using the mapping -- #
    for original_labels_pair, new_label in mapping.items():
        # -- Change the original label to the new label and set everything else to background (= 0) -- #
        # -- Change the old label with the negative new_label, so there are no side effects if new_label is one of old labels -- #
        img_array[img_array == int([x.strip() for x in original_labels_pair.split('-->')][1])] = -int(new_label)

    # -- Set everything that is greater than 0 to 0 -- #
    img_array[img_array > 0] = 0    # Set all old labels to background

    # -- Invert all negative labels to get the desired (transformed) mask -- #
    img_array = np.absolute(img_array)

    # -- Convert the transformed numpy array back to a SimpleITK image and save the new mask by overwriting the old one -- #
    mask = sitk.GetImageFromArray(img_array)
    
    # -- Return the new mask image -- #
    return mask
            

def main():
    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add arguments -- #
    parser = argparse.ArgumentParser()
    
    # NOTE: The input task should have the same structure as for the conventional nnUNet!
    parser.add_argument("-t_in", "--tasks_in_path", nargs="+", help="Specify one or a list of paths to tasks TaskXX_TASKNAME folders.", required=True)
    parser.add_argument("-t_out", "--tasks_out_ids", nargs="+", type=int, help="Specify the unique task ids for the output folders. "
                                                                               "Since the task ids of the input folder are already used "
                                                                               "with the original (unchanged) masked this is required."
                                                                               "Keep in mind that IDs need to be unique --> This will be tested!", required=True)
    parser.add_argument("-m", "--mapping_files_path", nargs="+", help="Specify a list of paths to the mapping (.json) files corresponding to the task ids.",
                        required=True)
    parser.add_argument("-c", "--channels", nargs="+", help="Specify which channels to extract. Use (possible) indices 0, 1, ... or \'all\'."
                                                            " When using multiple tasks, consider that it is used for each task. Further consider"
                                                            " that the indices are all 0-based. If not specified, all channels will be extracted", required=False)
    parser.add_argument("-p", required=False, default=default_num_threads, type=int,
                        help="Use this to specify how many processes are used to run the script. "
                             "Default is %d" % default_num_threads)
    parser.add_argument("--no_pp", "--disable_plan_preprocess_tasks", required=False, action='store_true', default=False,
                        help="Set this if the plan and preprocessing step for each task using nnUNet_plan_and_preprocess "
                             "should not be performed after a transformation. "
                             "Default: After each task is transformed, nnUNet_plan_and_preprocess is performed.")

    
    # -----------------------------------------------------------------------------------------
    #                           Mapping .json strcutre definition
    # The mapping file should be a .json file and needs to have the following structure:
    #   {
    #        "old_label_description --> old_label": new_label,
    #        ...
    #   }
    # For example changing in Hippocampus Posterior to 1 and Anterior to 2:
    #   {
    #        "Posterior --> 2": 1,
    #        "Anterior --> 1": 2
    #   }
    # NOTE: Everything that is not included in the mapping will be set to background,
    #       ie. the following will only set Posterior to 1 and Anterior will be set to 0:
    #   {
    #        "Posterior --> 2": 1
    #   }
    # NOTE: Spaces before or after --> is not mandatory and will be stripped either way.
    # -----------------------------------------------------------------------------------------


    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser arguments -- #
    args = parser.parse_args()
    tasks_in = args.tasks_in_path    # List of the paths to the tasks
    task_out = args.tasks_out_ids   # List of the tasks IDs
    mappings = args.mapping_files_path       # List of the paths to the mapping files
    channels = args.channels       # List of the channels to extract
    disable_pp = args.no_pp # Flag that specifies if nnUNet_plan_and_preprocess should be performed

    # -- Sanity checks for input -- #
    if task_out is not None:
        assert len(tasks_in) == len(task_out), "Number of input tasks should be equal to the number of specified output tasks IDs."
    assert len(tasks_in) == len(mappings), "Number of input tasks should be equal to the number of provided mappings for transformation."

    # -- Check that each mapping path ends with .json -- #
    for mapping in mappings:
        assert mapping[-5:] == '.json', "The mapping files should be in .json format."

    # -- If channels is not set or ['all'], set it to all
    if channels is None or channels[0] == 'all':
        channels = 'all'
    else: # Ensure that the list contains only integers
        channels = list(map(int, channels))
        

    # ------------------------------------------------
    # Transform masks based on the provided mappings
    # ------------------------------------------------
    # -- Loop through the input tasks and transform the label masks -- #
    for idx, in_task in enumerate(tasks_in):
        # -- Extract task name --> copied from nnUNet split4d function (nnunet/experiment_planning/utils.py) -- #
        full_task_name = in_task.split("/")[-1]
        task_name = full_task_name[7:]
        mod_in_task = in_task

        # -- Before doing anything further, check that the tasks are unique -- #
        out_task = task_out[idx]
        assert ("Task%03.0d_" % out_task + task_name not in os.listdir(preprocessing_output_dir)\
                and "Task%03.0d_" % out_task + task_name not in os.listdir(nnUNet_raw_data)\
                and "Task%03.0d_" % out_task + task_name not in os.listdir(nnUNet_cropped_data)),\
                    "The task is not unique, use another one or delete all tasks in preprocessing_output_dir, nnUNet_raw_data and nnUNet_cropped_data.."
        
        # -- Load the corresponding mapping -- #
        mapping = load_json(mappings[idx])

        # -- Update user and use function from nnUNet to check for the right dataset format -- #
        print("Checking if the provided dataset has a Decathlon-like structure..")
        crawl_and_remove_hidden_from_decathlon(in_task)

        # -- Select the desired channels from the dataset if not all are desired -- #
        if channels != 'all':
            print("Extracting only the desired channels from the scans..")
            mod_in_task = _extract_desired_channels(in_task, full_task_name, channels)

        # -- Load dataset.json that includes informations about the dataset, for instance the different labels that need to be changed -- #
        dataset_info = load_json(join(in_task, 'dataset.json'))

        # -- Rename the original dataset.json file -- #
        os.rename(join(in_task, 'dataset.json'), join(in_task, 'tmp_dataset.json'))

        # -- Updating modalities in dataset file if channels have been selected -- #
        if channels != 'all':
            # -- Copy the dict keys, otherwise an error will pop up while looping through dict and deleting in parallel -- #
            modalities = list(dataset_info['modality'].keys())
            # -- Loop through the modality keys and delete them if they were not selected -- #
            for key in modalities:
                if int(key) not in channels: # If it is not in the desired selection delete the entry
                    # -- Delete the modality entry -- #
                    del dataset_info['modality'][key]
                    # -- Delete from quantitative as well ? -- # 
                    dataset_info['quantitative'].remove(int(key))
        
        # -- Save the transformed dataset as a file that is then used for planning etc. -- #
        save_json(dataset_info, join(in_task, 'dataset.json'))
        
        # -- Use nnUNet provided split_4d function that is used for Decathlon data but check before if task is unique! -- #
        print("Perform 4D split..")
        split_4d(mod_in_task, args.p, out_task)

        # -- Delete the copied and modified dataset.json file -- #
        os.remove(join(in_task, 'dataset.json'))

        # -- Rename the original dataset.json file  back to original name -- #
        os.rename(join(in_task, 'tmp_dataset.json'), join(in_task, 'dataset.json'))

        # -- Define the base for correct loading and saving -- #
        base = join(nnUNet_raw_data, "Task%03.0d_" % out_task + task_name)

        # -- Do not to forget to change the name from the new generated task by split_4d -- #
        if '_mod' in mod_in_task: # Modified task name because the generated folder is being used (correctly) after channel selection
            os.rename(base+'_mod', base)

        # -- Check if the provided mapping matches the dataset_info. A mapping key consist of pairs of the old label description and the corresponding label -- #
        # -- whereas the values represent the new label. All labels that are not listed in the mapping are classified as background! -- #
        # -- In parallel, build the new labels dictionary that will be changed in the dataset_info file after transformation -- #
        new_labels = {'0': dataset_info['labels']['0']}   # Add the background label to new_labels since this will never change
        for original_labels_pair, new_label in mapping.items():
            # -- Extract the label description and label value of the original label from the mapping -- #
            old_label_description, old_label = [x.strip() for x in original_labels_pair.split('-->')]

            # -- Check if the labels description is in the dataset_info under the specified label, if not raise error -- #
            assert dataset_info['labels'][old_label] == old_label_description, "The provided mapping of labels can not be "\
                                                                               "resolved since it does not match the datasets information .json file."

            # -- Add new label to new_labels -- #
            new_labels[str(new_label)] = old_label_description
        
        # -- Extract all mask paths from the preprocessed directory -- #
        masks = [x for x in os.listdir(join(base, 'labelsTr')) if '._' not in x]

        # -- Loop through all prediction masks and change the labels based on original_labels_pair, new_labels -- #
        for mask in tqdm(masks, ascii=True, desc="Transforming masks of task {}:".format(task_name)):
            # -- Load the mask using SimpleITK and convert it to a numpy array for transformation -- #
            m_img = sitk.ReadImage(join(base, 'labelsTr', mask))

            # -- Transform the image mask -- #
            m_img = _perform_transformation_on_mask_using_mapping(m_img, mapping)

            # -- Save the image -- #
            sitk.WriteImage(m_img, join(base, 'labelsTr', mask))
        
        # -- Now, when this is finished, update the dataset.json file -- #
        dataset_info['labels'] = new_labels
        save_json(dataset_info, join(base, 'dataset.json'))

        # -- Do not forget to delete the copied directory if it exists -- #
        if mod_in_task != in_task:
            delete_dir_con(mod_in_task) # Delete the whole folder

        # -- Plan and preprocess the new dataset if the flag is not set -- #
        if not disable_pp:
            # -- Update user -- #
            print("Performing planning and preprocessing of task {}..".format("Task%03.0d_" % out_task + task_name))
            # -- Execute the nnUNet_plan_and_preprocess command -- #
            os.system('nnUNet_plan_and_preprocess -t ' + str(out_task))


if __name__ == "__main__":
    main()