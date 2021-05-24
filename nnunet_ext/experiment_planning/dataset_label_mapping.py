#########################################################################################################
#----------This class represents the Transformation of Decathlon(-like) datasets by using a mapping-----#
#----------to change the labels masks. This is based on the dataset.json of the provided dataset.-------#
#########################################################################################################

import os
import argparse
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.utils import split_4d
from nnunet_ext.paths import preprocessing_output_dir, nnUNet_raw_data
from nnunet.experiment_planning.nnUNet_convert_decathlon_task import crawl_and_remove_hidden_from_decathlon

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
        img_array[img_array == int(original_labels_pair.split(' --> ')[1])] = -int(new_label)

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
    # -----------------------------------------------------------------------------------------


    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser arguments -- #
    args = parser.parse_args()
    tasks_in = args.tasks_in_path    # List of the paths to the tasks
    task_out = args.tasks_out_ids   # List of the tasks IDss
    mappings = args.mapping_files_path       # List of the paths to the mapping files
    disable_pp = args.no_pp # Flag that specifies if nnUNet_plan_and_preprocess should be performed

    # -- Sanity checks for input -- #
    if task_out is not None:
        assert len(tasks_in) == len(task_out), "Number of input tasks should be equal to the number of specified output tasks IDs."
    assert len(tasks_in) == len(mappings), "Number of input tasks should be equal to the number of provided mappings for transformation."

    # -- Check that each mapping path ends with .json -- #
    for mapping in mappings:
        assert mapping[-5:] == '.json', "The mapping files should be in .json format."


    # ------------------------------------------------
    # Transform masks based on the provided mappings
    # ------------------------------------------------
    # -- Loop through the input tasks and transform the label masks -- #
    for idx, in_task in enumerate(tasks_in):
        # -- Load the corresponding mapping -- #
        mapping = load_json(mappings[idx])

        # -- Update user and use function from nnUNet to check for the right dataset format -- #
        print("Checking if the provided dataset has a Decathlon-like structure..")
        crawl_and_remove_hidden_from_decathlon(in_task)

        # -- Extract task name --> copied from nnUNet split4d function (nnunet/experiment_planning/utils.py) -- #
        full_task_name = in_task.split("/")[-1]
        task_name = full_task_name[7:]

        # -- Use nnUNet provided split_4d function that is used for Decathlon data but check before if task is unique! -- #
        out_task = task_out[idx]
        assert ("Task%03.0d_" % out_task + task_name not in os.listdir(preprocessing_output_dir)\
                and "Task%03.0d_" % out_task + task_name not in os.listdir(nnUNet_raw_data)), "The task is not unique, use another one.."
        print("Perform 4D split..")
        split_4d(in_task, args.p, out_task)

        # -- Define the base for correct loading and saving -- #
        # -- Load dataset.json that includes informations about the dataset, for instance the different labels that need to be changed -- #
        base = join(nnUNet_raw_data, "Task%03.0d_" % out_task + task_name)
        dataset_info = load_json(join(base, 'dataset.json'))

        # -- Check if the provided mapping matches the dataset_info. A mapping key consist of pairs of the old label description and the corresponding label -- #
        # -- whereas the values represent the new label. All labels that are not listed in the mapping are classified as background! -- #
        # -- In parallel, build the new labels dictionary that will be changed in the dataset_info file after transformation -- #
        new_labels = {'0': dataset_info['labels']['0']}   # Add the background label to new_labels since this will never change
        for original_labels_pair, new_label in mapping.items():
            # -- Extract the label description and label value of the original label from the mapping -- #
            old_label_description, old_label = original_labels_pair.split(' --> ')

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

        # -- Plan and preprocess the new dataset if the flag is not set -- #
        if not disable_pp:
            # -- Update user -- #
            print("Performing planning and preprocessing of task {}..".format("Task%03.0d_" % out_task + task_name))
            # -- Execute the nnUNet_plan_and_preprocess command -- #
            os.system('nnUNet_plan_and_preprocess -t ' + str(out_task))