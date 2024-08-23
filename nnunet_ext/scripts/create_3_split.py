#use this file to split your favorite dataset into train, val and test by copying to a new task
from nnunet_ext.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier, nnUNet_raw_data
from distutils.dir_util import copy_tree
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join, load_pickle, write_pickle
import os, random


def do_split(old_task_name: str, new_task_name:str, NUM_TEST_DATA: float): 

    
    os.symlink(join(nnUNet_raw_data, old_task_name), join(nnUNet_raw_data, new_task_name))    

    random.seed(123454321)
    maybe_mkdir_p(join(preprocessing_output_dir, new_task_name))
    for file in os.listdir(join(preprocessing_output_dir, old_task_name)):
        if os.path.basename(file) == "splits_final.pkl":
            splits_final = load_pickle(join(preprocessing_output_dir, old_task_name, file))

            new_splits = []
            for fold in range(len(splits_final)):
                d = dict()
                d["val"] = splits_final[fold]["val"]

                random.shuffle(splits_final[fold]["train"])
                num_test_data = int(len(splits_final[fold]["train"]) * NUM_TEST_DATA)
                d["test"] = splits_final[fold]["train"][:num_test_data]
                d["train"] = splits_final[fold]["train"][num_test_data:]

                assert len(d["train"]) + len(d["val"]) + len(d["test"]) == len(splits_final[fold]["train"]) + len(splits_final[fold]["val"])
                assert set(d["train"]).isdisjoint(set(d["test"]))
                print(len(d["train"]), len(d["val"]), len(d["test"]))

                new_splits.append(d)
            write_pickle(new_splits, join(preprocessing_output_dir, new_task_name, "splits_final.pkl"))
        else:
            os.symlink(join(preprocessing_output_dir, old_task_name, file), join(preprocessing_output_dir, new_task_name, os.path.basename(file)))



if __name__ == '__main__':
    do_split("Task505_Prostate_MEDSeg", "Task506_Prostate_MEDSeg_split", 0.3)