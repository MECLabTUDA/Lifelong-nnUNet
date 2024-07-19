#use this file to split your favorite dataset into train, val and test by copying to a new task
from nnunet_ext.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier, nnUNet_raw_data
from distutils.dir_util import copy_tree
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join, load_pickle, write_pickle
import os, random


def do(TASK: int, NUM_TEST_DATA): 
    if TASK == 0:
        old_task_name = "Task008_mHeartA"
        new_task_name = "Task108_mHeartA"
    elif TASK == 1:
        old_task_name = "Task009_mHeartB"
        new_task_name = "Task109_mHeartB"
    elif TASK == 2:
        old_task_name = "Task011_Prostate-BIDMC"
        new_task_name = "Task111_Prostate-BIDMC"
    elif TASK == 3:
        old_task_name = "Task012_Prostate-I2CVB"
        new_task_name = "Task112_Prostate-I2CVB"
    elif TASK == 4:
        old_task_name = "Task013_Prostate-HK"
        new_task_name = "Task113_Prostate-HK"
    elif TASK == 5:
        old_task_name = "Task015_Prostate-UCL"
        new_task_name = "Task115_Prostate-UCL"
    elif TASK == 6:
        old_task_name = "Task016_Prostate-RUNMC"
        new_task_name = "Task116_Prostate-RUNMC"
    elif TASK == 7:
        old_task_name = "Task097_DecathHip"
        new_task_name = "Task197_DecathHip"
    elif TASK == 8:
        old_task_name = "Task098_Dryad"
        new_task_name = "Task198_Dryad"
    elif TASK == 9:
        old_task_name = "Task099_HarP"
        new_task_name = "Task199_HarP"
    elif TASK == 10:
        old_task_name = "Task201_BraTS1"
        new_task_name = "Task301_BraTS1"
    elif TASK == 11:
        old_task_name = "Task204_BraTS4"
        new_task_name = "Task304_BraTS4"
    elif TASK == 12:
        old_task_name = "Task206_BraTS6"
        new_task_name = "Task306_BraTS6"
    elif TASK == 13:
        old_task_name = "Task213_BraTS13"
        new_task_name = "Task313_BraTS13"
    elif TASK == 14:
        old_task_name = "Task216_BraTS16"
        new_task_name = "Task316_BraTS16"
    elif TASK == 15:
        old_task_name = "Task218_BraTS18"
        new_task_name = "Task318_BraTS18"
    elif TASK == 16:
        old_task_name = "Task220_BraTS20"
        new_task_name = "Task320_BraTS20"
    elif TASK == 17:
        old_task_name = "Task221_BraTS21"
        new_task_name = "Task321_BraTS21"
    else:
        raise ValueError("Unknown task")

    
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
    
    NUM_TEST_DATA = 0.3
    for TASK in [2,7]:
        do(TASK, NUM_TEST_DATA)