# nnU-Net Continual Learning extension: Delete preprocessed and planned tasks 

This is a general description on how to delete tasks that have been planned and preprocessed using the nnU-Net commands.

1. The extension can only be used after it has been succesfully installed *-- including all dependencies --* following the instructions from [here](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/README.md#installation). Further, all relevant paths have to be set, so the nnU-Net and the nnU-Net extension can extract the right directories for training, preprocessing, storing etc. as described [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

2. Let's assume we use `Task04_Hippocampus`. In order for all preprocessed and planned folders with the name corresponding `Task004_Hippocampus` to be deleted, the original data, ie. `Task04_Hippocampus` needs to be converted using
    ```bash
    nnUNet_convert_decathlon_task -i $nnUNet_raw_data_base/Task04_Hippocampus
    ```
    and then preprocessed and planned using: 
    ```bash
    nnUNet_plan_and_preprocess -t 4
    ```

3. The commands `nnUNet_convert_decathlon_task` and `nnUNet_plan_and_preprocess` create three folders at three different locations. For the Hippocampus example those are:
    * `$nnUNet_raw_data_base/nnUNet_raw_data/Task004_Hippocampus`
    * `$nnUNet_preprocessed/Task004_Hippocampus`
    * `$nnUNet_raw_data_base/nnUNet_cropped_data/Task004_Hippocampus`

    The following command deletes all those three folders at once without the user needing to remove them by hand:
    ```bash
    nnUNet_delete_tasks -t 4
    ```
    Before panning and preprocessing a task again, eg. after an alteration, it is important to remove those already preprocessed and planned folders since a task can only be planned and preprocessed if those three corresponding paths do not exist. Note that only the preprocessed and planned data gets removed, but not the original data, in this example `Task04_Hippocampus`, located at `$nnUNet_raw_data_base/Task04_Hippocampus`.

    Using this command with multiple tasks like
    ```bash
    nnUNet_delete_tasks -t t_1 t_2 ... t_n
    ```
    those tasks t_1, ... t_n are getting removed by using only one command.