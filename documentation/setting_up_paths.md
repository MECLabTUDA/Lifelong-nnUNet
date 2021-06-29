# nnU-Net Extensions: Setting up paths

The process of setting up the paths for nnU-Net extensions is nearly the same as for the conventional nnU-Net from [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md). The paths for the conventinonal nnU-Net need to be set in all cases, luckily by using the extension there is not that much to add. If the user wants to perform or replicate one of the presented experiments by using the provided evaluations for a CL method, a fourth path needs to be set. If the evaluation method will not be used, only the conventional nnU-Net paths need to be set.

The nnU-Net CL extension relies on the same environment variables as the conventional nnU-Net in order to access the raw and preprocessed data and to store the trained model weights. As mentioned earlier, a fourth environment variable is necessary to be able to know where the results of the Evaluation should be stored if the Evaluation will be performed.

1. `nnUNet_raw_data_base`: This environment variable will be used to access the raw, cropped, not preprocessed data by the nnU-Net and the nnU-Net CL extension. As same as for the conventional nnU-Net, it is crucial that the user provides the data in a Decathlon-like structure, as described [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md). The tree structure is the same as for the conventional nnU-Net (extracted from [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)):

        nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart
        ├── dataset.json
        ├── imagesTr
        │   ├── la_003_0000.nii.gz
        │   ├── la_004_0000.nii.gz
        │   ├── ...
        ├── imagesTs
        │   ├── la_001_0000.nii.gz
        │   ├── la_002_0000.nii.gz
        │   ├── ...
        └── labelsTr
            ├── la_003.nii.gz
            ├── la_004.nii.gz
            ├── ...

        nnUNet_raw_data_base/nnUNet_raw_data/Task005_Prostate/
        ├── dataset.json
        ├── imagesTr
        │   ├── prostate_00_0000.nii.gz
        │   ├── prostate_00_0001.nii.gz
        │   ├── ...
        ├── imagesTs
        │   ├── prostate_03_0000.nii.gz
        │   ├── prostate_03_0001.nii.gz
        │   ├── ...
        └── labelsTr
            ├── prostate_00.nii.gz
            ├── prostate_01.nii.gz
            ├── ...

2. `nnUNet_preprocessed`: The preprocessed data will be stored in this specified path. The function that enables the changing of labels based on a provided mapping *-- [see here](change_mask_labels.md) --* stores the results there as well. During training, the preprocessed data will be used.

3. `RESULTS_FOLDER`: The path provided in this variable will be used to store the weights of the trained models, again used by both, nnU-Net and nnU-Net CL extension.

4. `EVALUATION_FOLDER`: This specifies where nnU-Net CL extension will store the results from the evaluation. This variable is not a necessesity and certainly does not need to be set if the evaluation method will not be used.

#### Note:
When using `RESULTS_FOLDER` and `EVALUATION_FOLDER`, the extension creates its own subfolder `nnUNet_ext`, so the different methods can be clearly distinguished from the methods/models trained using the conventional nnU-Net *-- stored in its own `nnUNet` folder --*.

### How to set environment variables
The easiest way to set the paths is in the `~/.bashrc`. If Anaconda is used as an environment, conda will certainly be initialized in this file, so it should already exist. The variables will simply be set by specifying them using the `export` command as follows:

```bash
export nnUNet_raw_data_base="<path_to_raw_data_base>"
export nnUNet_preprocessed="<path_to_preprocessed_data>"
export RESULTS_FOLDER="<path_to_results_folder>"
export EVALUATION_FOLDER="<path_to_evaluation_folder>"
```

The export commands can simply be added at the bottom or top of the `~/.bashrc` file. All terminals that were running during the changing of the `~/.bashrc` file need to be reloaded by using the `source ~/.bashrc` command in order for the changes to take effect. Then the corresponding environment needs to be reactivated as well using `source activate <your_anaconda_env>`. Note that this method will set the paths permamently and they are set for each terminal session.

An alternative, temporary way of setting the variables is by using the terminal directly. The same `export` commands need to be executed sequentially using the terminal. These changes only work for the terminal in which the commands have been executed in and will vanish after the terminal session is closed. So with this alternative, the paths need to be set for each terminal, every time the terminal has been closed.

To check *-- in both cases --* if the environment variables are set, simply run `echo $<variable_name>` in the corresponding terminal. If the paths are correctly set, the corresponding path should be displayed in the terminal.
