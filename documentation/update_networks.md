# nnU-Net Continual Learning extension: Updating the networks checkpoint

This is a general description on how to update a networks checkpoint, for instance the stored paths in the checkpoint which might be invalid due to a relocation of the models/checkpoint folder or even when using a provided pre-trained (extension) network which was trained on a different machine. This might also be the case if the network is accessible at the path it has been trained on, however the user executing for example the [evaluation script](evaluation.md) using this network has no permission to write to the paths that are specified within those checkpoints. Moving the checkpoint to a location where the user has permission without changing the stored paths is a good way to start, however only relocating does not change the fact the stored paths are still the old ones leading to the old directories. In all cases, the internal paths of the checkpoints have to be altered in order to enable a smooth execution and usage of networks, indifferent whether used for inference, evaluation or anything else.
This of course does not apply if the user that trained the network wants to use it afterwards with the exception of a relocation of the networks or a withdrawal of permission to write to those corresponding directories.


### Command Line Arguments
The following arguments and flags can be set when using this provided functionality.

| tag_name | description | required | choices | default | 
|:-:|-|:-:|:-:|:-:|
| first argument, no tag | Specify the network to use. | yes | `2d`, `3d_fullres`, ... | -- |
| second argument, no tag | Specify the network_trainer to use. | yes | `nnUNetTrainerV2`, `nnUNetTrainerEWC`, `nnViTUNetTrainer`, ... | -- |
| `-p` | Specify the plans identifier. Only change this if you created a custom experiment planner. | no | -- | `nnUNetPlansv2.1` from [nnunet_ext/paths.py](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/paths.py#L10) |
| `-trained_on` | Specify a list of task ids to train on (ids or names). Each of these ids must have a matching folder TaskXXX_TASKNAME in the raw data folder. | yes | -- | -- |
| `-use_model` or `--use` | Specify a list of task ids that specify the exact network that should be used for evaluation. Each of these ids must, have a matching folder 'TaskXXX_' in the raw. | no | -- | -- |
| `-f` or `--folds` | Specify on which folds to train on. Use a fold between `0, 1, ..., 4` or `all`. | no | -- | -- |
| `-rw` or `--replace_with` | First specify the part to replace and the path it will be replaced with. Only takes two arguments. | yes | -- | -- |
| `--use_vit` | If this is set, the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) will be used instead of the [Generic_UNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py#L167). Note that then the flags `-v` and `-v_type` should be set accordingly if applicable. | no | -- | `False` |
| `--task_specific_ln` | If this is set, the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) will be used instead of the [Generic_UNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py#L167) has task specific Layer Norms. | no | -- | `False` |
| `-v` or `--version` | Select the ViT input building version. Currently there are only three possibilities: `1`, `2`, `3` or `4`. For further references with regards to the versions, see the [docs](https://github.com/camgbus/Lifelong-nnUNet/blob/ViT_U-Net/documentation/ViT_U-Net.md). | no | `1`, `2`, `3`, `4` | `1` |
| `-v_type` or `--vit_type` | Specify the ViT architecture. Currently there are only three possibilities: `base`, `large` or `huge`. | no | `base`, `large`, `huge` | `base` |
 `--no_transfer_heads` | Set this flag if a new head should not be initialized using the last head during training. | no | -- | `False` |
| `--do_LSA` | Set this flag if Locality Self-Attention should be used for the ViT. | no | -- | `False` |
| `--do_SPT` | Set this flag if Shifted Patch Tokenization should be used for the ViT. | no | -- | `False` |
| `--no_pod` | Set this flag if the POD embedding has been included in the loss calculation (only for our own methods). | no | -- | `False` |
| `-r` | Use this if all subfolders should be scanned for checkpoints. If this is set, all other flags for specifying one simple model will not be considered then. | no | -- | `False` |
| `-rm` | Use this if all subfolders should be scanned for checkpoints. If this is set, then only the checkpoints wrt to the network_trainer are used. | no | -- | `False` |
| `-h` or `--help` | Simply shows help on which arguments can and should be used. | -- | -- | -- |

When talking about lists in command lines, this does not mean to provide a real list, like values in brackets *--* `[.., .., ...]`  *--*, but rather does it mean to provide an enumeration of values *--* `val_1 val2 val3 ...` *--*.

Note that we did specifically created the `-rw/--replace_with` flag for the user to provide the desired substring and its replacement. We could have omitted this flag by extracting the old nnU-Net paths based on the stored paths and replacing them with the ones set by the user. However the user might also have a different location for the trained data folder, ie. set by `RESULTS_FOLDER` and the raw data `nnUNet_raw_data_base` or preprocessed data `nnUNet_preprocessed`. For this and out of simplicity, the user has to provide the substring and its replacement.

It is also important to note that this method if used in the wrong way might damage the checkpoint pickle file thus leading to a checkpoint that can not be used, so this method should be applied in an appropriate manner. Another important thing about the substrings is that they *-- in this case we use this method for replacing paths --* should be long enough without creating conflicts with any other parts of the checkpoint that should not be replaced, eg. replacing the string `init` with `/new_path/` would change the dictionaries key of the checkpoint pickle file and thus result in an error during the restoring process of the network using this very checkpoint. Rather specify a longer string, like `'init/data/user/...` to be replaced with `/new_path/user/...` to ensure the replacement at the correct and intended places. Also ensure that the sub-paths provided with the `-rw/--replace_with` flag start and end with the path seperator `/` to avoid any other complications. One might want to replace the sub-path `/test/user/admin` with `/new/user/user_xyz`, however a path like `/test/user/admin_root/` will be replaced resulting in the updated path `/new/user/user_xyz_root/` which surely is not intended. Providing the arguments like `/test/user/admin/` and `/new/user/user_xyz/` would solve the issue. 

To use this functionality, the user needs to extract some informations on his/her own to know which parts should be replaced using this function. The following code snippet prints the arguments that were used for the initialization of the specific model corresponding to the checkpoint. 

```python
from batchgenerators.utilities.file_and_folder_operations import load_pickle
chkpt_info = load_pickle(<path_to_chkpt_with_model.pkl_in_it>)
print(chkpt_info['init'])
# print(chkpt_info)
```

As mentioned previously, the user might have used different base paths for `RESULTS_FOLDER` and other folders like `nnUNet_preprocessed` eg. The print command printing the arguments for the initialization should be sufficient, however cross checking it with the whole checkpoint, ie. ensuring that there are no other paths that have to be changed, would not harm in any way *-- second print command --*. Knowing which parts the user wants to replace, we can provide multiple use cases showing how to apply this function. As one might notice, this function can be used to manipulate any kind of string within the checkpoint recursively, so be careful and as thorough as possible when applying this function *-- maybe create a backup of the checkpoint(s) when using the function for the first time to avoid a total screw up --*.

### Exemplary use cases
In the following, a few examples are shown representing possible use cases on how to use the provided functionality to update/modifiy the checkpoints pickle files. For the following use cases, the user always wants to replace the `/home/user/admin/` sub-path with the updated `/home/test_env/user/user_xyz/` sub-path. We further solely focus on the training sequence of `Task011_XYZ`, `Task012_XYZ` and `Task013_XYZ` using the 3d network settings. If one wants to replace multiple parts since one used different paths for every base folder, then the function has to be executed multiple times based on the amount of different paths, of course with adapted `-rw/--replace_with` flag.

The easiest use cases are the ones where the user uses the `-r` or `-rm` flag. The difference being that `-r` walks through the whole directory solely based on the provided Task IDs (`-trained_on` flag). The `-rm` flag expects a little more arguments. It scans through the whole directory as well, but only modifies the checkpoints of the specific trainer.

Let's assume the user wants to modify all checkpoints, ie. the checkpoints of all models trained on the Tasks `11`, `12`, and `13`. The corresponding command looks like the following:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_update_checkpoints 3d_fullres _ -trained_on 11 12 13 -r
                                                  -rw /home/user/admin/ /home/test_env/user/user_xyz/
```
Note that the user has to provide the network_trainer although it will not be considered, so this can be set to whatever the user wants since it will not be checked any further as long as something is provided. To demonstrate this, we used the wildcart character `_`.

Let's assume the user wants to modify all checkppoints of networks that were trained using the [EWC Trainer](ewc_training.md), without considering if the [Generic_UNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py#L167) architecture or [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) has been used.
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_update_checkpoints 3d_fullres nnUNetTrainerEWC -trained_on 11 12 13
                                                  -rm -rw /home/user/admin/ /home/test_env/user/user_xyz/
```
Note that for the first two use cases, all other additional arguments can be set but are irrelevant since they are never used, ie. those two methods will iterate over all folds that the network has been trained on.

Let's assume the user wants to modify the checkpoint of a single model. For this we assume the network was again trained on the Tasks `11`, `12`, and `13` and the user only wants to modify the network trained on `11` and `12` using the [MiB Trainer](mib_training.md) on fold 2 and 4.
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_update_checkpoints 3d_fullres nnUNetTrainerMiB -trained_on 11 12 13 -use 11 12
                                                  -rm -f 2 4 -rw /home/user/admin/ /home/test_env/user/user_xyz/
                                                  [--use_vit -v <VERSION> -v_type <TYPE> ...]
```
All the so far provided examples use the [Generic_UNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py#L167) architecture as foundation, however as proposed in the Command Line Arguments, one can use our proposed [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) from the [ViT_U-Net branch](https://github.com/camgbus/Lifelong-nnUNet/tree/ViT_U-Net) instead. The following example uses Version 2 (out of 4) of the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) specifying the Vision Transformer itself as a huge Transformer, ie. the biggest one (out of 3 types). Further, Shifted Patch Tokenization (SPT) but not Locality Self-Attention (LSA) *-- as proposed [here](https://arxiv.org/pdf/2112.13492v1.pdf) --* has been used. More informations with regard to the ViT_U-Net architecture can be found [here](https://github.com/camgbus/Lifelong-nnUNet/blob/ViT_U-Net/documentation/ViT_U-Net.md):
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_update_checkpoints 3d_fullres nnUNetTrainerMiB -trained_on 11 12 13 -use 11 12
                                                  -rm -f 2 4 -rw /home/user/admin/ /home/test_env/user/user_xyz/
                                                  --use_vit -v 2 -v_type huge --do_SPT
                                                  [--do_LSA --no_transfer_heads ...]
```
