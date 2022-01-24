# nnU-Net Continual Learning extension: Multi-Head Training

This is a general description on how to use the Multi-Head Trainer to train the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) on a sequence of tasks/datasets. The extension can only be used after it has been succesfully installed *-- including all dependencies --* following the instructions from [here](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/README.md#installation). Further, all relevant paths have to be set, so the nnU-Net and the nnU-Net extension can extract the right directories for training, preprocessing, storing etc. as described [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md). Before training with any Trainer, ensure that the Task is properly prepared, planned and preprocessed as described in the example [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/training_example_Hippocampus.md).


### Command Line Arguments
The following arguments and flags represent the base arguments for all extension Trainers. Thus, they are extended with further arguments with respect to the corresponding Trainer. The arguments for the original nnUNet Trainer are the following and are still valid for the Multi-Head Trainer.

| tag_name | description | required | choices | default | 
|:-:|-|:-:|:-:|:-:|
| first argument, no tag | Specify the network to use. | yes | `2d`, `3d_fullres` | -- |
| `-val` or `--validation_only` | Use this if you want to only run the validation. This will validate each model of the sequential pipeline, so they should have been saved and not deleted. | no | -- | `False` |
| `-p` | Specify the plans identifier. Only change this if you created a custom experiment planner. | no | -- | `nnUNetPlansv2.1` from [nnunet_ext/paths.py](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/paths.py#L10) |
| `--use_compressed_data` | If `use_compressed_data` is set, the training cases will not be decompressed. Reading compressed data is much more CPU and RAM intensive. | no | -- | `False` |
| `--deterministic` | Makes training deterministic, but reduces training speed substantially. Deterministic training will make you overfit to some random seed. | no | -- | `False` |
| `-npz` | If this is set, then nnUNet will export npz files of predicted segmentations in the validation as well. This is needed to run the ensembling step. | no | -- | `False` |
| `--find_lr` | This is a flag provided by the nnUNet but is not used. | no | -- | `False` |
| `--valbest` | This is a flag provided by the nnUNet but should not be used. | no | -- | `False` |
| `--fp32` | This flag is used to disable mixed precision training and run fp32. | no | -- | `False` |
| `--val_folder` | Specify the name of the validation folders. | no | -- | `validation_raw` |
| `--disable_postprocessing_on_folds` | This flag should be set if the user wants to omit the postprocessing on each fold. | no | -- | `False` |
| `--val_disable_overwrite` | This can be set so validation does overwrite existing segmentations. | no | -- | `True` |
| `--disable_next_stage_pred` | This flag is used to specify if the next stages should be predicted. | no | -- | `False` |
| `-pretrained_weights` | This flag is provided by the nnUNet but only in beta version. | no | -- | `None` |

The following arguments are specifically added for all Trainers, including the Multi-Head Trainer thus representing the more interesting part of flags:

| tag_name | description | required | choices | default | 
|:-:|-|:-:|:-:|:-:|
| `-t` or `--task_ids` | Specify a list of task ids to train on (ids or names). Each of these ids must have a matching folder TaskXXX_TASKNAME in the raw data folder. | yes | -- | -- |
| `-f` or `--folds` | Specify on which folds to train on. Use a fold between `0, 1, ..., 4` or `all`. | yes | -- | -- |
| `-d` or `--device` | Try to train the model on the GPU device with <GPU_ID>. Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well. Default: Only GPU device with ID 0 will be used. | no | -- | `0` |
| `-s` or `--split_at` | Specify the path in the network in which the split will be performed. Use a single string for it, specify between layers using a dot `.` notation. This is a required field and no default will be set. Use the same names as present in the desired network architecture. | yes | -- | -- |
| `-num_epochs` | Specify the number of epochs to train the model. | no | -- | `500` |
| `-save_interval` | Specify after which epoch interval to update the saved data. | no | -- | `25` |
| `--store_csv` | Set this flag if the validation data and any other data if applicable should be stored as a .csv file as well. | no | -- | `False` |
| `--init_seq` | Specify if the first task from `-t` is already trained and represents an init `network_trainer` to do (extensional) training on or not. If so, `-initialize_with_network_trainer` needs to be provided as well. | no | -- | `False` |
| `-initialize_with_network_trainer` | Specify the `network_trainer` that should be used as a foundation to start training sequentially. The `network_trainer` of the first provided task needs to be finished with training and either a (extensional) `network_trainer` or a standard `nnUNetTrainerv2`. | no | -- | `None` |
| `-used_identifier_in_init_network_trainer` | Specify the identifier that should be used for the `network_trainer` that is used as a foundation to start training sequentially. | no | -- | `nnUNetPlansv2.1` from [nnunet_ext/paths.py](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/paths.py#L10) |
| `--disable_saving` | If set, nnU-Net will not save any parameter files (except a temporary checkpoint that will be removed at the end of the training). Useful for development when you are only interested in the results and want to save some disk space. Further for sequential tasks the intermediate model won't be saved then, remeber that. | no | -- | `False` |
| `--use_vit` | If this is set, the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) will be used instead of the [Generic_UNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py#L167). Note that then the flags `-v`, `-v_type` and `--use_mult_gpus` should be set accordingly if applicable. | no | -- | `False` |
| `--task_specific_ln` | If this is set, the Generic_ViT_UNet will have task specific Layer Norms. | no | -- | `False` |
| `--use_mult_gpus` | If this is set, the ViT model will be placed onto a second GPU. When this is set, more than one GPU have to be provided when using `-d`. | no | -- | `False` |
| `-v` or `--version` | Select the ViT input building version. Currently there are only three possibilities: `1`, `2`, `3` or `4`. For further references with regards to the versions, see the [docs](https://github.com/camgbus/Lifelong-nnUNet/blob/ViT_U-Net/documentation/ViT_U-Net.md). | no | `1`, `2`, `3`, `4` | `1` |
| `-v_type` or `--vit_type` | Specify the ViT architecture. Currently there are only three possibilities: `base`, `large` or `huge`. | no | `base`, `large`, `huge` | `base` |
| `--do_LSA` | Set this flag if Locality Self-Attention should be used for the ViT. | no | -- | `False` |
| `--do_SPT` | Set this flag if Shifted Patch Tokenization should be used for the ViT. | no | -- | `False` |
| `--no_transfer_heads` | Set this flag if a new head should not be initialized using the last head during training. | no | -- | `False` |
| `-h` or `--help` | Simply shows help on which arguments can and should be used. | -- | -- | -- |

When talking about lists in command lines, this does not mean to provide a real list, like values in brackets *--* `[.., .., ...]`  *--*, but rather does it mean to provide an enumeration of values *--* `val_1 val2 val3 ...` *--*.


### Exemplary use cases
In the following, a few examples are shown representing possible use cases on how to use the Multi Head Trainer.

One of the easiest and simplest example is to simply train on a bunch of tasks, for example `Task011_XYZ`, `Task012_XYZ` and `Task013_XYZ`. Each task should be trained for 250 epochs, whereas every 25th epoch, the stored data is updated and the results are stored in a `.csv` file. Additionally, the network should be split at the `seg_outputs` part of the network and trained on the GPU with ID <GPU_ID> (can be one or multilpe IDs):
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_multihead 3d_fullres -t 11 12 13 -f 0
                                               -num_epoch 250 -d <GPU_ID> -save_interval 25 -s seg_outputs --store_csv
                                               [--use_vit -v <VERSION> -v_type <TYPE> ...]
```

One more complex example showing how to define a deeper split using the `.` notation would be the following, where the settings are the same as before, but the split is now in the context block of the network. Note that for setting an appropriate split, the user needs to know the networks structure or the splitting might not work:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_multihead 3d_fullres -t 11 12 13 -f 0
                                               -num_epoch 250 -d <GPU_ID> -save_interval 25
                                               -s conv_blocks_context.0.blocks.1 --store_csv
                                               [--use_vit -v <VERSION> -v_type <TYPE> ...]
```

All the so far provided examples use the [Generic_UNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py#L167) architecture as foundation, however as proposed in the Command Line Arguments, one can use our proposed [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) from the [ViT_U-Net branch](https://github.com/camgbus/Lifelong-nnUNet/tree/ViT_U-Net) instead. The following example uses Version 1 (out of 4) of the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) specifying the Vision Transformer itself as a base Transformer, ie. the smallest one (out of 3 types). More informations with regard to the ViT_U-Net architecture can be found [here](https://github.com/camgbus/Lifelong-nnUNet/blob/ViT_U-Net/documentation/ViT_U-Net.md):
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_multihead 3d_fullres -t 11 12 13 -f 0
                                               -num_epoch 250 -d <GPU_ID> -save_interval 25 -s seg_outputs --store_csv
                                               --use_vit -v 1 -v_type base [--use_mult_gpus ...]
```

In a next use case, the same settings as in the previous one apply, except that the LayerNorm layers of the ViT should be task specific while using the Shifted Path Tokenization and Locality Self-Attention method as proposed [here](https://arxiv.org/pdf/2112.13492v1.pdf):
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_multihead 3d_fullres -t 11 12 13 -f 0 --task_specific_ln
                                               -num_epoch 250 -d <GPU_ID> -save_interval 25 -s seg_outputs --store_csv
                                               --use_vit -v 1 -v_type base --do_LSA --do_SPT
                                               [--use_mult_gpus ...]                
```

Last but not least, the following example shows how to use a pre-trained nnU-Net as a foundation (trained on `Task011_XYZ` with `nnUNetTrainerV2` Trainer) to continue training on using new tasks (`Task012_XYZ` and `Task013_XYZ`). Note that this has not been used and thus not tested yet:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_multihead 3d_fullres -t 11 12 13 -f 0
                                               -num_epoch 250 -d <GPU_ID> -save_interval 25
                                               -s seg_outputs --store_csv --init_seq
                                               -initialize_with_network_trainer nnUNetTrainerV2
                                               -used_identifier_in_init_network_trainer nnUNetPlansv2.1
                                               [--use_vit -v <VERSION> -v_type <TYPE> ...]
```

Note that the `--no_transfer_heads` flag makes sense to use in combination with the Multi-Head Trainer, otherwise this will be a classical Transfer Learning case represented by the Sequential Trainer, described [here](sequential_training.md). This flag might make more sense when using the EWC or LwF Trainer as shown [here](ewc_training.md) and [here](lwf_training.md).
