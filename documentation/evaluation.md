# nnU-Net Continual Learning extension: Performing evaluation on trained networks

This is a general description on how to perform evaluation on trained networks using the nnU-Net extension. After successfull training of one or multiple methods, one might want to evaluate the network and its performance based on all the datasets it has been trained on. For this a specific command is provided that helps the user to specify which network should be evaluated on which datasets, ie. not linked to the actual dataset(s) the network is trained on.

1. The network that should be used to perform evaluation on needs to be trained. This can be either performed using the conventional nnU-Net as shown in [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/training_example_Hippocampus.md) or using one of the provided extension methods.

2. After sucessful training, the Evaluator can be used to evaluate the trained network with any data, ie. Task that has been preprocessed and planned. Note that the Evaluator always uses the data from the validation part of the split, ie. not the data from the training part of the split. For this, the  `splits_final` file located under `$nnUNet_preprocessed/<TASK>/splits_final.pkl` will be used.

3. The results of the evaluation can be found in the specified `EVALUATION_FOLDER`. This folder has the same structure as the `RESULTS_FOLDER`.

In the following, the command line arguments are presented and further discussed providing some exemplary use cases.

### Command Line Arguments
The following Command Line Arguments can be set.

| tag_name | description | required | choices | default | 
|:-:|-|:-:|:-:|:-:|
| first argument, no tag | Specify the network to use. | yes | `2d`, `3d_fullres` | -- |
| second argument, no tag | Specify the network_trainer to use. | no | `nnUNetTrainerV2`, `nnUNetTrainerEWC`, `nnViTUNetTrainer`, ... | -- |
| `-p` | Specify the plans identifier. Only change this if you created a custom experiment planner. | no | -- | `nnUNetPlansv2.1` from [nnunet_ext/paths.py](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/paths.py#L10) |
| `-f` or `--folds` | Specify on which folds to train on. Use a fold between `0, 1, ..., 4` or `all`. | yes | -- | -- |
| `-trained_on` | Specify a list of task ids the network has trained with to specify the correct path to the networks. Each of these ids must, have a matching folder 'TaskXXX_' in the raw data folder. | yes | -- | -- |
| `-use_model` or `--use` | Specify a list of task ids that specify the exact network that should be used for evaluation. Each of these ids must, have a matching folder 'TaskXXX_' in the raw. | yes | -- | -- |
| `-use_head` | Specify which head to use for the evaluation in case the task is not a registered head. When using a non nn-UNet extension, that is not necessary. If this is not set, always the latest trained head will be used. | no | -- | `None` |
| `--fp32_used` | Specify if mixed precision has been used during training or not. | no | -- | `False` |
| `-evaluate_on` | Specify a list of task ids the network will be evaluated on. Each of these ids must, have a matching folder 'TaskXXX_' in the raw data folder. | yes | -- | -- |
| `-d` or `--device` | Try to train the model on the GPU device with <GPU_ID>. Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well. Default: Only GPU device with ID 0 will be used. | no | -- | `0` |
| `--store_csv` | Set this flag if the validation data and any other data if applicable should be stored as a .csv file as well. Default: .csv are not created. | no | -- | `False` |
| `--use_vit` | If this is set, the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) will be used instead of the [Generic_UNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py#L167). Note that then the flags `-v`, `-v_type`, `use_mult_gpus` and `--use_mult_gpus` should be set accordingly if applicable. | no | -- | `False` |
| `--use_mult_gpus` | If this is set, the ViT model will be placed onto a second GPU. When this is set, more than one GPU needs to be provided when using `-d`. | no | -- | `False` |
| `-v` or `--version` | Select the ViT input building version. Currently there are only three possibilities: `1`, `2` or `3`. For further references with regards to the versions, see the [docs](https://github.com/camgbus/Lifelong-nnUNet/blob/ViT_U-Net/documentation/ViT_U-Net.md). | no | `1`, `2`, `3` | `1` |
| `-v_type` or `--vit_type` | Specify the ViT architecture. Currently there are only three possibilities: `base`, `large` or `huge`. | no | `base`, `large`, `huge` | `base` |
| `--no_transfer_heads` | Set this flag if a new head should not be initialized using the last head during training. | no | -- | `False` |
| `--always_use_last_head` | If this is set, during the evaluation, always the last head will be used, for every dataset the evaluation is performed on. When an extension network was trained with the `--no_transfer_heads` flag then this should be set as well. Otherwise, the corresponding head to the dataset will be used if available or the last trained head instead. | no | -- | `False` |
| `--do_LSA` | Set this flag if Locality Self-Attention has been used for the ViT. | no | -- | `False` |
| `--do_SPT` | Set this flag if Shifted Patch Tokenization has been used for the ViT. | no | -- | `False` |
| `--include_training_data` | Set this flag if the evaluation should also be done on the training data. | no | -- | `False` |
| `-h` or `--help` | Simply shows help on which arguments can and should be used. | -- | -- | -- |


### Exemplary use cases
In the following, a few examples are shown representing possible use cases on how to use the Evaluator.

The easiest use case is to use the Evaluator on a network trained using the conventional nnU-Net, here: `nnUNetTrainerV2`. Let's assume the network was trained on the Hippocampus task and one might want to test the generalization of the just trained network by performing an evaluation on this trained network with a different, but still Hippocampus dataset (for this matter specified as `Task_014_Hip`). The evaluation on the trained Hippocampus task and `Task_014_Hip` would like the following:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_evaluate 3d_fullres nnUNetTrainerV2 -trained_on 4 -f 0
                                        -use_model 4 -evaluate_on 4 14 -d <GPU_ID> --store_csv
```
Note that `-trained_on` and `-use_model` only specify the task the network has been trained on, ie. not a list since the `nnUNetTrainerV2` can only train on one task.

Let's assume a [Sequential Trainer](sequential_training.md) has been used to train on `Task011_XYZ`, `Task012_XYZ` and `Task013_XYZ` sequentially, only using fold 0. In the `RESULTS_FOLDER` are now three different folders with models, ie.:
* Model trained on `Task011_XYZ` with only one head being `Task011_XYZ`
* Model trained on `Task011_XYZ` and then `Task012_XYZ` with two heads being `Task011_XYZ` and `Task012_XYZ`
* Model trained on `Task011_XYZ`, then `Task012_XYZ` and finally `Task013_XYZ` having three heads being `Task011_XYZ`, `Task012_XYZ` and `Task013_XYZ`

The user now wants to evaluate the first model, only trained on `Task011_XYZ` and the last model, ie. trained on all three tasks. For every evaluation, the model should be evaluated on all three tasks, ie. `Task011_XYZ`, `Task012_XYZ` and `Task013_XYZ`. Note that `-trained_on` specifies all tasks the `SequentialTrainer` has been initialized with, ie. if the user trained using `-t 11 12 13` than `-trained_on` has to specify the same tasks so the Evaluator navigates to the correct folder containing all model subfolders. Otherwise there might either be a mixup or an error will be thrown because of a wrong path leading nowhere. Further, during evaluation, only the validation split of a task will be used, not the training split.
Since the use case specifies two different evaluations, two different commands have to be executed:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_evaluate 3d_fullres nnUNetTrainerSequential -trained_on 11 12 13 -f 0
                                        -use_model 11 -evaluate_on 11 12 13 -d <GPU_ID> --store_csv
                                        --always_use_last_head [--use_vit -v <VERSION> -v_type <TYPE> ...]
```
for the evaluation of the first model and
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_evaluate 3d_fullres nnUNetTrainerSequential -trained_on 11 12 13 -f 0
                                        -use_model 11 12 13 -evaluate_on 11 12 13 -d <GPU_ID> --store_csv
                                        --always_use_last_head [--use_vit -v <VERSION> -v_type <TYPE> ...]
```
for the evaluation of the last model trained on all tasks. If the user wants to perform the evaluation on the model trained on `Task011_XYZ` and `Task012_XYZ`, then only the `-use_model` flag has to be updated, specifying those tasks, ie. `-use_model 11 12`. Note that a model combination like `-use_model 11 13 12` or `-use_model 11 13` or `-use_model 12 11` does not exist, except the Sequential Trainer has been used in this order, ie. `-t 11 13 12` for `-use_model 11 13 12` and `-use_model 11 13 12` and `-t 12 11 12` for `-use_model 12 11`. However the tasks one performs the evaluation with only have to be planned and preprocessed, but not trained on, ie. if a `Task017_XYZ` would exist, one can use it for the evaluation as well, specifying it in the `-evaluate_on` flag (`-evaluate_on 11 12 13 17` eg.). The order of the tasks to perform evaluation with is irrelevant as well. The `--always_use_last_head` flag specifies that only the last head will be used for the evaluation. This makes sense for a Sequential setting since in such settings, one normally only has the last model, in terms of head. If this would not be set, than for every task, the corresponding heads will be used. If a task has no corresponding head, then the head specified with the `-use_head` flag will be used instead, as shown in the next example.

Let's assume a [EWC Trainer](ewc_training.md) has been used and instead of the [Generic_UNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py#L167) architecture as foundation, one used our proposed [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) from the [ViT_U-Net branch](https://github.com/camgbus/Lifelong-nnUNet/tree/ViT_U-Net) instead. The following example uses Version 1 (out of 3) of the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) specifying the Vision Transformer itself as a base Transformer, ie. the smallest one (out of 3 types). More informations with regard to the ViT_U-Net architecture can be found [here](https://github.com/camgbus/Lifelong-nnUNet/blob/ViT_U-Net/documentation/ViT_U-Net.md). The evaluation of the model trained on all tasks (same settings as above) and all folds using only tasks `Task011_XYZ` and `Task013_XYZ` in combination with `Task017_XYZ` *-- a task the model has not been trained on --* would look like the following:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_evaluate 3d_fullres nnUNetTrainerEWC -trained_on 11 12 13 -f all
                                        -use_model 11 12 13 -evaluate_on 11 13 17 -d <GPU_ID> --store_csv
                                        --use_vit -v 1 -v_type base -use_head 12
                                        [--use_mult_gpus ...]
```
Note that the model is trained on `Task011_XYZ`, `Task012_XYZ` and `Task013_XYZ`, however there does not exist a head for `Task017_XYZ`. The `-use_head` is used to specify that the head of `Task012_XYZ` should be used for the evaluation on `Task017_XYZ`. If this flag is not set, then always the last head will be used, in this setting it would be the one corresponding to `Task013_XYZ`. When using the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14), then all ViT related flags have to be set the same as set during training in order to select the correct model during evaluation.
In a next use case, the same settings as in the previous one apply, except that the LayerNorm layers of the ViT should be task specific and the network was trained using multiple GPUs. Additionally, the evaluation should be performed on the validation and training data.:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_evaluate 3d_fullres nnUNetTrainerEWC -trained_on 11 12 13 -f all
                                        -use_model 11 12 13 -evaluate_on 11 13 17 -d <GPU_ID> --store_csv
                                        --use_vit -v 1 -v_type base --use_mult_gpus --task_specific_ln     
                                        --include_training_data [--always_use_last_head ...]                    
```

so last but not least, one might also want to evaluate for the specified models always using the same head. In the exception
the following example shows how to use a pre-trained nnU-Net as a foundation (trained on `Task011_XYZ` with `nnUNetTrainerV2` Trainer) to continue training on using new tasks (`Task012_XYZ` and `Task013_XYZ`). Note that this has not been used and thus not tested yet:
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

Note that the `--no_transfer_heads` flag has to be set for all those networks that were trained with the `--no_transfer_heads` flag.
