# nnU-Net Continual Learning extension: Training using Riemannian Walk (RW)

This is a general description on how to use the RW Trainer to train the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) on a sequence of tasks/datasets using the RW method proposed in the [paper](https://arxiv.org/pdf/1801.10112.pdf). The extension can only be used after it has been succesfully installed *-- including all dependencies --* following the instructions from [here](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/README.md#installation). Further, all relevant paths have to be set, so the nnU-Net and the nnU-Net extension can extract the right directories for training, preprocessing, storing etc. as described [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md). Before training with any Trainer, ensure that the Task is properly prepared, planned and preprocessed as described in the example [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/training_example_Hippocampus.md).

### Command Line Arguments
The exact same Command Line Arguments as presented in the [Multi-Head](multihead_training.md) Trainer apply for the RW Trainer as well. Additionally, further parameters can be set:

| tag_name | description | required | choices | default | 
|:-:|-|:-:|:-:|:-:|
| `-rw_alpha` | Specify the alpha parameter that is used to calculate the Fisher values --> should be [0, 1]. | no | -- | `0.9` |
| `-rw_lambda` | Specify the importance of the previous tasks for the RW method using the EWC regularization. | no | -- | `0.4` |
| `-update_after` | Specify after which iteration (batch iteration, not epoch) the fisher values are updated/calculated. | no | -- | `10` |

### Exemplary use cases
In the following, a few examples are shown representing possible use cases on how to use the RW Trainer.

One of the easiest and simplest example is to simply train on a bunch of tasks, for example `Task011_XYZ`, `Task012_XYZ` and `Task013_XYZ`. Each task should be trained for 250 epochs, whereas every 25th epoch, the stored data is updated and the results are stored in a `.csv` file. Additionally, the network should be split at the `seg_outputs` part of the network and trained on the GPU with ID <GPU_ID> (can be one or multilpe IDs). The importance of the previous tasks for the RW method should be set to 0.21 instead of 0.4 with a new alpha of 0.77:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_rw 3d_fullres -t 11 12 13 -f 0 -rw_lambda 0.21 -rw_alpha 0.77
                                        -num_epoch 250 -d <GPU_ID> -save_interval 25 -s seg_outputs --store_csv
                                        [--use_vit -v <VERSION> -v_type <TYPE> -update_after <PERIOD> ...]
```

The following example uses Version 1 (out of 3) of the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) using the smallest one (out of 3 types). More informations with regard to the ViT_U-Net architecture can be found [here](https://github.com/camgbus/Lifelong-nnUNet/blob/ViT_U-Net/documentation/ViT_U-Net.md). Further, after every 5th epoch, the fisher and score values should be updated instead of every 10th:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_rw 3d_fullres -t 11 12 13 -f 0 -update_after 5
                                        -num_epoch 250 -d <GPU_ID> -save_interval 25 -s seg_outputs --store_csv
                                        --use_vit -v 1 -v_type base
                                        [-rw_alpha <VALUE>  -rw_lambda <VALUE> --use_mult_gpus ...]
                             
```

Note that the `--no_transfer_heads` flag can be used with this Trainer. If it is set, the previous head will not be used as an initialization of the new head, ie. the head from the initial split from Multi-Head Module is used as initialization of the new head. If it is not set *-- as in all use cases above --*, the previously trained head will be used as an initialization of the new head.
