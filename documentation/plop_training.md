# nnU-Net Continual Learning extension: Training using Pseudo-labeling and LOcal Pod (PLOP)

This is a general description on how to use the PLOP Trainer to train the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) on a sequence of tasks/datasets using the PLOP method proposed in the [paper](https://arxiv.org/pdf/2011.11390.pdf). The extension can only be used after it has been succesfully installed *-- including all dependencies --* following the instructions from [here](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/README.md#installation). Further, all relevant paths have to be set, so the nnU-Net and the nnU-Net extension can extract the right directories for training, preprocessing, storing etc. as described [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md). Before training with any Trainer, ensure that the Task is properly prepared, planned and preprocessed as described in the example [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/training_example_Hippocampus.md).

### Command Line Arguments
The exact same Command Line Arguments as presented in the [Multi-Head](multihead_training.md) Trainer apply for the PLOP Trainer as well. The hyperparameter $\lambda$ for the PLOP Loss and the number of scales $S$ can be set too using Command Line Arguments.

| tag_name | description | required | choices | default | 
|:-:|-|:-:|:-:|:-:|
| `-pod_lambda` | Specify the lambda weighting for the distillation loss. | no | -- | `0.01` |
| `-pod_scales` | Specify the number of scales for the PLOP method. | no | -- | `3` |

### Exemplary use cases
In the following, a few examples are shown representing possible use cases on how to use the PLOP Trainer.

One of the easiest and simplest example is to simply train on a bunch of tasks, for example `Task011_XYZ`, `Task012_XYZ` and `Task013_XYZ`. Each task should be trained for 250 epochs, whereas every 25th epoch, the stored data is updated and the results are stored in a `.csv` file. Additionally, the network should be split at the `seg_outputs` part of the network and trained on the GPU with ID <GPU_ID> (can be one or multilpe IDs). The importance of the previous tasks for the PLOP method should be set to 0.01 instead of 0.05:
```bash
          ~ $ source ~/.bashrc
          ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_plop 3d_fullres -t 11 12 13 -f 0 -pod_lambda 0.01
                             -num_epoch 250 -d <GPU_ID> -save_interval 25 -s seg_outputs --store_csv
                             [--use_vit -v <VERSION> -v_type <TYPE> -pod_scales <SCALE> ...]
```

The following example uses Version 1 (out of 3) of the [Generic_ViT_UNet](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/nnunet_ext/network_architecture/generic_ViT_UNet.py#L14) using the smallest one (out of 3 types). More informations with regard to the ViT_U-Net architecture can be found [here](https://github.com/camgbus/Lifelong-nnUNet/blob/ViT_U-Net/documentation/ViT_U-Net.md). Further, only 2 scales should be used instead of 3 and no POD embedding should be included in the loss calculation:
```bash
          ~ $ source ~/.bashrc
          ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_plop 3d_fullres -t 11 12 13 -f 0
                             -num_epoch 250 -d <GPU_ID> -save_interval 25 -s seg_outputs --store_csv
                             --use_vit -v 1 -v_type base -pod_scales 2
                             [-pod_lambda <VALUE> --use_mult_gpus ...]
                             
```

Note that the `--transfer_heads` flag can be used with this Trainer. If it is set, the previous head will be used as an initialization of the new head. If it is not set *-- as in all use cases above --*, the head from the initial split from Multi-Head Module is used as initialization of the new head.
