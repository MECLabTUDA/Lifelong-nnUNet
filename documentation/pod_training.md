# nnU-Net Continual Learning extension: Training using Pooled Outputs Distillation

This is a general description on how to use the POD Trainer to train the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) on a sequence of tasks/datasets using Pooled Outputs Distillation as proposed in [here](https://arxiv.org/pdf/2011.11390.pdf) and [here](https://arxiv.org/pdf/2004.13513.pdf). The extension can only be used after it has been succesfully installed *-- including all dependencies --* following the instructions from [here](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/README.md#installation). Further, all relevant paths have to be set, so the nnU-Net and the nnU-Net extension can extract the right directories for training, preprocessing, storing etc. as described [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md). Before training with any Trainer, ensure that the Task is properly prepared, planned and preprocessed as described in the example [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/training_example_Hippocampus.md).

### Command Line Arguments
The exact same Command Line Arguments as presented in the [PLOP](plop_training.md) Trainer apply for the POD Trainer as well. 

### Exemplary use cases
In the following, a few examples are shown representing possible use cases on how to use the POD Trainer.

One of the easiest and simplest example is to simply train on a bunch of tasks, for example `Task011_XYZ`, `Task012_XYZ` and `Task013_XYZ`. Each task should be trained for 250 epochs, whereas every 25th epoch, the stored data is updated and the results are stored in a `.csv` file. Additionally, the network should be split at the `seg_outputs` part of the network and trained on the GPU with ID <GPU_ID> (can be one or multilpe IDs). The importance of the previous tasks for the PLOP method should be set to 0.01 instead of 0.05:
```bash
                    ~ $ source ~/.bashrc
                    ~ $ source activate <your_anaconda_env>
(<your_anaconda_env>) $ nnUNet_train_pod 3d_fullres -t 11 12 13 -f 0 -pod_lambda 0.01
                                         -num_epoch 250 -d <GPU_ID> -save_interval 25 -s seg_outputs --store_csv
                                         [--use_vit -v <VERSION> -v_type <TYPE> -pod_scales <SCALE> ...]
```

Note that the `--no_transfer_heads` flag can be used with this Trainer. If it is set, the previous head will not be used as an initialization of the new head, ie. the head from the initial split from Multi-Head Module is used as initialization of the new head. If it is not set *-- as in all use cases above --*, the previously trained head will be used as an initialization of the new head.
