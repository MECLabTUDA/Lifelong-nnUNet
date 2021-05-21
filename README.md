# nnU-Net Continual Learning extensions

This specific branch of the [nnUNet_extensions repository](https://github.com/camgbus/nnUNet_extensions) contains multiple Continnual Learning (CL) extensions for the nnUNet framework.

## Table Of Contents

[Introduction](#introduction)

[Installation](#installation)

[Extensions](#extensions)
  * [Sequential Training](#sequential-training)
  * [Rehearsal Training](#rehearsal-training)
  * [Elastic Weight Consolidation](#elastic-weight-consolidation)
  * [Learning Without Forgetting](#learning-without-forgetting)
  * [TBD](#tbd)

[Documentations](#documentations)

[License](#license)

## Introduction
TODO

## Installation
The simplest and most convenient way to install everything can be achieved by using Anaconda:

1. Create a Python3.8 environment as follows: `conda create -n <your_anaconda_env> python=3.8` and activate the environment.
2. Install CUDA and PyTorch through conda with the command specified by https://pytorch.org/. The command for Linux was at the time `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`.
3. Navigate to the project root (where setup.py lives)
4. Execute `pip install -r requirements.txt` to install all required packages. With this step, the [original nnU-Net](https://github.com/MIC-DKFZ/nnUNet) will be installed as well, so all commands described there will work as well.
5. Set your paths as described in the original [nnU-Net repository](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).
6. Execute `pytest` to ensure that everything is working. All tests should work, however one test specifically tests if at least one GPU is present, so this one might fail if no GPU is installed.

NOTE: Maybe create own setting-up patsh file since location or name might change from original repo..


## Extensions
This nnU-Net Continual Learning extension provides multiple CL methods that can be used for training different models, which are introduced in the following sections.

### Sequential Training
TODO
### Rehearsal Training
TODO
### Elastic Weight Consolidation
TODO
### Learning Without Forgetting
TODO
### TBD
TODO

## Documentations
In the [documentation folder](/nnunet_ext/documentation) of this branch are multiple readme files *-- for each extension one --* that describes in general how and which arguments can be set for the specific extension followed by a setp-by-step example that can be easily replicated to understand the presented extension. Further, in each of the extensions implementation, the source code is extensively commented in such a way, that others that work with this extensions are able to follow the construction and pipeline of these models/trainers to be able to further extend it or use it in other projects.

## License
Set license
