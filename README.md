# Lifelong-nnUNet

This repository aims to extend the popular [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework with methods that allow for a safer on-the-wild use. This includes functionality for continual learning, out-of-distribution detection and self-supervision. 

Currently, each of these functionalities is found under a specific branch, such as the `continual_learning` branch you are currently at.

The supported nnUNet version is specified in the `requirements.txt` file. Please note that, at times, files are replicated from this version and adapted as needed. If you wish to use a newer nnUNet version, please make sure that all adapted files are consistent with that version. For the current `continual_learning` branch, this does not apply, ie. no files are replicated.


## Table Of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Required Paths](#required-paths)
4. [Mapping Datasets to Other Labels](#mapping-datasets-to-other-labels)
5. [License](#license)


## Introduction

This branch currently includes the following methods for Continual Learning:
* Sequential Training
* Rehearsal Training
* Elastic Weight Consolidation
* Learning Without Forgetting

For instructions on how to run these please see [here](documentation/continual_learning.md).


## Installation

The simplest way to install all dependencies is by using [Anaconda](https://conda.io/projects/conda/en/latest/index.html):

1. Create a Python 3.9 environment as `conda create -n <your_conda_env> python=3.9` and activate it as `conda activate  <your_conda_env>`.
2. Install CUDA and PyTorch through conda with the command specified by https://pytorch.org/. The command for Linux was at the time `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`. At least PyTorch version 1.6 is required, and the code was last tested with version 1.9. Pytorch and TorchVision versions can be specified during the installation as `conda install pytorch==<X.X.X> torchvision==<X.X.X> cudatoolkit=<X.X> -c pytorch`.
3. Navigate to the project root (where `setup.py` lives).
4. Execute `pip install -r requirements.txt` to install all required packages. With this step, the [original nnUNet](https://github.com/MIC-DKFZ/nnUNet) will be installed as well, so all commands described there will work. Please note that the nnUNet commit to install is specified in requirements.txt, the code may not work for other versions of the nnUNet.
5. Set your paths as described [here](documentation/setting_up_paths.md). You should set [these](#required-paths) paths.
6. Execute `pytest` to ensure that everything is working. All tests should work, however one test specifically tests if at least one GPU is present, so this one might fail if no GPU is installed.


## Required Paths

Following environment variables must be set for all Lifelong-nnUNet branches:

* nnUNet_raw_data_base
* nnUNet_preprocessed
* RESULTS_FOLDER
* EVALUATION_FOLDER

Refer to [this file](documentation/setting_up_paths.md) for a description of how to set these.


## Mapping Datasets to Other Labels

In certain cases you may wish to change the meaning of certain labels or merge different labels in order to harmonize label structures between datasets. Please refer to [this file](documentation/change_mask_labels.md) for instructions on how to do this.


## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
