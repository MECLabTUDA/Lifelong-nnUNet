# Lifelong-nnUNet: Continual Learning (CL)

Lifelong-nnUNet can only be used after it has been succesfully installed *-- including all dependencies --* following [these instructions](../README.md#installation). Further, all relevant paths have to be set so that the right directories for training, preprocessing, storing etc. can be extracted. This process is described [here](setting_up_paths.md). Note that this repository/branch only represents an extension of the original [nnUNet Framework](https://github.com/MIC-DKFZ/nnUNet), thus all nnunet commands should still work.


## General folder structure for CL extension
The CL extension creates seperate folders for all provided extension networks, to seperate the results *-- trained models and evaluation results--* from the results of the actual nnU-Net Framework. For this sake, in the folder where the trained networks are stored, the ones from the original nnU-Net Framework are stored under `nnUNet/...` and the provided methods from the CL extension under `nnUNet_ext/...`. The same goes for the results from the evaluation as presented in the corresponding [documentation](evaluation.md). 


## Implemented Methods
This Lifelong-nnUNet CL extension provides multiple Baseline and CL methods that can be used for training different models, which are introduced in the following sections.

### Baselines -- Multi-Head and Sequential Training
The simplest extension which can be referred as one of the Baselines (along with the Sequential Training) is the Multi-Head Training process. All provided Trainers are based on the Multi-Head Trainer and thus follow the same idea. Every network is split into a shared body and task specific heads. More details regarding the Multi-Head Architecture can be found [here](multihead_architecture.md). Based on this architecture, for every new task the network is trained on, a new head will be introduced and trained. From this point, there are two main methods on how to introduce a new head, thus resulting in two simple non-CL trainers, referred to as Baselines. Every head has the same structure, however the most important thing of those task specific heads are the weights and biases that are learned throughout the training of this head with the specific dataset/task. The difference between the two Baselines lies in the setting of those weights and biases:
*   Internally, when splitting a Generic U-Net into a body and the very first head, the *state_dict* of this head will be stored in a variable. If the weights and biases of a new head are set to the ones directly stored after the initial splitting of the network, then this is referred to as a Multi-Head Training process. This process and how to train a network like this is further described [here](multihead_training.md).
*   The second option is to use the last trained head as initialization for the new head, thus a classical sequential method which can be compared to a sort of transfer learning. More details can be found [here](sequential_training.md).

### CL methods
When it comes to CL methods, this extension provides three methods in total that use the [Multi-Head Architecture](multihead_architecture.md) in order to train on a sequence of tasks. 

#### Rehearsal Training
The simplest way to tackle the phenomenon of *Catastrophic Forgetting*, thus enabling Lifelong-Learning is the inclusion of data from previous tasks with the goal of maintaining the learned knowledge from previously seen data. This training process is also known as a simple Rehearsal method and is further described [here](rehearsal_training.md).

#### Elastic Weight Consolidation and Learning without Forgetting
Two, for image classification developed, known CL method are Elastic Weight Consolidation (EWC) and Learning without Forgetting (LwF) as introduced in the corresponding publications ([EWC](https://arxiv.org/pdf/1612.00796.pdf) and [LwF](https://arxiv.org/pdf/1606.09282.pdf)). The implementation of those methods was conducted very close to the original publication and implementation (if applicable). More details on the corresponding Trainers can be found [here](ewc_training.md) (EWC) and [here](lwf_training.md) (LwF).


## How to evaluate the trained networks
After successfull training of a or multiple methods, one might want to evaluate the network and its performance based on all the datasets it has been trained on. For this a specific command is provided that helps the user to specify which network should be evaluated on which datasets, i.e. not linked to the actual dataset(s) the network is trained on. More details are described [here](evaluation.md), however for the evaluation to work, the corresponding fourth path needs to be set as described in [here](setting_up_paths.md).


## CL extension commands
In general, all provided terminal commands are listet in [setup.py](https://github.com/camgbus/Lifelong-nnUNet/blob/continual_learning/setup.py). We made sure that those commands listed there do not interfere with the ones provided from the original nnU-Net Framework from [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/setup.py). For every method, specific commands need to be provided which can either be extracted from the corresponding source code or by executing the command in combination with the `-h` or `--help` flag which displays all parameters and a corresponding description. In general, we always provide a `-d` or `--device` flag that can be used to specifically set the preferred GPU device(s) that should be used for the corresponding command. This flag is not used in the orignal nnU-Net Framework. However, if the device is not set, always the GPU with ID 0 will be used. An alternative solution would be to start the desired command with `CUDA_VISIBLE_DEVICES=X nnUNet_...` resulting in only using the device with ID X.