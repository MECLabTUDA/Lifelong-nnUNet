#########################################################################################################
#----------This class represents the nnUNet trainer for LWF training. Implementation--------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

# -- The implementation of this method is based on the following Source Code: -- #
# -- https://github.com/arunmallya/packnet/blob/master/src/lwf.py. -- #
# -- It represents the method proposed in the paper https://arxiv.org/pdf/1606.09282.pdf -- #
# -- The original implementation from https://github.com/lizhitwo/LearningWithoutForgetting -- #
# -- refers to the one that is used in this class, so when citing, cite both -- #

import copy
import torch
from itertools import tee
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossLWF as LWFLoss
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead


class nnUNetTrainerLWF(nnUNetTrainerMultiHead): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='lwf', lwf_temperature=2.0, tasks_list_with_char=None,
                 mixed_precision=True, save_csv=True):
        r"""Constructor of LWF trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         save_csv)
        # -- Ensure that at least two GPUs are provided -- #
        assert torch.cuda.device_count() > 1, "For the LwF Trainer, at least two GPUs need to be provided."

        # -- Set the temperature variable for the LWF Loss calculation during training -- #
        self.lwf_temperature = lwf_temperature

        # -- Add seed in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        if already_trained_on is not None:
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                # -- Add the lwf temperature and checkpoint settings -- #
                self.already_trained_on[str(self.fold)]['used_lwf_temperature'] = self.lwf_temperature
                self.already_trained_on[str(self.fold)]['orig_checkpoint_should_exist'] = False
                self.already_trained_on[str(self.fold)]['tasks_at_time_of_orig_checkpoint'] = list()
                self.already_trained_on[str(self.fold)]['active_task_at_time_of_orig_checkpoint'] = None
            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['used_lwf_temperature', 'orig_checkpoint_should_exist',\
                        'tasks_at_time_of_orig_checkpoint', 'active_task_at_time_of_orig_checkpoint']
                # -- Check that everything is provided as expected -- #
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            # -- Update lwf_temperature in trained on file for restoring to be able to ensure that lwf_temperature can not be changed during training -- #
            self.already_trained_on[str(self.fold)]['used_lwf_temperature'] = self.lwf_temperature
            # -- Update original model checkpoint informations in the already_trained_on -- #
            self.already_trained_on[str(self.fold)]['orig_checkpoint_should_exist'] = False
            self.already_trained_on[str(self.fold)]['tasks_at_time_of_orig_checkpoint'] = list()
            self.already_trained_on[str(self.fold)]['active_task_at_time_of_orig_checkpoint'] = None

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, already_trained_on, use_progress, identifier, extension,
                          lwf_temperature, tasks_list_with_char, mixed_precision)

        # -- Initialize a variable for the copied mh_network at the beginning of every task -- #
        self.orig_mh_network = None

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None):
        r"""Overwrite the initialize function so the correct Loss function for the LWF method can be set.
            NOTE: The previous tasks are already set in self.mh_network, so everything is in self.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path)
 
        # -- Now we can set the orig model but only if the mh_network has more than one task -- #
        if len(list(self.mh_network.heads.keys())) > 1: # Do this since we need one historical task otherwise it makes no sense
            self.orig_mh_network = copy.deepcopy(self.mh_network)

        # -- Reset self.loss from MultipleOutputLoss2 to DC_and_CE_loss so the LWF Loss can be initialized properly -- #
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (LWF) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- NOTE: The predictions of the previous models need to be updated after each iteration -- #
        self.loss = LWFLoss(self.loss, self.ds_loss_weights, list(), list(), self.lwf_temperature)
    
    def reinitialize(self, task):
        r"""This function is used to reinitialize the Multi Head Trainer when a new task is trained for the LWF Trainer.
            The most important thing here is that it sets the orig_mh_network as well for the LWF method.
        """ 
        # -- Execute the super function -- # 
        super().reinitialize(task)
        
        # -- Now we can certainly set the orig model -- #
        self.orig_mh_network = copy.deepcopy(self.mh_network)

        # -- If this task is already in the orig network remove it, since we are going to train it and it should not be in orig -- #
        if task in self.orig_mh_network.heads:
            del self.orig_mh_network.heads[task]

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, gpu_id=0):
        r"""This function needs to be changed for the LWF method, since all previously trained models will be used
            to predict the same batch as the current model we train on. These results go then into the Loss function
            to compute the Loss as proposed in the paper.
        """
        # -- Check if we are performing an iteration for validation purposes only, then we do not have to do all the following -- #
        if do_backprop and self.orig_mh_network is not None: # --> only do this if we want backpropagation, ie. during training and we have at least one task
            # -- Initialize empty list in which the predictions of the previous models will be put -- #
            pred_logits = list()
            target_logits = list()
            for i in range(2): # --> Two times, once with original and once with the current running model
                # -- Loop through tasks and load the corresponding model to make predictions -- #
                for task in list(self.mh_network.heads.keys())[:-1]:    # Skip the current task we're currently training on
                    # -- Activate the model accordingly to task using the orig or the updated model -- #
                    if i == 0:
                        self.network = copy.deepcopy(self.orig_mh_network.assemble_model(task))
                    else:
                        self.network = copy.deepcopy(self.mh_network.assemble_model(task)).cuda(1)

                    # -- Set network to eval -- #
                    self.network.eval()
                    
                    # -- Create a copy from the data_generator so the data_generator won't be touched. -- #
                    # -- This way, each previous task uses the same batch, as well as the model that will train -- #
                    # -- using the data_generator and thus same batch. -- #
                    data = tee(data_generator, 1)[0]

                    # -- Extract the current batch from data -- #
                    x = next(data)
                    x = maybe_to_torch(x['data'])

                    # -- Put the data on GPU if possible --> model is already on GPU -- #
                    if i == 0:
                        x = to_cuda(x, gpu_id=gpu_id)
                    else:
                        x = to_cuda(x, gpu_id=1)

                    # -- Make predictions using the loaded model and data -- #
                    task_logit = self.network(x)[0]

                    # -- Flatten the task logit since it has more than one output tensor based on the network structure -- #
                    task_logit_flat = list()
                    for task in task_logit:
                        # -- Append the task to the flat list so the list only contains of tensors -- #
                        task_logit_flat.append(task.cuda(gpu_id))   # --> Ensure that all tasks are on the same GPU for the loss calculation

                    # -- Append the result to target_logits or pred_logits -- #
                    if i == 0:
                        target_logits.extend(task_logit_flat)
                    else:
                        pred_logits.extend(task_logit_flat)

            # -- Update the previous predictions list in the Loss function -- #
            self.loss.update_logits(pred_logits, target_logits)

        # -- Reset the network to the current task -- #
        self.network = self.mh_network.assemble_model(self.task)
        
        # -- Put model into train mode -- #
        self.network.train()

        # -- Run iteration as usual and return the loss -- #
        ret = super().run_iteration(data_generator, do_backprop, run_online_evaluation)

        # -- Return the result -- #
        return ret

    def save_checkpoint(self, fname, save_optimizer=True):
        r"""Overwrite the parent class, since we want to store the original network as well along with the current running
            model.
        """
        # -- Use parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
        super().save_checkpoint(fname, save_optimizer)

        # -- Store the original model as well, only if model is not empty -- #
        if self.orig_mh_network is not None:
            # -- Set the network to the full MultiHead_Module network to save everything in the class not only the current model -- #
            self.network = self.orig_mh_network

            # -- Set the flag in already_trained_on BEFORE creating a checkpoint since this will otherwise be missing -- #
            # -- Set the flag to True -- #
            self.already_trained_on[str(self.fold)]['orig_checkpoint_should_exist'] = True
            # -- Add the current head keys for restoring (is in correct order due to OrderedDict type of heads) -- #
            self.already_trained_on[str(self.fold)]['tasks_at_time_of_orig_checkpoint'] = list(self.orig_mh_network.heads.keys())
            # -- Add the current active task for restoring -- #
            self.already_trained_on[str(self.fold)]['active_task_at_time_of_orig_checkpoint'] = self.orig_mh_network.active_task
            # -- Save the updated dictionary as a json file -- #
            save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))
            # -- Update self.init_tasks so the storing works properly -- #
            self.update_init_args()

            # -- Use grand parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
            super(nnUNetTrainerV2, self).save_checkpoint(fname, save_optimizer)

            # -- Reset network to the assembled model to continue training -- #
            self.network = self.mh_network.model

    def load_checkpoint_ram(self, checkpoint, train=True):
        r"""Overwrite the parent class, since we want to restore the original network as well along with the current running
            model.
        """
        # -- Restore the running model network first using super class -- #
        super().load_checkpoint_ram(checkpoint, train)

        # -- If the original model checkpoint exists, load it -- #
        if self.already_trained_on[str(self.fold)]['orig_checkpoint_should_exist']:
            # -- For all tasks, create a corresponding head, otherwise the restoring would not work due to mismatching weights -- #
            self.orig_mh_network.add_n_tasks_and_activate(self.already_trained_on[str(self.fold)]['tasks_at_time_of_orig_checkpoint'],
                                                          self.already_trained_on[str(self.fold)]['active_task_at_time_of_orig_checkpoint'])
            
            # -- Set the network to the full MultiHead_Module network to restore everything -- #
            self.network = self.orig_mh_network

            # -- Use grand parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
            super(nnUNetTrainerV2, self).load_checkpoint_ram(checkpoint, train)

            # -- Reset the running model to train on -- #
            self.network = self.mh_network.model