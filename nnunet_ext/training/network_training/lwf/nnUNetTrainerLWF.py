#########################################################################################################
#----------This class represents the nnUNet trainer for LWF training. Implementation--------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

# -- The implementation of this method is based on the following Source Code: -- #
# -- https://github.com/arunmallya/packnet/blob/master/src/lwf.py. -- #
# -- It represents the method proposed in the paper https://arxiv.org/pdf/1606.09282.pdf -- #
# -- The original implementation from https://github.com/lizhitwo/LearningWithoutForgetting -- #
# -- refers to the one that is used in this class, so when citing, cite both -- #

import torch
from itertools import tee
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLoss2LWF as LWFLoss


class nnUNetTrainerLWF(nnUNetTrainerSequential): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None,
                 identifier=default_plans_identifier, extension='lwf', lwf_temperature=2.0, tasks_joined_name=None, trainer_class_name=None):
        r"""Constructor of LWF trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16,
                         save_interval, already_trained_on, identifier, extension, tasks_joined_name, trainer_class_name)

        # -- Set the temperature variable for the LWF Loss calculation during training -- #
        self.lwf_temperature = lwf_temperature

        # -- Update lwf_temperature in trained on file fore restoring to be able to ensure that lwf_temperature can not be changed during training -- #
        self.already_trained_on[str(self.fold)]['used_lwf_temperature'] = self.lwf_temperature

    def run_training(self, task):
        r"""Perform training using lwf trainer. Simply executes training method of parent class (nnUNetTrainerSequential)
            while calculating the predictions using the previous task models.
        """
        # -- Choose the right loss function (LWF) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        self.loss = LWFLoss(self.loss, self.ds_loss_weights, list(), self.lwf_temperature)

        # -- Execute the training for the desired epochs -- #
        ret = super().run_training()       # Execute training from parent class

        return ret  # Finished with training for the specific task

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        r"""This function needs to be changed for the EWC method, since it is very important, even
            crucial to update the current models network parameters that will be used in the loss function
            after each iteration, and not after each epoch! If this will not be done after each iteration
            the EWC loss calculation would always use the network parameters that were initialized before the
            first epoch took place which is wrong because it should be always the one of the current iteration.
            It is the same with the loss, we do not calculate the loss once every epoch, but with every iteration (batch).
        """
        # -- Initialize empty list in which the predictions of the previous models will be put -- #
        prev_trainer_res = list()

        # -- Loop through tasks and load the corresponding model to make predictions -- #
        for task in self.already_trained_on[str(self.fold)]['finished_training_on']:
            # -- Load model of previous task -- #
            model = task #TODO
            
            # -- Put model on GPU and set it to evaluation so it can be properly used for inference -- #
            model.eval()

            # -- Create a copy from the data_generator so the data_generator won't be touched. -- #
            # -- This way, each previous task uses the same batch, as well as the model that will train -- #
            # -- using the data_generator and thus same batch. -- #
            data = tee(data_generator, n=1)

            # -- Extract the current batch from data -- #
            x = next(data)
            x = maybe_to_torch(x['data'])

            # -- Put the model and data on GPU if possible -- #
            if torch.cuda.is_available():
                x = to_cuda(x)
                model = to_cuda(model)

            # -- Make predictions using the loaded model and data -- #
            task_logit = model(x)

            # -- Append the result to prev_trainer_res -- #
            prev_trainer_res.append(task_logit)

        # -- Update the previous predictions list in the Loss class -- #
        self.loss.update_prev_trainer_predictions(prev_trainer_res)

        # -- Run iteration as usual and return the loss -- #
        return super().run_iteration(data_generator, do_backprop, run_online_evaluation)