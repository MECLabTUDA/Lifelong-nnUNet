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
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
#from nnunet_ext.utilities.load_prev_trainers import get_prev_trainers
#from nnunet_ext.utilities.helpful_functions import join_texts_with_char
#from nnunet_ext.run.default_configuration import get_default_configuration
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

        # -- Set the temperature variable for the LWF Loss calculation during training -- #
        self.lwf_temperature = lwf_temperature

        # -- Update lwf_temperature in trained on file fore restoring to be able to ensure that lwf_temperature can not be changed during training -- #
        self.already_trained_on[str(self.fold)]['used_lwf_temperature'] = self.lwf_temperature

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, already_trained_on, use_progress, identifier, extension,
                          lwf_temperature, tasks_list_with_char, mixed_precision)

        # -- Initialize a ModuleList in which all previous tasks models will be stored -- #
        #self.prev_trainer_list = torch.nn.ModuleList()

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None):
        r"""Overwrite the initialize function so the correct Loss function for the LWF method can be set.
            NOTE: The previous tasks are already set in self.mh_network, so everything is in self.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path)
 
        #  -- If this trainer is initialized for training, then load the models, else it is just an initialization as a prev_trainer -- #
        #if training:
            # -- Update the log -- #
            #self.print_to_log_file("Start initializing all previous models so they can be used for the LWF loss calculation.")   

            # -- Extract previous tasks -- #
            #previous_tasks = self.already_trained_on[str(self.fold)]['finished_training_on']
            
            # -- Load previous trained models -- #
            #if len(previous_tasks) != 0:
                # -- Load the previous task models -- #
            #    self.prev_trainer_list = get_prev_trainers(previous_tasks,
            #                                               network_name=self.network_name,
            #                                               tasks_joined_name=self.tasks_joined_name,
            #                                               already_trained_on=self.already_trained_on,
            #                                               fold=self.fold,
            #                                               extension=self.extension,
            #                                               prev_trainer=prev_trainer)

        # -- Reset self.loss from MultipleOutputLoss2 to DC_and_CE_loss so the LWF Loss can be initialized properly -- #
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (LWF) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- NOTE: The predictions of the previous models need to be updated after each iteration -- #
        self.loss = LWFLoss(self.loss, self.ds_loss_weights, list(), self.lwf_temperature)

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, gpu_id=0):
        r"""This function needs to be changed for the LWF method, since all previously trained models will be used
            to predict the same batch as the current model we train on. These results go then into the Loss function
            to compute the Loss as proposed in the paper.
            TODO: Change so this is not done when doing evaluation on every nth epoch!!!!!!!!!!!!!!!!!!!!!!
        """
        # -- Initialize empty list in which the predictions of the previous models will be put -- #
        prev_task_models_res = list()

        # -- Loop through tasks and load the corresponding model to make predictions -- #
        for task in list(self.mh_network.heads.keys())[:-1]:    # Skip the current task we're currently training on
            # -- Activate the model accordingly to task -- #
            self.network = self.mh_network.assemble_model(task)

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
            if torch.cuda.is_available():
                x = to_cuda(x, gpu_id=gpu_id)
                #prev_trainer = prev_trainer.cuda(gpu_id)

            # -- Make predictions using the loaded model and data -- #
            task_logit = self.network(x)

            # -- Append the result to prev_task_models_res -- #
            prev_task_models_res.append(task_logit)

        # -- Update the previous predictions list in the Loss function -- #
        self.loss.update_prev_trainer_predictions(prev_task_models_res)

        # -- Reset the network to the current task -- #
        self.network = self.mh_network.assemble_model(self.task)
        
        # -- Put model into train mode -- #
        self.network.train()

        # -- Run iteration as usual and return the loss -- #
        return super().run_iteration(data_generator, do_backprop, run_online_evaluation)