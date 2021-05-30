#########################################################################################################
#----------This class represents the nnUNet trainer for EWC training. Implementation--------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

# -- The implementation of this method is based on the following Source Code: -- #
# -- https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb. -- #
# -- It represents the method proposed in the paper https://arxiv.org/pdf/1612.00796.pdf -- #

#from itertools import tee
from nnunet_ext.utilities.load_prev_trainers import get_prev_trainers
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLoss2EWC as EWCLoss
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential


class nnUNetTrainerEWC(nnUNetTrainerSequential): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None,
                 identifier=default_plans_identifier, extension='ewc', ewc_lambda=0.4, tasks_joined_name=None,
                 trainer_class_name=None):
        r"""Constructor of EWC trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16,
                         save_interval, already_trained_on, identifier, extension, tasks_joined_name, trainer_class_name)

        # -- Set the importance variable for the EWC Loss calculation during training -- #
        self.ewc_lambda = ewc_lambda

        # -- Update ewc_lambda in trained on file fore restoring to be able to ensure that ewc_lambda can not be changed during training -- #
        self.already_trained_on[str(self.fold)]['used_ewc_lambda'] = self.ewc_lambda
        
        """
        # -- If already_trained_on is not None, this is a restoring, so add fisher and params if the fold is freshly initialized -- #
        if already_trained_on is not None: 
            # -- If the fisher in the current fold does not exists initialize it -- #
            if self.already_trained_on[str(self.fold)].get('fisher', None) is None: # Fold has been freshly initialized
                self.already_trained_on[str(self.fold)]['fisher'] = dict()
                self.already_trained_on[str(self.fold)]['params'] = dict()
        else:
            self.already_trained_on[str(self.fold)]['fisher'] = dict()
            self.already_trained_on[str(self.fold)]['params'] = dict()
        """

        # -- Initialize a variable that includes all model parameters of the last iteration -- #
        #self.network_params_last_iteration = None

        # -- Initialize dicts that hold the fisher and param values -- #
        self.fisher = dict()
        self.params = dict()

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer=None):
        r"""Overwrite the initialize function so the correct Loss function for the EWC method can be set.
        """
        import torch
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer)

        # -- If this trainer is initialized for training, then load the models, else it is just an initialization as a prev_trainer -- #
        if training:
            # -- Update the log -- #
            self.print_to_log_file("Start initializing all previous models so they can be used for the EWC loss calculation.")   

            # -- Extract previous tasks -- #
            previous_tasks = self.already_trained_on[str(self.fold)]['finished_training_on']
            
            # -- Build fisher and params dictionaries based on previous trained models -- #
            if len(previous_tasks) != 0:
                # -- Load the previous task models -- #
                prev_models = get_prev_trainers(previous_tasks,
                                                network_name=self.network_name,
                                                tasks_joined_name=self.tasks_joined_name,
                                                already_trained_on=self.already_trained_on,
                                                fold=self.fold,
                                                extension=self.extension,
                                                prev_trainer=prev_trainer)
        
                # -- Set fisher and params accordingly -- #
                for idx, model in enumerate(prev_models):
                    # -- Extract the current task --> prev_models are in same order added to the list as the task from the provided list -- #
                    c_task = self.already_trained_on[str(self.fold)]['finished_training_on'][idx]
                    # -- Initialize the task in fisher and params -- #
                    self.fisher[c_task] = dict()
                    self.params[c_task] = dict()
                    
                    # -- Loop through the extracted models parameters and calculate fisher and params -- #
                    for name, param in model.named_parameters():
                        print(param.grad is None)
                        if param.grad is None:
                            continue
                        self.fisher[c_task][name] = param.grad.data.clone().pow(2)
                        self.params[c_task][name] = param.data.clone()

        # -- Reset self.loss from MultipleOutputLoss2 to DC_and_CE_loss so the EWC Loss can be initialized properly -- #
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (EWC) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters after each iteration .. -- #
        self.loss = EWCLoss(self.loss, self.ds_loss_weights,
                            self.already_trained_on[str(self.fold)]['finished_training_on'],
                            self.ewc_lambda,
                            self.fisher,
                            self.params,
                            self.network.named_parameters())

    def initialize_network(self):
        r"""Initialize the network using parent class and extract fisher and param values if a model
            for initialization was provided.
        """
        # -- Set a variable that helps to determine how to calculate the fisher and params -- #
        #fisher_possible = self.trainer is not None
            
        # -- Initialize from beginning and start training, since no model is provided -- #
        super().initialize_network() # --> This updates the corresponding variables automatically since we inherit this class
        
        """
        # -- Calculate fisher and extract params a trainer was provided, ie. we have a nnUNetTrainerV2 as initialization (pre-trained) -- #
        if fisher_possible:
            # -- Extract the task the provided network has been trained on so far -- #
            task = self.already_trained_on[str(self.fold)]['finished_training_on'][0] # --> Only one task can be in this list at this point

            # -- Add the task to fisher and params if it does not exist -- #
            if self.fisher.get(task, None) is None: # If fisher does not exist, params does not exist as well
                self.fisher[task] = dict()
                self.params[task] = dict()

            # -- Set fisher and params in current fold -- #
            for name, param in self.network.named_parameters():
                # -- Update the fisher and params dict -- #
                self.fisher[task][name] = param.grad.data.clone().pow(2)
                self.params[task][name] = param.data.clone()
        
            # -- Set own network to trainer.network to use this pre-trained model if it exists -- #
            self.network = self.trainer.network
        """

    
    def run_training(self, task):
        r"""Perform training using ewc trainer. Simply executes training method of parent class (nnUNetTrainerSequential)
            while updating fisher and params dicts.
            NOTE: This class expects that the trainer is already initialized, if not, the calling class will initialize,
                  however the class we inherit from has another initialize function, that does not set the number of epochs
                  to train, so it will be 500 and it does not set a prev_trainer. The prev_trainer will be set to None!
                  --> Initialize the trainer using your desired num_epochs and prev_trainer before calling run_training.  
        """
        # -- Execute the training for the desired epochs -- #
        ret = super().run_training(task)       # Execute training from parent class --> already_trained_on will be updated there

        """
        # -- Update fisher and params in case the training is not finished yet -- #
        # -- Add the task to fisher and param dicts to store the parameters of the trained model -- #
        self.fisher[task] = dict()
        self.params[task] = dict()
        
        # -- Set fisher and params in current fold from last iteration --> final model parameters -- #
        for name, param in self.network_params_last_iteration:
            # -- Update the fisher and params dict -- #
            if param.grad is None:
                continue
            self.fisher[task][name] = param.grad.data.clone().pow(2)
            self.params[task][name] = param.data.clone()
        """

        return ret  # Finished with training for the specific task

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        r"""This function needs to be changed for the EWC method, since it is very important, even
            crucial to update the current models network parameters that will be used in the loss function
            after each iteration, and not after each epoch! If this will not be done after each iteration
            the EWC loss calculation would always use the network parameters that were initialized before the
            first epoch took place which is wrong because it should be always the one of the current iteration.
            It is the same with the loss, we do not calculate the loss once every epoch, but with every iteration (batch).
        """
        # -- Before running iteration and calculating the loss, update the parameters for the loss in next iteration -- #
        # -- Do not do this directly after the super().run_iteration, since the super function detaches the value which -- #
        # -- results in gradients that are all 0 afterwards --> Loss results will not be the same -- #
        self.loss.update_network_params(self.network.named_parameters())

        # -- Run iteration as usual -- #
        loss = super().run_iteration(data_generator, do_backprop, run_online_evaluation)

        # -- Clone the network parameters (generator) so they can be included in the already_trained_on file when training is finished -- #
        #self.network_params_last_iteration = tee(self.network.named_parameters(), 1)[0]

        # -- Return the loss -- #
        return loss