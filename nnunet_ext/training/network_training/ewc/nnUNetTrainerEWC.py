#########################################################################################################
#----------This class represents the nnUNet trainer for EWC training. Implementation--------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

# -- The implementation of this method is based on the following Source Code: -- #
# -- https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb. -- #
# -- It represents the method proposed in the paper https://arxiv.org/pdf/1612.00796.pdf -- #

from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLoss2EWC as EWCLoss


class nnUNetTrainerEWC(nnUNetTrainerSequential): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None,
                 identifier=default_plans_identifier, extension='ewc', ewc_lambda=0.4, tasks_joined_name=None, trainer_class_name=None):
        r"""Constructor of EWC trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16,
                         save_interval, already_trained_on, identifier, extension, tasks_joined_name, trainer_class_name)

        # -- Set the importance variable for the EWC Loss calculation during training -- #
        self.ewc_lambda = ewc_lambda

        # -- Update ewc_lambda in trained on file fore restoring to be able to ensure that ewc_lambda can not be changed during training -- #
        self.already_trained_on[str(self.fold)]['used_ewc_lambda'] = self.ewc_lambda

        # -- If already_trained_on is not None, this is a restoring, so add fisher and params if the fold is freshly initialized -- #
        if already_trained_on is not None: 
            # -- If the fisher in the current fold does not exists initialize it -- #
            if self.already_trained_on[str(self.fold)].get('fisher', None) is None: # Fold has been freshly initialized
                self.already_trained_on[str(self.fold)]['fisher'] = dict()
                self.already_trained_on[str(self.fold)]['params'] = dict()
        else:
            self.already_trained_on[str(self.fold)]['fisher'] = dict()
            self.already_trained_on[str(self.fold)]['params'] = dict()

    def initialize_network(self):
        r"""Initialize the network using parent class and extract fisher and param values if a model
            for initialization was provided.
        """
        # -- Set a variable that helps to determine how to calculate the fisher and params -- #
        fisher_possible = self.trainer is not None
            
        # -- Initialize from beginning and start training, since no model is provided -- #
        super().initialize_network() # --> This updates the corresponding variables automatically since we inherit this class
            
        # -- Calculate fisher and extract params a trainer was provided, ie. we have a nnUNetTrainerV2 as initialization (pre-trained) -- #
        if fisher_possible:
            # -- Extract the task the provided network has been trained on so far -- #
            task = self.already_trained_on[str(self.fold)]['finished_training_on'][0] # --> Only one task can be in this list at this point

            # -- Set fisher and params in current fold -- #
            for name, param in self.network.named_parameters():
                # -- Update the fisher and params dict -- #
                self.already_trained_on[str(self.fold)]['fisher'][task][name] = param.grad.data.clone().pow(2)
                self.already_trained_on[str(self.fold)]['params'][task][name] = param.data.clone()
        
            # -- Set own network to trainer.network to use this pre-trained model if it exists -- #
            self.network = self.trainer.network

    """
    #------------------------------------------ Partially copied by original implementation ------------------------------------------#
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        For each iteration, the Loss Function is differently calculated. It is defined in the EWC paper
           and is simply: loss_{t} + ewc_lambda * \sum_{i} F_{i}(param_{i} - param_{t-1, i})**2.
           The function itself is the same as from the inherited class, however the calculation of the loss
           that influence the backpropagation is adapted.
        
        # -- Extract data and target -- #
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        # -- Put data and target to cuda if possible -- #
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        # -- Zero out the gradients -- #
        self.optimizer.zero_grad()

        # -- Check if model should be compressed to floating point 16 --> then autocast -- #
        if self.fp16:
            with autocast():
                # -- Calculate output and delete data -- #
                output = self.network(data)
                del data

                # -- Calculate the current loss and
                loss = self.loss(output, target)

                # -- Simply update the loss as proposed in the paper and return this loss to the calling function instead -- #
                # -- Loop through the tasks the model has already been trained on -- #
                for task in self.already_trained_on[str(self.fold)]['finished_training_on']:
                    for name, param in self.network.named_parameters(): # Get named parameters of the current model
                        # -- Extract corresponding fisher and param values -- #
                        fisher_value = self.already_trained_on[str(self.fold)]['fisher'][task][name]
                        param_value = self.already_trained_on[str(self.fold)]['params'][task][name]

                        # -- loss = loss_{t} + ewc_lambda * \sum_{i} F_{i}(param_{i} - param_{t-1, i})**2 -- #
                        loss = loss + self.ewc_lambda * (fisher_value * (param_value - param).pow(2)).sum()

            if do_backprop:
                self.amp_grad_scaler.scale(loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            loss = self.loss(output, target)

            # -- Simply update the loss as proposed in the paper and return this loss to the calling function instead -- #
            # -- Loop through the tasks the model has already been trained on -- #
            for task in self.already_trained_on[str(self.fold)]['finished_training_on']:
                for name, param in self.network.named_parameters(): # Get named parameters of the current model
                    # -- Extract corresponding fisher and param values -- #
                    fisher_value = self.already_trained_on[str(self.fold)]['fisher'][task][name]
                    param_value = self.already_trained_on[str(self.fold)]['params'][task][name]

                    # -- loss = loss_{t} + ewc_lambda * \sum_{i} F_{i}(param_{i} - param_{t-1, i})**2 -- #
                    loss = loss + self.ewc_lambda * (fisher_value * (param_value - param).pow(2)).sum()

            if do_backprop:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        # -- Return the updated loss value -- #
        return loss.detach().cpu().numpy()
    #------------------------------------------ Partially copied by original implementation ------------------------------------------#
    """

    def run_training(self, task):
        r"""Perform training using ewc trainer. Simply executes training method of parent class (nnUNetTrainerSequential)
            while updating fisher and params dicts.
        """
        # -- Choose the right loss function (EWC) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters after each iteration .. -- #
        self.loss = EWCLoss(self.loss, self.ds_loss_weights,
                            self.already_trained_on[str(self.fold)]['finished_training_on'],
                            self.ewc_lambda,
                            self.already_trained_on[str(self.fold)]['fisher'],
                            self.already_trained_on[str(self.fold)]['params'],
                            self.network.named_parameters())

        # -- Execute the training for the desired epochs -- #
        ret = super().run_training(task)       # Execute training from parent class --> already_trained_on will be updated there

        # -- The gradients of the freshly trained network (model) can be used to calculate fisher for the current task -- #
        for name, param in self.network.named_parameters():
            # -- Update the fisher and params dict -- #
            self.already_trained_on[str(self.fold)]['fisher'][task][name] = param.grad.data.clone().pow(2)
            self.already_trained_on[str(self.fold)]['params'][task][name] = param.data.clone()

        # -- Save the updated dictionary as a json file -- #
        save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))

        return ret  # Finished with training for the specific task

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        r"""This function needs to be changed for the EWC method, since it is very important, even
            crucial to update the current models network parameters that will be used in the loss function
            after each iteration, and not after each epoch! If this will not be done after each iteration
            the EWC loss calculation would always use the network parameters that were initialized before the
            first epoch took place which is wrong because it should be always the one of the current iteration.
            It is the same with the loss, we do not calculate the loss once every epoch, but with every iteration (batch).
        """
        # -- Run iteration as usual -- #
        loss = super().run_iteration(data_generator, do_backprop, run_online_evaluation)

        # -- After running iteration and calculating the loss, update the parameters for the loss in next iteration -- #
        self.loss.update_network_params(self.network.named_parameters())

        # -- Return the loss -- #
        return loss