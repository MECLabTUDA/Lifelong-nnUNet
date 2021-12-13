#########################################################################################################
#----------This class represents the nnUNet trainer for EWC training. Implementation--------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

# -- The implementation of this method is based on the following Source Code: -- #
# -- https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb. -- #
# -- It represents the method proposed in the paper https://arxiv.org/pdf/1612.00796.pdf -- #

import torch
from time import time
from torch.cuda.amp import autocast
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossEWC as EWCLoss
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead


class nnUNetTrainerEWC(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='ewc', ewc_lambda=0.4, tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=False):
        r"""Constructor of EWC trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads)

        # -- Set the importance variable for the EWC Loss calculation during training -- #
        self.ewc_lambda = ewc_lambda

        # -- Add seed in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        if already_trained_on is not None:
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                # -- Add EWC specific entries to already_trained_on -- #
                self.already_trained_on[str(self.fold)]['used_ewc_lambda'] = self.ewc_lambda
                self.already_trained_on[str(self.fold)]['fisher_at'] = None
                self.already_trained_on[str(self.fold)]['params_at'] = None
            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['used_ewc_lambda', 'fisher_at', 'params_at']
                # -- Check that everything is provided as expected -- #
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            # -- Update ewc_lambda in trained on file fore restoring to be able to ensure that ewc_lambda can not be changed during training -- #
            self.already_trained_on[str(self.fold)]['used_ewc_lambda'] = self.ewc_lambda
            # -- Add fisher and params if the fold is freshly initialized -- #
            self.already_trained_on[str(self.fold)]['fisher_at'] = None
            self.already_trained_on[str(self.fold)]['params_at'] = None

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                          ewc_lambda, tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, self.vit_type,
                          version, split_gpu, transfer_heads)

        # -- Initialize dicts that hold the fisher and param values -- #
        if self.already_trained_on[str(self.fold)]['fisher_at'] is None or self.already_trained_on[str(self.fold)]['params_at'] is None:
            self.fisher = dict()
            self.params = dict()
        else:
            self.fisher = load_pickle(self.already_trained_on[str(self.fold)]['fisher_at'])
            self.params = load_pickle(self.already_trained_on[str(self.fold)]['params_at'])

            # -- Put data on GPU since the data is moved to CPU before it is stored -- #
            for task in self.fisher.keys():
                for key in self.fisher[task].keys():
                    to_cuda(self.fisher[task][key])
            for task in self.params.keys():
                for key in self.params[task].keys():
                    to_cuda(self.params[task][key])

        # -- Define the path where the fisher and param values should be stored/restored -- #
        self.ewc_data_path = join(self.trained_on_path, 'ewc_data')

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None):
        r"""Overwrite the initialize function so the correct Loss function for the EWC method can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path)
        
        # -- If this trainer has already trained on other tasks, then extract the fisher and params -- #
        if prev_trainer_path is not None and self.already_trained_on[str(self.fold)]['fisher_at'] is not None\
                                         and self.already_trained_on[str(self.fold)]['params_at'] is not None:
            self.fisher = load_pickle(self.already_trained_on[str(self.fold)]['fisher_at'])
            self.params = load_pickle(self.already_trained_on[str(self.fold)]['params_at'])

            # -- Put data on GPU since the data is moved to CPU before it is stored -- #
            for task in self.fisher.keys():
                for key in self.fisher[task].keys():
                    to_cuda(self.fisher[task][key])
            for task in self.params.keys():
                for key in self.params[task].keys():
                    to_cuda(self.params[task][key])
        
        # -- Reset self.loss from MultipleOutputLoss2 to DC_and_CE_loss so the EWC Loss can be initialized properly -- #
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (EWC) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters during each iteration -- #
        self.loss = EWCLoss(self.loss, self.ds_loss_weights,
                            self.ewc_lambda,
                            self.fisher,
                            self.params,
                            self.network.named_parameters())

    def reinitialize(self, task):
        r"""This function is used to reinitialize the Multi Head Trainer when a new task is trained for the EWC Trainer.
            The most important thing here is that it sets the fisher and param values accordingly in the loss.
            This should only be called when a new task is used --> by that time the new loss applies..
        """
        # -- Execute the super function -- # 
        super().reinitialize(task, False)

        # -- Print Loss update -- #
        self.print_to_log_file("I am using EWC loss now")
        
        # -- Put data on GPU since the data is moved to CPU before it is stored -- #
        for task in self.fisher.keys():
            for key in self.fisher[task].keys():
                to_cuda(self.fisher[task][key])
        for task in self.params.keys():
            for key in self.params[task].keys():
                to_cuda(self.params[task][key])

        # -- Update the fisher and param values in the loss function -- #
        self.loss.update_ewc_params(self.fisher, self.params)

    def run_training(self, task, output_folder):
        r"""Perform training using ewc trainer. Simply executes training method of parent class (nnUNetTrainerSequential)
            while updating fisher and params dicts.
            NOTE: This class expects that the trainer is already initialized, if not, the calling class will initialize,
                  however the class we inherit from has another initialize function, that does not set the number of epochs
                  to train, so it will be 500 and it does not set a prev_trainer. The prev_trainer will be set to None!
                  --> Initialize the trainer using your desired num_epochs and prev_trainer before calling run_training.  
        """
        # -- If there is at least one head and the current task is not in the heads, the network has finished on one task -- #
        # -- In such a case the fisher/param values should exist and should not be empty -- #
        if len(self.mh_network.heads) > 0 and task not in self.mh_network.heads:
            assert len(self.fisher) == len(self.mh_network.heads) and len(self.params) == len(self.mh_network.heads),\
            "The number of tasks in the fisher/param values are not as expected --> should be the same as in the Multi Head network."

        # -- Execute the training for the desired epochs -- #
        ret = super().run_training(task, output_folder)  # Execute training from parent class --> already_trained_on will be updated there
        
        # -- Define the fisher and params after the training -- #
        self.fisher[task] = dict()
        self.params[task] = dict()
        
        # -- Run forward and backward pass without optimizer step and extract the gradients -- #
        # -- --> optimizer step updates the weights -- #
        # -- This will update the parameters for the current task in self.fisher and self.params -- #
        self.after_train()

        # -- Put data from GPU to CPU before storing them in files -- #
        for task in self.fisher.keys():
            for key in self.fisher[task].keys():
                self.fisher[task][key].cpu()
        for task in self.params.keys():
            for key in self.params[task].keys():
                self.params[task][key].cpu()

        # -- Dump both dicts as pkl files -- #
        maybe_mkdir_p(self.ewc_data_path)
        write_pickle(self.fisher, join(self.ewc_data_path, 'fisher_values.pkl'))
        write_pickle(self.params, join(self.ewc_data_path, 'param_values.pkl'))

        if self.already_trained_on[str(self.fold)]['fisher_at'] is None or self.already_trained_on[str(self.fold)]['params_at'] is None:
            # -- Update the already_trained_on file that the values exist if necessary -- #
            self.already_trained_on[str(self.fold)]['fisher_at'] = join(self.ewc_data_path, 'fisher_values.pkl')
            self.already_trained_on[str(self.fold)]['params_at'] = join(self.ewc_data_path, 'param_values.pkl')
            
            # -- Save the updated dictionary as a json file -- #
            save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))
            # -- Update self.init_tasks so the storing works properly -- #
            self.update_init_args()
            # -- Resave the final model pkl file so the already trained on is updated there as well -- #
            self.save_init_args(join(self.output_folder, "model_final_checkpoint.model"))
        
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
        
        # -- After running one iteration and calculating the loss, update the parameters of the loss for the next iteration -- #
        # -- NOTE: The gradients DO exist even after the loss detaching of the super function, however the loss function -- #
        # --       does not need them, since they are only necessary for the Fisher values that are calculated once the -- #
        # --       training is done performing an epoch with no optimizer steps --> see after_train() for that -- #
        self.loss.update_network_params(self.network.named_parameters())
        
        # -- Return the loss -- #
        return loss

    def after_train(self):
        r"""This function needs to be executed once the training of the current task is finished.
            The function will use the same data to generate the gradients again and setting the
            models parameters.
        """
        # -- Update the log -- #
        self.print_to_log_file("Running one last epoch without changing the weights to extract Fisher and Parameter values...")
        start_time = time()
        #------------------------------------------ Partially copied from original implementation ------------------------------------------#
        # -- Put the network in train mode and kill gradients -- #
        self.network.train()
        self.optimizer.zero_grad()

        # -- Do loop through the data based on the number of batches -- # self.tr_gen
        for _ in range(self.num_batches_per_epoch):
            self.optimizer.zero_grad()
            # -- Extract the data -- #
            data_dict = next(self.tr_gen)
            data = data_dict['data']
            target = data_dict['target']

            # -- Push data to GPU -- #
            data = maybe_to_torch(data)
            target = maybe_to_torch(target)
            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)

            # -- Respect the fact if the user wants to autocast during training -- #
            if self.fp16:
                with autocast():
                    output = self.network(data)
                    del data
                    loss = self.loss(output, target)
                # -- Do backpropagation but do NOT update the weights -- #
                self.amp_grad_scaler.scale(loss).backward()
            else:
                output = self.network(data)
                del data
                loss = self.loss(output, target)
                # -- Do backpropagation but do NOT update the weights -- #
                loss.backward()

            del target

        # -- Set fisher and params in current fold from last iteration --> final model parameters -- #
        for name, param in self.network.named_parameters():
            # -- Update the fisher and params dict -- #
            if param.grad is None:
                self.fisher[self.task][name] = torch.tensor([1], device='cuda:0')
            else:
                self.fisher[self.task][name] = param.grad.data.clone().pow(2)
            self.params[self.task][name] = param.data.clone()

        # -- Discard the calculated loss -- #
        del loss
        #------------------------------------------ Partially copied from original implementation ------------------------------------------#
        # -- Update the log -- #
        self.print_to_log_file("Extraction and saving of Fisher and Parameter values took %.2f seconds" % (time() - start_time))