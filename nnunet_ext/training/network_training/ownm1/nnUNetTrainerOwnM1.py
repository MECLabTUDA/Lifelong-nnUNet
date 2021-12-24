# EWC only on ViT, MiB KD loss; POD only on heads if desired

#########################################################################################################
#--------------------This class represents the nnUNet trainer for our own training.---------------------#
#########################################################################################################

# -- This implementation represents our own method -- #
from time import time
import os, copy, torch
from operator import attrgetter
from torch.cuda.amp import autocast
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.training.network_training.ewc.nnUNetTrainerEWC import nnUNetTrainerEWC
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossOwn1 as OwnLoss
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead


class nnUNetTrainerOwnM1(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='ownm1', ewc_lambda=0.4, mib_alpha=1., lkd=10, pod_lambda=1e-2,
                 scales=3, tasks_list_with_char=None, mixed_precision=True, save_csv=True, del_log=False, use_vit=True,
                 vit_type='base', version=1, split_gpu=False, transfer_heads=True, ViT_task_specific_ln=False, do_pod=True):
        r"""Constructor of our own trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads, ViT_task_specific_ln)
        
        # -- Remove the old directory -- #
        try:
            # -- Only remove if empty -- #
            os.rmdir(self.trained_on_path)
        except:
            pass

        # -- Update the folder names including indicating if POD is used or not -- #
        if do_pod:
            self.output_folder = join(self.output_folder, 'pod')
            self.trained_on_path = join(self.trained_on_path, 'pod')
        else:
            self.output_folder = join(self.output_folder, 'no_pod')
            self.trained_on_path = join(self.trained_on_path, 'no_pod')

        # -- Create the folder if necessary -- #
        maybe_mkdir_p(self.trained_on_path)

        # -- Define a variable that specifies the hyperparameters for this trainer --> this is used for the parameter search method -- #
        self.hyperparams = {'mib_alpha': float, 'lkd': float, 'pod_lambda': float, 'scales': int, 'ewc_lambda': float}
        
        # -- Set the parameters used for Loss calculation during training -- #
        self.ewc_lambda = ewc_lambda
        self.pod_lambda = pod_lambda
        self.do_pod = do_pod
        self.alpha = mib_alpha
        self.scales = scales
        self.lkd = lkd

        # -- Add flags in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        if already_trained_on is not None:
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                # -- Add the parameter and checkpoint settings -- #
                self.already_trained_on[str(self.fold)]['fisher_at'] = None
                self.already_trained_on[str(self.fold)]['params_at'] = None
                self.already_trained_on[str(self.fold)]['used_lkd'] = self.lkd
                self.already_trained_on[str(self.fold)]['used_alpha'] = self.alpha
                self.already_trained_on[str(self.fold)]['used_scales'] = self.scales
                self.already_trained_on[str(self.fold)]['used_pod_lambda'] = self.pod_lambda
                self.already_trained_on[str(self.fold)]['used_batch_size'] = self.batch_size
                self.already_trained_on[str(self.fold)]['used_ewc_lambda'] = self.ewc_lambda
            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['fisher_at', 'params_at', 'used_alpha', 'used_lkd', 'used_pod_lambda', 'used_scales', 'used_batch_size', 'used_ewc_lambda']
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            # -- Update settings in trained on file for restoring to be able to ensure that scales can not be changed during training -- #
            self.already_trained_on[str(self.fold)]['fisher_at'] = None
            self.already_trained_on[str(self.fold)]['params_at'] = None
            self.already_trained_on[str(self.fold)]['used_lkd'] = self.lkd
            self.already_trained_on[str(self.fold)]['used_alpha'] = self.alpha
            self.already_trained_on[str(self.fold)]['used_scales'] = self.scales
            self.already_trained_on[str(self.fold)]['used_pod_lambda'] = self.pod_lambda
            self.already_trained_on[str(self.fold)]['used_batch_size'] = self.batch_size
            self.already_trained_on[str(self.fold)]['used_ewc_lambda'] = self.ewc_lambda

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                          ewc_lambda, mib_alpha, lkd, pod_lambda, scales, tasks_list_with_char, mixed_precision, save_csv,
                          del_log, use_vit, self.vit_type, version, split_gpu, transfer_heads, ViT_task_specific_ln)

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
        self.ewc_data_path = join(self.trained_on_path, 'ewc_data_ownm1')

        if self.do_pod:
            # -- Define the place holders for our results from the previous model on the current data -- #
            self.old_interm_results = dict()

            # -- Define empty dict for the current intermediate results during training -- #
            self.interm_results = dict()

        # -- Define a flag to indicate if the loss is switched or not -- #
        self.switched = False

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite the initialize function so the correct Loss function for the EWC method can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)

        # -- Update self.trainer_path -- #
        if prev_trainer_path is not None and not call_for_eval:
            if self.do_pod:
                self.trainer_path = join(self.trainer_path.split(os.sep)[:-1], 'pod', "fold_%s" % str(self.fold))
            else:
                self.trainer_path = join(self.trainer_path.split(os.sep)[:-1], 'no_pod', "fold_%s" % str(self.fold))
        else:   # If for eval, then this is a nnUNetTrainerV2 whereas the path is not build as implemented in _build_output_path
            self.trainer_path = prev_trainer_path

        # -- Reset the batch size to something that should fit for every network, so something small but not too small. -- #
        # -- Otherwise the sizes for the convolutional outputs (ie. the batch dim) don't match and they have to -- #
        if self.do_pod:
            self.batch_size = 100
        self.already_trained_on[str(self.fold)]['used_batch_size'] = self.batch_size
        
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
        
        # -- Create a backup loss, so we can switch between original and own loss -- #
        self.loss_orig = copy.deepcopy(self.loss)
        # self.switched = False

        # -- Reset self.loss from MultipleOutputLoss2 to DC_and_CE_loss so the EWC Loss can be initialized properly -- #
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (Own Method) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters during each iteration -- #
        self.own_loss = OwnLoss(self.loss, self.ds_loss_weights, self.alpha, self.lkd, self.ewc_lambda,
                                self.fisher, self.params, self.network.named_parameters(), True, ['ViT'],
                                True, self.pod_lambda, self.scales, self.do_pod)

    def reinitialize(self, task):
        r"""This function is used to reinitialize the Multi Head Trainer when a new task is trained for our own Trainer.
        """
        # -- Execute the super function -- # 
        super().reinitialize(task, False)

        # -- Print Loss update -- #
        self.print_to_log_file("I am using my own loss now")
        
        # -- Put data on GPU since the data is moved to CPU before it is stored -- #
        for task in self.fisher.keys():
            for key in self.fisher[task].keys():
                to_cuda(self.fisher[task][key])
        for task in self.params.keys():
            for key in self.params[task].keys():
                to_cuda(self.params[task][key])

        # -- Update the fisher and param values in the loss function -- #
        if self.switched:
            self.loss.update_fisher_params(self.fisher, self.params, False)
        else:
            self.own_loss.update_fisher_params(self.fisher, self.params, False)

    def run_training(self, task, output_folder):
        r"""Perform training .
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

        # -- Create a deepcopy of the previous, ie. currently set model if we do PLOP training -- #
        if task not in self.mh_network.heads:
            self.network_old = copy.deepcopy(self.network)
            if self.split_gpu and not self.use_vit:
                self.network_old.cuda(1)    # Put on second GPU

            if self.do_pod:
                # -- Register the hook here as well -- #
                self.register_forward_hooks(old=True)
        
        # -- Execute the training for the desired epochs -- #
        if self.do_pod:
            output_folder = join(self._build_output_path(output_folder, False), 'pod', "fold_%s" % str(self.fold))
        else:
            output_folder = join(self._build_output_path(output_folder, False), 'no_pod', "fold_%s" % str(self.fold))
        ret = super().run_training(task, output_folder, build_folder=False)  # Execute training from parent class --> already_trained_on will be updated there
        
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

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True):
        r"""This function needs to be changed for the our own method.
        """
        # -- Ensure that the first task is trained as usual and the validation without the plop loss as well -- #
        if self.task in self.mh_network.heads and len(self.mh_network.heads) == 1 or run_online_evaluation: # The very first task
            # -- Use the original loss for this -- #
            self.loss = self.loss_orig
            self.switched = False
            # -- Run iteration as usual using parent class -- #
            loss = super().run_iteration(data_generator, do_backprop, run_online_evaluation, detach)
            # -- NOTE: If this is called during _perform_validation, run_online_evaluation is true --> Does not matter -- #
            # --       which loss is used, since we only calculate Dice and IoU and do not keep track of the loss -- #
        else:   # --> More than one head, ie. trained on more than one task  --> use PLOP
            if not self.switched:
                # -- Switch to own loss -- #
                self.loss = self.own_loss
                if self.do_pod:
                    # -- We are at a further sequence of training, so we train using the PLOP method -- #
                    self.register_forward_hooks()   # --> Just in case it is not already done, ie. after first task training!
                self.switched = True
            #------------------------------------------ Partially copied from original implementation ------------------------------------------#
            # -- Extract data -- #
            data_dict = next(data_generator)
            data = data_dict['data']
            target = data_dict['target']
            # -- Transform data to torch if necessary -- #
            data = maybe_to_torch(data)
            target = maybe_to_torch(target)
            # -- Put data on GPU -- #
            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)

            self.optimizer.zero_grad()

            if self.fp16:
                with autocast():
                    output = self.network(data) # --> self.interm_results is filled with intermediate result now!
                    # -- Extract the old results using the old network -- #
                    if self.split_gpu and not self.use_vit:
                        data = to_cuda(data, gpu_id=1)
                    output_o = self.network_old(data) # --> self.old_interm_results is filled with intermediate result now!
                    del data
                    if self.do_pod:
                        # -- Update the loss with the data -- #
                        self.loss.update_plop_params(self.old_interm_results, self.interm_results)
                    loss = self.loss(output, output_o, target)

                if do_backprop:
                    self.amp_grad_scaler.scale(loss).backward()
                    self.amp_grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
            else:
                output = self.network(data)
                if self.split_gpu and not self.use_vit:
                    data = to_cuda(data, gpu_id=1)
                output_o = self.network_old(data)
                del data
                if self.do_pod:
                    # -- Update the loss with the data -- #
                    self.loss.update_plop_params(self.old_interm_results, self.interm_results)
                loss = self.loss(output, output_o, target)

                if do_backprop:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.optimizer.step()

            if run_online_evaluation:
                self.run_online_evaluation(output, target)

            del target
            #------------------------------------------ Partially copied from original implementation ------------------------------------------#
    
            # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
            if do_backprop:
                self.mh_network.update_after_iteration()

            # -- Detach the loss -- #
            if detach:
                loss = loss.detach().cpu().numpy()

            if self.do_pod:
                # -- Empty the dicts -- #
                self.old_interm_results = dict()
                self.interm_results = dict()
        
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
        # -- Execute the function from EWC Trainer -- #
        nnUNetTrainerEWC.after_train(self)

        # -- Only keep the ones with the matching case in it to save time and space -- #
        for task in list(self.fisher.keys()):
            for key in list(self.fisher[task].keys()):
                if 'ViT' not in key:
                    # -- Remove the entry -- #
                    del self.fisher[task][key]
        for task in list(self.params.keys()):
            for key in list(self.params[task].keys()):
                if 'ViT' not in key:
                    # -- Remove the entry -- #
                    del self.params[task][key]

        # -- Storing and putting everything on CPU before is done in super class after this function is called -- #

    def register_forward_hooks(self, old=False):
        r"""This function sets the forward hooks for every convolutional layer in the network.
            The old parameter indicates that the old network should be used to register the hooks.
        """
        # -- Set the correct network to use -- #
        use_network = self.network_old if old else self.network

        # -- Extract all module names that are of any convolutional type -- #
        module_names = [name for name, module in use_network.named_modules() if 'conv.Conv' in str(type(module))]

        # -- Register hooks -- #
        for mod in module_names:
            # -- Only register hooks for the segmentation heads -- #
            if 'seg_outputs' in mod:
                attrgetter(mod)(use_network).register_forward_hook(self._get_activation(mod, old))

    def _get_activation(self, name, old=False):
        r"""This function returns the hook given a specific (module) name that needs to be
            registered to the module before calling it with actual data.
        """
        def hook(model, input, output):
            if old:
                self.old_interm_results[name]  = output.detach()     # Store the output in the dict at corresponding name
            else:
                self.interm_results[name] = output.detach()     # Store the output in the dict at corresponding name
        return hook
