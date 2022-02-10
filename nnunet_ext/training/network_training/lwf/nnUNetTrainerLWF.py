#########################################################################################################
#----------This class represents the nnUNet trainer for LWF training. Implementation--------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

# -- The implementation of this method is based on the following Source Code: -- #
# -- https://github.com/arunmallya/packnet/blob/master/src/lwf.py. -- #
# -- It represents the method proposed in the paper https://arxiv.org/pdf/1606.09282.pdf -- #
# -- The original implementation from https://github.com/lizhitwo/LearningWithoutForgetting -- #
# -- refers to the one that is used in this class, so when citing, cite both -- #

import copy, torch
from time import time
from itertools import tee
from torch.cuda.amp import autocast
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.utilities.helpful_functions import calculate_target_logits
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossLWF as LwFloss
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {'lwf_temperature': float}

class nnUNetTrainerLWF(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='lwf', lwf_temperature=2.0, tasks_list_with_char=None,
                 mixed_precision=True, save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False,
                 transfer_heads=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of LwF trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads,
                         ViT_task_specific_ln, do_LSA, do_SPT, network, use_param_split)
        
        # -- Set the temperature variable for the LWF Loss calculation during training -- #
        self.lwf_temperature = lwf_temperature

        # -- Add seed in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        if already_trained_on is not None:
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                # -- Add the lwf temperature and checkpoint settings -- #
                self.already_trained_on[str(self.fold)]['used_lwf_temperature'] = self.lwf_temperature
                self.already_trained_on[str(self.fold)]['freeze_run_finished'] = False
                self.already_trained_on[str(self.fold)]['ftasks_at_time_of_checkpoint'] = list()
                self.already_trained_on[str(self.fold)]['factive_task_at_time_of_checkpoint'] = None
            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['used_lwf_temperature', 'freeze_run_finished', 'ftasks_at_time_of_checkpoint',\
                        'factive_task_at_time_of_checkpoint', 'freezed_model_at']
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            # -- Update lwf_temperature in trained on file for restoring to be able to ensure that lwf_temperature can not be changed during training -- #
            self.already_trained_on[str(self.fold)]['used_lwf_temperature'] = self.lwf_temperature
            self.already_trained_on[str(self.fold)]['freeze_run_finished'] = False
            self.already_trained_on[str(self.fold)]['ftasks_at_time_of_checkpoint'] = list()
            self.already_trained_on[str(self.fold)]['factive_task_at_time_of_checkpoint'] = None
            self.already_trained_on[str(self.fold)]['freezed_model_at'] = None

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, already_trained_on, use_progress, identifier, extension,
                          lwf_temperature, tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, self.vit_type,
                          version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT)

        # -- Define a variable for the original and LwF loss -- #
        self.loss_orig = None
        self.LwFloss = None

        # -- Define a flag that is used in self.run_iteration(..) to identify how to perform the iteration -- #
        self.freeze_run = not self.already_trained_on[str(self.fold)]['freeze_run_finished']
        
        # -- Define the running idx to keep track at which batch we are -- #
        self.batch_idx = 0

        # -- Use this flag to indicate if perform validation is called, if so, always use the run_iteration -- #
        # -- of the grand parent class with the non LwF loss -- #
        self.do_val = False

        # -- Define an empty dict for the target_logits -- #
        self.target_logits = dict()

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite the initialize function so the correct Loss function for the LWF method can be set.
            NOTE: The previous tasks are already set in self.mh_network, so everything is in self.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)
 
        # -- Create a backup loss, so we can switch between original and LwF loss -- #
        self.loss_orig = copy.deepcopy(self.loss)

        # -- Define a loss_base for the LwFloss so it can be initialized properly -- #
        loss_base = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (LwF) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- NOTE: The predictions of the previous models need to be updated after each iteration -- #
        self.LwFloss = LwFloss(loss_base, self.ds_loss_weights, list(), list(), self.lwf_temperature)

    def reinitialize(self, task):
        r"""This function is used to reinitialize the Trainer when a new task is trained for the LwF Trainer.
            The most important thing here is that it sets the intermediate results accordingly in the loss.
            This should only be called when a new task is used --> by that time the new loss applies..
        """
        # -- Execute the super function -- #
        if len(self.mh_network.heads) == 1 and task in self.mh_network.heads:
            super().reinitialize(task, True)
        else:
            super().reinitialize(task, False)

            # -- Print Loss update -- #
            self.print_to_log_file("I am using LwF loss now")

    def run_training(self, task, output_folder):
        r"""Perform training using LwF Trainer. Simply executes training method of parent class
            while updating trained_on.pkl file. It is important to provide the right path, in which the results
            for the desired task should be stored.
            NOTE: If the task does not exist, a new head will be initialized using the init_head from the initialization
                  of the class. This new head is saved under task and will then be trained.
        """
        # -- Update the self.output_folder, otherwise the data will always be in the same folder for every task -- #
        # -- and everything will be overwritten over and over again -- #
        # -- Do this after reinitialization since the function might change the path -- #
        self.output_folder = join(self._build_output_path(output_folder, False), "fold_%s" % str(self.fold))

        # -- Make the directory so there will no problems when trying to save some files -- #
        maybe_mkdir_p(self.output_folder)

        # -- Create the dataloaders again, if they are still from the last task -- #
        if self.task != task:
            # -- Recreate the dataloaders for training and validation -- #
            self.reinitialize(task)
            # -- Now reset self.task to the current task -- #
            self.task = task

        # -- Add the current task to the self.already_trained_on dict in case of restoring -- #
        self.update_save_trained_on_json(task, False)   # Add task to start_training

        # -- Register the task if it does not exist in one of the heads -- #
        if task not in self.mh_network.heads:
            self.mh_network.add_new_task(task, use_init=not self.transfer_heads)
            # -- Reset the freeze_run flag since this task is a new one -- #
            self.freeze_run = True

        # -- Register the task in the ViT if task specific ViT is used -- #
        if self.use_vit and self.ViT_task_specific_ln:
            if task not in self.network.ViT.norm:
                self.network.ViT.register_new_task(task)
                # -- Update self.mh_network.model as well since we now have a set of new LNs -- #
                self.mh_network.model = copy.deepcopy(self.network)
            # -- Set the correct task_name for training -- #
            self.network.ViT.use_task(task)

        # -- Delete the trainer_model (used for restoring) -- #
        self.trainer_model = None

        # -- If only one head exists, this is the very first task to train, so train it conventionally --> nothing to do with LwF -- #
        if len(self.mh_network.heads) == 1:
            # -- Set everything correct in case of restoring -- #
            self.already_trained_on[str(self.fold)]['freeze_run_finished'] = True
            # -- Set the correct freeze_run and loss -- #
            self.freeze_run = False
            self.loss = self.loss_orig
            # -- Train -- #
            self.network = self.mh_network.assemble_model(task)
            # -- Set the correct task_name for training -- #
            if self.use_vit and self.ViT_task_specific_ln:
                self.network.ViT.use_task(task)
            ret = super().run_training(task, output_folder)
            self.freeze_run = True
        else:
            # -- If the model was initialized on task A but we are training on a later task, restore the model correctly -- #
            if self.already_trained_on[str(self.fold)]['freeze_run_finished']:
                self._load_model_and_update_target_logits()
                # -- Update the log -- #
                self.print_to_log_file("Start training on task {} using LwF method.".format(task))
                
            # -- If we have to continue with the freeze run, then do so -- #
            elif self.freeze_run:
                # -- Activate the model based on task freezing the body accordingly -- #
                self.network = self.mh_network.assemble_model(task, freeze_body=True)
                # -- Set the correct task_name for training -- #
                if self.use_vit and self.ViT_task_specific_ln:
                    self.network.ViT.use_task(task)
                # -- Train this task for n epochs without updating the body while using the standard loss function (not LwF one) -- #
                # -- Set the correct loss -- #
                self.loss = self.loss_orig
                # -- Update the log -- #
                self.print_to_log_file("Start training on task {} with freezed body.".format(task))
                # -- Train for n epochs -- #
                ret = super(nnUNetTrainerV2, self).run_training()
                # -- Update the log -- #
                self.print_to_log_file("Finished training on task {} with freezed body and start training for every head using LwF loss.".format(task))
                
                # -- Clean up -- #
                # -- Before returning, reset the self.epoch variable, otherwise the following task will only be trained for the last epoch -- #
                self.epoch = 0
                # -- Empty the lists that are tracking losses etc., since this will lead to conflicts in additional tasks durig plotting -- #
                # -- Do not worry about it, the right data is stored during checkpoints and will be restored as well, but after -- #
                # -- a task is finished and before the next one starts, the data needs to be emptied otherwise its added to the lists. -- #
                self.all_tr_losses = []
                self.all_val_losses = []
                self.all_val_losses_tr_mode = []
                self.all_val_eval_metrics = []
                self.validation_results = dict()
            
                # -- Set freeze_run to false, so we can train the whole body again -- #
                self.freeze_run = False

                # -- Store this mh_network for restoring purposes since the generators are not the same once the algorithm -- #
                # -- starts from the beginning --> so do not store the target_logits but the network and during restoring -- #
                # -- calculate the target_logits again -- #
                # -- Set the network to the full MultiHead_Module network to save everything in the class not only the current model -- #
                self.print_to_log_file("Store the finished model for restoring purposes as freezed_model since the other model will be overwritten during the following LwF training")
                self.network = self.mh_network
                # -- Set the flag to True -- #
                self.already_trained_on[str(self.fold)]['freeze_run_finished'] = True
                # -- Add the current head keys for restoring (is in correct order due to OrderedDict type of heads) -- #
                self.already_trained_on[str(self.fold)]['ftasks_at_time_of_checkpoint'] = list(self.mh_network.heads.keys())
                # -- Add the current active task for restoring -- #
                self.already_trained_on[str(self.fold)]['factive_task_at_time_of_checkpoint'] = self.mh_network.active_task
                # -- Save the updated dictionary as a json file -- #
                save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.pkl'))
                # -- Update self.init_tasks so the storing works properly -- #
                self.update_init_args()
                # -- Use grand parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
                super(nnUNetTrainerV2, self).save_checkpoint(join(self.output_folder, "model_freezed.model"), False)
                # -- Add the current path for restoring purposes -- #
                self.already_trained_on[str(self.fold)]['freezed_model_at'] = join(self.output_folder, "model_freezed.model")
                # -- Reset network to the assembled model to continue training however unfreeze the body now -- #
                self.network = self.mh_network.assemble_model(task, freeze_body=False)
                # -- Set the correct task_name for training -- #
                if self.use_vit and self.ViT_task_specific_ln:
                    self.network.ViT.use_task(task)
                # -- Update the log -- #
                self.print_to_log_file("Calculate the target_logits before training..")
                start_time = time()
                # -- Calculate the target_logits -- #
                self.target_logits = calculate_target_logits(self.mh_network, self.tr_gen, self.num_batches_per_epoch, self.fp16)
                # -- Update the log -- #
                self.print_to_log_file("Calculation of the target_logits took %.2f seconds" % (time() - start_time))

            # -- Put model into train mode -- #
            self.network.train()

            # -- Train for n epochs using every head and the LwF loss function in one iteration -- #
            # -- Set the correct loss -- #
            self.loss = self.LwFloss

            # -- Train for n epochs -- #
            ret = super().run_training(task, output_folder)

        # -- Reset everything for next task since the information is no longer necessary -- #
        # -- Reset the val_metrics_exist flag since the training is finished and restoring will fail otherwise -- #
        self.already_trained_on[str(self.fold)]['val_metrics_should_exist'] = False
        # -- Set the flag to False -- #
        self.already_trained_on[str(self.fold)]['freeze_run_finished'] = False
        # -- Add the current head keys for restoring (is in correct order due to OrderedDict type of heads) -- #
        self.already_trained_on[str(self.fold)]['ftasks_at_time_of_checkpoint'] = list()
        # -- Add the current active task for restoring -- #
        self.already_trained_on[str(self.fold)]['factive_task_at_time_of_checkpoint'] = None
        # -- Add the current path of the freezed_model -- #
        self.already_trained_on[str(self.fold)]['freezed_model_at'] = None

        # -- Add task to finished_training -- #
        self.update_save_trained_on_json(task, True)
        # -- Resave the final model pkl file so the already trained on is updated there as well -- #
        self.save_init_args(join(self.output_folder, "model_final_checkpoint.model"))

        # -- When model trained on second task and the self.new_trainer is still not updated, then update it -- #
        if self.new_trainer and len(self.already_trained_on) > 1:
            self.new_trainer = False

        # -- Before returning, reset the self.epoch variable, otherwise the following task will only be trained for the last epoch -- #
        self.epoch = 0

        # -- Empty the lists that are tracking losses etc., since this will lead to conflicts in additional tasks durig plotting -- #
        # -- Do not worry about it, the right data is stored during checkpoints and will be restored as well, but after -- #
        # -- a task is finished and before the next one starts, the data needs to be emptied otherwise its added to the lists. -- #
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []
        self.validation_results = dict()

        return ret  # Finished with training for the specific task

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, *args, **kwargs):
        r"""This function needs to be changed for the LWF method, since all previously trained models will be used
            to predict the same batch as the current model we train on. These results go then into the Loss function
            to compute the Loss as proposed in the paper.
        """
        if self.freeze_run or self.do_val:
            # -- Run iteration as usual and return the loss (use nnUNetTrainerV2) -- #
            ret = super(nnUNetTrainerV2, self).run_iteration(data_generator, do_backprop, run_online_evaluation)
            # -- Update the Multi Head Network after one iteration since backprop is performed (during training) -- #
            if do_backprop: # Check this, since after every training a validation is performed as well
                self.mh_network.update_after_iteration(update_body=False)   # Remeber we are in freeze_run, ie. body should not be updated
        else:
            # -- Check if we are performing an iteration for validation purposes only, then we do not have to do all the following -- #
            if len(self.mh_network.heads) > 1: # --> only do this if we want backpropagation, ie. during training and we have at least one task
                # Run per head and use LWF loss while updating the corresponding logits!
                for task in list(self.mh_network.heads.keys()):
                    # -- Build the current network -- #
                    self.network = self.mh_network.assemble_model(task)
                    # -- Set the correct task_name for training -- #
                    if self.use_vit and self.ViT_task_specific_ln:
                        self.network.ViT.use_task(task)
                    # -- Remove the softmax layer at the end by replacing the corresponding element with an identity function -- #
                    self.network.inference_apply_nonlin = lambda x: x
                    # -- Set network to eval -- #
                    self.network.eval()
                    # -- Create a copy from the data_generator so the data_generator won't be touched. -- #
                    # -- This way, each previous task uses the same batch, as well as the model that will train -- #
                    # -- using the data_generator and thus same batch. -- #
                    data = tee(data_generator, 1)[0]
                    # -- Extract the current batch from data -- #
                    x = next(data)
                    x = maybe_to_torch(x['data'])
                    if torch.cuda.is_available():
                        x = to_cuda(x)

                    # -- Make predictions using the loaded model and data -- #
                    if self.fp16:
                        with autocast():
                            output = self.network(x)[0]
                    else:
                        output = self.network(x)[0]
                        
                    # -- Do detach the output so the loss has no effect on the old network during backward step -- #
                    pred_logits = output.detach().cpu()

                    del x, output
                    # -- Update the LwF loss -- #
                    self.loss.update_logits(pred_logits, self.target_logits[task][self.batch_idx % 250])  # Use modulo since self.batch_idx is a running number
                    # -- Add the softmax layer again by replacing the corresponding element with softmax_helper -- #
                    self.network.inference_apply_nonlin = softmax_helper
                    # -- Put model into train mode -- #
                    self.network.train()
                    # -- Run iteration as usual and return the loss -- #
                    ret = super().run_iteration(tee(data_generator, 1)[0], do_backprop, run_online_evaluation)

                # -- Get next element in the generator since we always use a deepcopy of the generator -- #
                _ = next(data_generator)
                # -- Add one to the running index so we know at which batch we currently are -- #
                self.batch_idx += 1
            else:
                # -- Run iteration as usual and return the loss --> This is used when having only one task -- #
                ret = super().run_iteration(data_generator, do_backprop, run_online_evaluation)

        # -- Return the result -- #
        return ret

    def on_epoch_end(self):
        """Overwrite this function for the LwF trainer.
        """
        if self.freeze_run:
            # -- Perform everything the grandparent class makes --> no validation per task is performed here -- #
            res = super(nnUNetTrainerV2, self).on_epoch_end()
        else:
            # -- Create copies before doing validation since we do not want to calculate the logits again -- #
            # -- Generators might be assembled in a random way -- #
            dl_tr_cpy = tee(self.dl_tr, 1)[0]
            dl_val_cpy = tee(self.dl_val, 1)[0]
            tr_gen_cpy = tee(self.tr_gen, 1)[0]
            val_gen_cpy = tee(self.val_gen, 1)[0]

            # -- Perform everything the parent class makes -- #
            # -- Validation per task will be performed, so the tr_gen are created per task, but might change -- #
            # -- if random is used within the generator creation process --> so create target_logits again -- #
            self.do_val = True
            b_loss = self.loss
            self.loss = self.loss_orig
            res = super().on_epoch_end()
            # -- Restore the way it was before -- #
            self.do_val = False
            self.loss = b_loss
            del b_loss

            # -- Restore the generators again -- #
            self.dl_tr = dl_tr_cpy
            self.dl_val = dl_val_cpy
            self.tr_gen = tr_gen_cpy
            self.val_gen = val_gen_cpy
            del dl_tr_cpy, dl_val_cpy, tr_gen_cpy, val_gen_cpy
            
            # -- Reset the network to the current task -- #
            self.network = self.mh_network.assemble_model(self.task)
            # -- Set the correct task_name for training -- #
            if self.use_vit and self.ViT_task_specific_ln:
                self.network.ViT.use_task(self.task)
            # -- Put model into train mode -- #
            self.network.train()

        # -- Return the result from the parent class -- #
        return res
        
    def save_checkpoint(self, fname, save_optimizer=True):
        r"""Overwrite the parent class, since we want to store the original network as well along with the current running
            model.
        """
        if self.freeze_run:
            # -- Use grand parent class to save checkpoint -- #
            super(nnUNetTrainerV2, self).save_checkpoint(fname, save_optimizer)
        else:
            # -- Use parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
            super().save_checkpoint(fname, save_optimizer)

    def _load_model_and_update_target_logits(self):
        r"""This function is used to restore the MultiHead Module with the freezed body to calculate the
            target_logits.
        """
        # -- Load the MultiHead Network -- #
        mh_network_cpy = copy.deepcopy(self.mh_network)
        mh_network_cpy.add_n_tasks_and_activate(self.already_trained_on[str(self.fold)]['ftasks_at_time_of_checkpoint'],
                                                self.already_trained_on[str(self.fold)]['factive_task_at_time_of_checkpoint'])
        # -- Set the network to the full MultiHead_Module network to restore everything -- #
        self.network = mh_network_cpy
        # -- Use parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
        saved_model = torch.load(self.already_trained_on[str(self.fold)]['freezed_model_at'], map_location=torch.device('cpu'))
        super(nnUNetTrainerV2, self).load_checkpoint_ram(saved_model, False)
        # -- Update the log -- #
        self.print_to_log_file("Calculate the target_logits..")
        start_time = time()
        # -- Calculate the target_logits -- #
        self.target_logits = calculate_target_logits(mh_network_cpy, self.tr_gen, self.num_batches_per_epoch, self.fp16)
        self.network = self.mh_network.model
        del mh_network_cpy
        # -- Update the log -- #
        self.print_to_log_file("Calculation of the target_logits took %.2f seconds" % (time() - start_time))