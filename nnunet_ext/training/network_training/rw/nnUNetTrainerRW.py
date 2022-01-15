#########################################################################################################
#----------------------This class represents the nnUNet trainer for RW training.-----------------------#
#########################################################################################################

# -- This implementation represents the method proposed in the paper https://arxiv.org/pdf/1801.10112.pdf -- #
# -- Original implementation in tensorflow can be found here: https://github.com/facebookresearch/agem -- #
# -- Used implementation for PyTorch from here: https://github.com/fcdl94/MiB/blob/master/utils/regularizer.py -- #

import torch
from nnunet.utilities.to_torch import to_cuda
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossRW as RWLoss
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

EPSILON = 1e-8
# EPSILON = 1e-8    # From MiB re-implementation: https://github.com/fcdl94/MiB/blob/master/utils/regularizer.py#L4
# EPSILON = 1e-32   # From original implementation: https://github.com/facebookresearch/agem/blob/main/model/model.py#L20

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {'rw_alpha': float, 'rw_lambda': float, 'fisher_update_after': int}

class nnUNetTrainerRW(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='rw', fisher_update_after=10, rw_alpha=0.9, rw_lambda=0.4, tasks_list_with_char=None,
                 mixed_precision=True, save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False,
                 transfer_heads=True, use_param_split=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False):
        r"""Constructor of RW trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension,
                         tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads,
                         use_param_split, ViT_task_specific_ln, do_LSA, do_SPT)
        
        # -- Set the alpha for moving average fisher calculation -- #
        self.alpha = rw_alpha
        self.rw_lambda = rw_lambda
        self.fisher_update_after = fisher_update_after
        assert self.alpha > 0 and self.alpha <= 1, "rw_alpha should be between 0 and 1: [0, 1]."

        # -- Add flags in trained on file for restoring to be able to ensure that seed can not be changed during training -- #
        if already_trained_on is not None:
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                # -- Add EWC specific entries to already_trained_on -- #
                self.already_trained_on[str(self.fold)]['used_alpha'] = self.alpha
                self.already_trained_on[str(self.fold)]['used_rw_lambda'] = self.rw_lambda
                self.already_trained_on[str(self.fold)]['update_fisher_after'] = self.fisher_update_after
                self.already_trained_on[str(self.fold)]['fisher_at'] = None
                self.already_trained_on[str(self.fold)]['params_at'] = None
                self.already_trained_on[str(self.fold)]['scores_at'] = None
            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['used_alpha', 'used_rw_lambda', 'update_fisher_after', 'fisher_at', 'params_at', 'scores_at']
                # -- Check that everything is provided as expected -- #
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            # -- Update rw_lambda in trained on file fore restoring to be able to ensure that rw_lambda can not be changed during training -- #
            self.already_trained_on[str(self.fold)]['used_alpha'] = self.alpha
            self.already_trained_on[str(self.fold)]['used_rw_lambda'] = self.rw_lambda
            self.already_trained_on[str(self.fold)]['update_fisher_after'] = self.fisher_update_after
            self.already_trained_on[str(self.fold)]['fisher_at'] = None
            self.already_trained_on[str(self.fold)]['params_at'] = None
            self.already_trained_on[str(self.fold)]['scores_at'] = None

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, already_trained_on, use_progress, identifier, extension, fisher_update_after,
                          rw_alpha, rw_lambda, tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, self.vit_type,
                          version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT)
        
        # -- Initialize dicts that hold the fisher and param values -- #
        if self.already_trained_on[str(self.fold)]['fisher_at'] is None\
        or self.already_trained_on[str(self.fold)]['params_at'] is None\
        or self.already_trained_on[str(self.fold)]['scores_at'] is None:
            self.fisher, self.params, self.scores = dict(), dict(), dict()
        else:
            self.fisher = load_pickle(self.already_trained_on[str(self.fold)]['fisher_at'])
            self.params = load_pickle(self.already_trained_on[str(self.fold)]['params_at'])
            self.scores = load_pickle(self.already_trained_on[str(self.fold)]['scores_at'])

        # -- Define the path where the fisher and param values should be stored/restored -- #
        self.rw_data_path = join(self.trained_on_path, 'rw_data')
        self.prev_param, self.prev_fisher, self.count = None, None, 0

    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""Overwrite the initialize function so the correct Loss function for the RW method can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)
        
        # -- Asert if self.fisher_update_after > than nr of epochs to train on -- #
        assert self.fisher_update_after < self.num_batches_per_epoch,\
            "How should the fisher values and importance scores be calculated if update_after is greater than the number of iterations per epochs.."
        
        # -- If this trainer has already trained on other tasks, then extract the fisher and params -- #
        if prev_trainer_path is not None and self.already_trained_on[str(self.fold)]['fisher_at'] is not None\
                                         and self.already_trained_on[str(self.fold)]['params_at'] is not None\
                                         and self.already_trained_on[str(self.fold)]['scores_at'] is not None:
            self.fisher = load_pickle(self.already_trained_on[str(self.fold)]['fisher_at'])
            self.params = load_pickle(self.already_trained_on[str(self.fold)]['params_at'])
            self.scores = load_pickle(self.already_trained_on[str(self.fold)]['scores_at'])

        # -- Reset self.loss from MultipleOutputLoss2 to DC_and_CE_loss so the RW Loss can be initialized properly -- #
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (RW) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters during each iteration -- #
        self.loss = RWLoss(self.loss, self.ds_loss_weights,
                           self.rw_lambda,
                           self.fisher, self.params, self.scores,
                           self.network.named_parameters())

    def reinitialize(self, task):
        r"""This function is used to reinitialize the Trainer when a new task is trained for the RW Trainer.
            The most important thing here is that it sets the intermediate results accordingly in the loss.
            This should only be called when a new task is used --> by that time the new loss applies..
        """
        # -- Execute the super function -- # 
        super().reinitialize(task, False)

        # -- Print Loss update -- #
        self.print_to_log_file("I am using RW loss now")
        
        # -- Put data on GPU since the data is moved to CPU before it is stored -- #
        for task in self.fisher.keys():
            for key in self.fisher[task].keys():
                to_cuda(self.fisher[task][key])
        for task in self.params.keys():
            for key in self.params[task].keys():
                to_cuda(self.params[task][key])
        for task in self.scores.keys():
            for key in self.scores[task].keys():
                to_cuda(self.scores[task][key])

        # -- Update the fisher, param and score values in the loss function -- #
        self.loss.update_rw_params(self.fisher, self.params, self.scores)

    def run_training(self, task, output_folder):
        r"""Perform training using RW trainer. Simply executes training method of parent class (nnUNetTrainerMultiHead)
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

        # -- Define the fisher and params before the training -- #
        self.params[task] = dict()

        # -- Set all fisher values to zero --> default to simply add the values easily -- #
        self.fisher[task] = {n: torch.zeros_like(p, device='cuda:0', requires_grad=False)
                      for n, p in self.network.named_parameters() if p.requires_grad}

        self.scores[task] = {n: torch.zeros_like(p, device='cuda:0', requires_grad=False)
                      for n, p in self.network.named_parameters() if p.requires_grad}

        # -- Execute the training for the desired epochs -- #
        ret = super().run_training(task, output_folder)  # Execute training from parent class --> already_trained_on will be updated there

        # -- Reset score param variable -- #
        self.prev_param, self.count = None, 0

        # -- Extract the current params as well -- #
        self._extract_params()

        # -- Normalize the fisher values to be in range 0 to 1 -- #
        values = [torch.max(val) for val in self.scores[self.task].values()] # --> only for the current task of course
        minim, maxim = min(values), max(values)
        for k, v in self.fisher[self.task].items():
            self.fisher[self.task][k] = (v - minim) / (maxim - minim + EPSILON)

        # -- Normalize the score values to be in range 0 to 1 -- #
        values = [torch.max(val) for val in self.scores[self.task].values()] # --> only for the current task of course
        minim, maxim = min(values), max(values)
        if len(self.already_trained_on[str(self.fold)]['finished_training_on']) == 1:    # At this stage we have the first scores
            for k, v in self.scores[self.task].items():
                # -- Normalize and scale the score so that division does not have an effect -- #
                curr_score_norm = (v - minim) / (maxim - minim + EPSILON)
                self.scores[self.task][k] = 2 * curr_score_norm
        elif len(self.already_trained_on[str(self.fold)]['finished_training_on']) > 2:   # Scores are now computed differently since they are averaged
            prev_scores = {k: v.clone() for k, v in self.scores[self.already_trained_on[str(self.fold)]['finished_training_on'][-1]].items()}
            for k, v in self.scores[self.task].items():
                # -- Normalize the score -- #
                curr_score_norm = (v - minim) / (maxim - minim + EPSILON)
                # -- Average the score to alleviate rigidity due to the accumulating sum of the scores otherwise -- #
                self.scores[self.task][k] = 0.5 * (prev_scores[k] + curr_score_norm)

        # -- Store the fisher and param values -- #
        self.save_f_p_s_values()

        return ret  # Finished with training for the specific task

    def on_epoch_end(self):
        """Use this function to save the fisher values after every nth epoch.
        """
        # -- Perform everything the parent class makes -- #
        res = super().on_epoch_end()
       
        # -- Store the fisher values in case training fails so we don't loose the values -- #
        self.save_f_p_s_values()

        # -- Return the result -- #
        return res

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""Run the iteration and then update the fisher and score values.
        """
        # -- Run the iteration but do not detach the loss -- #
        loss = super().run_iteration(data_generator, do_backprop, run_online_evaluation, False, no_loss)
        # -- Update the values -- #
        self._update_f_s_values()
        # -- Now detach and return the loss if desired -- #
        if detach and loss is not None: # --> Loss is None when using Evaluator, so we need to prevent this or an error gets thrown
            loss = loss.detach().cpu().numpy()
        return loss

    def _update_f_s_values(self):
        r"""This function updates after every desired epoch the fisher and score values as proposed in the paper.
            Call this function before the loss is detached, otherwise the gradients are 0.
        """
        # -- Only do this every fisher_update_after epoch -- #
        if self.count % self.fisher_update_after == 0:
            # -- Update the scores -- #
            if self.prev_param is not None:
                # -- Update the importance score using distance in Riemannian Manifold -- #
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        # -- Get parameter difference from old param and current param t -- #
                        delta = param.grad.detach() * (self.prev_param[name].to(param.device) - param.detach())
                        # -- Calculate score denominator -- #
                        den = 0.5 * self.fisher[self.task][name] * (param.detach() - self.prev_param[name].to(param.device)).pow(2) + EPSILON
                        # -- Score: delat(L) / 0.5*F_t*delta(param)^2 --> only positive or zero values -- #
                        scores = (delta / den)
                        scores[scores < 0] = 0  # Ensure no negative values
                        # -- Update the scores -- #
                        self.scores[self.task][name] += scores

            # -- Update the prev params -- #
            self.prev_param = {k: torch.clone(v).detach().cpu() for k, v in self.network.named_parameters() if v.grad is not None}

            # -- Update the fisher values -- #
            for name, param in self.network.named_parameters():
                # -- F_t = alpha * F_t + (1-alpha) * F_t-1
                if param.grad is not None:
                    f_t = param.grad.data.clone().pow(2)
                    f_to = self.fisher[self.task][name] if self.fisher[self.task][name] is not None else torch.tensor([0], device='cuda:0')
                    self.fisher[self.task][name] = (self.alpha * f_t) + ((1 - self.alpha) * f_to)
        
        # -- Increase our count variable -- #
        self.count += 1

    def save_f_p_s_values(self):
        r"""This function stores the fisher, param and score values.
        """
        # -- Put data from GPU to CPU before storing them in files -- #
        for task in self.fisher.keys():
            for key in self.fisher[task].keys():
                self.fisher[task][key].cpu()
        for task in self.params.keys():
            for key in self.params[task].keys():
                self.params[task][key].cpu()
        for task in self.scores.keys():
            for key in self.scores[task].keys():
                self.scores[task][key].cpu()

        # -- Dump both dicts as pkl files -- #
        maybe_mkdir_p(self.rw_data_path)
        write_pickle(self.fisher, join(self.rw_data_path, 'fisher_values.pkl'))
        write_pickle(self.params, join(self.rw_data_path, 'param_values.pkl'))
        write_pickle(self.scores, join(self.rw_data_path, 'score_values.pkl'))

        if self.already_trained_on[str(self.fold)]['fisher_at'] is None\
        or self.already_trained_on[str(self.fold)]['params_at'] is None\
        or self.already_trained_on[str(self.fold)]['scores_at'] is None:
            # -- Update the already_trained_on file that the values exist if necessary -- #
            self.already_trained_on[str(self.fold)]['fisher_at'] = join(self.rw_data_path, 'fisher_values.pkl')
            self.already_trained_on[str(self.fold)]['params_at'] = join(self.rw_data_path, 'param_values.pkl')
            self.already_trained_on[str(self.fold)]['scores_at'] = join(self.rw_data_path, 'score_values.pkl')
            
            # -- Save the updated dictionary as a json file -- #
            save_json(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.json'))
            # -- Update self.init_tasks so the storing works properly -- #
            self.update_init_args()
            # -- Resave the final model pkl file so the already trained on is updated there as well -- #
            self.save_init_args(join(self.output_folder, "model_final_checkpoint.model"))

        # -- Put data from CPU back to GPU -- #
        for task in self.fisher.keys():
            for key in self.fisher[task].keys():
                if key == self.task:
                    to_cuda(self.fisher[task][key])
        for task in self.params.keys():
            for key in self.params[task].keys():
                if key == self.task:
                    to_cuda(self.params[task][key])
        for task in self.scores.keys():
            for key in self.scores[task].keys():
                if key == self.task:
                    to_cuda(self.scores[task][key])

    def _extract_params(self):
        r"""This function is used to extract the parameters after the finished training.
        """
        # -- Set params in current fold from last iteration --> final model parameters -- #
        for name, param in self.network.named_parameters():
            # -- Update the params dict -- #
            self.params[self.task][name] = param.data.clone()
