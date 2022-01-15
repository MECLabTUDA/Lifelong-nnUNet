#########################################################################################################
#----------This class represents a Parameter Searcher that can be used to find suitable parameter-------#
#-----------values to train a network with to achieve good results based on the tested params.----------#
#########################################################################################################

import pandas as pd
from multiprocessing import Pool
from collections import OrderedDict
import random, itertools, tqdm, glob
from nnunet_ext.utilities.helpful_functions import *
from nnunet_ext.experiment.experiment import Experiment
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.paths import param_search_output_dir, default_plans_identifier

class ParamSearcher():
    r"""Class that can be used to perform a parameter search using a specific extension that uses Hyperparameters.
    """
    def __init__(self, network, network_trainer, tasks_list_with_char, version=1, vit_type='base', eval_mode_for_lns='last_lns', fold=0,
                 plans_identifier=default_plans_identifier, mixed_precision=True, extension='multihead', val_folder='validation_raw',
                 split_at=None, transfer_heads=False, use_vit=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, do_pod=False,
                 search_mode='grid', grid_picks=None, rand_range=None, rand_pick=None, rand_seed=None, always_use_last_head=True, npz=False,
                 perform_validation=False, continue_training=False, unpack_data=True, deterministic=False, save_interval=5, num_epochs=100,
                 fp16=True, find_lr=False, valbest=False, disable_postprocessing_on_folds=False, split_gpu=False, fixate_params=None,
                 val_disable_overwrite=True, disable_next_stage_pred=False, run_in_parallel=False):
        r"""Constructor for parameter searcher. Use the constructor of an Experiment since they are very similar.
        """
        # -- Ensure everything is correct transmitted -- #
        if search_mode == 'random':
            assert rand_range is not None and rand_pick is not None,\
                "The user selected a random search method but does not provide the value ranges (and/or) nr of allowed value picks for the search.."
            # -- Set the seed -- #
            if rand_seed is not None:
                random.seed(rand_seed)
            # -- Define the self param values -- #
            self.params_to_tune = list(rand_range.keys())
        else:
            assert grid_picks is not None, "The user selected a grid search method but does not provide the value list for the search.."
            # -- Define the self param values -- #
            self.params_to_tune = list(grid_picks.keys())

        # -- Set all parameter search relevant attributes -- #
        self.summary = None
        self.main_sum = None
        self.rand_pick = rand_pick
        self.rand_seed = rand_seed
        self.rand_range = rand_range
        self.grid_picks = grid_picks
        self.search_mode = search_mode
        self.fixate_params = fixate_params if fixate_params is not None else dict()

        # -- Set tasks_joined_name for validation dataset building -- #
        self.tasks_joined_name = join_texts_with_char(tasks_list_with_char[0], tasks_list_with_char[1])
        # -- Everything is fit for now, start building the output_folder, ie. the root one -- #
        self.output_base = os.path.join(param_search_output_dir, network, self.tasks_joined_name)
        # -- Within this base folder, we have a folder for the experiments and one for the evaluation results -- #
        self.output_exp = os.path.join(self.output_base, 'experiments')
        self.output_eval = os.path.join(self.output_base, 'evaluation')

        # -- Define the experiment arguments -- #
        self.exp_args = {'network': network, 'network_trainer': network_trainer, 'tasks_list_with_char': tasks_list_with_char,
                         'version': version, 'vit_type': vit_type, 'eval_mode_for_lns': eval_mode_for_lns, 'fold': fold, 'plans_identifier': plans_identifier,
                         'mixed_precision': mixed_precision, 'extension': extension, 'save_interval': save_interval, 'val_folder': val_folder,
                         'split_at': split_at, 'transfer_heads': transfer_heads, 'use_vit': use_vit, 'ViT_task_specific_ln': ViT_task_specific_ln,
                         'do_LSA': do_LSA, 'do_SPT': do_SPT, 'do_pod': do_pod, 'always_use_last_head': always_use_last_head, 'npz': npz, 'use_param_split': True,
                         'output_exp': self.output_exp, 'output_eval': self.output_eval, 'perform_validation': perform_validation, 'params_to_tune': self.params_to_tune,
                         'continue_training': continue_training, 'unpack_data': unpack_data, 'deterministic': deterministic, 'save_interval': save_interval,
                         'num_epochs': num_epochs, 'fp16': fp16, 'find_lr': find_lr, 'valbest': valbest, 'disable_postprocessing_on_folds': disable_postprocessing_on_folds,
                         'split_gpu': split_gpu, 'val_disable_overwrite': val_disable_overwrite, 'disable_next_stage_pred': disable_next_stage_pred}

        # -- Do an initialization like the one of an Experiment -- #
        Experiment.__init__(self, **self.exp_args)
        # Experiment.__init__(self, network, network_trainer, tasks_list_with_char,
        # version, vit_type, eval_mode_for_lns, fold, plans_identifier, mixed_precision, extension, save_interval,
        # val_folder, split_at, transfer_heads, use_vit, ViT_task_specific_ln, do_LSA, do_SPT, do_pod, always_use_last_head,
        # npz, self.output_base, self.output_exp, self.output_eval, perform_validation, continue_training, unpack_data,
        # deterministic, save_interval, num_epochs, fp16, find_lr, valbest, disable_postprocessing_on_folds, split_gpu,
        # val_disable_overwrite, disable_next_stage_pred)

        # -- Remove the arguments that are not relevant for the parameter search class -- #
        del self.trainer_map, self.hyperparams, self.evaluator, self.basic_eval_args, self.use_param_split

        # -- Define a empty dictionary that is used for backup purposes -- #
        self.backup_information = dict()

        self.run_in_parallel = run_in_parallel
        
        # -- Load the Backup file if it exists -- #
        backup_file = os.path.join(self.output_base, 'backup.pkl')
        if os.path.isfile(backup_file):
            self.backup_information = load_pickle(backup_file)
        else:
            # -- Assert if continue training is set, since no backup is stored -- #
            assert not self.continue_training, "There is no backup file, yet the user wants to continue with the parameter search.. Remove the -c flag."

        if self.continue_training:
            # TODO: If -c load the existing summary --> Check if the matching case works!!
            self.summary = glob.glob(os.path.join(self.output_base, 'parameter_search_summary*.txt'))
            assert len(self.summary) == 1, "There are no or more than one summary file, how can this be when using -c?"
            self.summary = self.summary[0]

            # TODO: When doing -c try to load main_sum using: self.main_sum = pd.read_csv(os.path.join(self.output_base, 'parameter_search_val_summary.csv'))
            # Assess if it should exist based on the backup file json
        
        # -- Create a summary file for this parameter search --> self.summary might be None, so provide all arguments -- #
        else:
            self.summary = print_to_log_file(self.summary, self.output_base, 'parameter_search_summary', "Starting with the parameter searching.. \n \n")
            self.summary = print_to_log_file(self.summary, "NOTE: The parameter search method uses its own splits.\
                                                            Those splits are generated from the original split file by splitting the training split again\
                                                            randomly into an 80:20 split. This data is then used for training and validation,\
                                                            whereas the validation data from the original split will never be used during parameter searching.")
        if not self.continue_trainnig:
            self.summary = print_to_log_file(self.summary, "The user wants to use the {} search method.".format(self.search_mode))
            self.summary = print_to_log_file(self.summary, "Building all possible settings for the experiments.")
            
        # -- Create the parameter combinations for the parameter search -- #
        self.experiments = OrderedDict()
        if self.search_mode == 'grid':
            # -- Create permutation between all possibilities in grid_picks -- #
            if len(self.params_to_tune) == 1:
                # -- No permutation possible --> simply add the experiments -- #
                hyperparam = list(self.params_to_tune)[0]
                for idx, value in enumerate(*self.grid_picks.values()):
                    self.experiments['exp_{}'.format(idx)] = {hyperparam: value, **self.fixate_params}
            else:   # -- Do permutation first -- #
                combinations = itertools.product(*self.grid_picks.values())
                for idx, combi in enumerate(combinations):
                    # -- Build the experiments dictionary -- #
                    experiment = dict()
                    for i, k in enumerate(self.params_to_tune):
                        experiment[k] = combi[i]
                    self.experiments['exp_{}'.format(idx)] = {**experiment, **fixate_params}
        else:
            # -- Do rand_pick times a random pick for every hyperparameter based on random_range attributes -- #
            picks = dict()
            for idx, param in enumerate(self.params_to_tune):
                picked = []
                for _ in range(self.rand_pick):
                    picked.append(round(random.uniform(*self.rand_range[param]), 3))    # Make a random pick and round to 3 decimals
                picks[param] = picked
            # -- Do permutation first -- #
            combinations = itertools.product(*picks.values())
            for idx, combi in enumerate(combinations):
                # -- Build the experiments dictionary -- #
                experiment = dict()
                for i, k in enumerate(self.params_to_tune):
                    experiment[k] = combi[i]
                self.experiments['exp_{}'.format(idx)] = {**experiment, **fixate_params}

        # -- Assert if there are no experiments to do -- #
        assert len(self.experiments.keys()) != 0, "Unfortunately, there are no experiments based on the users arguments.."
        
        # -- Paste all experiments into the log file -- #
        if not self.continue_trainnig:
            # -- Build the string summarizing all experiments with the corresponding settings -- #
            exp_sum = ''
            for k, v in self.experiments.items():
                exp_sum += str(k) + ' := '
                for k_, v_ in v.items():
                    exp_sum += str(k_) + ':' + str(v_) + ', '
                exp_sum = exp_sum[:-2] + '\n'
            self.summary = print_to_log_file(self.summary, "There are {} experiments: \n {}".format(len(self.experiments.keys()), exp_sum))
         
        # -- Add the experiments to the backup file -- #
        # TODO
        
        # TODO: Everything around backup for restoring (high level not complex to know which top run again and which not) ...
        #       Modify parallelizing thing, now only consider the ones that are not finished and add -c flag to them and the ones not even started...

    def start_searching(self):
        r"""This function performs the actual parameter search. If more than one GPU is provided and the flag
            in_parallel is set, this function will run the experiments in parallel. If -c has been set, then
            it will be continued with training form where it stopped.
        """
        # -- Make those directories if they don't exist yet -- #
        maybe_mkdir_p(self.output_exp)
        maybe_mkdir_p(self.output_eval)

        # -- Initialize the experiment object -- #
        exp_ = Experiment(**self.exp_args)
        
        # -- Create a single experiment if this is not done in paralllel -- #
        if not self.run_in_parallel:
            # -- Run the experiment for all provided settings -- #
            for exp, sets in self.experiments.items():
                self.summary = print_to_log_file(self.summary, "Start running the experiment {} using trainer {}.".format(exp, self.network_trainer))
                # -- Run the experiment -- #
                available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
                e_id, eval_res_pth = exp_.run_experiment(exp_id = exp, settings = sets, settings_in_folder_name = True, gpu_ids = available_gpus)
                self.summary = print_to_log_file(self.summary, 'Finished the experiment.')
                
                # -- Join and save the results -- #
                self._join_save_results(eval_res_pth, e_id)

                # -- Add finished experiment to backup file -- #
                # TODO
        else:
            # -- Extract the list of GPUs -- #
            gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
            available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
            # -- Reduce the nr of GPUs is split_gpu has been set, then every experiment needs two GPUs -- #
            if self.split_gpu:
                gpus //= 2

            # -- Do in parallel -- #
            pool = Pool(process = gpus) # N workers based on the amount of GPUs per process
            # -- Build the runs so we can specifically set the GPUs and ensure parallel -- #
            # -- processes don't use the same GPUs -- #
            runs = list()   # --> Put settings in there as lists, ie. nr gpus experiments per list in this list

            # -- Loop through the number of GPUs and build the experiments -- #
            for run_index in range(0, len(self.experiments.keys()), gpus):
                # -- Loop through the GPUs and assign them to specific experiments in this run -- #
                args = list()
                if self.split_gpu:
                    idx = 0
                    for gpu_ids in zip(*[iter(available_gpus)]*2):   # --> fetch GPU ID tuples
                        exp_id = 'exp_{}'.format(run_index + idx)
                        exp_args = self.experiments[exp_id]
                        kwargs = {'exp_id': exp_id, 'settings': exp_args, 'settings_in_folder_name': True, 'gpu_ids': list(gpu_ids)}
                        args.append(kwargs)
                        idx += 1    # Use this because we go in two steps instead of one
                        # -- If the experiments and step size does not match perfectly there might be an overflow, so catch it -- #
                        if (run_index + idx) > len(self.experiments.keys()):
                            break
                else:
                    for gpu_index, gpu_id in enumerate(available_gpus):
                        exp_id = 'exp_{}'.format(run_index + gpu_index)
                        exp_args = self.experiments[exp_id]
                        kwargs = {'exp_id': exp_id, 'settings': exp_args, 'settings_in_folder_name': True, 'gpu_ids': [gpu_id]}
                        args.append(kwargs)
                        # -- If the experiments and step size does not match perfectly there might be an overflow, so catch it -- #
                        if (run_index + gpu_index) > len(self.experiments.keys()):
                            break

                # -- Add those n experiment settings to the runs -- #
                runs.append(args)

            # -- Run the experiments -- #
            with tqdm(total = len(self.experiments.keys())) as pbar:
                for args in runs:    # --> Do this or the Pool might trigger two experiments using the same GPUs..
                    # -- Updating the summary log -- #
                    for kwargs in args:
                        self.summary = print_to_log_file(self.summary, "Start running the experiment {} using trainer {} (parallel mode).".format(kwargs['exp_id'], self.network_trainer))
                    
                    # -- NOTE: args is a list of kwargs so they have to be unpacked before fed to the function -- #
                    # -- NOTE: Since we are dealing with a for loop, there are no conflicts when writing files within the loop. -- #
                    # --       Every experiment is processed and the results are joined sequentially once they are done. -- #
                    for e_id, eval_res_pth in pool.imap_unordered(lambda kwargs: exp_.run_experiment(**kwargs), args):
                        # -- Update the tqdm bar -- #
                        pbar.update()

                        # -- Add finished experiment to backup file -- #
                        # TODO

                        # -- Join and save the results -- #
                        self._join_save_results(eval_res_pth, e_id)

            # -- Once done, close the Pool and its workes and wait until that's done -- #
            pool.close()
            pool.join()

            # -- Update the summary log file -- #
            self.summary = print_to_log_file(self.summary, 'Finished with all experiments. \nThe parameter searching is completed.')

    def _join_save_results(self, eval_res_pth, e_id):
        # -- Load the evaluation results -- #
        assert len(eval_res_pth) == 1, "There are no or more than one validation file(s). There should only be one file.."
        res = pd.read_csv(eval_res_pth[0], sep='\t')
        # -- Add the e_id as column to the res -- #
        res['experiment'] = e_id
        # -- Append them to out main summary and sort them by there experiment ID -- #
        self.main_sum = pd.concat([self.main_sum, res]) if self.main_sum is not None else res
        self.main_sum.sort_values('experiment')
        # -- Dump the self.main_sum dataframe with the summary of all experiments -- #
        self.main_sum.to_csv(os.path.join(self.output_base, 'parameter_search_val_summary.csv'))

        # -- Update the summary log file -- #
        self.summary = print_to_log_file(self.summary, 'Finished experiment {} using trainer {}.'.format(e_id, self.network_trainer))
        