#########################################################################################################
#----------This class represents a Parameter Searcher that can be used to find suitable parameter-------#
#-----------values to train a network with to achieve good results based on the tested params.----------#
#########################################################################################################

import pandas as pd
from collections import OrderedDict
import random, itertools, tqdm
from multiprocessing import Process, Queue
from nnunet_ext.utilities.helpful_functions import *
from nnunet_ext.experiment.experiment import Experiment
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.paths import param_search_output_dir, default_plans_identifier
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

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
        self.do_pod = do_pod
        self.rand_pick = rand_pick
        self.rand_seed = rand_seed
        self.rand_range = rand_range
        self.grid_picks = grid_picks
        self.search_mode = search_mode
        self.run_in_parallel = run_in_parallel
        self.continue_training = continue_training
        self.fixate_params = fixate_params if fixate_params is not None else dict()

        # -- Define the experiment arguments -- #
        self.exp_args = {'network': network, 'network_trainer': network_trainer, 'tasks_list_with_char': tasks_list_with_char,
                         'version': version, 'vit_type': vit_type, 'eval_mode_for_lns': eval_mode_for_lns, 'fold': fold, 'plans_identifier': plans_identifier,
                         'mixed_precision': mixed_precision, 'extension': extension, 'save_interval': save_interval, 'val_folder': val_folder,
                         'split_at': split_at, 'transfer_heads': transfer_heads, 'use_vit': use_vit, 'ViT_task_specific_ln': ViT_task_specific_ln,
                         'do_LSA': do_LSA, 'do_SPT': do_SPT, 'do_pod': do_pod, 'always_use_last_head': always_use_last_head, 'npz': npz, 'use_param_split': True,
                         'output_exp': '', 'output_eval': '', 'perform_validation': perform_validation, 'show_progress_tr_bar': False,
                         'unpack_data': unpack_data, 'deterministic': deterministic, 'save_interval': save_interval, 'param_call': True,
                         'num_epochs': num_epochs, 'fp16': fp16, 'find_lr': find_lr, 'valbest': valbest, 'disable_postprocessing_on_folds': disable_postprocessing_on_folds,
                         'split_gpu': split_gpu, 'val_disable_overwrite': val_disable_overwrite, 'disable_next_stage_pred': disable_next_stage_pred}

        # -- Do an initialization like the one of an Experiment -- #
        Experiment.__init__(self, **self.exp_args)

        # -- Remove the arguments that are not relevant for the parameter search class -- #
        del self.hyperparams, self.evaluator, self.basic_eval_args, self.param_split, self.use_progress_bar, self.param_call

        # -- Set tasks_joined_name for validation dataset building -- #
        self.tasks_joined_name = join_texts_with_char(tasks_list_with_char[0], tasks_list_with_char[1])
        # -- Everything is fit for now, start building the output_folder, ie. the root one -- #
        self.output_base = os.path.join(param_search_output_dir, network, self.tasks_joined_name, self.network_trainer + "__" + self.plans_identifier)
        self.output_base = nnUNetTrainerMultiHead._build_output_path(self, self.output_base, False)
        
        # -- Within this base folder, we have a folder for the experiments and one for the evaluation results -- #
        if 'OwnM' in self.network_trainer:
            self.output_base = os.path.join(self.output_base, 'pod' if self.do_pod else 'no_pod')
        self.output_exp = os.path.join(self.output_base, 'experiments')
        self.output_eval = os.path.join(self.output_base, 'evaluation')

        # -- Update the exp_args -- #
        self.exp_args['output_exp'] = self.output_exp
        self.exp_args['output_eval'] = self.output_eval

        # -- Define a empty dictionary that is used for backup purposes -- #
        self.backup_information = dict()
        new_backup = False
        self.e_id_fix = 0
        
        # -- Load the Backup file if it exists -- #
        self.backup_file = os.path.join(self.output_base, 'backup.pkl')
        if self.continue_training and os.path.isfile(self.backup_file):
            self.backup_information = load_pickle(self.backup_file)
            self.summary = self.backup_information['sum_log']
            # -- Try to load main_sum -- #
            if os.path.isfile(self.backup_information['main_sum_csv']):
                self.main_sum = pd.read_csv(self.backup_information['main_sum_csv'], sep='\t', header=0)
        if self.continue_training and not os.path.isfile(self.backup_file):
            assert False, "The user sets the -c flag but there is no backup file to begin with.."
        if not self.continue_training and os.path.isfile(self.backup_file):
            exp_folders = [x for x in os.listdir(self.output_exp) if 'exp_' in x]
            if len(exp_folders) > 0:
                self.e_id_fix = max([int(x.split('_')[-1]) for x in exp_folders]) + 1   # Add 1 since the e_id in loop start with 0
                self.backup_information = load_pickle(self.backup_file)
                self.summary = self.backup_information['sum_log']
                assert 'main_sum_csv' in self.backup_information,\
                    'Your backup information file at {} is not complete/corrupt, delete the corresponding folder since no experiments were run yet and restart the process..'.format(self.backup_file)
                # -- Try to load main_sum -- #
                if os.path.isfile(self.backup_information['main_sum_csv']):
                    self.main_sum = pd.read_csv(self.backup_information['main_sum_csv'], sep='\t', header=0)
            else:
                new_backup = True
        if not self.continue_training and not os.path.isfile(self.backup_file) or new_backup:   # Fresh experiments
            # -- Assert if continue training is set, since no backup is stored -- #
            self.backup_information['main_sum_csv'] = os.path.join(self.output_base, 'parameter_search_val_summary.csv')
            self.backup_information['started_experiments'] = set()
            self.backup_information['finished_experiments'] = set()

        # -- Create a summary file for this parameter search --> self.summary might be None, so provide all arguments -- #
        if not self.continue_training and self.e_id_fix == 0:   # --> only print this once
            self.summary = print_to_log_file(self.summary, self.output_base, 'parameter_search_summary', "Starting with the parameter searching.. \n")
            self.summary = print_to_log_file(self.summary, None, '', "NOTE: The parameter search method uses its own splits. " +\
                                                            "Those splits are generated from the original split file by splitting the training split again " +\
                                                            "randomly into an 80:20 split. This data is then used for training and validation, " +\
                                                            "whereas the validation data from the original split will never be used during parameter searching.")
            self.summary = print_to_log_file(self.summary, None, '', "The user wants to use the {} search method.".format(self.search_mode))
            self.summary = print_to_log_file(self.summary, None, '', "Building all possible settings for the experiments.")
            # -- Store the path to the summary log file -- #
            self.backup_information['sum_log'] = self.summary
            
        # -- Create the parameter combinations for the parameter search -- #
        if self.continue_training:
            self.experiments = self.backup_information['all_experiments']
        else:   # -- Build the experiments and store them
            if self.e_id_fix == 0:  # There are no other experiments in the folder
                self.experiments = OrderedDict()
            else:   # already done a params search but maybe with different settings
                self.experiments = self.backup_information['all_experiments']
            if self.search_mode == 'grid':
                # -- Create permutation between all possibilities in grid_picks -- #
                if len(self.params_to_tune) == 1:
                    # -- No permutation possible --> simply add the experiments -- #
                    hyperparam = list(self.params_to_tune)[0]
                    for idx, value in enumerate(*self.grid_picks.values()):
                        self.experiments['exp_{}'.format(self.e_id_fix+idx)] = {hyperparam: value, **self.fixate_params}
                else:   # -- Do permutation first -- #
                    combinations = itertools.product(*self.grid_picks.values())
                    for idx, combi in enumerate(combinations):
                        # -- Build the experiments dictionary -- #
                        experiment = dict()
                        for i, k in enumerate(self.params_to_tune):
                            experiment[k] = combi[i]
                        self.experiments['exp_{}'.format(self.e_id_fix+idx)] = {**experiment, **fixate_params}
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
                    self.experiments['exp_{}'.format(self.e_id_fix+idx)] = {**experiment, **fixate_params}

            # -- Assert if there are no experiments to do -- #
            assert len(self.experiments.keys()) != 0, "Unfortunately, there are no experiments based on the users arguments.."

            # -- Clean the experiments so no exp is done twice if the settings are equal -- #
            dup = 0
            sets_ = list()
            res = OrderedDict()
            for e_, sets in tuple(self.experiments.items()):
                # -- Build the settings as a set, so we can add them to the list and it will be invariant to the hyperparam order -- #
                sets_mod = set()
                for k, v in sets.items():
                    out = str(k) + '=' + str(v)    # hyperparam = value
                    sets_mod.add(out)
                    
                if sets_mod in sets_: # It already exists, so do not add it to res and increase dup by one -- #
                    msg = ''
                    for k, v in sets.items():
                        msg += str(k) + ':' + str(v) + ', '
                    self.summary = print_to_log_file(self.summary, None, '', "The experimet {} with setting {} is already trained or in process..".format(e_, msg[:-2]))
                    dup += 1
                else:   
                    res['exp_{}'.format(int(e_.split('_')[-1]) - dup)] = sets
                    sets_.append(sets_mod)

            self.experiments = res
            del res, dup, sets_

            # -- Add the experiments to the backup_information -- #
            self.backup_information['all_experiments'] = self.experiments

        # -- Paste all experiments into the log file -- #
        if not self.continue_training and self.e_id_fix == 0:
            # -- Build the string summarizing all experiments with the corresponding settings -- #
            exp_sum = ''
            for k, v in self.experiments.items():
                exp_sum += str(k) + ' := '
                for k_, v_ in v.items():
                    exp_sum += str(k_) + ':' + str(v_) + ', '
                exp_sum = exp_sum[:-2] + '\n'
            self.summary = print_to_log_file(self.summary, None, '', "There is/are {} experiments:\n{}".format(len(self.experiments.keys()), exp_sum))

        # -- Paste all experiments into the log file -- #
        if not self.continue_training and self.e_id_fix != 0:
            # -- Build the string summarizing all new experiments with the corresponding settings -- #
            new_exps = {k:v for k, v in self.experiments.items() if k not in [x for x in os.listdir(self.output_exp) if 'exp_' in x]}
            exp_sum = ''
            for k, v in new_exps.items():
                exp_sum += str(k) + ' := '
                for k_, v_ in v.items():
                    exp_sum += str(k_) + ':' + str(v_) + ', '
                exp_sum = exp_sum[:-2] + '\n'
            self.summary = print_to_log_file(self.summary, None, '', "There is/are {} new experiments:\n{}".format(len(new_exps.keys()), exp_sum))

        # -- Store the backup file -- #
        self._store_backup_file()
        
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

        if self.continue_training or self.e_id_fix != 0:
            # -- Distinguish between -c experiments and the experiments that have not even started later in the loops -- #
            self.experiments = {k: v for k, v in self.experiments.items() if k not in self.backup_information['finished_experiments']}
            
        # -- Create a single experiment if this is not done in paralllel -- #
        if not self.run_in_parallel:
            # -- Run the experiment for all provided settings -- #
            for exp, sets in self.experiments.items():
                self.summary = print_to_log_file(self.summary, None, '', "Start running the experiment {} using trainer {}.".format(exp, self.network_trainer))
                # -- Add the experiment to the set of started experiments --> there are no duplicates -- #
                # -- Store the backup file -- #
                self._store_backup_file()
                cont = exp in self.backup_information['started_experiments']    # Flag if continue with training or from beginning
                self.backup_information['started_experiments'].add(exp)
                # -- Run the experiment -- #
                available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
                e_id, eval_res_pth = exp_.run_experiment(exp_id = exp, settings = sets, settings_in_folder_name = True, gpu_ids = available_gpus, continue_tr = cont)
                self.summary = print_to_log_file(self.summary, None, '', 'Finished the experiment.')
                
                # -- Join and save the results -- #
                self._join_save_results(e_id, eval_res_pth)

                # -- Add finished experiment to backup file -- #
                self.backup_information['finished_experiments'].add(e_id)
                # -- Remove the experiment from the list of started experiments -- #
                self.backup_information['started_experiments'].remove(exp)
                # -- Store the backup file -- #
                self._store_backup_file()
        else:
            # -- Extract the list of GPUs -- #
            gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
            available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
            # -- Reduce the nr of GPUs is split_gpu has been set, then every experiment needs two GPUs -- #
            if self.split_gpu:
                gpus //= 2

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
                        exp_id = 'exp_{}'.format(self.e_id_fix + run_index + idx)
                        exp_args = self.experiments[exp_id]
                        cont = exp_id in self.backup_information['started_experiments']    # Flag if continue with training or from beginning
                        kwargs = {'exp_id': exp_id, 'settings': exp_args, 'settings_in_folder_name': True, 'gpu_ids': list(gpu_ids), 'continue_tr': cont}
                        args.append(kwargs)
                        idx += 1    # Use this because we go in two steps instead of one
                        # -- If the experiments and step size does not match perfectly there might be an overflow, so catch it -- #
                        if (run_index + idx) > len(self.experiments.keys())-1:
                            break
                else:
                    for gpu_index, gpu_id in enumerate(available_gpus):
                        # -- If the experiments and step size does not match perfectly there might be an overflow, so catch it -- #
                        if (run_index + gpu_index) > len(self.experiments.keys())-1:
                            break
                        exp_id = 'exp_{}'.format(self.e_id_fix + run_index + gpu_index)
                        exp_args = self.experiments[exp_id]
                        cont = exp_id in self.backup_information['started_experiments']    # Flag if continue with training or from beginning
                        kwargs = {'exp_id': exp_id, 'settings': exp_args, 'settings_in_folder_name': True, 'gpu_ids': list(gpu_id), 'continue_tr': cont}
                        args.append(kwargs)

                # -- Add those n experiment settings to the runs -- #
                runs.append(args)

            # -- Run the experiments -- #
            self.queue = Queue()    # Use this to keep track of the results per experiment (https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce)
            with tqdm.tqdm(total = len(self.experiments.keys()), desc = 'Experiments') as pbar:
                for args in runs:    # --> Do this or the Pool might trigger two experiments using the same GPUs..
                    
                    # -- NOTE: args is a list of kwargs so they have to be unpacked before fed to the function -- #
                    # -- NOTE: Since we are dealing with a for loop, there are no conflicts when writing files within the loop. -- #
                    # --       Every experiment is processed and the results are joined sequentially once they are done. -- #
                    
                    # -- Updating the summary log -- #
                    for kwargs in args:
                        self.summary = print_to_log_file(self.summary, None, '', "Start running the experiment {} using trainer {} (parallel mode).".format(kwargs['exp_id'], self.network_trainer))
                    
                    # -- Do in parallel using processes -- #
                    proc = list()
                    for kwargs in args:
                        # -- Add the experiment to the set of started experiments --> there are no duplicates -- #
                        self.backup_information['started_experiments'].add(kwargs['exp_id'])
                        # -- Store the backup file -- #
                        self._store_backup_file()
                        # -- Create a process and start it -- #
                        p = Process(target=self._parallel_exp_execution, kwargs = {'exp_': exp_, **kwargs})
                        p.start()
                        proc.append(p)
                    
                    for p in proc:  # Fetch the results
                        e_id, eval_res_pth = self.queue.get()
                        pbar.update()
                        # -- Join and save the results -- #
                        self._join_save_results(e_id, eval_res_pth)
                        
                        # -- Add finished experiment to backup file -- #
                        self.backup_information['finished_experiments'].add(e_id)
                        # -- Remove the experiment from the list of started experiments -- #
                        self.backup_information['started_experiments'].remove(e_id)
                        # -- Store the backup file -- #
                        self._store_backup_file()

                    for p in proc:
                        p.join()

            # -- Update the summary log file -- #
            self.summary = print_to_log_file(self.summary, None, '', 'Finished with all experiments.\nThe parameter searching is completed.\n')
            

    def _parallel_exp_execution(self, exp_, **kwargs):
        r"""This function is only used when running in parallel since this omits the prints that are created during the
            experiment runs.
        """
        # -- While not allowing any print execute the experiment -- #
        with suppress_stdout():
            e_id, eval_res_pth = exp_.run_experiment(**kwargs)
        # -- Return the results outside the while loop -- #
        self.queue.put((e_id, eval_res_pth))

    def _store_backup_file(self):
        r"""Call this function if the backup file is modified and has to be stored again for the changes to take effect.
        """
        # -- Dump the backup_information file -- #
        write_pickle(self.backup_information, self.backup_file)

    def _join_save_results(self, e_id, eval_res_pth):
        r"""This function is used to join the different results from all experiments into one dataframe while saving it.
        """
        # -- Load the evaluation results -- #
        res = list()
        for res_p in eval_res_pth:
            df = pd.read_csv(res_p, sep='\t', header=0)
            res.append(df)
        # -- Append all loaded csvs -- #
        res = pd.concat(res, ignore_index=True)
        # -- Add the e_id as column to the res -- #
        res['experiment'] = e_id
        # -- Add the settings as column to the res -- #
        sets = ''
        for set, val in self.experiments[e_id].items():
            sets += str(set) + ':' + str(val) + ', '
        res['settings'] = sets[:-2]
        # -- Append them to out main summary -- #
        self.main_sum = pd.concat([self.main_sum, res]) if self.main_sum is not None else res
        # -- Rearrange the dataframe -- #
        eval_col = self.main_sum.columns.values.tolist()
        eval_col.remove("experiment")
        eval_col.remove("settings")
        self.main_sum = self.main_sum[["experiment", "settings", *eval_col]]
        # -- Sort by experiment ID and dump the self.main_sum dataframe with the summary of all experiments -- #
        self.main_sum.sort_values('experiment')
        dumpDataFrameToCsv(self.main_sum, self.output_base, 'parameter_search_val_summary')

        # -- Update the summary log file -- #
        self.summary = print_to_log_file(self.summary, None, '', 'Finished experiment {} using trainer {}.'.format(e_id, self.network_trainer))