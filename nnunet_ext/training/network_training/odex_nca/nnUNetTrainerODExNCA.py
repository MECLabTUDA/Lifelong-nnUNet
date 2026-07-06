#########################################################################################################
#----------------------This class represents the nnUNet trainer for ODEx with NCA.-----------------------#
#########################################################################################################

# -- This implementation represents the method proposed in the-- #

from torch import nn
import numpy as np
import copy, torch
import SimpleITK as sitk
from torch.amp import autocast
from collections import OrderedDict
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.utilities.nd_softmax import softmax_helper
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.utilities.helpful_functions import *
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.inference.predict import predict_from_folder
from nnunet_ext.paths import preprocessing_output_dir, default_plans_identifier

from nnunet_ext.network_architecture.nca.OctreeNCA3D import OctreeNCA3D
from nnunet_ext.network_architecture.nca.OctreeNCA2D import OctreeNCA2D

import traceback
import inspect


class nnUNetTrainerODExNCA(nnUNetTrainerV2):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='odex_nca', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, nca=False, network=None, use_param_split=False):
        r"""Constructor of Odex Trainer
        """
        # -- Initialize using parent class -- #
        
        # -- Create a backup of the original output folder that is provided -- #
        self.output_folder_orig = output_folder

        output_folder = self._build_output_path(output_folder, False)

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

        self.task = task                # name of the model that is referred to as a task name
        self.split = split              # provided split
        self.identifier = identifier    # identifier for building the .pkl file for restoring states
        self.fold = fold                # for tracking and saving in self.already_trained_on file
        self.csv = save_csv             # flag if saving validation metrics every nth epoch
        self.del_log = del_log          # flag if the log should be removed or not
        self.extension = extension      # Set the extension for output file
        self.active_task = task

        # -- Set trainer_class_name -- #
        self.trainer_class_name = self.__class__.__name__

        # -- Initialize or set self.already_trained_on dictionary to keep track of the trained tasks so far for restoring -- #
        if already_trained_on is not None:
            self.already_trained_on = already_trained_on    # Use provided already_trained on
            # -- If the current fold does not exists initialize it -- #
            if self.already_trained_on.get(str(self.fold), None) is None:
                self.already_trained_on[str(self.fold)] = {'finished_training_on': list(), 'start_training_on': None, 'finished_validation_on': list(),
                                                           'used_identifier': self.identifier, 'prev_trainer': [self.trainer_class_name], 'val_metrics_should_exist': False,
                                                           'checkpoint_should_exist': False, 'tasks_at_time_of_checkpoint': list(),
                                                           'active_task_at_time_of_checkpoint': None}  # Add current fold as new entry
            else: # It exists, then check if everything is in it
                # -- Define a list of all expected keys that should be in the already_trained_on dict for the current fold -- #
                keys = ['finished_training_on', 'start_training_on', 'finished_validation_on', 'used_identifier', 'prev_trainer',\
                        'val_metrics_should_exist', 'checkpoint_should_exist','tasks_at_time_of_checkpoint',\
                        'active_task_at_time_of_checkpoint']
                # -- Check that everything is provided as expected -- #
                assert all(key in self.already_trained_on[str(self.fold)] for key in keys),\
                    "The provided already_trained_on dictionary does not contain all necessary elements"
        else:
            self.already_trained_on = {str(self.fold): {'finished_training_on': list(), 'start_training_on': None, 'finished_validation_on': list(),
                                                        'used_identifier': self.identifier, 'prev_trainer': [self.trainer_class_name], 'val_metrics_should_exist': False,
                                                        'checkpoint_should_exist' : False, 'tasks_at_time_of_checkpoint': list(),
                                                        'active_task_at_time_of_checkpoint': None}}

        self.trained_on_path = os.path.dirname(os.path.dirname(os.path.realpath(self.output_folder_orig)))
        self.trained_on_path = self._build_output_path(self.trained_on_path, True)

        # -- Create the folder if necessary -- #
        maybe_mkdir_p(self.trained_on_path)

        # -- Set save_every, so the super trainer class creates checkpoint individually and the validation metrics will be filtered accordingly -- #
        self.save_every = save_interval

        # -- Initialize subject_names list that is used to store the subject names for every nth evaluation -- #
        self.subject_names_raw = list() # Store the names as is, ie. not cleaned (removed duplicates etc.) --> For evaluation necessary

        # -- Extract network_name that might come in handy at a later stage -- #
        # -- For more details on how self.output_folder is built look at get_default_configuration -- #
        self.network_name = network
        assert self.network_name is not None, "Please provide the network setting that is used.."
    

        # -- Set if the model should be compressed as floating point 16 -- #
        self.mixed_precision = mixed_precision


        # -- Ensure that it is a tuple and that the first element is a list and second element a string -- #
        assert isinstance(tasks_list_with_char, tuple) and isinstance(tasks_list_with_char[0], list) and isinstance(tasks_list_with_char[1], str),\
             "tasks_list_with_char should be a tuple consisting of a list of tasks as the first and a string "+\
             "representing the character that is used to join the tasks as the second element.."
        
        # -- Store the tuple consisting of a list with tasks and the character that should be used to join the tasks -- #
        self.tasks_list_with_char = tasks_list_with_char
 
        # -- Set tasks_joined_name for validation dataset building -- #
        self.tasks_joined_name = join_texts_with_char(self.tasks_list_with_char[0], self.tasks_list_with_char[1])

        # -- Define a dictionary for the metrics for validation after every nth epoch -- #
        self.validation_results = dict()

        # -- If -c is used, the self.validation_results need to be restored as well -- #
        # -- Check if the val_metrics should exist -- #
        if self.already_trained_on[str(self.fold)]['val_metrics_should_exist']:
            try:
                # -- Try to load the file -- #
                self.validation_results = load_json(join(self.output_folder, 'val_metrics.json'))
            except: # File does not exist
                assert False, "The val_metrics.json file could not be loaded although it is expected to exist given the current state of the model."

        # -- Set use_prograss_bar if desired so a progress will be shown in the terminal -- #
        self.use_progress_bar = use_progress

        # -- Define an empty trainer_model -- #
        self.trainer_model = None

        # -- Define flag for evaluation (per batch or per subject) -- #
        self.eval_batch = True

        # -- Set the flag if the param_split should be used instead of the general split -- #
        # -- Only set this to True if the parameter search method is used -- #
        self.param_split = use_param_split

        self.nca = nca
        if nca:
            self.initial_lr = 1e-3

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                          tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, vit_type, version, split_gpu,
                          transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT, nca)


        self.NQM_dict = dict()
        self.model_pool = nn.ModuleDict()


    def do_split(self):
        r"""Modify the original function. This enables the loading of the split
            for the parameter search method if the flag is set.
            When using the parameter search method, the split will be modified in such a way, that the
            training set of the original split will be split by 80:20. The original val_set will not be used
            during training or validation, ie. never when doing parameter search. However the split contains
            those under the test_set flag.
        """
        # -- Copied from original implementation and modified -- #
        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            # -- Set the splits file path -- #
            p_splits_file = join(self.dataset_directory, "splits_param_search.pkl")
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            if not isfile(p_splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split for parameter searching based on already existing split file {}...".format(splits_file))
                # -- Define new split list -- #
                new_splits = list()
                # -- Go through all defined folds and resplit the train set into 80:20 data split -- #
                for split in splits:
                    # -- Extract the names of the files in the train split -- #
                    train = split['train']
                    # -- Split the train randomly in train and val and add it to the new_splits -- #
                    train_, val_ = train_test_split(train, random_state=3299, test_size=0.2)
                    new_splits.append(OrderedDict())
                    new_splits[-1]['train'] = train_
                    new_splits[-1]['val'] = val_
                    new_splits[-1]['test'] = split['val']   # --> Only for storing, should be never used
                # -- Save the new splits file -- #
                save_pickle(new_splits, p_splits_file)

            if self.param_split:
                splits = load_pickle(p_splits_file)

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                    % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20/64:16:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]

                # -- Split the train randomly in train and val again if self.param_split is desired -- #
                if self.param_split:
                    tr_keys, val_keys = train_test_split(tr_keys, random_state=3299, test_size=0.2)
                # else:
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                    % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
        # -- Copied from original implementation and modified -- #

    def process_plans(self, plans):
        r"""Modify the original function. This just reduces the batch_size by half and manages the correct initialization of the network.
        """
        # -- Initialize using parent class -- #
        super().process_plans(plans)

        # -- Reduce the batch_size by half after it has been set by super class --> only if ViT is used -- #
        # -- Do this so it fits onto GPU --> if it still does not, model needs to be put onto multiple GPUs -- #
        if self.nca and self.threeD:
            self.batch_size = 2

     
    def maybe_update_lr(self, epoch=None):
        if not self.nca:
            return super().maybe_update_lr(epoch)
        self.lr_scheduler.step(epoch)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def initialize_optimizer_and_scheduler(self):
        r"""Update this so params without gradients are never updated during training --> Don't forget to recall
            if the gradients are changed during training..
        """
        assert self.network is not None, "self.initialize_network must be called first"

        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=0)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_num_epochs, eta_min=1e-6)


    def initialize(self, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None, call_for_eval=False):
        r"""
            Copied from MultiHeadTrainer
            Overwrite parent function, since we want to include a prev_trainer that is used as a base for ODEx NCA Trainer.
            Further the num_epochs should be set by the user if desired.
        """
        # -- The Trainer embodies the actual model that will be used as foundation to continue training on -- #
        # -- It should be already initialized since the output_folder will be used. If it is None, the model will be initialized and trained. -- #
        # -- Further, the trainer needs to be of class nnUNetTrainerV2 or nnUNetTrainerMultiHead for this method, nothing else. -- #
        # -- Set prev_trainer_path correctly as a string and not any class instance -- #
        if prev_trainer_path is not None and not call_for_eval:
            self.trainer_path = join(self._build_output_path(prev_trainer_path, False), "fold_%s" % str(self.fold))
        else:   # If for eval, then this is a nnUNetTrainerV2 whereas the path is not build as implemented in _build_output_path
            self.trainer_path = prev_trainer_path
        
        # -- Initialize using super class -- #
        super().initialize(training, force_load_plans) # --> This updates the corresponding variables automatically since we inherit this class

        # -- Set nr_epochs to provided number -- #
        self.max_num_epochs = num_epochs

        # -- Initialize the trained_on_tasks and load trained_on_folds -- #
        trained_on_tasks = list()
        trained_on_folds = self.already_trained_on.get(str(self.fold), list())
        
        # -- Reset the trained_on_tasks if the trained_on_folds exist for the current fold -- #
        if isinstance(trained_on_folds, dict):
            trained_on_tasks = trained_on_folds.get('finished_training_on', list())

        # -- The new_trainer indicates if the model is a new multi head model, -- #
        # -- ie. if it has been trained on only one task so far (True) or on more than one (False) -- #
        if len(trained_on_tasks) > 1:
            self.new_trainer = False
        else:
            self.new_trainer = True


    def initialize_network(self):
        print("???? initialize network odex")

        num_levels = len(self.net_num_pool_op_kernel_sizes)
        #base_num_steps = int(3 * max(self.patch_size / 2**num_levels))

        num_steps = [5] * (num_levels-1) + [20]

        model = None

        if self.threeD:
            num_steps = [6,7,8,9,10,20]
            num_steps = num_steps[-num_levels:]
            assert len(num_steps) == num_levels, f"num_steps: {num_steps}, num_levels: {num_levels}"
            model = OctreeNCA3D(num_channels=16, 
                                num_input_channels=self.num_input_channels,
                                num_classes=self.num_classes,
                                hidden_size=64,
                                fire_rate=0.5,
                                num_steps=num_steps,
                                num_levels=num_levels,
                                pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes)
        else:
            model = OctreeNCA2D(num_channels=16, 
                                num_input_channels=self.num_input_channels,
                                num_classes=self.num_classes,
                                hidden_size=64,
                                fire_rate=0.5,
                                num_steps=num_steps,
                                num_levels=num_levels)

        self.network = model

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


    def run_training(self, task, output_folder):
        r"""Overwrite super class to adapt for ood detection
        """

        if len(self.NQM_dict) != 1:
            self.active_task = task

        self.output_folder = join(self._build_output_path(output_folder, False), "fold_%s" % str(self.fold))
        maybe_mkdir_p(self.output_folder)

        # -- Run training using parent class -- #
        ret = super().run_training()

        nqm_list_of_current_task = self.compute_nqm_of_task(task, self)
        
        nqm_list_of_current_task.sort()
        task_threshold = nqm_list_of_current_task[int(len(nqm_list_of_current_task)*0.9)]


        # compute NQM for task and save it in 
        self.NQM_dict[task] = task_threshold # TODO: use real NQM
        print(f"-- NQM dict: {self.NQM_dict}")
        print(f"-- current model pool: {self.model_pool.keys()}, active task: {self.active_task}")

        ###### Copied from MultiHeadNetworkTrainer
        # -- Reset the val_metrics_exist flag since the training is finished and restoring will fail otherwise -- #
        self.already_trained_on[str(self.fold)]['val_metrics_should_exist'] = False

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


        # -- Return the result -- #
        return ret
    
    def compute_nqm_of_task(evaluate_on, model, include_training_data=False):
        
        print(f"-----------   output folder: {self.output_folder}")

        input_folder = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', evaluate_on, 'imagesTr')
        
        print(f"-----------   input folder: {input_folder}")

        # setting parameters for predict_from_folder()
        
        lowres_segmentations = None
        save_npz = False
        enable_tta = False
        mixed_precision = True
        num_threads_preprocessing = 1
        num_parts = 1
        part_id = 0
        num_threads_nifti_save = 2
        step_size = 0.5

        params_ext = {
            'use_head': None,
            'always_use_last_head': False,
            'extension': "odex_nca",
            'param_split': False,
            'network': "3d_fullres",
            'network_trainer': "nnUNetTrainerODExNCA",
            'use_model': [f"{task}"],
            'tasks_list_with_char': ["Task198_T1threesplit"],
            'plans_identifier': default_plans_identifier,
            'vit_type': "base",
            'version': 1
        }

        evaluate_on = task
        
        for i in range(10):
            nqm_tmp_folder = self.output_folder + f"/nqm_it_{i}"
            predict_from_folder(params_ext, "None", input_folder, nqm_tmp_folder, [self.fold], save_npz, num_threads_preprocessing,
                    num_threads_nifti_save, lowres_segmentations, part_id, num_parts, enable_tta,
                    overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None,
                    mixed_precision=mixed_precision,
                    step_size=step_size, no_load=True, trainer=model, params=[None], plans_path_=self.plans_file)

        
        dataset_directory = join(preprocessing_output_dir, evaluate_on)
        splits_final = load_pickle(join(dataset_directory, "splits_final.pkl"))

        ground_truth_folder: str = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', evaluate_on, 'labelsTr')
        

        if include_training_data:
            cases_to_perform_evaluation_on = []
            for s in splits_final[self.fold].keys():
                cases_to_perform_evaluation_on.extend(splits_final[self.fold][s])
        else:
            cases_to_perform_evaluation_on = []
            for s in splits_final[self.fold].keys():
                if s != 'train':
                    cases_to_perform_evaluation_on.extend(splits_final[self.fold][s])

        print(f"splits_final: {splits_final[self.fold]}")
        print("original training cases:", splits_final[self.fold]['train'])
        print("performing validation on:", cases_to_perform_evaluation_on)
        nqm_list_of_current_task = []
        for case in cases_to_perform_evaluation_on:
            file_name = case + ".nii.gz"
            #there must be a corresponding entry in inference_folder
            
            ensemble = []
            for i in range(10):
                assert isfile(join(self.output_folder, f"nqm_it_{i}", file_name))
                ensemble.append(sitk.GetArrayFromImage(sitk.ReadImage(join(self.output_folder, f"nqm_it_{i}", file_name))))
            ensemble = np.stack(ensemble, axis=0)
            mean = np.sum(ensemble, axis=0) / ensemble.shape[0]
            stdd = 0
            for id in range(ensemble.shape[0]):
                img = ensemble[id] - mean
                img = np.power(img, 2)
                stdd = stdd + img
            stdd = stdd / ensemble.shape[0]
            stdd = np.sqrt(stdd)
            nqm_score = np.sum(stdd) / np.sum(mean)
            nqm_list_of_current_task.append(nqm_score)
            print("NQM Score: ", nqm_score)
        
        return nqm_list_of_current_task

    def _build_output_path(self, output_folder, meta_data=False):
        r"""This function is used to build the output folder path during training when a new task is started.
            If the path is not adjusted given this method, the data files are scattered all over the place at
            different folders where they don't belong.
        """
        # -- First of all remove the fold_ from the path -- #
        if 'fold_' in output_folder.split(os.path.sep)[-1]:
            output_folder = os.path.join(*output_folder.split(os.path.sep)[:-1])
            
        # -- Generic_UNet will be used so update the path accordingly -- #
        #if Generic_UNet.__name__ != output_folder.split(os.path.sep)[-1] and Generic_UNet.__name__ not in output_folder:
        if not meta_data:
            output_folder = os.path.join(output_folder, Generic_UNet.__name__)
        else:   # --> Path were meta data is stored, i.e. already_trained_on file
            output_folder = os.path.join(output_folder, 'metadata', Generic_UNet.__name__)

        # -- In every case, change the current folder in such a way that there is an indication if transfer_heads was true or false -- #
        if 'MH' not in output_folder and 'SEQ' not in output_folder:
            output_folder = os.path.join(output_folder, 'SEQ')
        
        print(f"???? output_folder: {output_folder}")
        # -- Return the folder -- #
        return output_folder
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function runs an iteration based on the underlying model. It returns the detached or undetached loss.
            The undetached loss might be important for methods that have to extract gradients without always copying
            the run_iteration function.
            NOTE: The calling class needs to set self.network according to the desired task, this is not done in this
                  function but expected by the user.
        """
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                if not no_loss:
                    l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            if not no_loss:
                l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        ## update after iteration
        if self.active_task not in self.model_pool:
            self.model_pool[self.active_task] = copy.deepcopy(self.network)
        self.model_pool[self.active_task].load_state_dict(self.network.state_dict())
        
        # -- Return the loss -- #
        if not no_loss:
            if detach:
                l = l.detach().cpu().numpy()
            return l

    def update_save_trained_on_json(self, task, finished=True):
        r"""This function updates the dictionary, if a model is trained for n different tasks, this list needs to be updated
            after each sucessful training of a task and stored accordingly! The 'finished' specifies if the task is finished training
            or just started for training.
            This function also saves the already_trained_on list as a pickle file under the path of the new model task (output_folder).
        """
        # -- Add the provided task at the end of the list, sort the list and dump it as pkl file -- #
        if finished:    # Task finished with training
            if task not in self.already_trained_on[str(self.fold)]['finished_training_on']:
                self.already_trained_on[str(self.fold)]['finished_training_on'].append(task)
            # -- Remove the task from start_training_on -- #
            self.already_trained_on[str(self.fold)]['start_training_on'] = None 
        else:   # Task started to train
            # -- Add the current task -- #
            self.already_trained_on[str(self.fold)]['start_training_on'] = task
            # -- Update the prev_trainer -- #
            if self.trainer_model is not None and len(self.already_trained_on[str(self.fold)]['prev_trainer']) == 1: # This is always the case when a pre-trained network is used as initialization
                self.already_trained_on[str(self.fold)]['prev_trainer'][-1:] = [self.trainer_model.__class__.__name__]  # --> The one from the used trainer
                self.already_trained_on[str(self.fold)]['prev_trainer'].append(self.trainer_class_name)                 # --> The current trainer we start training with
        # -- Update the used_identifier -- #
        self.already_trained_on[str(self.fold)]['used_identifier'] = self.identifier

        # -- Save the updated dictionary as a pkl file -- #
        write_pickle(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.pkl'))
        # -- Update self.init_tasks so the storing works properly -- #
        self.update_init_args()

    def save_checkpoint(self, fname, save_optimizer=True):
        r"""Overwrite the parent class, since we want to store the body and heads along with the current activated model
            and not only the current network we train on. If the class uses an old_model, we have to store this as well.
            The old model should be stored in self.network_old, always!
        """
        # -- Set the network to the full MultiHead_Module network to save everything in the class not only the current model -- #
        #self.network = self.mh_network

        print(f"saving checkpoint under {fname}")
        #traceback.print_stack()

        # -- Set the flag to True -- #
        self.already_trained_on[str(self.fold)]['checkpoint_should_exist'] = True
        # -- Add the current head keys for restoring (is in correct order due to OrderedDict type of heads) -- #
        self.already_trained_on[str(self.fold)]['tasks_at_time_of_checkpoint'] = list(self.NQM_dict.keys())
        # -- Add the current active task for restoring -- #
        self.already_trained_on[str(self.fold)]['active_task_at_time_of_checkpoint'] = len(self.NQM_dict)
        # -- Save the updated dictionary as a json file -- #
        write_pickle(self.already_trained_on, join(self.trained_on_path, self.extension+'_trained_on.pkl'))
        # -- Update self.init_tasks so the storing works properly -- #
        self.update_init_args()

        # -- Use parent class to save checkpoint for MultiHead_Module model consisting of self.model, self.body and self.heads -- #
        super().save_checkpoint(fname, save_optimizer) 

    def update_init_args(self):
        r"""This function is used to update the init_args variable that is saved during checkpoint storing.
            During this update, only the already_trained_on will be updated.
        """
        # -- Transform tuple to list -- #
        init = list(self.init_args)
        # -- Update already_trained_on (has position 12 --> if that changes, than change this here as well) -- #
        init[12] = self.already_trained_on
        # -- Transform list back to tuple -- #
        self.init_args = tuple(init)

    def save_init_args(self, fname):
        r"""This function needs to be executed after a finished training, if some arguments are changed once the final mode
            is stored. Those results need to be updated as well or there might be a problem when trying to restore later on.
        """
        #------------------------------------------ Copied from original implementation ------------------------------------------#
        # -- Save the results in case the script gets interrupted etc. for proper restoring -- #
        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans
        # -- Dump the file -- #
        write_pickle(info, fname + ".pkl")
        #------------------------------------------ Copied from original implementation ------------------------------------------#
