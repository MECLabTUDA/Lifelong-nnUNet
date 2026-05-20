#########################################################################################################
#----------------------This class represents the nnUNet trainer for ODEx with NCA.-----------------------#
#########################################################################################################

# -- This implementation represents the method proposed in the-- #

import copy, torch
from torch.cuda.amp import autocast
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

from nnunet_ext.network_architecture.nca.OctreeNCA3D import OctreeNCA3D
from nnunet_ext.network_architecture.nca.OctreeNCA2D import OctreeNCA2D


class nnUNetTrainerODExNCA(nnUNetTrainerV2):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='odex_nca', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, nca=False, network=None, use_param_split=False):
        r"""Constructor of Odex Trainer
        """
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)
        
        # -- Create a backup of the original output folder that is provided -- #
        self.output_folder_orig = output_folder

        output_folder = self._build_output_path(output_folder, False)

        self.task = task                # name of the model that is referred to as a task name
        self.split = split              # provided split
        self.identifier = identifier    # identifier for building the .pkl file for restoring states
        self.fold = fold                # for tracking and saving in self.already_trained_on file
        self.csv = save_csv             # flag if saving validation metrics every nth epoch
        self.del_log = del_log          # flag if the log should be removed or not
        self.extension = extension      # Set the extension for output file

        # -- Set trainer_class_name -- #
        self.trainer_class_name = self.__class__.__name__

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
                          tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, self.vit_type, version, split_gpu,
                          transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT, nca)


        self.NQM_dict = dict()
        self.model_pool = dict()


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
        # -- Create a deepcopy of the previous, ie. currently set model if we do PLOP training -- #
        if task not in self.mh_network.heads:
            print("???? new task")


        if self.threeD:
            num_steps = [6,7,8,9,10,20]
            num_steps = num_steps[-num_levels:]
            assert len(num_steps) == num_levels, f"num_steps: {num_steps}, num_levels: {num_levels}"
            self.network = OctreeNCA3D(num_channels=16, 
                                num_input_channels=self.num_input_channels,
                                num_classes=self.num_classes,
                                hidden_size=64,
                                fire_rate=0.5,
                                num_steps=num_steps,
                                num_levels=num_levels,
                                pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes)
        else:
            self.network = OctreeNCA2D(num_channels=16, 
                                num_input_channels=self.num_input_channels,
                                num_classes=self.num_classes,
                                hidden_size=64,
                                fire_rate=0.5,
                                num_steps=num_steps,
                                num_levels=num_levels)


    def run_training(self, task, output_folder):
        r"""Overwrite super class to adapt for ood detection
        """

        

        # -- Run training using parent class -- #
        ret = super().run_training(task, output_folder)

        # compute NQM for task and save it in 
        self.NQM_dict[task] = 0.2 * (1 + len(self.NQM_dict)) # TODO: use real NQM

        # -- Return the result -- #
        return ret
