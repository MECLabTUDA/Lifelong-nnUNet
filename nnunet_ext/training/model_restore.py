#########################################################################################################
#----------This class represents the model_restore module for the nnUNet extension. Implementation------#
#----------inspired by original implementation (--> model_restore), copied code is marked as such.------#
#########################################################################################################

import torch
import torch.nn as nn
import importlib, pkgutil, nnunet, nnunet_ext
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.run.default_configuration import get_default_configuration as get_default_configuration_orig

# -- Import the trainer classes -- #
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

def recursive_find_python_class_file(folder, trainer_name, current_module):
    r"""This returns the file to import, but not the actual class within this file.
        Modified implementation of https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/model_restore.py
    """
    tr = None
    for _, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            mod = importlib.import_module(current_module + "." + modname)
            if hasattr(mod, trainer_name):
                break

    if tr is None:
        for _, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return mod

def restore_model(pkl_file, checkpoint=None, train=False, fp16=True, use_extension=False, extension_type='multihead', del_log=False, param_search=False, network=None):
    """ This function is modified to work for the nnU-Net extension as well and ensures a correct loading of trainers
        for both (conventional and extension). Use del_log when using this for evaluation to remove the then created log_file
        during intialization.
        This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
        nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
        is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
        The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
        :param pkl_file:
        :param checkpoint:
        :param train:
        :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
        :return:
    """
    # -- Load the provided pickle file -- #
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    
    # -- Reset arguments if a Generic_ViT_UNet is used -- #
    if use_extension and nnViTUNetTrainer.__name__ in pkl_file:
        # Only occurs during evaluation when building a MH network which sets a wrong extension_type
        extension_type = None
    
    # -- Set search_in and base_module given the current arguments -- #
    if use_extension:   # -- Extension search in nnunet_ext
        if extension_type is None: # Should only be None when using the Generic_ViT_UNet
            search_in = (nnunet_ext.__path__[0], "training", "network_training")
            base_module = 'nnunet_ext.training.network_training'
        else:
            search_in = (nnunet_ext.__path__[0], "training", "network_training", extension_type)
            base_module = 'nnunet_ext.training.network_training.' + extension_type
    else:   # -- No extension search in nnunet
        search_in = (nnunet.__path__[0], "training", "network_training")
        base_module = 'nnunet.training.network_training'

    # -- Search for the trainer class based on search_in, name of the trainer and base_module -- #
    tr = recursive_find_python_class([join(*search_in)], name, current_module=base_module)

    # -------------------- From nnUNet implementation (modifed, but same output) -------------------- #
    if tr is None:
        try:
            import meddec
            search_in = join(meddec.__path__[0], "model_training")
            tr = recursive_find_python_class([search_in], name, current_module="meddec.model_training")
        except ImportError:
            pass
        
    if tr is None:
        raise RuntimeError("Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
                           "is not located there, please move it or change the code of restore_model. Your model "
                           "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
                           "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))
    assert issubclass(tr, nnUNetTrainer), "The network trainer was found but is not a subclass of nnUNetTrainer. " \
                                          "Please make it so!"
    # -------------------- From nnUNet implementation (modifed, but same output) -------------------- #

    # -- Set the trainer -- #
    # -- For old models that were trained before moving the models to another location -- #
    if '/gris/gris-f/homestud/' in init[2]:
        init = list(init)
        init[2] = init[2].replace('/gris/gris-f/homestud/', '/local/scratch/')
        init[4] = init[4].replace('/gris/gris-f/homestud/', '/local/scratch/')
        init[5] = init[5].replace('/gris/gris-f/homestud/', '/local/scratch/')
        if isinstance(init[12], dict):
            for i in init[12].keys():
                try:
                    init[12][i]['fisher_at'] = init[12][i]['fisher_at'].replace('/gris/gris-f/homestud/', '/local/scratch/')
                    init[12][i]['params_at'] = init[12][i]['params_at'].replace('/gris/gris-f/homestud/', '/local/scratch/')
                except:
                    pass

    if use_extension and extension_type is not None:    # Only for extensions, with the exception of ViT_U-Net
        assert network is not None, "Please provide the network setting that is used.."
        trainer = tr(*init, network=network)
        trainer.del_log = del_log
        trainer.param_split = param_search
    else:
        trainer = tr(*init)
    trainer.initialize(train)

    # -------------------- From nnUNet implementation (modifed, but same output) -------------------- #
    if fp16 is not None:
        trainer.fp16 = fp16
    
    # -- Backup the patch_size before loading the plans -- #
    patch_size = trainer.patch_size

    # -- NOTE: This loads the plan file from the current task! The patch size is high likely to change -- #
    # --       which is why we created a backup to reset the patch size after plans processing --> we -- #
    # --       do not change the plans file since this is based on the data and if we initialize a MH with -- #
    # --       this task, we want it to fit to the initialized data, however for every following trainer, -- #
    # --       the patch size can not change since the model structure is not fit for it and thus fail during -- #
    # --       the forward pass in the skip connections! Keep this in mind -- # 
    trainer.process_plans(info['plans'])
    
    # -- Restore the patch size -- #
    trainer.patch_size = patch_size
    
    if checkpoint is not None:
        # -- Note, when restoring we don't need the old network --> this is only called once training is finished -- #
        trainer.load_checkpoint(checkpoint, train)

    return trainer
    # -------------------- From nnUNet implementation (modifed, but same output) -------------------- #


# -------------------- Arguments are different from the original nnUNet -------------------- #
def load_model_and_checkpoint_files(params, folder, folds=None, mixed_precision=None, checkpoint_name="model_best"):
    """
    This will restore the model from the checkpoint in the specified fold.
    """
    t_fold = folds[0] # Fold must be specified
    if isinstance(folds, str):
        folds = [join(folder, "all")]
        assert isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [join(folder, "all")]
        else:
            folds = [join(folder, "fold_%d" % i) for i in folds]
        assert all([isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [join(folder, "fold_%d" % folds)]
        assert all([isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds. Type: %s. Expected: list of int, int, str or None", str(type(folds)))

    # Restore trainer (extension logic)
    trainer_path = folds[0]
    checkpoint = join(trainer_path, "%s.model.pkl" % checkpoint_name)
    pkl_file = checkpoint + ".pkl"
    use_extension = not 'nnUNetTrainerV2' in trainer_path
    trainer = restore_model(pkl_file, checkpoint, train=False, fp16=mixed_precision,\
                            use_extension=use_extension, extension_type=params['extension'], del_log=True,\
                            param_search=params['param_split'], network=params['network'])

    # -- If this is a conventional nn-Unet Trainer, then make a MultiHead Trainer out of it, so we can use the _perform_validation function -- #
    if not use_extension or nnViTUNetTrainer.__name__ in trainer_path:
        # -- Ensure that use_model only contains one task for the conventional Trainer -- #
        assert len(params['tasks_list_with_char'][0]) == 1, "When trained with {}, only one task could have been used for training, not {} since this is no extension.".format(params['network_trainer'], len(params['use_model']))
        # -- Store the epoch of the trainer to set it correct after initialization of the MultiHead Trainer -- #
        epoch = trainer.epoch
        # -- Extract the necessary information of the current Trainer to build a MultiHead Trainer -- #
        # -- NOTE: We can use the get_default_configuration from nnunet and not nnunet_ext since the information -- #
        # --       we need can be extracted from there as well without as much 'knowledge' since we do not need -- #
        # --       everything for a MultiHead Trainer -- #
        if 'nnUNetTrainerV2' in trainer_path:
            plans_file, prev_trainer_path, dataset_directory, batch_dice, stage, \
            _ = get_default_configuration_orig(params['network'], params['tasks_list_with_char'][0][0], params['network_trainer'], params['plans_identifier'])
        else:   # ViT_U-Net
            plans_file, prev_trainer_path, dataset_directory, batch_dice, stage, \
            _ = get_default_configuration(params['network'], params['tasks_list_with_char'][0][0], None, params['network_trainer'], None,
                                            params['plans_identifier'], extension_type=None)
            # -- Modify prev_trainer_path based on desired version and ViT type -- #
            if not nnViTUNetTrainer.__name__+'V' in prev_trainer_path:
                prev_trainer_path = prev_trainer_path.replace(nnViTUNetTrainer.__name__, nnViTUNetTrainer.__name__+params['version'])
            if params['vit_type'] != prev_trainer_path.split(os.path.sep)[-1] and params['vit_type'] not in prev_trainer_path:
                prev_trainer_path = os.path.join(prev_trainer_path, params['vit_type'])

        # -- Build a simple MultiHead Trainer so we can use the perform validation function without re-coding it -- #
        trainer = nnUNetTrainerMultiHead('seg_outputs', params['tasks_list_with_char'][0][0], plans_file, t_fold, output_folder=trainer_path,\
                                            dataset_directory=dataset_directory, tasks_list_with_char=(params['tasks_list_with_char'][0], params['tasks_list_with_char'][1]),\
                                            batch_dice=batch_dice, stage=stage, already_trained_on=None, use_param_split=params['param_split'], network=params['network'])
        trainer.initialize(False, num_epochs=0, prev_trainer_path=prev_trainer_path, call_for_eval=True)
        # -- Reset the epoch -- #
        trainer.epoch = epoch
        # -- Remove the Generic_UNet/MH part from the ouptput folder -- #
        if 'nnUNetTrainerV2' in trainer.output_folder:
            fold_ = trainer.output_folder.split(os.path.sep)[-1]
            trainer.output_folder = join(os.path.sep, *trainer.output_folder.split(os.path.sep)[:-3], fold_)

    # -- Set trainer output path -- #
    trainer.output_folder = trainer_path
    os.makedirs(trainer.output_folder, exist_ok=True)
        
    # -- Adapt the already_trained_on with only the prev_trainer part since this is necessary for the validation part -- #
    trainer.already_trained_on[str(t_fold)]['prev_trainer'] = [nnUNetTrainerMultiHead.__name__]*len(params['tasks'])
        
    # -- Set the head based on the users input -- #
    if params['use_head'] is None:
        use_head = list(trainer.mh_network.heads.keys())[-1]
    
    # -- Create a new log_file in the evaluation folder based on changed output_folder -- #
    trainer.print_to_log_file("The {} model trained on {} will be used for this evaluation with the {} head.".format(params['network_trainer'], ', '.join(params['tasks_list_with_char'][0]), params['use_head']))
    trainer.print_to_log_file("The used checkpoint can be found at {}.".format(join(trainer_path, "model_final_checkpoint.model")))
    trainer.print_to_log_file("Start performing evaluation on fold {} for the following tasks: {}.\n".format(t_fold, ', '.join(params['tasks'])))

    # -- Delete all heads except the last one if it is a Sequential Trainer, since then always the last head should be used -- #
    if params['always_use_last_head']:
        # -- Create new heads dict that only contains the last head -- #
        last_name = list(trainer.mh_network.heads.keys())[-1]
        last_head = trainer.mh_network.heads[last_name]
        trainer.mh_network.heads = nn.ModuleDict()
        trainer.mh_network.heads[last_name] = last_head

    # Additional logic from the original function
    all_best_model_files = [join(i, "%s.model" % checkpoint_name) for i in folds]
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params