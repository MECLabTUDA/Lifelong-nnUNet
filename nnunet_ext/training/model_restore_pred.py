#########################################################################################################
# This class represents the model_restore module for the nnUNet extension. Unlike the original 
# implementation, we separate this from model_restore to avoid circular imports.
#########################################################################################################

import torch, os
from nnunet_ext.training.model_restore import restore_model
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.run.default_configuration import get_default_configuration as get_default_configuration_orig

# -- Import the trainer classes -- #
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
import sys

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
    checkpoint = join(trainer_path, "%s.model" % checkpoint_name)
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
        
    # -- Set the head based on the users input -- #
    use_head = params['use_head']
    if params['use_head'] is None:
        use_head = list(trainer.mh_network.heads.keys())[-1]
    
    trainer.network = trainer.mh_network.assemble_model(use_head)

    # -- Set the correct task_name for training -- #
    if trainer.use_vit and trainer.ViT_task_specific_ln:
        trainer.network.ViT.use_task(use_head)

    # -- Create a new log_file in the evaluation folder based on changed output_folder -- #
    trainer.print_to_log_file("The {} model trained on {} will be used for this evaluation with the {} head.".format(params['network_trainer'], ', '.join(params['tasks_list_with_char'][0]), params['use_head']))
    trainer.print_to_log_file("The used checkpoint can be found at {}.".format(join(trainer_path, "model_final_checkpoint.model")))

    # Additional logic from the original function
    all_best_model_files = [join(i, "%s.model" % checkpoint_name) for i in folds]
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params, all_best_model_files