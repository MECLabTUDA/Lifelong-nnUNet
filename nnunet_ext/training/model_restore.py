#########################################################################################################
#----------This class represents the model_restore module for the nnUNet extension. Implementation------#
#----------inspired by original implementation (--> model_restore), copied code is marked as such.------#
#########################################################################################################

import importlib, pkgutil, nnunet, nnunet_ext
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

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

def restore_model(pkl_file, checkpoint=None, train=False, fp16=True, use_extension=False, extension_type='multihead', del_log=False, param_search=False, network=None, mcdo=-1):
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

    if use_extension and extension_type is not None:    # Only for extensions, with the exception of ViT_U-Net
        assert network is not None, "Please provide the network setting that is used.."
        trainer = tr(*init, network=network)
        trainer.del_log = del_log
        trainer.param_split = param_search
    else:
        trainer = tr(*init)
    # Some trainers do not yet support mcdo
    try:
        if network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']:
            trainer.initialize(train, mcdo=mcdo, network_arch='generic')
        else:
            trainer.initialize(train, mcdo=mcdo, network_arch=network)
    except:
        trainer.initialize(train)

    # -------------------- From nnUNet implementation (modifed, but same output) -------------------- #
    if fp16 is not None:
        trainer.fp16 = fp16
    
    # -- Backup the patch_size before loading the plans -- #
    patch_size = trainer.patch_size
    net_num_pool_op_kernel_sizes = trainer.net_num_pool_op_kernel_sizes

    # -- NOTE: This loads the plan file from the current task! The patch size is high likely to change -- #
    # --       which is why we created a backup to reset the patch size after plans processing --> we -- #
    # --       do not change the plans file since this is based on the data and if we initialize a MH with -- #
    # --       this task, we want it to fit to the initialized data, however for every following trainer, -- #
    # --       the patch size can not change since the model structure is not fit for it and thus fail during -- #
    # --       the forward pass in the skip connections! Keep this in mind -- # 
    trainer.process_plans(info['plans'])
    
    # -- Restore the patch size -- #
    trainer.patch_size = patch_size
    trainer.net_num_pool_op_kernel_sizes = net_num_pool_op_kernel_sizes
    
    if checkpoint is not None:
        # -- Note, when restoring we don't need the old network --> this is only called once training is finished -- #
        trainer.load_checkpoint(checkpoint, train)

    return trainer
    # -------------------- From nnUNet implementation (modifed, but same output) -------------------- #