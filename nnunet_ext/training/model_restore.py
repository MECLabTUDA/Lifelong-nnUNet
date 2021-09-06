#########################################################################################################
#----------This class represents the model_restore module for the nnUNet extension. Implementation------#
#----------inspired by original implementation (--> model_restore), copied code is marked as such.------#
#########################################################################################################

import nnunet, nnunet_ext
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

def restore_model(pkl_file, checkpoint=None, train=False, fp16=True, use_extension=False, extension_type='multihead'):
    """ This function is modified to work for the nnU-Net extension. When using this to restore Multi head Network always
        set train to True.
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
    trainer = tr(*init)
    trainer.initialize(train)

    # -------------------- From nnUNet implementation (modifed, but same output) -------------------- #
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer
    # -------------------- From nnUNet implementation (modifed, but same output) -------------------- #