#########################################################################################################
#----------Alternative entry points without specifying parameters in the console.-----------------------#
#----------See https://github.com/MIC-DKFZ/nnUNet and --------------------------------------------------#
#----------https://github.com/MIC-DKFZ/nnUNet/blob/master/setup.py for details -------------------------#
#########################################################################################################

import sys
import os
import nnunet.experiment_planning.nnUNet_plan_and_preprocess as plan_and_preprocess
import nnunet_ext.nnunet.inference.predict_simple as predict_simple
import nnunet.run.run_training as run_training
from nnunet_ext.utilities.helpful_functions import join_texts_with_char

# 3d_cascade_fullres requires 3d_lowres to be completed, and uses
# trainer nnUNetTrainerV2CascadeFullRes
models = ['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres']


def set_devices(devices=[0]):
    """Set cuda devices"""
    for idx, c in enumerate(devices):
        devices[idx] = str(c)  # Change type from int to str, otherwise join_texts_with_char will throw an error
    cuda = join_texts_with_char(devices, ',')    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

def nnUNet_plan_and_preprocess(args=None, task_id=None, verify_dataset_integrity=True):
    if args is None:
        args = ['']  # To simulate the path to the called file
        args += ['-t', task_id]
        if verify_dataset_integrity:
            args += ['--verify_dataset_integrity']
    sys.argv = args  # Necessary because the main files don't include args=None
    plan_and_preprocess.main()

def nnUNet_predict(args=None, input_path=None, output_path=None, task_id=None,
    model_type='3d_fullres', checkpoint='model_final_checkpoint', fold_ix=0, disable_tta=True,
    extract_outputs=False, mcdo=-1, softmaxed=True):
    if args is None:
        args = ['']
        args += ['-i', input_path]
        args += ['-o', output_path]
        args += ['-t', task_id, '-f', str(fold_ix), '-m', model_type, '-chk', checkpoint]
        if disable_tta:
            args += ['--disable_tta']
        if extract_outputs or mcdo>-1:
            # Instead of predictions, outputs are extracted
            args += ['--output_probabilities']
        if not softmaxed:
            args += ['--no_softmax']  #TODO
        # If mcdo > -1, Dropout is activated and output ix "mcdo" is extracted
        args += ['-mcdo', str(mcdo)]
    sys.argv = args
    predict_simple.main()

def nnUNet_train(args=None, model_type='3d_fullres', trainer='nnUNetTrainerV2', task_id=None, fold_ix=0):
    if args is None:
        args = ['']
        args += [model_type, trainer, task_id, str(fold_ix)]
    sys.argv = args
    run_training.main()
