#########################################################################################################
#----------Alternative entry points without specifying parameters in the console.-----------------------#
#----------See https://github.com/MIC-DKFZ/nnUNet and --------------------------------------------------#
#----------https://github.com/MIC-DKFZ/nnUNet/blob/master/setup.py for details -------------------------#
#########################################################################################################

import sys
import nnunet.experiment_planning.nnUNet_plan_and_preprocess as plan_and_preprocess
import nnunet.inference.predict_simple as predict_simple
import nnunet.run.run_training as run_training

# 3d_cascade_fullres requires 3d_lowres to be completed, and uses
# trainer nnUNetTrainerV2CascadeFullRes
models = ['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres']

def nnUNet_plan_and_preprocess(args=None, task_id=None, verify_dataset_integrity=True):
    if args is None:
        args = ['']  # To simulate the path to the called file
        args += ['-t', task_id]
        if verify_dataset_integrity:
            args += ['--verify_dataset_integrity']
    sys.argv = args  # Necessary because the main files don't include args=None
    plan_and_preprocess.main()

def nnUNet_predict(args=None, input_path=None, output_path=None, task_id=None,
    model_type='3d_fullres', checkpoint='model_final_checkpoint', fold_ix=0, disable_tta=True):
    if args is None:
        args = ['']
        args += ['-i', input_path]
        args += ['-o', output_path]
        args += ['-t', task_id, '-f', fold_ix, '-m', model_type, '-chk', checkpoint]
        if disable_tta:
            args += ['--disable_tta']
    sys.argv = args
    predict_simple.main()

def nnUNet_train(args=None, model_type='3d_fullres', trainer='nnUNetTrainerV2', task_id=None, fold_ix=0):
    if args is None:
        args = ['']
        args += [model_type, trainer, task_id, fold_ix]
    sys.argv = args
    run_training.main()
