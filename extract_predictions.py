import os
import sys
from nnunet_ext.run.run_inference import run_inference
from nnunet_ext.scripts.update_checkpoints import modify_checkpoints
import nnunet.experiment_planning.nnUNet_plan_and_preprocess as exp_pp

arch = '3d_fullres'
fold = '0'
device = '0'
"""
# Prostate
trained_on = [79, 78, 77, 76]
use_model = [79]
use_heads = [79]
trainers = ["nnUNetTrainerSequential"]
evaluate_on_tasks = [79]

"""
# Hyppocampus
trained_on = [99, 98, 97]
use_model = [99]
use_heads = [99]
trainers = ["nnUNetTrainerSequential"]#, "nnUNetTrainerEWC", "nnUNetTrainerLWF", "nnUNetTrainerRehearsal", "nnUNetTrainerMiB", "nnUNetTrainerRW"]
evaluate_on_tasks = [99]


# Adapt checkpoint
# This is always the same path for the stored models, repeat for the new_checkpoints iterating over env variables
current_checkpoint_root = '/local/scratch/aranem/Lifelong-nnUNet-storage/'
old_vars = [os.path.join(current_checkpoint_root, var_name) for var_name in ['nnUNet_raw_data_base', 'nnUNet_preprocessed', 'nnUNet_trained_models', 'nnUNet_models_evaluation']]
new_vars = [os.environ['nnUNet_raw_data_base'], os.environ['nnUNet_preprocessed'], os.environ['RESULTS_FOLDER'], os.environ['EVALUATION_FOLDER']]
for trainer in trainers:
    for current_checkpoint, new_checkpoint in zip(old_vars, new_vars):
        args = [sys.argv[0], arch, trainer, '-trained_on']
        args += [str(x) for x in trained_on]
        args += ['-f', fold, '-r', '-rw', current_checkpoint, new_checkpoint, '-use']
        args += [str(x) for x in use_model]
        sys.argv = args
        modify_checkpoints()

# Preprocess relevant datasets, if needed
"""
for evaluate_on in evaluate_on_tasks: 
    sys.argv = [sys.argv[0], '-t', str(evaluate_on)]
exp_pp.main()
sys.argv = [sys.argv[0], '-t', str(trained_on[0])]
exp_pp.main()
"""

# Extract predictions
for head in use_heads:
    for trainer in trainers:
        for evaluate_on in evaluate_on_tasks:
            args = [sys.argv[0], arch, trainer, '-trained_on']
            args += [str(x) for x in trained_on]
            args += ['-f', fold, '-use_model']
            args += [str(x) for x in use_model]
            args += ['-evaluate_on', str(evaluate_on), '-d', device, '-use_head', str(head), '--enable_tta']
            # print(args)
            sys.argv = args
            run_inference()