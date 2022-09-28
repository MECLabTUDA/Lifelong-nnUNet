import os
import sys
from nnunet_ext.run.run_inference import run_inference
from nnunet_ext.scripts.update_checkpoints import modify_checkpoints
import nnunet.experiment_planning.nnUNet_plan_and_preprocess as exp_pp

arch = '2d'
fold = '0'
device = '1'
"""
# Prostate
trained_on = [79, 78, 77, 76]
use_model = [79]
use_heads = [79]
trainers = ["nnUNetTrainerSequential"]
evaluate_on_tasks = [79]

"""
# Hyppocampus
trained_on = [34]
use_model = [34]
use_heads = [34]
trainers = ["nnViTUNetTrainer"] #["nnUNetTrainerSequential", "nnUNetTrainerEWC", "nnUNetTrainerLWF", "nnUNetTrainerRehearsal", "nnUNetTrainerMiB", "nnUNetTrainerRW"]
evaluate_on_tasks = [34]


# Adapt checkpoint
# This is always the same path for the stored models, repeat for the new_checkpoints iterating over env variables
current_checkpoint_root = '/local/scratch/aranem/Lifelong-nnUNet-storage/'
old_vars = [os.path.join(current_checkpoint_root, var_name) for var_name in ['nnUNet_raw_data_base', 'nnUNet_preprocessed', 'nnUNet_trained_models', 'nnUNet_models_evaluation']]
new_vars = [os.environ['nnUNet_raw_data_base'], os.environ['nnUNet_preprocessed'], os.environ['RESULTS_FOLDER'], os.environ['EVALUATION_FOLDER']]

# for trainer in trainers:
#     for current_checkpoint, new_checkpoint in zip(old_vars, new_vars):
#         args = [sys.argv[0], arch, trainer, '-trained_on']
#         args += [str(x) for x in trained_on]
#         args += ['-f', fold, '-r', '-rw', current_checkpoint, new_checkpoint, '-use']
#         args += [str(x) for x in use_model]
#         sys.argv = args
#         modify_checkpoints()

# Preprocess relevant datasets, if needed
"""
for evaluate_on in evaluate_on_tasks: 
    sys.argv = [sys.argv[0], '-t', str(evaluate_on)]
exp_pp.main()
sys.argv = [sys.argv[0], '-t', str(trained_on[0])]
exp_pp.main()
"""

# out = '/local/scratch/aranem/Lifelong-nnUNet-storage/CMR_sub_2d_vit_train_data_segs'
model = 'ViT_Voxing_Image_To_BEV_Network_full_nnUNet_ViT'
out = f'/local/scratch/aranem/Lifelong-nnUNet-storage/registration_eval/{model}'
in_ = '/local/scratch/aranem/Lifelong-nnUNet-storage/nnUNet_raw/nnUNet_raw_data/Task034_OASIS/imagesTs'
# out = '/home/aranem_locale/Storage/Lifelong-nnUNet-storage/CMR_sub_2d_vitv3'
# in_ = '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/Lifelong-nnUNet-storage/nnUNet_raw/nnUNet_raw_data/Task041_CMR_val/imagesTs/'
# in_ = '/local/scratch/aranem/Lifelong-nnUNet-storage/nnUNet_raw/nnUNet_raw_data/Task040_CMR/imagesTr'

# Extract predictions
for head in use_heads:
    for trainer in trainers:
        for evaluate_on in evaluate_on_tasks:
            args = [sys.argv[0], arch, trainer, '-trained_on']
            args += [str(x) for x in trained_on]
            args += ['-f', fold, '-use_model']
            args += [str(x) for x in use_model]
            args += ['--no_plans']
            args += ['-reg', 'ViT_Voxing']
            args += ['--use_vit', '-v', '3']
            args += ['-i', in_, '-evaluate_on', str(evaluate_on), '-d', device, '-use_head', str(head), '--enable_tta', '-o', out]
            # print(args)
            sys.argv = args
            run_inference()