import os, sys
from nnunet_ext.run.run_inference import run_inference


# Adapt checkpoint
# This is always the same path for the stored models, repeat for the new_checkpoints iterating over env variables
current_checkpoint_root = '/local/scratch/aranem/Lifelong-nnUNet-storage/'
old_vars = [os.path.join(current_checkpoint_root, var_name) for var_name in ['nnUNet_raw_data_base', 'nnUNet_preprocessed', 'nnUNet_trained_models', 'nnUNet_models_evaluation']]
new_vars = [os.environ['nnUNet_raw_data_base'], os.environ['nnUNet_preprocessed'], os.environ['RESULTS_FOLDER'], os.environ['EVALUATION_FOLDER']]


model = 'Voxel_UNet_ViT_12b_12msa_2x2_attnscale'
arch = '2d'
fold = '0'
device = '0'

# OASIS
# trained_on = [34]
# use_model = [34]
# use_heads = [34]
# trainers = ["nnViTUNetTrainer"]
# evaluate_on_tasks = [34]
# out = f'/local/scratch/aranem/Lifelong-nnUNet-storage/IPMI_2023/IPMI_2023_evaluation/Task034_OASIS/{model}'
# in_ = '/local/scratch/aranem/Lifelong-nnUNet-storage/nnUNet_raw/nnUNet_raw_data/Task034_OASIS/imagesTs'


# AbdomenMRCT
trained_on = [30]
use_model = [30]
use_heads = [30]
trainers = ["nnViTUNetTrainer"]
evaluate_on_tasks = [30]
out = f'/local/scratch/aranem/Lifelong-nnUNet-storage/IPMI_2023/IPMI_2023_evaluation/Task030_AbdomenMRCT/{model}'
in_ = '/local/scratch/aranem/Lifelong-nnUNet-storage/nnUNet_raw/nnUNet_raw_data/Task030_AbdomenMRCT/imagesTs'


# Extract predictions
for head in use_heads:
    for trainer in trainers:
        for evaluate_on in evaluate_on_tasks:
            args = [sys.argv[0], arch, trainer, '-trained_on']
            args += [str(x) for x in trained_on]
            args += ['-f', fold, '-use_model']
            args += [str(x) for x in use_model]
            args += ['--no_plans']
            args += ['-reg', 'VoxelMorph_ViT']
            # args += ['-reg', 'VoxelMorph']
            args += ['-i', in_, '-evaluate_on', str(evaluate_on), '-d', device, '-use_head', str(head), '--enable_tta', '-o', out]
            sys.argv = args
            run_inference()