import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/val_metrics_eval.csv"


def hippocampus_gt_1():
    rehearsal_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    config1 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/first_layer/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    config2 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/between_encoder_decoder/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    config3 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/middle_encoder/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    config4 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/beginning_decoder/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    
    
    
    sequential_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    combinations = ["initialization",
                    "Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path': rehearsal_evaluation_path,
    'trainer': "nnUNetTrainerRehearsal",
    'name': "Rehearsal"}
    feature_rehearsal1 = {'eval_path': config1,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(1)"}
    feature_rehearsal2 = {'eval_path': config2,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(2)"}
    feature_rehearsal3 = {'eval_path': config3,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(3)"}
    feature_rehearsal4 = {'eval_path': config4,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(4)"}
    sequential = {'eval_path': sequential_evaluation_path,
    'trainer': "nnUNetTrainerSequential",
    'name': "Sequential"}
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, sequential]
    return trainers, combinations


def cardiac_gt_1():
    # mask_1: left ventricle
    # mask_2: myocardium
    # mask_3: right ventricle


    rehearsal_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    config1 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/first_layer/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    config2 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/between_encoder_decoder/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    config3 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/middle_encoder/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    config4 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/beginning_decoder/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    
    
    sequential_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    combinations = ["initialization",
                    "Task008_mHeartA",
                    "Task008_mHeartA_Task009_mHeartB"]
    rehearsal = {'eval_path': rehearsal_evaluation_path,
    'trainer': "nnUNetTrainerRehearsal",
    'name': "Rehearsal"}
    feature_rehearsal1 = {'eval_path': config1,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(1)"}
    feature_rehearsal2 = {'eval_path': config2,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(2)"}
    feature_rehearsal3 = {'eval_path': config3,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(3)"}
    feature_rehearsal4 = {'eval_path': config4,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(4)"}
    sequential = {'eval_path': sequential_evaluation_path,
    'trainer': "nnUNetTrainerSequential",
    'name': "Sequential"}
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, sequential]
    return trainers, combinations




def prostate_gt_1():
    rehearsal_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    config1 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/first_layer/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    config2 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/between_encoder_decoder/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    config3 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/middle_encoder/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    config4 = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/beginning_decoder/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    
    
    sequential_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    combinations = ["initialization",
                    "Task011_Prostate-BIDMC",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]
    rehearsal = {'eval_path': rehearsal_evaluation_path,
    'trainer': "nnUNetTrainerRehearsal",
    'name': "Rehearsal"}
    feature_rehearsal1 = {'eval_path': config1,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(1)"}
    feature_rehearsal2 = {'eval_path': config2,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(2)"}
    feature_rehearsal3 = {'eval_path': config3,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(3)"}
    feature_rehearsal4 = {'eval_path': config4,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "(4)"}
    sequential = {'eval_path': sequential_evaluation_path,
    'trainer': "nnUNetTrainerSequential",
    'name': "Sequential"}
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, sequential]
    return trainers, combinations
























trainers, combinations = prostate_gt_1()
data = []
## data needs to have ["case_name", "last task trained", "task the case belongs to", "value"]
for i, task in enumerate(combinations):
    for trainer in trainers:
        frame = pd.read_csv(os.path.join(trainer['eval_path'], task, trainer['trainer'] + END), sep="\t")
        frame = frame.drop(frame[frame['metric'] != 'Dice'].index)
        frame = frame.drop(frame[frame['seg_mask'] != "mask_1"].index)
        frame['Epoch'] = i * 250
        frame['Trainer'] = trainer['name']
        data.append(frame)
        print(frame)

data = pd.concat(data)
data = data.rename(columns={"value": "Dice"})


ax = sns.lineplot(x="Epoch", y="Dice",
             hue="Trainer", 
             style="Task",
             data=data, 
             errorbar=None
             )

plt.xticks(np.arange(0, len(combinations) * 250, 250))
plt.yticks(np.arange(0, 1.2, 0.2))


sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))




plt.savefig("plot.png", bbox_inches='tight')

