import copy
import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plot_colors
import ood_eval_helper
END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/val_metrics_eval.csv"
END_TRAIN = "__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/val_metrics.csv"
from plot_utils import rename_tasks
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 12})


END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/val_metrics_eval.csv"
END_TRAIN = "__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/val_metrics.csv"

combinations = ["Task097_DecathHip",
                "Task097_DecathHip_Task098_Dryad",
                "Task097_DecathHip_Task098_Dryad_Task099_HarP"]

combinations_splitted = ["Task197_DecathHip",
                "Task197_DecathHip_Task198_Dryad",
                "Task197_DecathHip_Task198_Dryad_Task199_HarP"]
METRIC = "Dice"
MASK = 'mask_1'
TASKS = ["Task197_DecathHip",
        "Task198_Dryad",
        "Task199_HarP"]



############### baselines ###############
rehearsal_3d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                'trainer': "nnUNetTrainerRehearsal",
                'name': "Rehearsal"#, 3D, w/ skips
}
sequential_3d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                'trainer': "nnUNetTrainerSequential",
                'name': "Sequential"
}
rehearsal_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                'trainer': "nnUNetTrainerRehearsal",
                'name': "Rehearsal"#, 2D, w/ skips, w/o freezing
}
sequential_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                'trainer': "nnUNetTrainerSequential",
                'name': "Sequential"#, 2D, w/ skips, w/o freezing
}
lwf_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                'trainer': "nnUNetTrainerLWF",
                'name': "LwF"#, 2D, w/ skips, w/o freezing
}













############### ablation on extraction layer ###############
def prostate_gt_1():
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal middle",
                  'code': "between encoder and decoder"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal shallow",#early
                  'code': "middle encoder"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/beginning_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal deep",#late
                  'code': "beginning decoder"
    }
    trainers = [rehearsal_3d, feature_rehearsal3, feature_rehearsal2, feature_rehearsal4, sequential_3d]
    return trainers, "Prostate, ground truth, 1.0"


############### ablation on location of feature extraction and amount of samples stored ###############
def prostate_gt():
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal middle 10\%",
                  'code': "between encoder and decoder, 0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal middle 25\%",
                  'code': "between encoder and decoder, 0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal middle 50\%",
                  'code': "between encoder and decoder, 0.5"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal shallow 10\%",
                  'code': "middle encoder, 0.1"
    }    
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal shallow 25\%",
                  'code': "middle encoder, 0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal shallow 50\%",
                  'code': "middle encoder, 0.5"
    }
    trainers = [rehearsal_3d, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential_3d]
    return trainers, "Prostate, ground truth"








def prostate_seeded():
    rehearsal_seeded_ood = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerRehearsal",
                    'name': "Rehearsal, Softmax",#, 2D, w/ skips, w/o freezing
                    'line_style': (3, 3),
                    'ood':{
                        'evaluator': ood_eval_helper.eval_softmax,
                        'segmentation_trainer': "nnUNetTrainerRehearsal",
                        'trained_on': TASKS,
                        'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                        'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation_ood_checkpoints",
                        'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                        'trainer': "nnUNetTrainerRehearsal",
                        'method': "uncertainty"
                    }
    }
    rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerRehearsal",
                    'name': "Rehearsal",#, 2D, w/ skips, w/o freezing
                    'line_style': (3, 3)
    }
    feature_rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerFeatureRehearsal2",
                    'name': "Feature Rehearsal",#, 2D, w/ skips, w/o freezing
                    'line_style': (3, 3)
    }
    upper_bound = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerRehearsalNoSkipsFrozen",
                    'name': "Rehearsal--",
                    'code': "upper bound",#, 2D, w/o skips, w/ freezing
                    'line_style': (3, 3)
    }
    ccvae_rehearsal_ood = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': r"\textbf{ccVAE (ours)}",#, 2D, w/ skips, w/o freezing
                    'code': "ccVAEr",
                    'ood':{
                        'evaluator': ood_eval_helper.eval_reconstruction,
                        'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                        'trained_on': TASKS,
                        'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                        'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                        'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                        'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                        'method': "vae_reconstruction"
                    }
    }
    ccvae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "CCVAEr"#, 2D, w/o skips, w/ freezing
    }
    ccvae_rehearsal_fixed = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation2",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': r"\textbf{ccVAEr}",#, 2D, w/o skips, w/ freezing
                    'code': "ccVAEr"#, 2D, w/o skips, w/ freezing
    }
    cvae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAEr"#, 2D, w/o skips, w/ freezing
    }
    ewc_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerEWC",
                    'name': "EWC"#, 2D, w/ skips, w/o freezing
    }
    ewc_seeded_softmax = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerEWC",
                    'name': "EWC, Softmax",#, 2D, w/ skips, w/o freezing
                    'code': "EWC",
                    'ood':{
                        'evaluator': ood_eval_helper.eval_softmax,
                        'segmentation_trainer': "nnUNetTrainerEWC",
                        'trained_on': TASKS,
                        'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                        'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation_ood_checkpoints",
                        'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                        'trainer': "nnUNetTrainerEWC",
                        'method': "uncertainty"
                    }
    }
    mib_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerMiB",
                    'name': "MiB"#, 2D, w/ skips, w/o freezing
    }
    mib_seeded_softmax = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerMiB",
                    'name': "MiB, Softmax",#, 2D, w/ skips, w/o freezing
                    'code': "MiB, Softmax",
                    'ood':{
                        'evaluator': ood_eval_helper.eval_softmax,
                        'segmentation_trainer': "nnUNetTrainerMiB",
                        'trained_on': TASKS,
                        'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                        'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation_ood_checkpoints",
                        'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                        'trainer': "nnUNetTrainerMiB",
                        'method': "uncertainty"
                    }
    }


    lwf_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerLWF",
                    'name': "LwF"#, 2D, w/ skips, w/o freezing
    }
    curl = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation2",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerCURL",
                    'name': "cURL"#, 2D, w/ skips, w/o freezing
    }
    sequential_seeded_softmax = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerSequential",
                    'name': "Sequential, Softmax",#, 2D, w/ skips, w/o freezing
                    'code': "Sequential",
                    'ood':{
                        'evaluator': ood_eval_helper.eval_softmax,
                        'segmentation_trainer': "nnUNetTrainerSequential",
                        'trained_on': TASKS,
                        'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                        'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation_ood_checkpoints",
                        'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                        'trainer': "nnUNetTrainerSequential",
                        'method': "uncertainty"
                    }
    }
    
    seg_dist_seeded_softmax = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerSequential",
                    'name': "Sequential, Seg. Dist.",#, 2D, w/ skips, w/o freezing
                    'code': "sequential",
                    'ood':{
                        'evaluator': ood_eval_helper.eval_segmentation_distortion,
                        'segmentation_trainer': "nnUNetTrainerSequential",
                        'trained_on': TASKS,
                        'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                        'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation_ood_checkpoints",
                        'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                        'trainer': "nnUNetTrainerSegDist",
                        'method': "segmentation_distortion_normalized"
                    }
    }

    seg_dist_pool_seeded_softmax = copy.deepcopy(seg_dist_seeded_softmax)
    seg_dist_pool_seeded_softmax['name'] = "Model Pool + Seg. Dist."
    seg_dist_pool_seeded_softmax['ood']['evaluator'] = ood_eval_helper.eval_segmentation_distortion_pool

    vae_rehearsal_no_conditioning_ood = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsNoConditioning",
                    'name': r"VAE",#, 2D, w/ skips, w/o freezing
                    'code': "VAE",
                    'ood':{
                        'evaluator': ood_eval_helper.eval_reconstruction,
                        'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsNoConditioning",
                        'trained_on': TASKS,
                        'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                        'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                        'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                        'trainer': "nnUNetTrainerVAERehearsalNoSkipsNoConditioning",
                        'method': "vae_reconstruction"
                    }
    }


    trainers = [ccvae_rehearsal_ood, sequential_seeded_softmax, mib_seeded_softmax, ewc_seeded_softmax, seg_dist_seeded_softmax, seg_dist_pool_seeded_softmax]
    trainers = [sequential_seeded_softmax, ewc_seeded_softmax, mib_seeded_softmax, seg_dist_pool_seeded_softmax, ccvae_rehearsal_ood, vae_rehearsal_no_conditioning_ood]
    #trainers = [vae_rehearsal_no_conditioning_ood]
    return trainers, "Hippocampus, seeded", combinations_splitted

def hippocampus_seeded():
    rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerRehearsal",
                    'name': "Rehearsal",#, 2D, w/ skips, w/o freezing
                    'line_style': (3, 3)
    }
    feature_rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerFeatureRehearsal2",
                    'code': "Feature Rehearsal",
                    'name': r"Feature rehearsal",#, 2D, w/ skips, w/o freezing
                    'line_style': (3, 3)
    }
    upper_bound = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerRehearsalNoSkipsFrozen",
                    'code': "upper bound",
                    'name': "Rehearsal--",#, 2D, w/o skips, w/ freezing
                    'line_style': (3, 3)
    }
    vae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': r"\textbf{ccVAEr}",#, 2D, w/o skips, w/ freezing
                    'code': "ccVAEr"#, 2D, w/o skips, w/ freezing
    }
    cvae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAEr"#, 2D, w/o skips, w/ freezing
    }
    lwf_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerLWF",
                    'name': "LwF"#, 2D, w/ skips, w/o freezing
    }
    ewc_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerEWC",
                    'name': "EWC"#, 2D, w/ skips, w/o freezing
    }
    mib_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerMiB",
                    'name': "MiB"#, 2D, w/ skips, w/o freezing
    }
    curl = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerCURL",
                    'name': "cURL"#, 2D, w/ skips, w/o freezing
    }
    sequential_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerSequential",
                    'name': "Sequential"#, 2D, w/ skips, w/o freezing
    }

    trainers = [rehearsal_seeded, feature_rehearsal_seeded, upper_bound, vae_rehearsal,cvae_rehearsal, curl, ewc_seeded, mib_seeded, lwf_seeded, sequential_seeded]
    return trainers, "Hippocampus, seeded", combinations_splitted





all_plots = [
    prostate_seeded
]

VISUALIZE_PER_TASK = False

configuration = prostate_seeded
t = configuration()
if len(t) == 3:
    trainers, title, combinations = t
else:
    trainers, title = t


data = []
mask = MASK
metric = METRIC

palette = [] #TODO: add palette

for trainer in trainers:
    #frame = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'], "initialization", trainer['trainer'] + END), sep="\t")
    #frame = frame.drop(frame[frame['metric'] != metric].index)
    #frame = frame.drop(frame[frame['seg_mask'] != mask].index)
    #frame['Epoch'] = 0
    #frame['Trainer'] = trainer['name']
    #frame['Task'] = frame['Task'].apply(rename_tasks)
    #data.append(frame)

    if 'code' in trainer.keys() and trainer['code'] in plot_colors.colors.keys():
        palette.append(plot_colors.colors[trainer['code']])
    elif trainer['name'] in plot_colors.colors.keys():
        palette.append(plot_colors.colors[trainer['name']])
    else:
        print(f"WARNING: trainer {trainer['name']} has no associated color")
        palette.append(list(np.random.choice(range(256), size=3) / 255))
        exit()

    INTERPOLATE = True

    thresholds = []

    for i, task in enumerate(combinations[:-1]):
        #if INTERPOLATE:
        #    if i == 0:
        #        continue

        if INTERPOLATE:
            array = zip([49,99,149,199,250], ["trained_49", "trained_99", "trained_149", "trained_199", "trained_final"])
        else:
            array = zip([250], ["trained_final"])

        for intermediate_train_step, path in array:

            #frame = pd.read_csv(os.path.join(trainer['eval_path_base'], path, trainer['eval_path_middle'], task, trainer['trainer'] + END), sep="\t")
            #frame = frame.drop(frame[frame['metric'] != metric].index)
            #frame = frame.drop(frame[frame['seg_mask'] != mask].index)

            trainer_copy = copy.deepcopy(trainer)
            if trainer['ood']['evaluator'] is not ood_eval_helper.eval_reconstruction:
                trainer_copy['ood']['eval_path_base_ood'] += f"/{path}"

            trainer_copy['ood']['eval_path_base_seg'] += f"/{path}"

            temp, _, threshold_per_task = trainer['ood']['evaluator'](trainer_copy['ood'], i, combinations[i], combinations, TASKS, "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", METRIC, MASK)
            
            
            #temp = temp[temp["Task"] != "Task199_HarP"]
            num_total_cases = len(temp)

            num_total_cases_per_task = dict()
            if VISUALIZE_PER_TASK:
                for task in TASKS:
                        num_total_cases_per_task[task] = len(temp[temp['Task'] == task])


            latest_threshold = threshold_per_task[-1]
            print("\n")
            if INTERPOLATE:
                temp = temp.rename(columns={"case": "subject_id"})
                #temp = temp.drop("value", axis=1)
                #temp = temp.merge(frame, on=["Task", "seg_mask", "metric", "subject_id"])

            if len(thresholds) > 0:
                threshold = max(*thresholds, latest_threshold)
            else:
                threshold = latest_threshold
            
            if intermediate_train_step == 250:
                thresholds.append(latest_threshold)

            frame = temp[temp['ood_score'] < threshold]
            num_cases_segmented = len(frame)
            #print(frame)
            #frame =frame.groupby(['Epoch']).mean().reset_index()
            rel_num_ood = (num_total_cases - num_cases_segmented) / num_total_cases
            
            if VISUALIZE_PER_TASK:
                for task in TASKS:
                    num_cases_segmented_in_task = len(frame[frame['Task'] == task])
                    frame[frame['Task'] == task]['value'] = .7 *frame[frame['Task'] == task]['value'] + .3 * num_cases_segmented_in_task / num_total_cases_per_task[task]  
            else:
                #frame['value'] = .7 *frame['value'] + .3 * num_cases_segmented / num_total_cases
                frame['value'] = 100*frame['value']# + .3 * num_cases_segmented / num_total_cases


            #print(frame)

            if False:                   #-> start after training on that repspective task
                b = [x in task for x in frame['Task']]
                assert(len(b) == len(frame))
                frame= frame[b]

            frame['Epoch'] = i * 250 + intermediate_train_step
            frame['Trainer'] = trainer['name']
            #frame['Task'] = frame['Task'].apply(rename_tasks)
            data.append(frame)
            #print(frame)

    

data = pd.concat(data)
#data['value'] = data["value"].apply(lambda x: 1-x)
data = data.rename(columns={"value": metric})
print(data)

if not VISUALIZE_PER_TASK:
    ax = sns.lineplot(
        data=data,
        x="Epoch", 
        y=metric,
        errorbar=None,
        #err_style="bars",
        hue="Trainer",
        marker="X",
        palette=palette,
        linewidth=3,
    )
    ax.tick_params(labelbottom=True)
    ax.grid(True)
    ax.set_xlabel("Epoch", visible=True, fontsize=15)
    ax.set_ylabel(r"Dice", visible=True, fontsize=15)
    ax.set_xticks([0, 250, 500])
else:
    # if select Dice from data where Epoch == $epoch and Trainer == $trainer and Task == $task is empty, then insert 0
    for trainer in trainers:
        trainer = trainer['name']
        for task in TASKS:
            for epoch in range(49, 500, 50):
                if epoch % 250 == 249:
                    epoch += 1
                if len(data[(data['Epoch'] == epoch) & (data['Trainer'] == trainer) & (data['Task'] == task)])==0:
                    data = data.append({'Epoch': epoch, 'Trainer': trainer, 'Task': task, metric: 0}, ignore_index=True)

    data['Task'] = data['Task'].apply(rename_tasks)
    ax = sns.relplot(
        data=data,
        x="Epoch", y=metric,
        hue="Trainer", #size="choice", 
        col="Task",
        kind="line", 
        #size_order=["T1", "T2"], 
        palette=palette,
        height=4, aspect=1, facet_kws=dict(sharex=True),
        errorbar=None, 
        col_wrap=3,
        marker="X",

        style="Trainer",
        dashes=[trainer['line_style'] if 'line_style' in trainer.keys() else (1, 0) for trainer in trainers],
        linewidth=3,
    )
    plt.subplots_adjust(hspace=0.2)
    for i, t in enumerate(ax.axes):
        t.tick_params(labelbottom=True)
        t.grid(True)
        t.set_xlabel("Epoch", visible=True)
        t.set_ylabel(r"\textbf{DiceID}", visible=True)
        t.set_xticks([0, 250, 500])
        t.get_xgridlines()[i].set_color('black')
        t.get_xgridlines()[i].set_linewidth(2)
    last_bbox = ax.axes[-1].get_tightbbox(for_layout_only=True)
    last_pos = ax.axes[-1].get_position()
    last_pos2 = ax.axes[-2].get_position()
    x = ax.axes[0].get_position().x1
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(last_pos2.x0, last_pos.y1), frameon=True)

#ax = sns.relplot(
#    data=data,
#    x="Epoch", y=metric,
#    hue="Trainer", #size="choice", 
#    col="Task",
#    kind="line", 
#    #size_order=["T1", "T2"], 
#    palette=palette,
#    height=4, aspect=1, facet_kws=dict(sharex=True),
#    errorbar=None, 
#    col_wrap=3,
#
#    style="Trainer",
#    dashes=[trainer['line_style'] if 'line_style' in trainer.keys() else (1, 0) for trainer in trainers],
#)

#ax.get_xgridlines()[i].set_color('black')
#ax.get_xgridlines()[i].set_linewidth(2)
#t.get_lines()[0].set_linestyle("--")


#sns.move_legend(ax, loc="upper left", frameon=True)

#plt.yscale("log")
#plt.subplots_adjust(hspace=0.22)
#plt.yticks([0.1, 0.2,0.4,0.6,0.8, 1])
#plt.gca().set_yticklabels([90, 80, 60, 40, 20, 0])
#plt.gca().invert_yaxis()

#plt.legend()
#ax.get_legend().remove()
plt.savefig(f"plot_hippocampus_with_ood.pdf", bbox_inches='tight')
print(f"plot_hippocampus_with_ood.pdf")
#plt.savefig(f"plots/continual_learning/prostate/{configuration.__name__}.pdf", bbox_inches='tight')
#print(f"plots/continual_learning/prostate/{configuration.__name__}.pdf")