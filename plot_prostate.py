import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plot_colors
END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/val_metrics_eval.csv"
END_TRAIN = "__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/val_metrics.csv"
from plot_utils import rename_tasks
plt.rcParams['text.usetex'] = True

combinations = ["Task011_Prostate-BIDMC",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]

combinations_splitted = ["Task111_Prostate-BIDMC",
                    "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB",
                    "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK",
                    "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL",
                    "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC"]
############### baselines ###############
rehearsal_3d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                'trainer': "nnUNetTrainerRehearsal",
                'name': "Rehearsal"
}
sequential_3d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                'trainer': "nnUNetTrainerSequential",
                'name': "Sequential"
}
rehearsal_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/2d/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                'trainer': "nnUNetTrainerRehearsal",
                'name': "Rehearsal"
}
sequential_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/2d/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                'trainer': "nnUNetTrainerSequential",
                'name': "Sequential"
}
lwf_2d = None

rehearsal_2d_no_skips_freeze = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/evaluation/",
                'eval_path_middle': "nnUNet_ext/2d/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                'trainer': "nnUNetTrainerRehearsalNoSkipsFrozen",
                'name': "Rehearsal, 2D, w/o skips, w/ freezing"
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


############### ablation on amount of samples stored ###############
def prostate_gt_encoder_decoder():
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.1",
                  'code': "between encoder and decoder, 0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.25",
                  'code': "between encoder and decoder, 0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.5",
                  'code': "between encoder and decoder, 0.5"
    }
    trainers = [rehearsal_3d, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, sequential_3d]
    return trainers, "Prostate, ground truth, between encoder and decoder"


############### ablation on amount of samples stored ###############
def prostate_gt_middle_encoder():
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.1",
                  'code': "middle encoder, 0.1"
    }    
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.25",
                  'code': "middle encoder, 0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.5",
                  'code': "middle encoder, 0.5"
    }
    trainers = [rehearsal_3d, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential_3d]
    return trainers, "Prostate, ground truth, middle encoder"


############### ablation on target method ###############
def prostate_middle_encoder_01():
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal ground truth",
                  'code': "middle encoder, ground truth"
    } 
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal distilled output",
                  'code': "middle encoder, distilled output"
    }
    trainers = [rehearsal_3d, feature_rehearsal1, feature_rehearsal2, sequential_3d]
    return trainers, "Prostate, middle encoder, 0.1"

def prostate_middle_encoder_025():
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "ground truth",
                  'code': "middle encoder, ground truth"
    } 
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "distilled output",
                  'code': "middle encoder, distilled output"
    } 
    trainers = [rehearsal_3d, feature_rehearsal1, feature_rehearsal2, sequential_3d]
    return trainers, "Prostate, middle encoder, 0.25"






############### ablation on VAE rehearsal methods ###############
def prostate_vae_rehearsal_2d_no_skips():
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal, distilled output, 3D, w/ skips, w/ freezing"
    }
    vae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips2/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing"
    }
    vae_rehearsal_double_conditional = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                  'name': "CCVAEr"
    }

    trainers = [rehearsal_2d, feature_rehearsal2, vae_rehearsal, vae_rehearsal_double_conditional, sequential_2d, rehearsal_2d_no_skips_freeze]
    #trainers = [vae_rehearsal, vae_rehearsal_double_conditional]
    return trainers, "Prostate"






def prostate_seeded():
    rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerRehearsal",
                    'name': "Rehearsal",#, 2D, w/ skips, w/o freezing
                    'line_style': (3, 3)
    }
    feature_rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerFeatureRehearsal2",
                    'name': "Feature Rehearsal",#, 2D, w/ skips, w/o freezing
                    'line_style': (2, 2)
    }
    upper_bound = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerRehearsalNoSkipsFrozen",
                    'name': "Rehearsal--",
                    'code': "upper bound",#, 2D, w/o skips, w/ freezing
                    'line_style': (3, 3)
    }
    ccvae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "CCVAEr"#, 2D, w/o skips, w/ freezing
    }
    ccvae_rehearsal_fixed = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation2",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': r"\textbf{ccVAEr}",#, 2D, w/o skips, w/ freezing
                    'code': "ccVAEr"#, 2D, w/o skips, w/ freezing
    }
    cvae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAEr"#, 2D, w/o skips, w/ freezing
    }
    ewc_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerEWC",
                    'name': "EWC"#, 2D, w/ skips, w/o freezing
    }
    mib_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerMiB",
                    'name': "MiB"#, 2D, w/ skips, w/o freezing
    }
    lwf_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerLWF",
                    'name': "LwF"#, 2D, w/ skips, w/o freezing
    }
    curl = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation2",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerCURL",
                    'name': "cURL"#, 2D, w/ skips, w/o freezing
    }
    sequential_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trainer': "nnUNetTrainerSequential",
                    'name': "Sequential"#, 2D, w/ skips, w/o freezing
    }

    #trainers = [rehearsal_seeded, feature_rehearsal_seeded, upper_bound, ccvae_rehearsal_fixed, cvae_rehearsal, curl, ewc_seeded, mib_seeded, lwf_seeded, sequential_seeded]
    #trainers = [feature_rehearsal_seeded, ccvae_rehearsal_fixed, ewc_seeded, mib_seeded, sequential_seeded]
    trainers = [sequential_seeded, ewc_seeded, mib_seeded, ccvae_rehearsal_fixed]
    return trainers, "Hippocampus, seeded", combinations_splitted






all_plots = [
    prostate_gt_1,
    prostate_gt,
    prostate_gt_encoder_decoder,
    prostate_gt_middle_encoder,
    prostate_middle_encoder_01,
    prostate_middle_encoder_025,
    prostate_vae_rehearsal_2d_no_skips,
    prostate_seeded
]


configuration = prostate_seeded
t = configuration()
if len(t) == 3:
    trainers, title, combinations = t
else:
    trainers, title = t


data = []
mask = "mask_1"
metric = 'Dice'

palette = [] #TODO: add palette
ONLY_ON_TRAINED = False

for trainer in trainers:
    frame = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'], "initialization", trainer['trainer'] + END), sep="\t")
    frame = frame.drop(frame[frame['metric'] != metric].index)
    frame = frame.drop(frame[frame['seg_mask'] != mask].index)
    frame['Epoch'] = 0
    frame['Trainer'] = trainer['name']
    frame['Task'] = frame['Task'].apply(rename_tasks)
    if not ONLY_ON_TRAINED:
        data.append(frame)

    if 'code' in trainer.keys() and trainer['code'] in plot_colors.colors.keys():
        palette.append(plot_colors.colors[trainer['code']])
    elif trainer['name'] in plot_colors.colors.keys():
        palette.append(plot_colors.colors[trainer['name']])
    else:
        print(f"WARNING: trainer {trainer['name']} has no associated color")
        palette.append(list(np.random.choice(range(256), size=3) / 255))

    for i, task in enumerate(combinations):
        for intermediate_train_step, path in zip([49,99,149,199,250], ["trained_49", "trained_99", "trained_149", "trained_199", "trained_final"]):

            frame = pd.read_csv(os.path.join(trainer['eval_path_base'], path, trainer['eval_path_middle'], task, trainer['trainer'] + END), sep="\t")
            frame = frame.drop(frame[frame['metric'] != metric].index)
            frame = frame.drop(frame[frame['seg_mask'] != mask].index)
            
            if ONLY_ON_TRAINED:                   #-> start after training on that repspective task
                b = [x in task for x in frame['Task']]
                assert(len(b) == len(frame))
                frame= frame[b]

            frame['Epoch'] = i * 250 + intermediate_train_step
            frame['Trainer'] = trainer['name']
            frame['Task'] = frame['Task'].apply(rename_tasks)
            data.append(frame)
            #print(frame)

    
ALWAYS_FROM_ZERO = True

data = pd.concat(data)
data['value'] = data["value"].apply(lambda x: 1-x)
data = data.rename(columns={"value": metric})

ax = sns.relplot(
    data=data,
    x="Epoch", y=metric,
    hue="Trainer", #size="choice", 
    col="Task",
    kind="line", 
    #size_order=["T1", "T2"], 
    palette=palette,
    height=4, aspect=1, facet_kws=dict(sharex=ALWAYS_FROM_ZERO),
    errorbar=None, 
    col_wrap=3,
    linewidth=3,
    marker="X",

    style="Trainer",
    dashes=[trainer['line_style'] if 'line_style' in trainer.keys() else (1, 0) for trainer in trainers],
)

for i, t in enumerate(ax.axes):
    t.tick_params(labelbottom=True)
    t.grid(True)
    t.set_xlabel("Epoch", visible=True)
    if ALWAYS_FROM_ZERO:
        t.set_xticks([0, 250, 500, 750, 1000, 1250])
        t.get_xgridlines()[i].set_color('black')
        t.get_xgridlines()[i].set_linewidth(2)
    else:
        t.set_xticks([0, 250, 500, 750, 1000][i:])
        t.set_xlim([[0, 250, 500, 750, 1000, 1250][i], 1250])



last_bbox = ax.axes[-1].get_tightbbox(for_layout_only=True)
last_pos = ax.axes[-1].get_position()
last_pos2 = ax.axes[-3].get_position()
x = ax.axes[0].get_position().x1
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(last_pos2.x0, last_pos.y1), frameon=True)

plt.yscale("log")
plt.subplots_adjust(hspace=0.22)
plt.yticks([0.1, 0.2,0.4,0.6,0.8, 1])
plt.gca().set_yticklabels([90, 80, 60, 40, 20, 0])
plt.gca().invert_yaxis()
#ax.fig.suptitle(title, fontsize=16)
#ax.fig.subplots_adjust(top=0.9)
plt.savefig(f"plots/continual_learning/prostate/{configuration.__name__}.svg", bbox_inches='tight')
print(f"plots/continual_learning/prostate/{configuration.__name__}.pdf")