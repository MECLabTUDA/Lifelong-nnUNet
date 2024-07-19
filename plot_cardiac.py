import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plot_colors
from plot_utils import rename_tasks, convert_epoch_string_to_int
plt.rcParams['text.usetex'] = True



END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/val_metrics_eval.csv"
END_TRAIN = "__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/val_metrics.csv"

combinations = ["Task008_mHeartA",
                "Task008_mHeartA_Task009_mHeartB"]

############### baselines ###############
rehearsal_3d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                'trainer': "nnUNetTrainerRehearsal",
                'name': "Rehearsal"#, 3D, w/ skips
}
sequential_3d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                'trainer': "nnUNetTrainerSequential",
                'name': "Sequential"
}
rehearsal_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/2d/Task008_mHeartA_Task009_mHeartB",
                'trainer': "nnUNetTrainerRehearsal",
                'name': "Rehearsal"#, 2D, w/ skips, w/o freezing
}
sequential_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/2d/Task008_mHeartA_Task009_mHeartB",
                'trainer': "nnUNetTrainerSequential",
                'name': "Sequential"#, 2D, w/ skips, w/o freezing
}
lwf_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                'eval_path_middle': "nnUNet_ext/2d/Task008_mHeartA_Task009_mHeartB",
                'trainer': "nnUNetTrainerLWF",
                'name': "LwF"#, 2D, w/ skips, w/o freezing
}



############### ablation on location of feature extraction ###############
def cardiac_gt_1():
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal middle",
                  'code': "between encoder and decoder"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal shallow",#early
                  'code': "middle encoder"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/beginning_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal deep",#late
                  'code': "beginning decoder"
    }
    trainers = [rehearsal_3d, feature_rehearsal3, feature_rehearsal2, feature_rehearsal4, sequential_3d]
    return trainers, "cardiac, ground truth, 1.0"


############### ablation on location of feature extraction and amount of samples stored ###############
def cardiac_gt():
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal middle 10\%",
                  'code': "between encoder and decoder, 0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal middle 25\%",
                  'code': "between encoder and decoder, 0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal middle 50\%",
                  'code': "between encoder and decoder, 0.5"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal shallow 10\%",
                  'code': "middle encoder, 0.1"
    }
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal shallow 25\%",
                  'code': "middle encoder, 0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"Feature rehearsal shallow 50\%",
                  'code': "middle encoder, 0.5"
    }
    trainers = [rehearsal_3d, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential_3d]
    return trainers, "cardiac, ground truth"


############### ablation on amount of samples stored ###############
def cardiac_gt_encoder_decoder():
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"10\%",
                  'code': "between encoder and decoder, 0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"25\%",
                  'code': "between encoder and decoder, 0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"50\%",
                  'code': "between encoder and decoder, 0.5"
    }
    trainers = [rehearsal_3d, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, sequential_3d]
    return trainers, "cardiac, ground truth, between encoder and decoder"


############### ablation on amount of samples stored ###############
def cardiac_gt_middle_encoder():
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"10\%",
                  'code': "middle encoder, 0.1"
    }
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"25\%",
                  'code': "middle encoder, 0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': r"50\%",
                  'code': "middle encoder, 0.5"
    }
    trainers = [rehearsal_3d, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential_3d]
    return trainers, "cardiac, ground truth, middle encoder"


############### ablation on target method ###############
def cardiac_middle_encoder_01():
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal ground truth",
                  'code': "middle encoder, ground truth"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal distilled output",
                  'code': "middle encoder, distilled output"
    }
    trainers = [rehearsal_3d, feature_rehearsal1, feature_rehearsal2, sequential_3d]
    return trainers, "cardiac, middle encoder, 0.1"

def cardiac_middle_encoder_025():
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal ground truth",
                  'code': "middle encoder, ground truth"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal distilled output",
                  'code': "middle encoder, distilled output"
    }
    trainers = [rehearsal_3d, feature_rehearsal1, feature_rehearsal2, sequential_3d]
    return trainers, "cardiac, middle encoder, 0.25"




############### ablation on VAE rehearsal methods ###############
def hippocampus_vae_rehearsal_2d_no_skips():
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal, distilled output, 3D, w/ skips, w/ freezing"
    }
    feature_rehearsal_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal, distilled output, 2D, w/ skips, w/ freezing"
    }
    feature_rehearsal_2d_no_skips = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsalNoSkips",
                  'name': "Feature rehearsal, distilled output, 2D, w/o skips, w/ freezing"
    }
    vae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing"
    }
    vae_rehearsal_2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips2/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing 2nd version"
    }
    vae_rehearsal_no_conditioning = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_conditional/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing, w/o conditioning"
    }
    vae_rehearsal_no_conditioning_2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkipsNoConditioning",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing, w/o conditioning 2nd version"
    }
    vae_rehearsal_large = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit",
                  'name': "CVAEr"#VAE rehearsal, 2D, w/o skips, w/ freezing, large VAE, force reinit
    }
    vae_rehearsal_large_double_conditional = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                  'name': "CCVAEr"#VAE rehearsal, 2D, w/o skips, w/ freezing, large VAE, double conditional
    }
    
    #trainers = [rehearsal_2d, feature_rehearsal_2d_no_skips, vae_rehearsal_large_double_conditional, sequential]
    trainers = [feature_rehearsal_2d_no_skips, vae_rehearsal_large, vae_rehearsal_large_double_conditional, sequential_2d, lwf_2d]
    trainers = [rehearsal_2d, feature_rehearsal_2d_no_skips, vae_rehearsal_large, vae_rehearsal_large_double_conditional, sequential_2d]
    return trainers, "Hippocampus"

def hippocampus_seeded():
    rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerRehearsal",
                    'name': "Rehearsal"#, 2D, w/ skips, w/o freezing
    }
    feature_rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerFeatureRehearsal2",
                    'code': "Feature Rehearsal",
                    'name': r"$z$-rehearsal"#, 2D, w/ skips, w/o freezing
    }
    upper_bound = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerRehearsalNoSkipsFrozen",
                    'code': "upper bound",
                    'name': "Rehearsal--"#, 2D, w/o skips, w/ freezing
    }
    vae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "CCVAEr"#, 2D, w/o skips, w/ freezing
    }
    cvae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                    'name': "CVAEr"#, 2D, w/o skips, w/ freezing
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
    return trainers, "Hippocampus, seeded", []



all_plots = [
    cardiac_gt_1,
    cardiac_gt,
    cardiac_gt_encoder_decoder,
    cardiac_gt_middle_encoder,
    cardiac_middle_encoder_01,
    cardiac_middle_encoder_025
]


configuration = cardiac_gt
t = configuration()
if len(t) == 3:
    trainers, title, combinations = t
else:
    trainers, title = t
    
data = []
mask = "mask_3"
metric = 'Dice'

palette = [] #TODO: add palette

for trainer in trainers:
    frame = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'], "initialization", trainer['trainer'] + END), sep="\t")
    frame = frame.drop(frame[frame['metric'] != metric].index)
    frame = frame.drop(frame[frame['seg_mask'] != mask].index)
    frame['Epoch'] = 0
    frame['Trainer'] = trainer['name']
    frame['Task'] = frame['Task'].apply(rename_tasks)
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

            if False:                   #-> start after training on that repspective task
                b = [x in task for x in frame['Task']]
                assert(len(b) == len(frame))
                frame= frame[b]

            frame['Epoch'] = i * 250 + intermediate_train_step
            frame['Trainer'] = trainer['name']
            frame['Task'] = frame['Task'].apply(rename_tasks)
            data.append(frame)
            #print(frame)

    

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
    height=4, aspect=1, facet_kws=dict(sharex=True),
    errorbar=None, 
    col_wrap=2,
    linewidth=1.5
)

for i, t in enumerate(ax.axes):
    t.tick_params(labelbottom=True)
    t.grid(True)
    t.set_xlabel("Epoch", visible=True)
    t.set_xticks([0, 250, 500])
    t.get_xgridlines()[i].set_color('black')
    t.get_xgridlines()[i].set_linewidth(2)

last_bbox = ax.axes[-1].get_tightbbox(for_layout_only=True)
last_pos = ax.axes[-1].get_position()
last_pos2 = ax.axes[-2].get_position()
x = ax.axes[0].get_position().x1
#sns.move_legend(ax, loc="upper left", bbox_to_anchor=(last_pos2.x0, last_pos.y1), frameon=True)
#sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 0),
#          fancybox=True, shadow=False, ncol=3, frameon=True)

#sns.move_legend(ax, bbox_to_anchor=(last_pos2.x0, last_pos.y1 + 0.06, last_pos.x1 - last_pos2.x0, 0.2), loc="lower left",
#                mode="expand", borderaxespad=0, ncol=3, frameon=True)
if len(trainers) in [1,2,3]:
    assert False
elif len(trainers) in [4,5,6]:
    sns.move_legend(ax, bbox_to_anchor=(last_pos2.x0, last_pos.y0 - 0.2 - 0.06, last_pos.x1 - last_pos2.x0, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=3, frameon=True)
elif len(trainers) in [7,8,9]:
    sns.move_legend(ax, bbox_to_anchor=(last_pos2.x0, last_pos.y0 - 0.25 - 0.06, last_pos.x1 - last_pos2.x0, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=3, frameon=True)
else:
    assert False, f"len(trainers) = {len(trainers)} not supported"

plt.yscale("log")
plt.subplots_adjust(hspace=0.22)
plt.yticks([0.1, 0.2,0.4,0.6,0.8, 1])
plt.gca().set_yticklabels([90, 80, 60, 40, 20, 0])
plt.gca().invert_yaxis()
#ax.fig.suptitle(title, fontsize=16)
#ax.fig.subplots_adjust(top=0.9)
plt.savefig(f"plots/continual_learning/cardiac/{configuration.__name__}_{mask}.pdf", bbox_inches='tight')
print(f"plots/continual_learning/cardiac/{configuration.__name__}_{mask}.pdf")