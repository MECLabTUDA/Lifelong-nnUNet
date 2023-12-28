import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plot_colors
END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/val_metrics_eval.csv"
END_TRAIN = "__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/val_metrics.csv"
from plot_utils import rename_tasks

#combinations = ["Task011_Prostate-BIDMC",
#                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
#                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
#                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
#                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]

combinations_splitted = ["Task306_BraTS6",
                    "Task306_BraTS6_Task313_BraTS13",
                    "Task306_BraTS6_Task313_BraTS13_Task316_BraTS16",
                    "Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20",
                    "Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21"]

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






def brats_seeded():
    rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trainer': "nnUNetTrainerRehearsal",
                    'name': "Rehearsal"#, 2D, w/ skips, w/o freezing
    }
    feature_rehearsal_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trainer': "nnUNetTrainerFeatureRehearsal2",
                    'name': "Feature Rehearsal"#, 2D, w/ skips, w/o freezing
    }
    upper_bound = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trainer': "nnUNetTrainerRehearsalNoSkipsFrozen",
                    'name': "upper bound"#, 2D, w/o skips, w/ freezing
    }
    vae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "CCVAEr"#, 2D, w/o skips, w/ freezing
    }
    cvae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                    'name': "CVAEr"#, 2D, w/o skips, w/ freezing
    }
    lwf_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trainer': "nnUNetTrainerLWF",
                    'name': "LwF"#, 2D, w/ skips, w/o freezing
    }
    ewc_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trainer': "nnUNetTrainerEWC",
                    'name': "EWC"#, 2D, w/ skips, w/o freezing
    }
    mib_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trainer': "nnUNetTrainerMiB",
                    'name': "MiB"#, 2D, w/ skips, w/o freezing
    }
    sequential_seeded = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trainer': "nnUNetTrainerSequential",
                    'name': "Sequential"#, 2D, w/ skips, w/o freezing
    }

    trainers = [rehearsal_seeded, feature_rehearsal_seeded, ewc_seeded, mib_seeded, sequential_seeded]
    return trainers, "Hippocampus, seeded", combinations_splitted









t = brats_seeded()
if len(t) == 3:
    trainers, title, combinations = t
else:
    trainers, title = t


data = []
mask = "mask_1"
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
    col_wrap=3
)

for i, t in enumerate(ax.axes):
    t.tick_params(labelbottom=True)
    t.grid(True)
    t.set_xlabel("Epoch", visible=True)
    t.set_xticks([0, 250, 500, 750, 1000, 1250])
    t.get_xgridlines()[i].set_color('black')
    t.get_xgridlines()[i].set_linewidth(2)


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
plt.savefig("plot_brats.png", bbox_inches='tight')