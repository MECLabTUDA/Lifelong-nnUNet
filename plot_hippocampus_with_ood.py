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






def hippocampus_seeded():
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

    trainers = [sequential_seeded_softmax, ewc_seeded_softmax, mib_seeded_softmax, seg_dist_pool_seeded_softmax, ccvae_rehearsal_ood]
    return trainers, "Hippocampus, seeded", combinations_splitted






VISUALIZE_PER_TASK = False

configuration = hippocampus_seeded
t = configuration()
if len(t) == 3:
    trainers, title, combinations = t
else:
    trainers, title = t


data = []
mask = MASK
metric = METRIC

palette = []

for trainer in trainers:

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

        if INTERPOLATE:
            array = zip([49,99,149,199,250], ["trained_49", "trained_99", "trained_149", "trained_199", "trained_final"])
        else:
            array = zip([250], ["trained_final"])

        for intermediate_train_step, path in array:

            trainer_copy = copy.deepcopy(trainer)
            if trainer['ood']['evaluator'] is not ood_eval_helper.eval_reconstruction:
                trainer_copy['ood']['eval_path_base_ood'] += f"/{path}"

            trainer_copy['ood']['eval_path_base_seg'] += f"/{path}"

            temp, _, threshold_per_task = trainer['ood']['evaluator'](trainer_copy['ood'], i, combinations[i], combinations, TASKS, "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", METRIC, MASK)
            
            
            num_total_cases = len(temp)

            num_total_cases_per_task = dict()
            if VISUALIZE_PER_TASK:
                for task in TASKS:
                        num_total_cases_per_task[task] = len(temp[temp['Task'] == task])


            latest_threshold = threshold_per_task[-1]
            print("\n")
            if INTERPOLATE:
                temp = temp.rename(columns={"case": "subject_id"})

            if len(thresholds) > 0:
                threshold = max(*thresholds, latest_threshold)
            else:
                threshold = latest_threshold
            
            if intermediate_train_step == 250:
                thresholds.append(latest_threshold)

            frame = temp[temp['ood_score'] < threshold]
            num_cases_segmented = len(frame)
            
            rel_num_ood = (num_total_cases - num_cases_segmented) / num_total_cases
            
            if VISUALIZE_PER_TASK:
                for task in TASKS:
                    num_cases_segmented_in_task = len(frame[frame['Task'] == task])
                    frame[frame['Task'] == task]['value'] = .7 *frame[frame['Task'] == task]['value'] + .3 * num_cases_segmented_in_task / num_total_cases_per_task[task]  
            else:
                frame['value'] = 100*frame['value']



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



plt.savefig(f"plot_hippocampus_with_ood.png", bbox_inches='tight')
print(f"plot_hippocampus_with_ood.png")