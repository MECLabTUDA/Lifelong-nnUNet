import scipy
import sklearn.metrics as metrics
import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plot_colors
from plot_utils import rename_tasks, convert_epoch_string_to_int

END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0"

combinations = ["Task197_DecathHip",
                "Task197_DecathHip_Task198_Dryad",
                "Task197_DecathHip_Task198_Dryad_Task199_HarP"]


segmentation_distortion = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerSegDist",
                'segmentation_trainer': "nnUNetTrainerSequential",
                'name': "Segmentation Distortion",
                'method': "segmentation_distortion"
}

segmentation_distortion_pool = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerSegDist",
                'name': "Segmentation Distortion + Model Pool",
                'method': "segmentation_distortion"
}

rehearsal_softmax = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerRehearsal",
                'segmentation_trainer': "nnUNetTrainerRehearsal",
                'name': "1-Softmax Rehearsal",
                'method': "uncertainty"
}

sequential_softmax = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerSequential",
                'segmentation_trainer': "nnUNetTrainerSequential",
                'name': "1-Softmax Sequential",
                'method': "uncertainty"
}

ccvae_vae_reconstruction = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                'name': "ccVAE reconstruction",
                'method': "vae_reconstruction"
}

cvae_vae_reconstruction = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkips",
                'name': "cVAE reconstruction",
                'method': "vae_reconstruction"
}

trainers = [rehearsal_softmax, sequential_softmax, cvae_vae_reconstruction, ccvae_vae_reconstruction]


result = pd.DataFrame(columns=["Method"] + combinations[:-1])

rows_list = []

for trainer in trainers:
    row = {"Method": f"{trainer['name']}"}
    for i, trained_tasks in enumerate(combinations[:-1]):
        ood_scores = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              trained_tasks, trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t")
        segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'],
                                                    trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
        segmentation_res = segmentation_res[segmentation_res['metric'] == 'Dice']
        segmentation_res = segmentation_res[segmentation_res['seg_mask'] == 'mask_1']
        segmentation_res['value'] = segmentation_res['value'].apply(lambda x: 1-x) # convert to error

        # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
        #ood_scores = ood_scores.rename(columns={'case': 'subject_id'})
        segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
        ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task'])

        ood_scores = ood_scores[ood_scores['split'] != 'train']
        # for each case take the one with the lowest ood_score
        ood_scores = ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')
        
        #plt.clf()
        #sns.scatterplot(data=ood_scores, x="ood_score", y="value", hue="is_ood")
        #plt.savefig(f"test.png")
        #exit()


        if "Model Pool" in trainer['name']:
            assert False, "Not implemented yet"
            all_ood_scores = []
            for model_in_pool in combinations[:i+1]:
                all_ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              trained_tasks, trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t"))
            all_ood_scores = pd.concat(all_ood_scores)
            all_ood_scores = all_ood_scores[all_ood_scores['split'] != 'train']
            all_ood_scores = all_ood_scores[all_ood_scores['assumed task_idx'] == 0]# those methods do not need to assume a task_idx
            all_ood_scores = all_ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')
            ood_scores = all_ood_scores

        corellation = scipy.stats.spearmanr(ood_scores['ood_score'], ood_scores['value'])
        print(corellation.statistic)
        row[trained_tasks] = corellation.statistic

    rows_list.append(row)

result = result.append(rows_list, ignore_index=True)
print(result)