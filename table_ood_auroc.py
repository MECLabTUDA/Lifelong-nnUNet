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
                'name': "1-Softmax Rehearsal",
                'method': "uncertainty"
}

sequential_softmax = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerSequential",
                'name': "1-Softmax Sequential",
                'method': "uncertainty"
}

ccvae_vae_reconstruction = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                'name': "ccVAE reconstruction",
                'method': "vae_reconstruction"
}

trainers = [segmentation_distortion,segmentation_distortion_pool, rehearsal_softmax, sequential_softmax, ccvae_vae_reconstruction]


result = pd.DataFrame(columns=["Method"] + combinations[:-1])

rows_list = []

for trainer in trainers:
    row = {"Method": f"{trainer['name']}"}
    for i, trained_tasks in enumerate(combinations[:-1]):
        ood_scores = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              trained_tasks, trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t")
        #print(ood_scores['ood_score'], ood_scores['is_ood'])

        ood_scores = ood_scores[ood_scores['split'] != 'train']
        # for each case take the one with the lowest ood_score
        ood_scores = ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')
        
        if "Model Pool" in trainer['name']:
            all_ood_scores = []
            for model_in_pool in combinations[:i+1]:
                all_ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              trained_tasks, trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t"))
            all_ood_scores = pd.concat(all_ood_scores)
            all_ood_scores = all_ood_scores[all_ood_scores['split'] != 'train']
            all_ood_scores = all_ood_scores[all_ood_scores['assumed task_idx'] == 0]# those methods do not need to assume a task_idx
            all_ood_scores = all_ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')
            ood_scores = all_ood_scores

        auroc = metrics.roc_auc_score(ood_scores['is_ood'], ood_scores['ood_score'])
        print(auroc)
        row[trained_tasks] = auroc

    rows_list.append(row)

result = result.append(rows_list, ignore_index=True)
print(result)