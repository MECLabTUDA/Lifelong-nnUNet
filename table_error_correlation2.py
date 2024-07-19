import scipy
import sklearn.metrics as metrics
import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plot_colors
from plot_utils import rename_tasks, convert_epoch_string_to_int
import warnings
warnings.filterwarnings("ignore")

METRIC = "Dice"
MASK = 'mask_1'

END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0"

TASKS = ["Task197_DecathHip",
            "Task198_Dryad",
            "Task199_HarP"]
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
                'segmentation_trainer': "nnUNetTrainerSequential",
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

cvae_vae_reconstruction = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkips",
                'name': "cVAE reconstruction",
                'method': "vae_reconstruction"
}

ccvae_vae_reconstruction = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                'name': "ccVAE reconstruction",
                'method': "vae_reconstruction"
}

cvae_vae_segmentation_distortion = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkips",
                'name': "cVAE reconstruction",
                'method': "segmentation_distortion"
}

ccvae_vae_segmentation_distortion = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                'name': "ccVAE reconstruction",
                'method': "segmentation_distortion"
}


cvae_vae_scaled_softmax = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkips",
                'name': "cVAE scaled Softmax",
                'methods': ["ood_scores_uncertainty_mse_temperature_threshold_0.018849379789815938",
                            "ood_scores_uncertainty_mse_temperature_threshold_0.018849379789815938_0.019073720282499856"],
                #'mse_cutoffs': [0.018849379789815938, 0.019073720282499856]
}

ccvae_vae_scaled_softmax = {'eval_path_base': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                'name': "ccVAE scaled Softmax",
                'methods': ["ood_scores_uncertainty_mse_temperature_threshold_0.022144482741647323",
                            "ood_scores_uncertainty_mse_temperature_threshold_0.022144482741647323_0.023007950314624513"],
}


def find_threshold(df):
    assert df['case'].is_unique, df
    # choose threshold such that 95% of the values in df['ood_score'] are below the threshold
    threshold = np.percentile(df['ood_score'], 95)
    return threshold



def eval_segmentation_distortion(trainer, last_task_idx, trained_tasks):
    ood_scores = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              trained_tasks, trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t")
    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'],
                                                trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
    segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
    segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]
    segmentation_res['value'] = segmentation_res['value'].apply(lambda x: 1-x) # convert to error

    ood_scores = ood_scores[ood_scores['assumed task_idx'] == last_task_idx]


    threshold_per_task = []
    for i in range(last_task_idx+1):
        # find the threshold such that 95% of the test data is considered in-distribution
        temp_for_threshold = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              combinations[i], trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t")
        temp_for_threshold = temp_for_threshold[temp_for_threshold['split'] == 'test']
        temp_for_threshold = temp_for_threshold[temp_for_threshold['assumed task_idx'] == i]
        temp_for_threshold = temp_for_threshold[temp_for_threshold['Task'] == TASKS[i]]
        threshold = find_threshold(temp_for_threshold)
        threshold_per_task.append(threshold)

    print("thresholds", threshold_per_task) 

    # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
    segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
    ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task'])

    ood_scores = ood_scores[ood_scores['split'] == 'val']

    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")



    corellation = scipy.stats.spearmanr(ood_scores['ood_score'], ood_scores['value'])
    print(corellation.statistic)
    return corellation.statistic


def eval_softmax(trainer, last_task_idx, trained_tasks):
    ood_scores = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              trained_tasks, trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t")
    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'],
                                                trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
    segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
    segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]
    segmentation_res['value'] = segmentation_res['value'].apply(lambda x: 1-x) # convert to error

    # find the threshold such that 95% of the test data is considered in-distribution
    threshold_per_task = []
    for i in range(last_task_idx+1):
        temp_for_threshold = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              combinations[i], trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t")
        # find the threshold such that 95% of the test data is considered in-distribution
        temp_for_threshold = temp_for_threshold[temp_for_threshold['split'] == 'test']
        temp_for_threshold = temp_for_threshold[temp_for_threshold['Task'] == TASKS[i]]
        threshold = find_threshold(temp_for_threshold)
        threshold_per_task.append(threshold)

    print("thresholds", threshold_per_task) 

    # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
    segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
    ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task'])

    ood_scores = ood_scores[ood_scores['split'] == 'val']

    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")

    corellation = scipy.stats.spearmanr(ood_scores['ood_score'], ood_scores['value'])
    print(corellation.statistic)
    return corellation.statistic


def eval_scaled_softmax(trainer, last_task_idx, trained_tasks):
    ood_scores = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              trained_tasks, trainer['trainer'] + END, 
                                              f"{trainer['methods'][last_task_idx]}.csv"), sep="\t")
    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'],
                                                trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
    segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
    segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]
    segmentation_res['value'] = segmentation_res['value'].apply(lambda x: 1-x) # convert to error

    ood_scores = ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')

    # find the threshold such that 95% of the test data is considered in-distribution
    threshold_per_task = []
    for i in range(last_task_idx+1):
        temp_for_threshold = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              combinations[i], trainer['trainer'] + END, f"{trainer['methods'][i]}.csv"), sep="\t")
        temp_for_threshold = temp_for_threshold[temp_for_threshold['split'] == 'test']
        temp_for_threshold = temp_for_threshold[temp_for_threshold['assumed task_idx'] == i]
        temp_for_threshold = temp_for_threshold[temp_for_threshold['Task'] == TASKS[i]]
        threshold = find_threshold(temp_for_threshold)
        threshold_per_task.append(threshold)

    print("thresholds", threshold_per_task) 

    # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
    segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
    ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task'])

    ood_scores = ood_scores[ood_scores['split'] == 'val']

    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")

    corellation = scipy.stats.spearmanr(ood_scores['ood_score'], ood_scores['value'])
    print(corellation.statistic)
    return corellation.statistic



def eval_reconstruction(trainer, last_task_idx ,trained_tasks):
    ood_scores = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              trained_tasks, trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t")
    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'],
                                                trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
    segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
    segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]
    segmentation_res['value'] = segmentation_res['value'].apply(lambda x: 1-x) # convert to error

    ood_scores = ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')


    # find the threshold such that 95% of the test data is considered in-distribution
    threshold_per_task = []
    for i in range(last_task_idx+1):
        temp_for_threshold = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                              combinations[i], trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t")
        temp_for_threshold = temp_for_threshold[temp_for_threshold['split'] == 'test']
        temp_for_threshold = temp_for_threshold[temp_for_threshold['assumed task_idx'] == i]
        temp_for_threshold = temp_for_threshold[temp_for_threshold['Task'] == TASKS[i]]
        threshold = find_threshold(temp_for_threshold)
        threshold_per_task.append(threshold)

    print("thresholds", threshold_per_task) 
    temp_for_threshold = ood_scores[ood_scores['split'] == 'test']
    temp_for_threshold = temp_for_threshold[temp_for_threshold['Task'] == TASKS[last_task_idx]]
    threshold = find_threshold(temp_for_threshold)
    print("treshold", threshold)

    # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
    segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
    ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task'])

    ood_scores = ood_scores[ood_scores['split'] == 'val']

    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")

    corellation = scipy.stats.spearmanr(ood_scores['ood_score'], ood_scores['value'])
    print(corellation.statistic)
    return corellation.statistic


def eval_segmentation_distortion_pool(trainer, last_task_idx, trained_tasks):
    all_ood_scores = []
    for i in range(last_task_idx+1):
        ood_scores = pd.read_csv(os.path.join(trainer['eval_path_base'], trainer['eval_path_middle'], 
                                                combinations[i], trainer['trainer'] + END, f"ood_scores_{trainer['method']}.csv"), sep="\t")
        ood_scores = ood_scores[ood_scores['assumed task_idx'] == i]
        all_ood_scores.append(ood_scores)
    ood_scores = pd.concat(all_ood_scores)


    all_segmentation_res = []
    for i in range(last_task_idx+1):
        segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'],
                                                    combinations[i], trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
        segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
        segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]
        segmentation_res['assumed task_idx'] = i
        segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
        all_segmentation_res.append(segmentation_res)
    segmentation_res = pd.concat(all_segmentation_res)
    segmentation_res['value'] = segmentation_res['value'].apply(lambda x: 1-x) # convert to error


    threshold_per_task = []
    for i in range(last_task_idx+1):
        # find the threshold such that 95% of the test data is considered in-distribution
        temp_for_threshold = ood_scores[ood_scores['split'] == 'test']
        temp_for_threshold = temp_for_threshold[temp_for_threshold['assumed task_idx'] == i]
        temp_for_threshold = temp_for_threshold[temp_for_threshold['Task'] == TASKS[i]]
        threshold = find_threshold(temp_for_threshold)
        threshold_per_task.append(threshold)

    print("thresholds", threshold_per_task) 

    # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
    segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
    ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task', 'assumed task_idx'])

    ood_scores = ood_scores[ood_scores['split'] == 'val']

    #each sample can only be in one task
    ood_scores = ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')


    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")

    corellation = scipy.stats.spearmanr(ood_scores['ood_score'], ood_scores['value'])
    print(corellation.statistic)
    return corellation.statistic






#print(eval_segmentation_distortion(segmentation_distortion, 1, combinations[1]))
#exit()



result = pd.DataFrame(columns=["Method"])
for i, trained_tasks in enumerate(combinations[:-1]):
    result[f'{trained_tasks} correlation'] = 0

print("\n\nSegmentation Distortion + Model Pool")

correlation_1 = eval_segmentation_distortion_pool(segmentation_distortion_pool, 0, combinations[0])
print("")
correlation_2 = eval_segmentation_distortion_pool(segmentation_distortion_pool, 1, combinations[1])
result.loc[len(result.index)] = ["Seg. Dist. + Model Pool", correlation_1, correlation_2]

print("\n\nSegmentation Distortion")

correlation_1 = eval_segmentation_distortion(segmentation_distortion, 0, combinations[0])
print("")
correlation_2 = eval_segmentation_distortion(segmentation_distortion, 1, combinations[1])
result.loc[len(result.index)] = ["Seg. Dist.", correlation_1, correlation_2]

print("\n\nRehearsal Softmax")

correlation_1 = eval_softmax(rehearsal_softmax, 0, combinations[0])
print("")
correlation_2 = eval_softmax(rehearsal_softmax, 1, combinations[1])
result.loc[len(result.index)] = ["1-Softmax Rehearsal", correlation_1, correlation_2]

print("\n\nSequential Softmax")

correlation_1 = eval_softmax(sequential_softmax, 0, combinations[0])
print("")
correlation_2 = eval_softmax(sequential_softmax, 1, combinations[1])
result.loc[len(result.index)] = ["1-Softmax Sequential", correlation_1, correlation_2]

print("\n\ncVAE reconstruction")

correlation_1 = eval_reconstruction(cvae_vae_reconstruction, 0, combinations[0])
print("")
correlation_2 = eval_reconstruction(cvae_vae_reconstruction, 1, combinations[1])
result.loc[len(result.index)] = ["cVAE reconstruction", correlation_1, correlation_2]

print("\n\nccVAE reconstruction")

correlation_1 = eval_reconstruction(ccvae_vae_reconstruction, 0, combinations[0])
print("")
correlation_2 = eval_reconstruction(ccvae_vae_reconstruction, 1, combinations[1])
result.loc[len(result.index)] = ["ccVAE reconstruction", correlation_1, correlation_2]

print("\n\nccVAE Segmentation Distortion")

correlation_1 = eval_segmentation_distortion(ccvae_vae_segmentation_distortion, 0, combinations[0])
print("")
correlation_2 = eval_segmentation_distortion(ccvae_vae_segmentation_distortion, 1, combinations[1])
result.loc[len(result.index)] = ["ccVAE Seg. Dist.", correlation_1, correlation_2]

print("\n\ncVAE Segmentation Distortion")

correlation_1 = eval_segmentation_distortion(cvae_vae_segmentation_distortion, 0, combinations[0])
print("")
correlation_2 = eval_segmentation_distortion(cvae_vae_segmentation_distortion, 1, combinations[1])
result.loc[len(result.index)] = ["cVAE Seg. Dist.", correlation_1, correlation_2]


print("\n\nccVAE Scaled Softmax")

correlation_1 = eval_scaled_softmax(ccvae_vae_scaled_softmax, 0, combinations[0])
print("")
correlation_2 = eval_scaled_softmax(ccvae_vae_scaled_softmax, 1, combinations[1])
result.loc[len(result.index)] = ["ccVAE scaled Softmax", correlation_1, correlation_2]

print("\n\ncVAE Scaled Softmax")

correlation_1 = eval_scaled_softmax(cvae_vae_scaled_softmax, 0, combinations[0])
print("")
correlation_2 = eval_scaled_softmax(cvae_vae_scaled_softmax, 1, combinations[1])
result.loc[len(result.index)] = ["cVAE scaled Softmax", correlation_1, correlation_2]

print(result)

#result.to_csv("table_error_correlation2.csv", index=False, sep="\t")

result = result.applymap(lambda x: round(x, 3) * 100 if not isinstance(x, str) else x)

#convert to latex
print(result.to_latex(index=False))

