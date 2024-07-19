import scipy
import sklearn.metrics as metrics
import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plot_colors
from plot_ood_new import rename_val, task_color
from plot_utils import rename_tasks, convert_epoch_string_to_int
import warnings
warnings.filterwarnings("ignore")

END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0"

def find_threshold(df):
    assert df['case'].is_unique, df
    # choose threshold such that 95% of the values in df['ood_score'] are below the threshold
    threshold = np.percentile(df['ood_score'], 95)
    return threshold


def eval_combination(trainer, last_task_idx, trained_tasks, val_only=True, merge_seg=True):
    id_cases = []
    for inner_trainer in trainer:
        ood_scores, threshold = inner_trainer['evaluator'](inner_trainer, last_task_idx, trained_tasks)
        ood_scores = ood_scores[ood_scores['ood_score']<threshold]
        id_cases.append(ood_scores)
    ood_scores = pd.concat(id_cases)
    # keep only the cases that are 3 times in ood_scores
    counts = ood_scores['case'].value_counts()
    counts = counts.to_frame()
    remove_cases = counts[counts['case'] < 3].index.tolist()
    ood_scores = ood_scores[~ood_scores['case'].isin(remove_cases)]
    return ood_scores, None

def insert_best_sbest(df, column, _max):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(column, str)
    assert isinstance(_max, bool)
    if _max:
        col_best_row =  (df[column].argsort().values[::-1])[0]
        col_sbest_row = (df[column].argsort().values[::-1])[1]
    else:
        col_best_row =  df[column].argsort()[0]
        col_sbest_row = df[column].argsort()[1]
    df.iloc[col_best_row, i] = "best" + "{:1.1f}".format(df.iloc[col_best_row, i])
    df.iloc[col_sbest_row, i] = "sbest" + "{:1.1f}".format(df.iloc[col_sbest_row, i])

def insert_best_sbest2(df, column, _max):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(column, str)
    assert isinstance(_max, bool)
    

    if _max:
        best_value =  np.sort(df[column].values)[-1]
        sbest_value = np.sort(df[column].values)[-2]
    else:
        best_value =  np.sort(df[column].values)[0]
        sbest_value = np.sort(df[column].values)[1]

    for row in range(len(df)):
        if df[column][row] == best_value:
            df.iloc[row, i] = "best" + "{:1.1f}".format(df.iloc[row, i])
        elif df[column][row] == sbest_value:
            df.iloc[row, i] = "sbest" + "{:1.1f}".format(df.iloc[row, i])
    #df.iloc[col_best_row, i] = "best" + "{:1.1f}".format(df.iloc[col_best_row, i])
    #df.iloc[col_sbest_row, i] = "sbest" + "{:1.1f}".format(df.iloc[col_sbest_row, i])

def get_all_thresholds(_trainer, last_task_idx):
    thresholds = []
    for i in range(last_task_idx+1):
        trained_on = _trainer['trained_on']
        trainer = _trainer['trainer']
        if 'methods' in _trainer:
            assert 'method' not in _trainer
            method = _trainer['methods'][i]
        else:
            assert 'methods' not in _trainer
            method = _trainer['method']
        df = pd.read_csv(os.path.join(_trainer['eval_path_base_ood'], "nnUNet_ext/2d", 
                                      '_'.join(trained_on), '_'.join(trained_on[:i+1]), f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", 
                                      trained_on[i], f"ood_scores_{method}.csv"), 
                         sep='\t')
        df = df[df['split'] == 'test']

        df = df.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')

        assert np.all(df['Task'] == trained_on[i]), df
        assert df['case'].is_unique, df
        # choose threshold such that 95% of the values in df['ood_score'] are below the threshold
        threshold = np.percentile(df['ood_score'], 95)
        thresholds.append(threshold)

    return thresholds


def eval_segmentation_distortion(trainer, last_task_idx, trained_tasks, val_only=True, merge_seg=True):
    ood_scores = []
    for task in TASKS:
        ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base_ood'], trainer['eval_path_middle'], 
                                                trained_tasks, trainer['trainer'] + END, task,
                                                f"ood_scores_{trainer['method']}.csv"), sep="\t"))
    ood_scores = pd.concat(ood_scores)
        

    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], "trained_final", trainer['eval_path_middle'],
                                                trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
    segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
    segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]

    ood_scores = ood_scores[ood_scores['assumed task_idx'] == last_task_idx]


    threshold_per_task = get_all_thresholds(trainer, last_task_idx)
    print("thresholds", threshold_per_task) 

    if merge_seg:
        assert val_only, "merge_seg only works with val_only=True"
        # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
        segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
        ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task'])

    if val_only:
        ood_scores = ood_scores[ood_scores['split'] == 'val']

    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")

    return ood_scores, max(threshold_per_task)


def eval_softmax(trainer, last_task_idx, trained_tasks, val_only=True, merge_seg=True):
    ood_scores = []
    for task in TASKS:
        ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base_ood'], trainer['eval_path_middle'], 
                                                trained_tasks, trainer['trainer'] + END, task,
                                                f"ood_scores_{trainer['method']}.csv"), sep="\t"))
    ood_scores = pd.concat(ood_scores)

    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], "trained_final", trainer['eval_path_middle'],
                                                trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
    segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
    segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]


    threshold_per_task = get_all_thresholds(trainer, last_task_idx)
    print("thresholds", threshold_per_task) 

    if merge_seg:
        assert val_only, "merge_seg only works with val_only=True"
        # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
        segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
        segmentation_res.fillna(1, inplace=True)
        ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task'])

    if val_only:
        ood_scores = ood_scores[ood_scores['split'] == 'val']

    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")

    return ood_scores, max(threshold_per_task)


def eval_scaled_softmax(trainer, last_task_idx, trained_tasks, val_only=True, merge_seg=True):
    ood_scores = []
    for task in TASKS:
        ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base_ood'], trainer['eval_path_middle'], 
                                                trained_tasks, trainer['trainer'] + END, task,
                                              f"ood_scores_{trainer['methods'][last_task_idx]}.csv"), sep="\t"))
    ood_scores = pd.concat(ood_scores)

    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], "trained_final", trainer['eval_path_middle'],
                                                trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
    segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
    segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]

    ood_scores = ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')

    # find the threshold such that 95% of the test data is considered in-distribution
    threshold_per_task = get_all_thresholds(trainer, last_task_idx)
    print("thresholds", threshold_per_task)

    if merge_seg:
        assert val_only, "merge_seg only works with val_only=True"
        # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
        segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
        ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task'])


    if val_only:
        ood_scores = ood_scores[ood_scores['split'] == 'val']


    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")

    return ood_scores, max(threshold_per_task)



def eval_reconstruction(trainer, last_task_idx ,trained_tasks, val_only=True, merge_seg=True):
    ood_scores = []
    for task in TASKS:
        ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base_ood'], trainer['eval_path_middle'], 
                                                trained_tasks, trainer['trainer'] + END, task,
                                                f"ood_scores_{trainer['method']}.csv"), sep="\t"))
    ood_scores = pd.concat(ood_scores)

    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], "trained_final", trainer['eval_path_middle'],
                                                trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
    segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
    segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]

    ood_scores = ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')


    threshold_per_task = get_all_thresholds(trainer, last_task_idx)
    print("thresholds", threshold_per_task)

    if merge_seg:
        assert val_only, "merge_seg only works with val_only=True"
        # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
        segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
        ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task'])

    if val_only:
        ood_scores = ood_scores[ood_scores['split'] == 'val']

    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")

    return ood_scores, max(threshold_per_task)


def eval_segmentation_distortion_pool(trainer, last_task_idx, trained_tasks, val_only=True, merge_seg=True):
    all_ood_scores = []
    for i in range(last_task_idx+1):
        ood_scores = []
        for task in TASKS:
            ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base_ood'], trainer['eval_path_middle'], 
                                                    combinations[i], trainer['trainer'] + END, task,
                                                    f"ood_scores_{trainer['method']}.csv"), sep="\t"))
        ood_scores = pd.concat(ood_scores)
        all_ood_scores.append(ood_scores)
    ood_scores = pd.concat(all_ood_scores)


    all_segmentation_res = []
    for i in range(last_task_idx+1):
        segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], "trained_final", trainer['eval_path_middle'],
                                                    combinations[i], trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
        segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
        segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]
        segmentation_res['assumed task_idx'] = i
        segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
        all_segmentation_res.append(segmentation_res)
    segmentation_res = pd.concat(all_segmentation_res)


    #threshold_per_task = []
    #for i in range(last_task_idx+1):
    #    # find the threshold such that 95% of the test data is considered in-distribution
    #    temp_for_threshold = ood_scores[ood_scores['split'] == 'test']
    #    temp_for_threshold = temp_for_threshold[temp_for_threshold['assumed task_idx'] == i]
    #    temp_for_threshold = temp_for_threshold[temp_for_threshold['Task'] == TASKS[i]]
    #    threshold = find_threshold(temp_for_threshold)
    #    threshold_per_task.append(threshold)
    #print("thresholds", threshold_per_task) 

    threshold_per_task = get_all_thresholds(trainer, last_task_idx)
    print("thresholds", threshold_per_task) 


    if merge_seg:
        assert val_only, "merge_seg only works with val_only=True"
        # join the dataframes where ood_scores['case'] == segmentation_res['subject_id'] and ood_scores['Task'] == segmentation_res['Task']
        segmentation_res = segmentation_res.rename(columns={'subject_id': 'case'})
        ood_scores = ood_scores.merge(segmentation_res, on=['case', 'Task', 'assumed task_idx'])

    if val_only:
        ood_scores = ood_scores[ood_scores['split'] == 'val']

    #each sample can only be in one task
    ood_scores = ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')


    assert ood_scores['case'].is_unique, ood_scores

    num_cases = len(ood_scores['case'])
    print(f"Number of cases: {num_cases}")

    #print(ood_scores)

    num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])
    print(f"Number of in distribution cases: {num_id_cases}")

    return ood_scores, max(threshold_per_task)


def eval_id_dice(trainer, last_task_idx, trained_tasks):
    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], "trained_final", trainer['eval_path_middle'],
                                                trained_tasks, trainer['segmentation_trainer'] + END, "val_metrics_eval.csv"), sep="\t")
    segmentation_res = segmentation_res[segmentation_res['metric'] == METRIC]
    segmentation_res = segmentation_res[segmentation_res['seg_mask'] == MASK]
    
    segmentation_res = segmentation_res[segmentation_res['Task'].isin(TASKS[:last_task_idx+1])]

    return segmentation_res['value'].mean()


############################################################################################
############################################################################################
############################################################################################
############################################################################################


anatomy = "prostate"

if anatomy == "hippocampus":
    METRIC = "Dice"
    MASK = 'mask_1'
    TASKS = ["Task197_DecathHip",
                "Task198_Dryad",
                "Task199_HarP"]
    combinations = ["Task197_DecathHip",
                    "Task197_DecathHip_Task198_Dryad",
                    #"Task197_DecathHip_Task198_Dryad_Task199_HarP"
                    ]

    segmentation_distortion = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSegDist",
                    'segmentation_trainer': "nnUNetTrainerSequential",
                    'name': "Segmentation Distortion",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion
    }

    segmentation_distortion_pool = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSegDist",
                    'segmentation_trainer': "nnUNetTrainerSequential",
                    'name': "Segmentation Distortion + Model Pool",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion_pool
    }

    rehearsal_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerRehearsal",
                    'segmentation_trainer': "nnUNetTrainerRehearsal",
                    'name': "1-Softmax Rehearsal",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }

    sequential_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSequential",
                    'segmentation_trainer': "nnUNetTrainerSequential",
                    'name': "1-Softmax Sequential",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }

    ewc_no_ood = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSequential",
                    'segmentation_trainer': "nnUNetTrainerEWC",
                    'name': "EWC",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }
    mib_no_ood = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSequential",
                    'segmentation_trainer': "nnUNetTrainerMiB",
                    'name': "MiB",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }

    cvae_vae_reconstruction = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAE reconstruction",
                    'method': "vae_reconstruction",
                    'evaluator': eval_reconstruction
    }

    ccvae_vae_reconstruction = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "ccVAE reconstruction",
                    'method': "vae_reconstruction",
                    'evaluator': eval_reconstruction
    }

    cvae_vae_segmentation_distortion = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAE Segmentation Distortion",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion
    }

    ccvae_vae_segmentation_distortion = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "ccVAE Segmentation Distortion",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion
    }


    cvae_vae_scaled_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAE scaled Softmax",
                    'methods': ["uncertainty_mse_temperature_threshold_0.02067640062210244",
                                "uncertainty_mse_temperature_threshold_0.02067640062210244_0.020159663562016105"],
                    'evaluator': eval_scaled_softmax
    }

    ccvae_vae_scaled_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "ccVAE scaled Softmax",
                    'methods': ["uncertainty_mse_temperature_threshold_0.022144482741647323",
                                "uncertainty_mse_temperature_threshold_0.022144482741647323_0.023007950314624513"],
                    'evaluator': eval_scaled_softmax
    }
elif anatomy == "prostate":
    METRIC = "Dice"
    MASK = 'mask_1'
    TASKS = ["Task111_Prostate-BIDMC",
                "Task112_Prostate-I2CVB",
                "Task113_Prostate-HK",
                "Task115_Prostate-UCL",
                "Task116_Prostate-RUNMC"]
    combinations = ["Task111_Prostate-BIDMC",
                    "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB",
                    "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK",
                    "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL",
                    #"Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC"
                    ]

    segmentation_distortion = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSegDist",
                    'segmentation_trainer': "nnUNetTrainerSequential",
                    'name': "Segmentation Distortion",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion
    }

    segmentation_distortion_pool = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSegDist",
                    'segmentation_trainer': "nnUNetTrainerSequential",
                    'name': "Segmentation Distortion + Model Pool",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion_pool
    }

    rehearsal_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerRehearsal",
                    'segmentation_trainer': "nnUNetTrainerRehearsal",
                    'name': "1-Softmax Rehearsal",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }

    sequential_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSequential",
                    'segmentation_trainer': "nnUNetTrainerSequential",
                    'name': "1-Softmax Sequential",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }

    ewc_no_ood = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSequential",
                    'segmentation_trainer': "nnUNetTrainerEWC",
                    'name': "EWC",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }
    
    mib_no_ood = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerSequential",
                    'segmentation_trainer': "nnUNetTrainerMiB",
                    'name': "MiB",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }

    cvae_vae_reconstruction = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAE reconstruction",
                    'method': "vae_reconstruction",
                    'evaluator': eval_reconstruction
    }

    ccvae_vae_reconstruction = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "ccVAE reconstruction",
                    'method': "vae_reconstruction",
                    'evaluator': eval_reconstruction
    }

    cvae_vae_segmentation_distortion = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAE Segmentation Distortion",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion
    }

    ccvae_vae_segmentation_distortion = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "ccVAE Segmentation Distortion",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion
    }


    cvae_vae_scaled_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAE scaled Softmax",
                    'methods': ["uncertainty_mse_temperature_threshold_0.031311719649957506",
                                "uncertainty_mse_temperature_threshold_0.031311719649957506_0.027911194720400073",
                                "uncertainty_mse_temperature_threshold_0.031311719649957506_0.027911194720400073_0.02023473611722388"],
                    'evaluator': eval_scaled_softmax
    }

    ccvae_vae_scaled_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation",
                    'eval_path_middle': "nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC",
                    'trained_on': TASKS,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "ccVAE scaled Softmax",
                    'methods': ["uncertainty_mse_temperature_threshold_0.03324513657335259",
                                "uncertainty_mse_temperature_threshold_0.03324513657335259_0.03449850807810431",
                                "uncertainty_mse_temperature_threshold_0.03324513657335259_0.03449850807810431_0.016187053324150227"],
                    'evaluator': eval_scaled_softmax
    }
elif anatomy == "brats":
    METRIC = "Dice"
    MASK = 'mask_3'
    trained_on = ["Task306_BraTS6",
                "Task313_BraTS13",
                "Task316_BraTS16",
                "Task320_BraTS20",
                "Task321_BraTS21"]
    TASKS = ["Task306_BraTS6",
                "Task313_BraTS13",
                "Task316_BraTS16",
                "Task320_BraTS20",
                "Task321_BraTS21",
                "Task200_BraTSOthers",
                "Task201_BraTS1",
                "Task204_BraTS4",
                "Task218_BraTS18"
                ]
    combinations = ["Task306_BraTS6",
                    "Task306_BraTS6_Task313_BraTS13",
                    "Task306_BraTS6_Task313_BraTS13_Task316_BraTS16",
                    "Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20",
                    #"Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21"
                    ]

    segmentation_distortion = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerSegDist",
                    'segmentation_trainer': "nnUNetTrainerSequential",
                    'name': "Segmentation Distortion",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion
    }

    segmentation_distortion_pool = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerSegDist",
                    'segmentation_trainer': "nnUNetTrainerSequential",
                    'name': "Segmentation Distortion + Model Pool",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion_pool
    }

    rehearsal_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerRehearsal",
                    'segmentation_trainer': "nnUNetTrainerRehearsal",
                    'name': "1-Softmax Rehearsal",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }

    sequential_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerSequential",
                    'segmentation_trainer': "nnUNetTrainerSequential",
                    'name': "1-Softmax Sequential",
                    'method': "uncertainty",
                    'evaluator': eval_softmax
    }

    cvae_vae_reconstruction = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAE reconstruction",
                    'method': "vae_reconstruction",
                    'evaluator': eval_reconstruction
    }

    ccvae_vae_reconstruction = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "ccVAE reconstruction",
                    'method': "vae_reconstruction",
                    'evaluator': eval_reconstruction
    }

    cvae_vae_segmentation_distortion = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAE Segmentation Distortion",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion
    }

    ccvae_vae_segmentation_distortion = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "ccVAE Segmentation Distortion",
                    'method': "segmentation_distortion_normalized",
                    'evaluator': eval_segmentation_distortion
    }


    cvae_vae_scaled_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsLarge",
                    'name': "cVAE scaled Softmax",
                    'methods': ["uncertainty_mse_temperature_threshold_0.027975266869065655",
                                "uncertainty_mse_temperature_threshold_0.027975266869065655_0.02087775344121095",
                                "uncertainty_mse_temperature_threshold_0.027975266869065655_0.02087775344121095_0.02538238980262286"],
                    'evaluator': eval_scaled_softmax
    }

    ccvae_vae_scaled_softmax = {
                    'eval_path_base_ood': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_base_seg': "/local/scratch/clmn1/master_thesis/seeded/evaluation3",
                    'eval_path_middle': "nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21",
                    'trained_on': trained_on,
                    'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'segmentation_trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                    'name': "ccVAE scaled Softmax",
                    'methods': ["uncertainty_mse_temperature_threshold_0.03407409618518855",
                                "uncertainty_mse_temperature_threshold_0.03407409618518855_0.024456772481037368",
                                "uncertainty_mse_temperature_threshold_0.03407409618518855_0.024456772481037368_0.02805355638918686"],
                    'evaluator': eval_scaled_softmax
    }
else:
    assert False, "anatomy not recognized" 


trainers = [segmentation_distortion, segmentation_distortion_pool, 
            rehearsal_softmax, sequential_softmax, 
            cvae_vae_reconstruction, ccvae_vae_reconstruction, 
            cvae_vae_segmentation_distortion, ccvae_vae_segmentation_distortion, 
            cvae_vae_scaled_softmax, ccvae_vae_scaled_softmax]

#trainers = [cvae_vae_reconstruction, ccvae_vae_reconstruction, 
#            cvae_vae_segmentation_distortion, ccvae_vae_segmentation_distortion, 
#            cvae_vae_scaled_softmax, ccvae_vae_scaled_softmax]
#
#trainers = [segmentation_distortion, segmentation_distortion_pool, 
#            sequential_softmax, 
#             ccvae_vae_reconstruction]

trainers = [sequential_softmax, segmentation_distortion_pool, ccvae_vae_reconstruction, ewc_no_ood, mib_no_ood]

#trainers = [segmentation_distortion_pool]

TABLE = "BWT_FWT" #dice, ECE, spearman, plot, ECE-spearman, dice_id_plot, Dice on ID, F1
ANNOTATE_BEST = len(trainers) > 1





if TABLE == "dice":

    results = []
    for trainer in trainers:
        res_dict = {
            "Method": trainer['name']
        }
        for i in range(len(combinations)-1):
            print("\n\n", trainer['name'], "trained on", combinations[i])
            ood_scores, threshold = trainer['evaluator'](trainer, i, combinations[i])
            num_cases = len(ood_scores['case'])
            num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])

            f1 = metrics.f1_score(ood_scores['is_ood'], ood_scores['ood_score'] > threshold)
            ood_scores = ood_scores[ood_scores['ood_score'] < threshold]
            dice = np.mean(ood_scores['value'])


            id_dice = eval_id_dice(trainer, i, combinations[i])


            #res_dict[f'{combinations[i]} Dice [ID]'] = id_dice
            res_dict[f'{combinations[i]} Dice'] = dice
            res_dict[f'{combinations[i]} ID'] = len(ood_scores['case']) / num_cases
            #res_dict[f'{combinations[i]} Class ID [true]'] = len(ood_scores['case']) / num_id_cases

        results.append(res_dict)

    df = pd.DataFrame.from_records(results)
    print(df)
    df.to_csv("table_ood_dice.csv", index=False, sep="\t")

    df = df.applymap(lambda x: round(x, 3) * 100 if not isinstance(x, str) else x)
    
    if ANNOTATE_BEST:
        for i, column in enumerate(df):
            if i == 0:
                continue
            #col_best_row = df[column].idxmax()
            insert_best_sbest2(df, column, True)


    #convert to latex
    print(df.to_latex(index=False).replace(" sbest", r" \sbest").replace(" best", r" \best"))

elif TABLE == "ECE":
    results = []
    for trainer in trainers:
        res_dict = {
            "Method": trainer['name']
        }
        for i in range(len(combinations)-1):
            print("\n\n", trainer['name'], "trained on", combinations[i])
            ood_scores, threshold = trainer['evaluator'](trainer, i, combinations[i])
            num_cases = len(ood_scores['case'])

            #min max normalize ood_scores['ood_score'] to [0,1]
            ood_scores['ood_score'] = (ood_scores['ood_score'] - min(ood_scores['ood_score'])) / (max(ood_scores['ood_score']) - min(ood_scores['ood_score']))
            ood_scores['ood_score'] = 1 - ood_scores['ood_score'] #convert to confidence

            #min max normalize ood_scores['value'] to [0,1]
            ood_scores['value'] = (ood_scores['value'] - min(ood_scores['value'])) / (max(ood_scores['value']) - min(ood_scores['value']))




            min_ood_score = min(ood_scores['ood_score'])
            max_ood_score = max(ood_scores['ood_score'])
            # sort the values in ood_scores['ood_score'] into 15 bins
            bins = np.linspace(min_ood_score, max_ood_score, 15)
            ood_scores['bin'] = np.digitize(ood_scores['ood_score'], bins)
            ece = []
            for bin in range(1, len(bins)):
                bin_df = ood_scores[ood_scores['bin'] == bin]
                if len(bin_df) == 0:
                    continue
                bin_acc = np.mean(bin_df['value'])
                bin_conf = np.mean(bin_df['ood_score'])
                bin_size = len(bin_df)
                ece.append(bin_size * np.abs(bin_acc - bin_conf))
            ece = np.sum(ece) / num_cases
            res_dict[f'{combinations[i]} ECE'] = round(ece,3)*100
        
        results.append(res_dict)
    df = pd.DataFrame.from_records(results)
    print(df)
    print(df.to_latex(index=False))

elif TABLE == "spearman":
    results = []
    for trainer in trainers:
        res_dict = {
            "Method": trainer['name']
        }
        for i in range(len(combinations)-1):
            print("\n\n", trainer['name'], "trained on", combinations[i])
            ood_scores, threshold = trainer['evaluator'](trainer, i, combinations[i])
            num_cases = len(ood_scores['case'])
            assert ood_scores['case'].is_unique, ood_scores

            correlation = scipy.stats.spearmanr(ood_scores['ood_score'], ood_scores['value'])



            res_dict[f'{combinations[i]} spearman'] = round(-correlation.statistic, 3)*100
        
        results.append(res_dict)
    df = pd.DataFrame.from_records(results)
    print(df)
    print(df.to_latex(index=False))

elif TABLE == "plot":
    assert len(trainers) == 1, "Only one trainer allowed for plotting"
    trainer = trainers[0]

    for i in range(len(combinations)-1):
        plt.clf()

        ood_scores, threshold = trainer['evaluator'](trainer, i, combinations[i], val_only=False, merge_seg=False)
        min_ood_score = 0
        max_ood_score = max(ood_scores['ood_score'])

        trained_on = trainer['trained_on'][:i+1]
        for task in TASKS:
            xs = np.linspace(min_ood_score, max_ood_score, 1000)


            #split_data = task in trained_on
            if task == trained_on[-1]:
                arr = ['train', 'test', 'val',]
            elif task in trained_on:
                arr = ['val']
            else:
                arr = [None]    # do not perform splitting
                
            for val in arr:
                subset_df = ood_scores[ood_scores['Task'] == task]
                if arr[0] is not None:
                    subset_df = subset_df[subset_df['split'] == val] #only validation data

                y = []
                for x in xs:
                    #count amount of values in subset_df where values at column 'ood_score' is less than x
                    y.append(len([v for v in subset_df['ood_score'] if v > x]) / len(subset_df))
                assert len(y) == len(xs)
                if task in trained_on:
                    if val=='train':
                        linestyle = 'solid'
                    elif val == 'val':
                        linestyle = 'dashed'
                    elif val == 'test':
                        linestyle = 'dotted'
                    sns.lineplot(x=xs, y=y, label=f"{rename_tasks(task)}, {rename_val(val)}", linestyle=linestyle, color=task_color(task))
                else:
                    sns.lineplot(x=xs, y=y, label=rename_tasks(task), linestyle='dashed', color=task_color(task))
        if threshold is not None:
            plt.axvline(x=threshold, color='black', linestyle='dashed')#, label="95% threshold")
        plt.xlabel(r"Threshold $\tau$")
        plt.ylabel("Amount of samples classified as OOD")
        title = f"{trainer['name']} trained on "
        for t in trained_on[:-1]:
            title += "\emph{" + rename_tasks(t) + "}, "
        title += "\emph{" + rename_tasks(trained_on[-1]) + "}"
        plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.savefig(f"plot_generic.pdf", bbox_inches='tight')
        plt.savefig(f"plots/ood_detection/{anatomy}/{trainer['name'].replace(' ', '_')}_{i}.pdf", bbox_inches='tight')

elif TABLE == "ECE-spearman":
    results = []
    for trainer in trainers:
        res_dict = {
            "Method": trainer['name']
        }
        for i in range(len(combinations)-1):
            print("\n\n", trainer['name'], "trained on", combinations[i])
            ood_scores, threshold = trainer['evaluator'](trainer, i, combinations[i])
            num_cases = len(ood_scores['case'])


            correlation = scipy.stats.spearmanr(ood_scores['ood_score'], ood_scores['value'])



            spearman_correlation = round(-correlation.statistic, 3)*100


            #min max normalize ood_scores['ood_score'] to [0,1]
            ood_scores['ood_score'] = (ood_scores['ood_score'] - min(ood_scores['ood_score'])) / (max(ood_scores['ood_score']) - min(ood_scores['ood_score']))
            ood_scores['ood_score'] = 1 - ood_scores['ood_score'] #convert to confidence

            #min max normalize ood_scores['value'] to [0,1]
            ood_scores['value'] = (ood_scores['value'] - min(ood_scores['value'])) / (max(ood_scores['value']) - min(ood_scores['value']))


            min_ood_score = min(ood_scores['ood_score'])
            max_ood_score = max(ood_scores['ood_score'])
            # sort the values in ood_scores['ood_score'] into 15 bins
            bins = np.linspace(min_ood_score, max_ood_score, 15)
            ood_scores['bin'] = np.digitize(ood_scores['ood_score'], bins)
            ece = []
            for bin in range(1, len(bins)):
                bin_df = ood_scores[ood_scores['bin'] == bin]
                if len(bin_df) == 0:
                    continue
                bin_acc = np.mean(bin_df['value'])
                bin_conf = np.mean(bin_df['ood_score'])
                bin_size = len(bin_df)
                ece.append(bin_size * np.abs(bin_acc - bin_conf))
            ece = np.sum(ece) / num_cases


            res_dict[f'{combinations[i]} ECE'] = round(ece,3)*100
            res_dict[f'{combinations[i]} spearman'] = spearman_correlation
        
        results.append(res_dict)
    df = pd.DataFrame.from_records(results)
    print(df)




    if ANNOTATE_BEST:
        for i, column in enumerate(df):
            if i == 0:
                continue
            #col_best_row = df[column].idxmax()
            if "ECE" in column:
                insert_best_sbest(df, column, False)
            elif "spearman" in column:
                insert_best_sbest(df, column, True)

    print(df.to_latex(index=False).replace(" sbest", r" \sbest").replace(" best", r" \best"))

elif TABLE == "dice_id_plot":

    for i in range(len(combinations)-1):
        i = 2
        plt.clf()
        for trainer in trainers:

            ood_scores, threshold = trainer['evaluator'](trainer, i, combinations[i], val_only=True, merge_seg=True)
            min_ood_score = min(ood_scores['ood_score'])
            max_ood_score = max(ood_scores['ood_score'])

            xs = np.linspace(min_ood_score, max_ood_score, 1000)
            num_all_cases = len(ood_scores['case'])
            dices = []
            IDs = []
            y = []
            dice_plus_id = []
            dice_minus_ood = []
            dice_times_id = []
            dice_plus_alpha_id = []
            alpha = 0.1
            for x in xs:
                subset_df = ood_scores[ood_scores['ood_score'] < x]
                if len(subset_df) == 0:
                    y.append(0)
                    dices.append(0)
                    IDs.append(0)
                    dice_plus_id.append(0)
                    dice_minus_ood.append(0)
                    dice_times_id.append(0)
                    dice_plus_alpha_id.append(0)
                else:
                    dice = np.mean(subset_df['value'])
                    num_id_cases = len(subset_df)
                    y.append( 0.5 * dice + .5* num_id_cases / num_all_cases)#* num_id_cases / num_all_cases)
                    dice_plus_id.append(0.5 * dice + .5* num_id_cases / num_all_cases)
                    dice_minus_ood.append(dice -  (num_all_cases - num_id_cases) / num_all_cases)
                    dice_times_id.append(dice * num_id_cases / num_all_cases)
                    dice_plus_alpha_id.append((1-alpha) * dice + alpha * num_id_cases / num_all_cases)
                    dices.append(dice)
                    IDs.append(num_id_cases / num_all_cases)

            #sns.lineplot(x=xs / max_ood_score, y=y, label=f"{trainer['name']}")
            xs_normalized = (xs - min_ood_score) / (max_ood_score - min_ood_score)
            __frame = pd.DataFrame({'x': xs_normalized, 'y': dice_plus_alpha_id})
            sns.lineplot(data=__frame, x="x", y="y", label=f"{trainer['name']}")
            #sns.lineplot(x=xs, y=dices, label="dices")
            #sns.lineplot(x=xs, y=IDs, label="IDs")
        break

    #if threshold is not None:
    #    plt.axvline(x=threshold, color='black', linestyle='dashed')#, label="95% threshold")
    plt.xlabel(r"Threshold $\tau$")
    plt.ylabel("Dice + ID")
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"plot_generic.pdf", bbox_inches='tight')
    print("save as generic.pdf")
    #plt.savefig(f"plots/ood_detection/{anatomy}/{trainer['name'].replace(' ', '_')}_{i}.pdf", bbox_inches='tight')

elif TABLE == "Dice on ID":
    # in a table report the Dice on the data that are ID (independent of the models decision)
    
    results = []
    for trainer in trainers:
        res_dict = {
            "Method": trainer['name']
        }
        for i in range(len(combinations)):
            print("\n\n", trainer['name'], "trained on", combinations[i])
            ood_scores, threshold = trainer['evaluator'](trainer, i, combinations[i])
            num_cases = len(ood_scores['case'])
            num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])

            f1 = metrics.f1_score(ood_scores['is_ood'], ood_scores['ood_score'] > threshold)

            ood_scores = ood_scores[ood_scores["is_ood"] == False]# take only ID data


            #ood_scores = ood_scores[ood_scores['ood_score'] < threshold]
            dice = np.mean(ood_scores['value'])


            #id_dice = eval_id_dice(trainer, i, combinations[i])


            #res_dict[f'{combinations[i]} Dice [ID]'] = id_dice
            res_dict[f'{combinations[i]} Dice'] = dice
            #res_dict[f'{combinations[i]} ID'] = len(ood_scores['case']) / num_cases
            #res_dict[f'{combinations[i]} Class ID [true]'] = len(ood_scores['case']) / num_id_cases

        results.append(res_dict)

    df = pd.DataFrame.from_records(results)
    print(df)
    df.to_csv("table_ood_dice.csv", index=False, sep="\t")

    df = df.applymap(lambda x: round(x, 3) * 100 if not isinstance(x, str) else x)
    
    if ANNOTATE_BEST:
        for i, column in enumerate(df):
            if i == 0:
                continue
            #col_best_row = df[column].idxmax()
            insert_best_sbest2(df, column, True)


    #convert to latex
    print(df.to_latex(index=False).replace(" sbest", r" \sbest").replace(" best", r" \best"))

elif TABLE == "F1":
    # in a table report the Dice on the data that are ID (independent of the models decision)
    
    results = []
    for trainer in trainers:
        res_dict = {
            "Method": trainer['name']
        }
        for i in range(len(combinations)):
            print("\n\n", trainer['name'], "trained on", combinations[i])
            ood_scores, threshold = trainer['evaluator'](trainer, i, combinations[i])
            num_cases = len(ood_scores['case'])
            num_id_cases = len(ood_scores[ood_scores['is_ood'] == False]['case'])

            f1 = metrics.f1_score(ood_scores['is_ood'], ood_scores['ood_score'] > threshold)

            #ood_scores = ood_scores[ood_scores["is_ood"] == False]# take only ID data

            print(ood_scores)

            #ood_scores = ood_scores[ood_scores['ood_score'] < threshold]
            dice = np.mean(ood_scores['value'])


            #id_dice = eval_id_dice(trainer, i, combinations[i])


            #res_dict[f'{combinations[i]} Dice [ID]'] = id_dice
            res_dict[f'{combinations[i]} F1'] = f1
            #res_dict[f'{combinations[i]} ID'] = len(ood_scores['case']) / num_cases
            #res_dict[f'{combinations[i]} Class ID [true]'] = len(ood_scores['case']) / num_id_cases

        results.append(res_dict)

    df = pd.DataFrame.from_records(results)
    print(df)
    df.to_csv("table_ood_dice.csv", index=False, sep="\t")

    df = df.applymap(lambda x: round(x, 3) * 100 if not isinstance(x, str) else x)
    
    if ANNOTATE_BEST:
        for i, column in enumerate(df):
            if i == 0:
                continue
            #col_best_row = df[column].idxmax()
            insert_best_sbest2(df, column, True)


    #convert to latex
    print(df.to_latex(index=False).replace(" sbest", r" \sbest").replace(" best", r" \best"))

elif TABLE == "BWT_FWT":
    res_arr = []

    sequential_performance = []
    sequential_performance_list = []
    for i in range(len(combinations)):
        ood_scores, threshold = sequential_softmax['evaluator'](sequential_softmax, i, combinations[i])
        assert np.all(ood_scores['split'] == 'val')
        temp = ood_scores[ood_scores['Task'] == TASKS[i]]
        sequential_performance.append(np.mean(temp['value']))
        sequential_performance_list.append(temp['value'])

    for trainer in trainers:
        res = {
            "Method": trainer['name']
        }

        init_performance = []

        p = os.path.join(trainer['eval_path_base_seg'], "trained_final", trainer['eval_path_middle'], 
                     "initialization", f"{trainer['segmentation_trainer']}__nnUNetPlansv2.1", "Generic_UNet", "SEQ", "head_None", "fold_0",
                     "val_metrics_all.csv")
        df = pd.read_csv(p, sep="\t")
        df = df[df["seg_mask"] == MASK]
        df = df[df["metric"] == "Dice"]
        for i, task in enumerate(TASKS):
            init_performance.append(np.mean(df[df["Task"] == task]["value"]))

        confusion_matrix = np.zeros((len(combinations), len(combinations)))
        confusion_matrix_list = []
        for i in range(len(combinations)):
            confusion_matrix_list.append([])
            for j in range(len(combinations)):
                confusion_matrix_list[-1].append([])


        for i in range(len(combinations)):
            print("\n\n", trainer['name'], "trained on", combinations[i])
            ood_scores, threshold = trainer['evaluator'](trainer, i, combinations[i])
            assert np.all(ood_scores['split'] == 'val')
            for k in range(len(combinations)):
                temp = ood_scores[ood_scores['Task'] == TASKS[k]]
                confusion_matrix[i, k] = np.mean(temp['value'])#trained on task i, eval on task k
                confusion_matrix_list[i][k] = temp['value']#trained on task i, eval on task k

        for i in range(len(combinations)):
            for j in range(len(combinations)):
                print(len(confusion_matrix_list[i][j]),end=" ")
            print("\n",end="")
        

        temp = []
        for i in range(len(combinations)):
            temp.extend(np.array(confusion_matrix_list[-1][i]).tolist())

        #res['Dice'] = f"{np.round(np.mean(confusion_matrix[-1, :]) * 100, 1)}#pm{np.round(np.std(confusion_matrix[-1, :]) * 100, 1)}"
        res['Dice'] = f"{np.round(np.mean(temp) * 100, 1)}#pm{np.round(np.std(temp) * 100, 1)}"

        temp = []
        normalizer = []
        for i in range(len(combinations)-1):
            #temp.append(confusion_matrix[-1, i] - confusion_matrix[i, i])

            #Lifelong nnU-Net
            #temp.append((confusion_matrix[-1, i] - confusion_matrix[i, i]) / confusion_matrix[i, i])
            assert len(confusion_matrix_list[-1][i]) == len(confusion_matrix_list[i][i])
            temp.extend((np.array(confusion_matrix_list[-1][i]) - np.array(confusion_matrix_list[i][i])).tolist())
            normalizer.extend(np.array(confusion_matrix_list[i][i]).tolist())
            assert len(temp) == len(normalizer)


        print(temp)
        print(normalizer)
        temp = np.array(temp)
        normalizer = np.array(normalizer)
        #temp = temp / np.mean(normalizer)
        print(temp)

        res['BWT'] = np.round(np.mean(temp) * 100, 1)
        res['BWT'] = f"{np.round(np.mean(temp) * 100, 1)}#pm{np.round(np.std(temp) * 100, 1)}"



        temp = []
        normalizer = []
        for i in range(1, len(combinations)):
            #temp.append(confusion_matrix[i-1, i] - init_performance[i])

            # Lifelong nnU-Net
            #temp.append((confusion_matrix[i, i] - sequential_performance[i]) / sequential_performance[i])
            temp.extend((np.array(confusion_matrix_list[i][i]) - np.array(sequential_performance_list[i])).tolist())
            normalizer.extend(np.array(sequential_performance_list[i]).tolist())
            assert len(temp) == len(normalizer)

        temp = np.array(temp)
        normalizer = np.array(normalizer)
        #temp = temp / normalizer
        
        res['FWT'] = np.round(np.mean(temp) * 100, 1)
        res['FWT'] = f"{np.round(np.mean(temp) * 100, 1)}#pm{np.round(np.std(temp) * 100, 1)}"
            
        res_arr.append(res)
    
    df = pd.DataFrame.from_records(res_arr)
    print(df.to_latex(index=False))
    #plot matrix
    
    exit()

    index = []
    for i in range(len(TASKS)-1):
        label = ""
        for task in TASKS[:i+1]:
            label += rename_tasks(task) + "\n"

        label = label.rstrip("\n")
        index.append(label)

    confusion_df = pd.DataFrame(confusion_matrix * 100, columns=[rename_tasks(t) for t in TASKS[:-1]], index=index)
    ax = sns.heatmap(confusion_df, annot=True, vmin=0, vmax=100, cmap=sns.light_palette("seagreen", as_cmap=True), 
                annot_kws={"size": 20, "color": "black"}, 
                linewidths=0.1, linecolor='black',
                square=True)
    plt.title(trainer['name'], fontsize=22)
    #plt.title("ccVAE", fontsize=22)
    plt.xlabel("Evaluated on", fontsize=18)
    plt.ylabel("Trained on", fontsize=18)
    plt.tick_params(left = False, bottom = False)
    plt.yticks(rotation=0, fontsize=16)
    plt.xticks(fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.savefig(f"plots/cl_conf/{anatomy}/mib_conf.svg", bbox_inches='tight')

    exit()

else:
    print(TABLE)
    assert False