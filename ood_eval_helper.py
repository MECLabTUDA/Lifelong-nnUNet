import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


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


def eval_segmentation_distortion(trainer, last_task_idx, trained_tasks, combinations, TASKS, END, METRIC, MASK, val_only=True, merge_seg=True):
    ood_scores = []
    for task in TASKS:
        ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base_ood'], trainer['eval_path_middle'], 
                                                trained_tasks, trainer['trainer'] + END, task,
                                                f"ood_scores_{trainer['method']}.csv"), sep="\t"))
        
    ood_scores = pd.concat(ood_scores)
        

    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], trainer['eval_path_middle'],
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

    return ood_scores, max(threshold_per_task), threshold_per_task


def eval_softmax(trainer, last_task_idx, trained_tasks, combinations, TASKS, END, METRIC, MASK, val_only=True, merge_seg=True):
    ood_scores = []
    for task in TASKS:
        ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base_ood'], trainer['eval_path_middle'], 
                                                trained_tasks, trainer['trainer'] + END, task,
                                                f"ood_scores_{trainer['method']}.csv"), sep="\t"))
    ood_scores = pd.concat(ood_scores)

    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], trainer['eval_path_middle'],
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

    return ood_scores, max(threshold_per_task), threshold_per_task


def eval_scaled_softmax(trainer, last_task_idx, trained_tasks, combinations, TASKS, END, METRIC, MASK, val_only=True, merge_seg=True):
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

    return ood_scores, max(threshold_per_task), threshold_per_task



def eval_reconstruction(trainer, last_task_idx ,trained_tasks, combinations, TASKS, END, METRIC, MASK, val_only=True, merge_seg=True):
    ood_scores = []
    for task in TASKS:
        ood_scores.append(pd.read_csv(os.path.join(trainer['eval_path_base_ood'], trainer['eval_path_middle'], 
                                                trained_tasks, trainer['trainer'] + END, task,
                                                f"ood_scores_{trainer['method']}.csv"), sep="\t"))
    ood_scores = pd.concat(ood_scores)

    segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], trainer['eval_path_middle'],
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

    return ood_scores, max(threshold_per_task), threshold_per_task


def eval_segmentation_distortion_pool(trainer, last_task_idx, trained_tasks, combinations, TASKS, END, METRIC, MASK, val_only=True, merge_seg=True):
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
        if i is not last_task_idx:
            parent_dir = os.path.abspath(os.path.join(trainer['eval_path_base_seg'], os.pardir))
            segmentation_res = pd.read_csv(os.path.join(parent_dir, "trained_final", trainer['eval_path_middle'],
                                                    combinations[i], trainer['segmentation_trainer'] + END, "val_metrics_all.csv"), sep="\t")
        else:
            segmentation_res = pd.read_csv(os.path.join(trainer['eval_path_base_seg'], trainer['eval_path_middle'],
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

    return ood_scores, max(threshold_per_task), threshold_per_task
