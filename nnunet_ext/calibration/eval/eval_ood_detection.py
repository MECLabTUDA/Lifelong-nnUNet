# ------------------------------------------------------------------------------
# Main file to extract OOD results
# ------------------------------------------------------------------------------

import os
import numpy as np
import math
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
import nnunet_ext.calibration.eval.plotting as plotting
from pers.utils.load_restore import pkl_load

# If the metric is a "confidence", invert
invert_uncertainty = ['MaxSoftmax', 'KL', 'TempScaling'] + ['TempScaling_{}'.format(t) for t in [1, 10, 100, 1000]]

def fetch_validation_cases(ds_names):
    val_subject_names = []
    for ds_name in ds_names:
        path=os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', ds_name, 'labelsTr')
        subjects = [f.replace('.nii.gz', '') for f in os.listdir(path)]
        val_subject_names += subjects
    return val_subject_names

def fetch_test_validation_cases(ds_names, fold_ix=4):
    val_subject_names = []
    test_subject_names = []
    for ds_name in ds_names:
        splits = pkl_load('splits_final.pkl', path=os.path.join(os.environ['nnUNet_preprocessed'], ds_name))
        val_subject_names += list(splits[fold_ix]['train'])
        test_subject_names += list(splits[fold_ix]['val'])
    return test_subject_names, val_subject_names

def results_dict_to_df(df_items):
    r"""Convert again to DF for easier plotting
    """
    df_data = []
    columns = list(df_items[0].keys())
    for df_item in df_items:
        df_data.append([df_item[col] for col in columns])
    return pd.DataFrame(df_data, columns=columns)

def load_results(eval_storage_path, method='MaxSoftmax', id_val=[], id_test=[], ood=[], 
    nr_val_cases=4, exclude_fold_val_cases=False, df_subdir=None):
    r"""Fetches results in df form and build a joint dictionary"""
    if exclude_fold_val_cases:
        test_cases, validation_cases = fetch_test_validation_cases(id_val)
    else:
        validation_cases = fetch_validation_cases(id_val)
    all_dfs_items = []
    for ds_key in id_val + id_test + ood:
        if df_subdir is None:
            eval_path = os.path.join(eval_storage_path, ds_key)
        else:
            eval_path = os.path.join(eval_storage_path, ds_key, df_subdir)
        df = pd.read_csv(os.path.join(eval_path, 'df.csv'), sep='\t')
        df = df[df.Method == method]
        df[["Uncertainty", "Dice", "IoU"]] = df[["Uncertainty", "Dice", "IoU"]].apply(pd.to_numeric)
        df_items = list(df.set_index(df.Subject).T.to_dict().values())
        # Pick validation cases from the ID Test dataset
        if len(validation_cases) < 1:
            raise
            validation_cases = [df_item['Subject'] for df_item in df_items]
            validation_cases = random.choices(validation_cases, k=nr_val_cases)
        # Make relevant modifications to the data structure
        for ix in range(len(df_items)):
            if method in invert_uncertainty:
                df_items[ix]['Uncertainty'] = 1 - df_items[ix]['Uncertainty']
            df_items[ix]['Dataset'] = ds_key
            if ds_key in ood:
                df_items[ix]['Dist'] = 'OOD'
            else:
                df_items[ix]['Dist'] = 'ID'
            if exclude_fold_val_cases:
                if ds_key in id_val and df_items[ix]['Subject'] not in test_cases:
                    df_items[ix]['Split'] = 'Val'
                else:
                    df_items[ix]['Split'] = 'Test'
            else:
                if (ds_key in id_val) or (ds_key in id_test and df_items[ix]['Subject'] in validation_cases):
                    df_items[ix]['Split'] = 'Val'
                else:
                    df_items[ix]['Split'] = 'Test'
        all_dfs_items += df_items
    return all_dfs_items

def get_tp_tn_fn_fp(items, boundary):
    tp = sum([item['Uncertainty']<=boundary and item['Dist']=='ID' for item in items])
    tn = sum([item['Uncertainty']>boundary and item['Dist']=='OOD' for item in items])

    fn = sum([item['Uncertainty']>boundary and item['Dist']=='ID' for item in items])
    fp = sum([item['Uncertainty']<=boundary and item['Dist']=='OOD' for item in items])
    return tp, tn, fn, fp

def find_boundary_iid_tpr(items, tpr=0.95):
    r"""Find threshold for which the true positive rate of classifying examples as
    iid is tpr. Usual tpr taken (e.g. in https://arxiv.org/pdf/1706.02690.pdf,
    https://arxiv.org/pdf/1807.03888v2.pdf) is 0.95.
    """
    uncertainties = []
    train_items = [x for x in items if x['Split']=='Val']
    uncertainties = [x['Uncertainty'] for x in train_items]
    sorted_uncertainties = sorted(uncertainties)
    ix_below_boundary = math.floor((len(uncertainties)-1)*tpr)
    ix_above_boundary = ix_below_boundary + 1
    boundary = (sorted_uncertainties[ix_below_boundary] + sorted_uncertainties[ix_above_boundary])/2
    return boundary

def detection_error(tp, tn, fn, fp):
    r"""Detection error, taken as described in https://arxiv.org/pdf/1706.02690.pdf.
    Error = 0.5(1 - TPR) + 0.5 FPR
    """
    error = 0.
    if (tp+fn) > 0:
        tpr = tp/(tp+fn)
        error += 0.5*(1-tpr)
    if (fp+tn) > 0:
        fpr = fp/(fp+tn)
        error += 0.5*fpr
    return error

def ood_detection_error(items, boundary):
    r"""Taken as described in https://arxiv.org/pdf/1706.02690.pdf.
    Error = 0.5(1-TPR) + 0.5(FPR).
    """
    test_items = [x for x in items if x['Split']=='Test']
    tp, tn, fn, fp = get_tp_tn_fn_fp(test_items, boundary)
    return detection_error(tp, tn, fn, fp)

def fpr(tp, tn, fn, fp):
    r"""False positive rate.
    """
    if (fp+tn) > 0:
        return fp/(fp+tn)
    else:
        return None

def ood_detection_fpr(items, boundary):
    r"""Probability that an ood example is misclassified as iid"""
    test_items = [x for x in items if x['Split']=='Test']
    tp, tn, fn, fp = get_tp_tn_fn_fp(test_items, boundary)
    return fpr(tp, tn, fn, fp)

def auroc_score(items):
    r"""Area under ROC"""
    test_items = [x for x in items if x['Split']=='Test']
    if 'NormedUncertainty' not in items[0]:
        set_normed_uncertainties(items)
    labels = [item['Dist']=='ID' for item in test_items]
    scores = [1 - item['NormedUncertainty'] for item in test_items]
    return roc_auc_score(labels, scores)

def avg_score_below_boundary(items, boundary, score='Dice'):
    test_items = [x for x in items if x['Split']=='Test']
    test_items_certain = [x for x in test_items if x['Uncertainty']<=boundary]
    scores = [x[score] for x in test_items_certain]
    if len(scores) > 0:
        return np.mean(scores), np.std(scores), len(test_items_certain)/len(test_items)
    else:
        return np.nan, np.nan, 0

def uncertainty_boundaries(items, mult=2):
    val_items = [item for item in items if item['Split'] == 'Val']
    uncertainties = [x['Uncertainty'] for x in val_items]
    min_b = max(-10000, min(uncertainties))
    return min_b, mult*max(uncertainties)

def set_normed_uncertainties(items):
    min_b, max_b = uncertainty_boundaries(items)
    for item in items:
        value = item['Uncertainty']
        val_possible_range = max(min(value, max_b), min_b)
        item['NormedUncertainty'] = (val_possible_range-min_b)/(max_b - min_b)

def norm_boundary(items, boundary):
    min_b, max_b = uncertainty_boundaries(items)
    val_possible_range = max(min(boundary, max_b), min_b)
    return (val_possible_range-min_b)/(max_b - min_b)

def set_confidences(items):
    for item in items:
        item['Confidence'] = 1 - item['NormedUncertainty']

def _get_calibration_bins(items, metric='Dice', nr_bins=10, start=0, end=1):
    test_items = [x for x in items if x['Split']=='Test']
    bin_boundaries = [start+((end-start)/(nr_bins))*(1+bin_ix) for bin_ix in range(nr_bins)]
    bin_confidences = [[] for ix in range(nr_bins)]
    bin_scores = [[] for ix in range(nr_bins)]

    # Normalize uncertainties
    if 'NormedUncertainty' not in items[0]:
        set_normed_uncertainties(items)
    if 'Confidence' not in items[0]:
        set_confidences(items)
    # Fill in bins
    for item in test_items:
        # Add confidence and score to respective bin
        confidence = item['Confidence']
        score = item[metric]
        subject_bin_ix = None
        for bin_ix, bin_max_confidence in enumerate(bin_boundaries):
            if confidence <= bin_max_confidence:
                subject_bin_ix = bin_ix
                break
        bin_confidences[subject_bin_ix].append(confidence)
        bin_scores[subject_bin_ix].append(score)
    return bin_boundaries, bin_confidences, bin_scores

def calibration_error(items, metric='Dice', nr_bins=10):
    r"""Calculate the expected calibration error (for segmentation,
    i.e. with 'metric')
    """
    bin_boundaries, bin_confidences, bin_scores = _get_calibration_bins(items, 
        metric=metric, nr_bins=nr_bins)
    # Calculate the expected calibration error
    ece = 0
    total_samples = sum(len(bin_scores[bin_ix]) for bin_ix in range(nr_bins))
    for bin_ix in range(nr_bins):
        samples_in_bin = len(bin_scores[bin_ix])
        if samples_in_bin > 0:
            avg_conf = np.mean(bin_confidences[bin_ix])
            avg_score = np.mean(bin_scores[bin_ix])
            bin_ece = (samples_in_bin/total_samples)*abs(avg_conf-avg_score)
            ece += bin_ece
    return ece

def evaluate_uncertainty_method(eval_storage_path, results_name='results', 
    methods=None, id_test=[], ood=[], id_val=[], exclude_fold_val_cases=False, df_subdir=None):
    r"""Returns a number of measures to assess OOD detection and calibration quality.
    """
    if methods is None:
        methods=['MaxSoftmax', 'MCDropout', 'KL', 'Mahalanobis', 'TTA'] + ['TempScaling_{}'.format(t) for t in [1, 10, 100, 1000]] + ['EnergyScoring_{}'.format(t) for t in [10, 100, 1000]]
    df_data = []
    for method in methods:
        #print('Method: {}'.format(method))
        items = load_results(eval_storage_path=eval_storage_path, method=method, 
            id_val=id_val, id_test=id_test, ood=ood, exclude_fold_val_cases=exclude_fold_val_cases, df_subdir=df_subdir)

        # Calibration error
        ece = calibration_error(items, metric='Dice')

        # Boundary TPR 0.95 on i.i.d. validation data
        boundary_TPR = find_boundary_iid_tpr(items, tpr=0.95)

        # OOD detection measures
        detection_error = ood_detection_error(items, boundary_TPR)
        fpr = ood_detection_fpr(items, boundary_TPR)
        auroc = auroc_score(items)
        dice_mean, dice_std, coverage = avg_score_below_boundary(items, boundary_TPR, score='Dice')
        iou_mean, iou_std, coverage = avg_score_below_boundary(items, boundary_TPR, score='IoU')

        df_data.append([method, ece, detection_error, fpr, auroc, dice_mean, dice_std, 
            iou_mean, iou_std, coverage, boundary_TPR])
        
    # join results for all methods
    df = pd.DataFrame(df_data, columns=['Method', 'ECE', 'Error', 'FPR', 'AUROC',
        'Dice_MEAN', 'Dice_STD', 'IoU_MEAN', 'IoU_STD', 'Coverage', 'Boundary'])
    if df_subdir is None:
        df.to_csv(os.path.join(eval_storage_path, results_name+'.csv'), sep='\t')
    else:
        df_final_path = os.path.join(eval_storage_path, df_subdir)
        if not os.path.exists(df_final_path):
            os.makedirs(df_final_path)
        df.to_csv(os.path.join(df_final_path, results_name+'.csv'), sep='\t')
    return df

def plot_method_scatter(df_with_boundary, eval_storage_path, method, id_test, 
    ood, id_val=[], exclude_fold_val_cases=False, better_ds_names=None, normalize=False, hue='Dist', df_subdir=None):
    r"""Plot a scatter of the uncertainty against the Dice
    """
    boundary = float(df_with_boundary.loc[df_with_boundary['Method'] == method]['Boundary'])
    items = load_results(eval_storage_path=eval_storage_path, 
        method=method, id_val=id_val, id_test=id_test, ood=ood, exclude_fold_val_cases=exclude_fold_val_cases, df_subdir=df_subdir)
    if normalize and 'NormedUncertainty' not in items[0]:
        set_normed_uncertainties(items)
        boundary = norm_boundary(items, boundary)
    items = [item for item in items if item['Split'] == 'Test']
    
    if better_ds_names:
        # Leave only the cases specified in the dict, and prettify names
        items = [item for item in items if item['Dataset'] in better_ds_names]
        for item in items:
            item['Dataset'] = better_ds_names[item['Dataset']]

    df = results_dict_to_df(items)

    if df_subdir is not None:
        eval_storage_path = os.path.join(eval_storage_path, df_subdir)
    plotting.plot_uncertainty_performance(df, metric='Dice', hue=hue, 
        style='Split', boundary=boundary, 
        save_name='uncertainty_vs_dice_{}'.format(method), save_path=eval_storage_path, normalize=normalize)

def plot_dataset_performance_boxplot(eval_storage_path, id_test, ood, better_ds_names=None):
    r"""Plot the base performance of the method.
    """
    items = load_results(eval_storage_path=eval_storage_path, 
        method='MaxSoftmax', id_test=id_test, ood=ood, nr_val_cases=0)
    
    if better_ds_names:
        for item in items:
            item['Dataset'] = better_ds_names[item['Dataset']]
            item['All'] = 'all'
    
    duplicated_items = []
    for item in items:
        for metric in ['Dice', 'IoU']:
            new_item = item.copy()
            new_item['Metric'] = metric
            new_item['Value'] = new_item[metric]
            duplicated_items.append(new_item)
    
    df = results_dict_to_df(duplicated_items)

    plotting.boxplot(df, x='Metric', y='Value', hue='Dataset', 
        save_path=eval_storage_path, file_name='boxplot_performance.png')

def plot_separation_boxplot(eval_storage_path, methods, id_test, ood, id_val=[], better_method_names=None):
    all_items = []
    for method in methods:
        items = load_results(eval_storage_path=eval_storage_path, 
            method=method, id_val=id_val, id_test=id_test, ood=ood)
        set_normed_uncertainties(items)
        if better_method_names:
            if method in better_method_names:
                for item in items:
                    item['Method'] = better_method_names[method]
        all_items += items

    df = results_dict_to_df(all_items)
    plotting.boxplot(df, x='Method', y='NormedUncertainty', hue='Dist', 
        save_path=eval_storage_path, file_name='boxplot_separation.png')