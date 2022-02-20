# ------------------------------------------------------------------------------
# Per-voxel evaluation metrics.
# ------------------------------------------------------------------------------

import os
import numpy as np
import json
from tqdm import tqdm
from nnunet_ext.calibration.eval.evaluate_uncertainties import comp_dice_score, threshold_uncertainty
from nnunet.evaluation.metrics import dice, jaccard
from nnunet_ext.calibration import utils
from numbers import Number

def evaluate_subjects(eval_path, base_names, predictions_path, targets_path, uncertainties_path=None, thresholds=None, label=1, part=None, avg=True, plot_uncertainties=True):
    subjects_eval_path = os.path.join(eval_path, 'subjects')
    if not os.path.isdir(subjects_eval_path):
        os.makedirs(subjects_eval_path)
    all_eval_dicts = dict()
    all_subjects_df = []
    for base_name in tqdm(base_names):
        # If evaluation for this subject exists, reload. Otherwise extract
        subject_eval_path = os.path.join(subjects_eval_path, base_name+'.json')
        try:
            with open(subject_eval_path, 'r') as json_file:
                all_eval_dicts[base_name] = json_file
        except:
            prediction_path = os.path.join(predictions_path, base_name+'.nii.gz')
            target_path = os.path.join(targets_path, base_name+'.nii.gz')

            uncertainty_path = None
            if uncertainties_path:
                if part:
                    uncertainty_path = os.path.join(uncertainties_path, '{}_{}_part_{}.nii.gz'.format(base_name, label, part))
                else:
                    uncertainty_path = os.path.join(uncertainties_path, '{}_{}.nii.gz'.format(base_name, label))
            subject_eval, subject_df = evaluate_subject(prediction_path, target_path, uncertainty_path=uncertainty_path, thresholds=thresholds, label=label, plot_uncertainties=plot_uncertainties)
            all_subjects_df += subject_df
            with open(subject_eval_path, 'w') as json_file:
                json.dump(subject_eval, json_file)
            all_eval_dicts[base_name] = subject_eval
    # Average evaluation for all subjects and store mean and std
    if avg:
        mean_dict, std_dict = avg_subject_evaluations(all_eval_dicts)
        with open(os.path.join(eval_path, 'MEAN.json'), 'w') as json_file:
            json.dump(mean_dict, json_file)
        with open(os.path.join(eval_path, 'STD.json'), 'w') as json_file:
            json.dump(std_dict, json_file)

def avg_subject_evaluations(eval_dicts):
    r"""Receives a dictionary with key -> dict values, where all dict values have the same structure.
    Averages these are returns a dict with 2 keys: MEAN and STD
    """
    mean_dict, std_dict = average_dictionaries(*eval_dicts.values())
    return mean_dict, std_dict

def evaluate_subject(prediction_path, target_path, uncertainty_path=None, thresholds=None, label=1, merge_labels={2: 1}, plot_uncertainties=False):
    prediction = utils.load_nifty(prediction_path)[0].astype(np.float16)
    ground_truth = utils.load_nifty(target_path)[0].astype(np.float16)
    uncertainty = None
    if uncertainty_path:
        uncertainty = utils.load_nifty(uncertainty_path)[0].astype(np.float16)
    case_eval, df = evaluate_case(prediction, ground_truth, uncertainty=uncertainty, thresholds=thresholds, label=label, merge_labels=merge_labels, plot_uncertainties=plot_uncertainties)
    return case_eval, df

def evaluate_case(prediction, ground_truth, uncertainty=None, thresholds=None, label=1, merge_labels=None, plot_uncertainties=False):
    prediction = np.rint(prediction)
    ground_truth = np.rint(ground_truth)
    prediction = prediction.astype(int)
    ground_truth = ground_truth.astype(int)
    if merge_labels:
        for old_label, new_label in merge_labels.items():
            ground_truth[ground_truth == old_label] = new_label
            prediction[prediction == old_label] = new_label
    prediction[prediction != label] = 0
    ground_truth[ground_truth != label] = 0
    prediction[prediction == label] = 1
    ground_truth[ground_truth == label] = 1
    # Returns a dictionary from thresholds to metrics
    df = None
    if uncertainty is not None and thresholds is not None:
        uncertainty = utils.normalize(uncertainty)
        case_scores = dict()
        for threshold in thresholds:
            thresholded_uncertainty = threshold_uncertainty(uncertainty, threshold)
            thresholded_uncertainty = thresholded_uncertainty.astype(int)
            metrices = comp_metrices_new(prediction, ground_truth, thresholded_uncertainty)
            case_scores[threshold] = metrices
            if plot_uncertainties:
                df = update_uncertainty_df(df, prediction, ground_truth, thresholded_uncertainty)
    # Returns a dictionary of metrics
    else:
        case_scores = comp_metrices_new(prediction, ground_truth, thresholded_uncertainty=None)
    return case_scores, df

def update_uncertainty_df(df, prediction, ground_truth, thresholded_uncertainty):
    pass

def comp_metrices_new(prediction, ground_truth, thresholded_uncertainty=None):
    result = dict()
    result['Dice'] = dice(test=prediction, reference=ground_truth)
    result['IoU'] = jaccard(test=prediction, reference=ground_truth)
    if thresholded_uncertainty is not None:
        tp_u, fp_u, tn_u, fn_u = comp_confusion_matrix_correctly_uncertain(prediction, ground_truth, thresholded_uncertainty)
        result['Dice_correctly_uncertain'] = comp_dice_score(tp_u, fp_u, tn_u, fn_u)
        tp_c, fp_c, tn_c, fn_c = comp_confusion_matrix_avoiding_uncertain(prediction, ground_truth, thresholded_uncertainty)
        result['Dice_avoiding_uncertain'] = comp_dice_score(tp_c, fp_c, tn_c, fn_c)
        result['coverage'] = sum([tp_c, fp_c, tn_c, fn_c])/ground_truth.size
    return result

def comp_confusion_matrix_correctly_uncertain(prediction, ground_truth, uncertainty):
    r"""Does the uncertainty matrix correctly classify when he prediction was wrong?
    tp: correctly identified good prediction
    fp: identified bad prediction as good
    fn: identified good prediction as bad
    tn: correctly identified bad prediction
    """
    tp = np.sum(((prediction == 1) & (ground_truth == 1) & (uncertainty == 0)))
    tp += np.sum(((prediction == 0) & (ground_truth == 0) & (uncertainty == 0)))

    fp = np.sum(((prediction == 1) & (ground_truth == 0) & (uncertainty == 0)))
    fp += np.sum(((prediction == 0) & (ground_truth == 1) & (uncertainty == 0)))

    fn = np.sum(((prediction == 1) & (ground_truth == 1) & (uncertainty == 1)))
    fn += np.sum(((prediction == 0) & (ground_truth == 0) & (uncertainty == 1)))

    tn = np.sum(((prediction == 1) & (ground_truth == 0) & (uncertainty == 1)))
    tn += np.sum(((prediction == 0) & (ground_truth == 1) & (uncertainty == 1)))

    return tp, fp, tn, fn

def comp_confusion_matrix_avoiding_uncertain(prediction, ground_truth, uncertainty):
    r"""Everything for which uncertainty is high is avoided in the calculation.
    """
    tp = ((prediction == 1) & (ground_truth == 1) & (uncertainty == 0))
    tn = ((prediction == 0) & (ground_truth == 0) & (uncertainty == 0))
    fp = ((prediction == 1) & (ground_truth == 0) & (uncertainty == 0))
    fn = ((prediction == 0) & (ground_truth == 1) & (uncertainty == 0))
    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)
    return tp, fp, tn, fn

def comp_iou_score(tp, fp, tn, fn):
    if tp + fp + fn == 0:
        if tn > 0:
            return 1
        else:
            return 0
    else:
        return tp/(tp+fp+fn)

def average_dictionaries(*dicts):
    r"""For several dictionaries, averages the values for which they share
    the same keys into a new dictionary, which is returned."""
    mean_dict = dict()
    std_dict = dict()
    # Find common keys
    common_keys = set(dicts[0].keys())
    if len(dicts) > 1:
        for d in dicts:
            common_keys = common_keys.intersection(d.keys())
    # Average values
    for common_key in common_keys:
        values = [d[common_key] for d in dicts]
        if all(isinstance(val, dict) for val in values):  # Rec. apply function for dictionaries
            mean_dict[common_key], std_dict[common_key] = average_dictionaries(*values)
        elif all(isinstance(val, Number) for val in values):  # Average numeric values
            mean_dict[common_key] = np.mean(values)
            std_dict[common_key] = np.std(values)
    return mean_dict, std_dict
