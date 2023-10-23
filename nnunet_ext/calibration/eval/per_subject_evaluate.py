# ------------------------------------------------------------------------------
# Per-subject evaluation metrics.
# ------------------------------------------------------------------------------

import os
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
from nnunet_ext.calibration import utils
from nnunet_ext.calibration.uncertainty_calc import softmax_uncertainty, dropout_uncertainty, mahalanobis_uncertainty, kl_uncertainty, energy_scoring, temp_scaled_uncertainty, tta_uncertainty
from nnunet_ext.calibration.eval.per_voxel_evaluate import comp_metrices_new, avg_subject_evaluations

def per_subject_eval_with_uncertainties(eval_path, base_names, predictions_path, 
    targets_path, outputs_path, non_softmaxed_outputs_path, MC_outputs_path, TTA_outputs_path,
    features_path, mahal_features, label=1, nr_labels=2, part=0, temperatures=[1, 10, 100], 
    methods=None, dist_files_name='', patch_size=[28, 256, 256]):

    if methods is None:
        methods = ['MaxSoftmax', 'MCDropout', 'TTA', 'Mahalanobis', 'TempScaling', 'KL', 'EnergyScoring']
    
    #print('\nGetting segmentation results')
    #eval_dict = per_subject_evaluation(eval_path, base_names, predictions_path, targets_path, label=label, avg=True)
    
    if 'MaxSoftmax' in methods:
        print('\nGetting MaxSoftmax uncertainties')
        softmax_uncertainty_dict = per_subject_max_softmax_uncertainty(eval_path, base_names, outputs_path, nr_labels=nr_labels, part=part)
    
    if 'MCDropout' in methods:
        print('\nGetting MCDropout uncertainties')
        mcdo_uncertainty_dict = per_subject_dropout_uncertainty(eval_path, base_names, MC_outputs_path, label=label, norm=False)
    
    if 'TTA' in methods:
        print('\nGetting TTA uncertainties')
        tta_uncertainty_dict = per_subject_tta_uncertainty(eval_path, base_names, TTA_outputs_path, label=label, norm=False)

    if 'Mahalanobis' in methods:
        print('\nGetting Mahalanobis uncertainties')
        spatial_mahal_uncertainty_dict = dict()
        for feature_key, feature_names in mahal_features.items():
            spatial_mahal_uncertainty_dict[feature_key] = per_subject_mahalanobis_uncertainty(eval_path, base_names, features_path, feature_key=feature_key, feature_names=feature_names, norm=False, dist_files_name=dist_files_name, patch_size=patch_size)
    
    if 'TempScaling' in methods:
        print('\nGetting Temperature-scaled uncertainties')
        temp_scaled_uncertainty_dict = dict()
        for temp in temperatures:
            temp_scaled_uncertainty_dict[temp] = per_subject_temp_scaled_uncertainty(eval_path, base_names, non_softmaxed_outputs_path, temp=temp, nr_labels=nr_labels, part=part, norm=False)
    
    if 'KL' in methods:
        print('\nGetting KL divergence uncertainties')
        kl_uncertainty_dict = per_subject_kl_uncertainty(eval_path, base_names, outputs_path, nr_labels=nr_labels, part=part, norm=False)

    if 'EnergyScoring' in methods:
        print('\nGetting Energy Scoring uncertainties')
        energy_scoring_dict = dict()
        for temp in temperatures:
            energy_scoring_dict[temp] = per_subject_energy_scoring(eval_path, base_names, non_softmaxed_outputs_path, temp=temp, nr_labels=nr_labels, part=part, norm=False)
    
    # Design a Pandas dataframe with this content
    data = []
    for base_name in base_names:
        dice = -1 #eval_dict[base_name]['Dice']
        iou = -1 # eval_dict[base_name]['IoU']
        if 'MaxSoftmax' in methods:
            data.append([base_name, dice, iou, softmax_uncertainty_dict[base_name], 'MaxSoftmax'])
        if 'MCDropout' in methods:
            data.append([base_name, dice, iou, mcdo_uncertainty_dict[base_name], 'MCDropout'])
        if 'TTA' in methods:
            data.append([base_name, dice, iou, tta_uncertainty_dict[base_name], 'TTA'])
        if 'TempScaling' in methods:
            for temp in temperatures:
                data.append([base_name, dice, iou, temp_scaled_uncertainty_dict[temp][base_name], 'TempScaling_{}'.format(temp)])
        if 'KL' in methods:
            data.append([base_name, dice, iou, kl_uncertainty_dict[base_name], 'KL'])
        if 'EnergyScoring' in methods:
            for temp in temperatures:
                data.append([base_name, dice, iou, energy_scoring_dict[temp][base_name], 'EnergyScoring_{}'.format(temp)])
        if 'Mahalanobis' in methods:
            for feature_key in mahal_features.keys():
                print(feature_key)
                print(list(spatial_mahal_uncertainty_dict.keys()))
                data.append([base_name, dice, iou, spatial_mahal_uncertainty_dict[feature_key][base_name], 'Mahalanobis_{}'.format(feature_key)])

    df = pd.DataFrame(data, columns=['Subject', 'Dice', 'IoU', 'Uncertainty', 'Method'])
    return df

def per_subject_max_softmax_uncertainty(eval_path, base_names, outputs_path, nr_labels=2, part=0, norm=False):
    full_path = os.path.join(eval_path, 'softmax_uncertainties.json')
    try:
        with open(full_path, 'r') as json_file:
            return json.load(json_file)
    except:
        uncertainties_dict = dict()
        for base_name in tqdm(base_names):
            #try:
            uncertainty = softmax_uncertainty(outputs_path, base_name, nr_labels=nr_labels, part=part, norm=norm)
            uncertainty = average_uncertainty(uncertainty)
            uncertainties_dict[base_name] = float(uncertainty)
            #except:
            #    uncertainties_dict[base_name] = 'Error'
        with open(full_path, 'w') as json_file:
            json.dump(uncertainties_dict, json_file)
        return uncertainties_dict

def per_subject_temp_scaled_uncertainty(eval_path, base_names, non_softmaxed_outputs_path, temp=1000, nr_labels=2, part=0, norm=False):
    full_path = os.path.join(eval_path, 'temp_scaled_uncertainties_{}.json'.format(temp))
    try:
        with open(full_path, 'r') as json_file:
            return json.load(json_file)
    except:
        uncertainties_dict = dict()
        for base_name in tqdm(base_names):
            #try:
            uncertainty = temp_scaled_uncertainty(non_softmaxed_outputs_path, base_name, temp=temp, nr_labels=nr_labels, part=part, norm=norm)
            uncertainty = average_uncertainty(uncertainty)
            uncertainties_dict[base_name] = float(uncertainty)
            #except:
            #    uncertainties_dict[base_name] = 'Error'
        with open(full_path, 'w') as json_file:
            json.dump(uncertainties_dict, json_file)
        return uncertainties_dict

def per_subject_kl_uncertainty(eval_path, base_names, outputs_path, nr_labels=2, part=0, norm=False):
    full_path = os.path.join(eval_path, 'kl_uncertainties.json')
    try:
        with open(full_path, 'r') as json_file:
            return json.load(json_file)
    except:
        uncertainties_dict = dict()
        for base_name in tqdm(base_names):
            #try:
            uncertainty = kl_uncertainty(outputs_path, base_name, nr_labels=nr_labels, part=part, norm=norm)
            uncertainty = average_uncertainty(uncertainty)
            uncertainties_dict[base_name] = float(uncertainty)
            #except:
            #    uncertainties_dict[base_name] = 'Error'
        with open(full_path, 'w') as json_file:
            json.dump(uncertainties_dict, json_file)
        return uncertainties_dict

def per_subject_energy_scoring(eval_path, base_names, non_softmaxed_outputs_path, temp=1000, nr_labels=2, part=0, norm=False):
    full_path = os.path.join(eval_path, 'energy_scoring_{}.json'.format(temp))
    try:
        with open(full_path, 'r') as json_file:
            return json.load(json_file)
    except:
        uncertainties_dict = dict()
        for base_name in tqdm(base_names):
            #try:
            uncertainty = energy_scoring(non_softmaxed_outputs_path, base_name, temp=temp, nr_labels=nr_labels, part=part, norm=norm)
            uncertainty = average_uncertainty(uncertainty)
            uncertainties_dict[base_name] = float(uncertainty)
            #except:
            #    uncertainties_dict[base_name] = 'Error'
        with open(full_path, 'w') as json_file:
            json.dump(uncertainties_dict, json_file)
        return uncertainties_dict

def per_subject_dropout_uncertainty(eval_path, base_names, MC_outputs_path, label=1, norm=False):
    full_path = os.path.join(eval_path, 'dropout_uncertainties.json')
    try:
        with open(full_path, 'r') as json_file:
            return json.load(json_file)
    except:
        uncertainties_dict = dict()
        for base_name in tqdm(base_names):
            #try:
            uncertainty = dropout_uncertainty(MC_outputs_path, base_name, label=label, norm=norm)
            uncertainty = average_uncertainty(uncertainty)
            uncertainties_dict[base_name] = float(uncertainty)
            #except:
            #    uncertainties_dict[base_name] = 'Error'
        with open(full_path, 'w') as json_file:
            json.dump(uncertainties_dict, json_file)
        return uncertainties_dict

def per_subject_tta_uncertainty(eval_path, base_names, TTA_outputs_path, label=1, norm=False):
    full_path = os.path.join(eval_path, 'tta_uncertainties.json')
    try:
        with open(full_path, 'r') as json_file:
            return json.load(json_file)
    except:
        uncertainties_dict = dict()
        for base_name in tqdm(base_names):
            #try:
            uncertainty = tta_uncertainty(TTA_outputs_path, base_name, label=label, norm=norm)
            uncertainty = average_uncertainty(uncertainty)
            uncertainties_dict[base_name] = float(uncertainty)
            #except:
            #    uncertainties_dict[base_name] = 'Error'
        with open(full_path, 'w') as json_file:
            json.dump(uncertainties_dict, json_file)
        return uncertainties_dict

def per_subject_mahalanobis_uncertainty(eval_path, base_names, features_path, feature_key, feature_names, norm=False, dist_files_name='', patch_size=[28, 256, 256]):
    full_path = os.path.join(eval_path, 'mahalanobis_uncertainties_{}_{}_{}.json'.format(feature_key, norm, dist_files_name))
    try:
        with open(full_path, 'r') as json_file:
            return json.load(json_file)
    except:
        uncertainties_dict = dict()
        for base_name in base_names:
            #try:
            uncertainties = []
            for feature_name in feature_names:
                uncertainty = mahalanobis_uncertainty(features_path, base_name, feature_name, norm=norm, dist_files_name=dist_files_name, patch_size=patch_size)
                uncertainty = average_uncertainty(uncertainty)
                uncertainties.append(uncertainty)
            uncertainties_dict[base_name] = float(sum(uncertainties))
            #except:
            #    uncertainties_dict[base_name] = 'Error'
        with open(full_path, 'w') as json_file:
            json.dump(uncertainties_dict, json_file)
        return uncertainties_dict

def average_uncertainty(uncertainty):
    r"""Calculate a subject-based score from an uncertainty array.
    """
    return np.mean(uncertainty.flatten())

def per_subject_evaluation(eval_path, base_names, predictions_path, targets_path, label=1, avg=True):
    r"""Returns a dictionary with a confidence assessment per subject.
    These are the confidence scores gathered from get_subject_confidence

    Returns (dict[str -> float]): dict[subject name -> confidence/rating]
    """
    subjects_eval_path = os.path.join(eval_path, 'subjects')
    if not os.path.isdir(subjects_eval_path):
        os.makedirs(subjects_eval_path)
    all_eval_dicts = dict()
    for base_name in tqdm(base_names):
        # If evaluation for this subject exists, reload. Otherwise extract
        subject_eval_path = os.path.join(subjects_eval_path, base_name+'.json')
        try:
            with open(subject_eval_path, 'r') as json_file:
                all_eval_dicts[base_name] = json.load(json_file)
        except:
            prediction_path = os.path.join(predictions_path, base_name+'.nii.gz')
            target_path = os.path.join(targets_path, base_name+'.nii.gz')
            subject_eval = evaluate_subject(prediction_path, target_path, label=label)
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
    return all_eval_dicts

def evaluate_subject(prediction_path, target_path, label=1, merge_labels=None):
    prediction = utils.load_nifty(target_path)[0].astype(np.float16)  # # utils.load_nifty(prediction_path)[0].astype(np.float16)  # # ignore path of predictions during testing
    ground_truth = utils.load_nifty(target_path)[0].astype(np.float16)
    case_eval = evaluate_case(prediction, ground_truth, label=label, merge_labels=merge_labels)
    return case_eval

def evaluate_case(prediction, ground_truth, label=1, merge_labels=None):
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
    case_scores = comp_metrices_new(prediction, ground_truth, thresholded_uncertainty=None)
    return case_scores

