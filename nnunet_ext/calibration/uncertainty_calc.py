# ------------------------------------------------------------------------------
# Calculate uncertainty/confidence masks from outputs.
# ------------------------------------------------------------------------------

import os
import numpy as np
import pickle
from nnunet_ext.calibration import utils
from tqdm import tqdm
import ast
from scipy.ndimage.filters import gaussian_filter
from math import log2
from nnunet_ext.calibration.mahalanobis.density_estimation import GaussianDensityEstimator
from scipy.special import softmax, logsumexp
from multiprocessing import Pool

def softmax_uncertainty(outputs_path, base_name, nr_labels=2, part=0, 
    norm=False, invert=False, store_mask=False):
    r"""Considers maximum softmax value as the confidence for each voxel.
    """
    uncertainty = []
    for label in range(nr_labels):
        uncertainty_path = os.path.join(outputs_path, '{}_{}_part_{}.nii.gz'.format(base_name, label, part))
        label_uncertainty = utils.load_nifty_no_metadata(uncertainty_path).astype(np.float16)
        uncertainty.append(label_uncertainty)
    uncertainty = np.stack(uncertainty, axis=0)
    uncertainty = np.max(uncertainty, axis=0)
    if norm:
        uncertainty = utils.normalize(uncertainty)
    return uncertainty

def dropout_uncertainty(MC_outputs_path, base_name, label=1, norm=False,
    store_npy=False, store_mask=False):
    r"""Considers the standard deviation between outputs as the uncertainty for 
    each voxel.
    """
    # Gather all relevant files
    full_path = os.path.join(MC_outputs_path, '{}_uncertainty.npy'.format(base_name))
    try:
        return np.load(full_path)
    except:
        uncertainty_file_paths = [f for f in os.listdir(MC_outputs_path) 
            if '{}_{}_part'.format(base_name, label) in f]
        uncertainty_file_paths = [os.path.join(MC_outputs_path, f) 
            for f in uncertainty_file_paths]
        predictions = []
        for output in uncertainty_file_paths:
            prediction = utils.load_nifty_no_metadata(output).astype(np.float16)
            predictions.append(prediction)
        predictions = np.stack(predictions)
        uncertainty = np.std(predictions, axis=0)
        if norm:
            uncertainty = utils.normalize(uncertainty)
        if store_npy:
            np.save(full_path, uncertainty)
        return uncertainty

def tta_uncertainty(TTA_outputs_path, base_name, label=1, norm=False,
    store_npy=False, store_mask=False):
    r"""Considers the standard deviation between outputs as the uncertainty for 
    each voxel.
    """
    # Gather all relevant files
    full_path = os.path.join(TTA_outputs_path, '{}_uncertainty.npy'.format(base_name))
    try:
        return np.load(full_path)
    except:
        uncertainty_file_paths = [f for f in os.listdir(TTA_outputs_path) 
            if '{}_{}_part'.format(base_name, label) in f]
        uncertainty_file_paths = [os.path.join(TTA_outputs_path, f) 
            for f in uncertainty_file_paths]
        predictions = []
        for output in uncertainty_file_paths:
            prediction = utils.load_nifty_no_metadata(output).astype(np.float16)
            predictions.append(prediction)
        predictions = np.stack(predictions)
        uncertainty = np.std(predictions, axis=0)
        if norm:
            uncertainty = utils.normalize(uncertainty)
        if store_npy:
            np.save(full_path, uncertainty)
        return uncertainty

def _kl_div_from_uniform(p, smoothing=0.0001):
    p = [float(x) if x>0 else smoothing for x in p]
    p = [x/sum(p) for x in p]
    q = [1/(len(p))]*len(p)
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def kl_uncertainty(outputs_path, base_name, nr_labels=2, part=0, norm=False,
    store_npy=False, invert=False, store_mask=False):
    r"""Considers the KL divergence from an uniform distribution as the 
    confidence for each voxel.
    """
    full_path = os.path.join(outputs_path, '{}_kl_uncertainty.npy'.format(base_name))
    #try:
    #    return np.load(full_path)
    #except:
    label_outputs = []
    for label in range(nr_labels):
        uncertainty_path = os.path.join(outputs_path, 
            '{}_{}_part_{}.nii.gz'.format(base_name, label, part))
        label_output = utils.load_nifty_no_metadata(uncertainty_path).astype(np.float16)
        label_outputs.append(label_output)
    kl_shape = label_outputs[0].shape
    kl = np.zeros(kl_shape)
    for ix in np.ndindex(kl_shape):
        kl[ix] = _kl_div_from_uniform([x[ix] for x in label_outputs])
    if norm:
        kl = utils.normalize(kl)
    if store_npy:
        np.save(full_path, kl)
    return kl

def energy_scoring(non_softmaxed_outputs_path, base_name, temp=1000, 
    nr_labels=2, part=0, norm=False, invert=False, store_mask=False):
    r"""Applies Energy Scoring.
    """
    outputs = []
    for label in range(nr_labels):
        output_path = os.path.join(non_softmaxed_outputs_path, 
            '{}_{}_part_{}.nii.gz'.format(base_name, label, part))
        output = utils.load_nifty_no_metadata(output_path).astype(np.float16)
        outputs.append(output)
    outputs = np.stack(outputs, axis=0)
    outputs /= temp
    uncertainty = - logsumexp(outputs, axis=0, keepdims=True) * temp
    if norm:
        uncertainty = utils.normalize(uncertainty)
    return uncertainty

def temp_scaled_uncertainty(non_softmaxed_outputs_path, base_name, temp=1000, 
    nr_labels=2, part=0, norm=False, invert=False, store_mask=False):
    r"""Applies temperature scaling before considering the max. softmax as voxel
    confidence.
    """
    outputs = []
    for label in range(nr_labels):
        output_path = os.path.join(non_softmaxed_outputs_path, 
            '{}_{}_part_{}.nii.gz'.format(base_name, label, part))
        output = utils.load_nifty_no_metadata(output_path).astype(np.float16)
        outputs.append(output)
    outputs = np.stack(outputs, axis=0)
    outputs /= temp
    outputs = softmax(outputs, axis=0)
    uncertainty = np.max(outputs, axis=0)
    if norm:
        uncertainty = utils.normalize(uncertainty)
    return uncertainty

def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    r"""Code replicated from the nnUNet (https://github.com/MIC-DKFZ/nnUNet) 
    neural network module, to avoid initializing."""
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def mahalanobis_uncertainty(features_path, base_name, feature_name, 
    patch_size=[28, 256, 256], use_gaussian=True, norm=False, dist_files_name=''):
    distances_full_path = os.path.join(features_path, base_name + dist_files_name + '_distances.pkl')
    feature_distances = pickle.load(open(distances_full_path, 'rb'))
    # Weight middle more heavily
    if use_gaussian:
        gaussian_importance_map = _get_gaussian(patch_size)
    # Get steps from patch keys
    patch_keys = [(ast.literal_eval(patch_key), patch_key) for patch_key in list(feature_distances.keys())]
    img_shape = (max(x[0][0] for x in patch_keys), max(x[0][1] for x in patch_keys), max(x[0][2] for x in patch_keys))
    aggregated_results = np.zeros(img_shape, dtype=np.float32)
    for patch_key, patch_key_str in patch_keys:
        ub_x, ub_y, ub_z = patch_key
        lb_x, lb_y, lb_z = ub_x-patch_size[0], ub_y-patch_size[1], ub_z-patch_size[2]
        distance = feature_distances[patch_key_str][feature_name]
        distance_patch = np.full(patch_size, float(distance))
        if use_gaussian:
            distance_patch *= gaussian_importance_map
        aggregated_results[lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += distance_patch
    if norm:
        aggregated_results = utils.normalize(aggregated_results)
    return aggregated_results

def get_feature_estimator(train_features):
    # Build arrays with shape (#samples, dims)
    train_features = np.stack(train_features)
    estimator = GaussianDensityEstimator()
    estimator.fit(train_features)
    return estimator

def save_distances_subject(base_name, features_path, feature_names, estimators, files_name=''):
    full_path = os.path.join(features_path, base_name)
    features = pickle.load(open(full_path, 'rb'))
    # Initialize distances dictionary
    distances_name = base_name.replace('.pkl', files_name+'_distances.pkl')
    distances_full_path = os.path.join(features_path, distances_name)
    distances = dict()
    for patch_key, value in features.items():
        distances[patch_key] = dict()
        for feature_name in feature_names:
            feature_value = value[feature_name]
            feature_value = feature_value.flatten()
            distances[patch_key][feature_name] = estimators[feature_name].get_mahalanobis(feature_value)
    pickle.dump(distances, open(distances_full_path, "wb"))

def estimate_multivariate_gaussian_save_distances(features_root_path, 
    train_ds_names, store_ds_names, feature_names=None, files_name='', train_ds_cases=None):
    r"""Estimates a distribution using the training data and saves the distances
    for all other provided datasets.
    """
    train_features = dict()

    # Extract features
    print('Extracting features')
    for ds_name in train_ds_names:
        features_path = os.path.join(features_root_path, ds_name)
        base_names = list(os.listdir(features_path))
        base_names = [x for x in base_names if 'pkl' in x and 
            'plans' not in x and 'distances' not in x]
            
        # Decide on a subset of cases to use for training, so val cases do not need
        # to be separated previously.
        if train_ds_cases is not None:
            base_names = [x for x in base_names if x.replace('.pkl', '') in train_ds_cases[ds_name]]
            assert len(base_names) > 0, 'There are no training cases'

        for base_name in base_names:
            full_path = os.path.join(features_path, base_name)
            features = pickle.load(open(full_path, 'rb'))
            for patch_key, value in features.items():
                for feature_key, feature_value in value.items():
                    if feature_key not in train_features:
                        train_features[feature_key] = []
                    flattened_feature = feature_value.flatten()
                    train_features[feature_key].append(flattened_feature)

    # Distances are calculated for these features
    if feature_names is None:
        feature_names = list(train_features.keys())

    # Fit multivariate Gaussian estimation
    print('Fitting estimators features')
    estimators = dict()
    for feature_name in tqdm(feature_names):
        # Build arrays with shape (#samples, dims)
        train_features[feature_name] = np.stack(train_features[feature_name])
        estimators[feature_name] = GaussianDensityEstimator()
        estimators[feature_name].fit(train_features[feature_name])
    #p = Pool(8)
    #estimators_lst = p.map(get_feature_estimator, [train_features[feature_name] for feature_name in feature_names])
    #p.close()
    #p.join()
    #estimators = {feature_name: estimator for feature_name, estimator in zip(feature_names, estimators_lst)}

    # Finally, for each other training set create a directory similar to the 
    # features but storing distances for each patch instead of features
    for ds_name in store_ds_names:
        print('Extracting distances for dataset {}'.format(ds_name))
        features_path = os.path.join(features_root_path, ds_name)
        base_names = list(os.listdir(features_path))
        base_names = [x for x in base_names if 'pkl' in x 
            and 'plans' not in x and 'distances' not in x]
        #p = Pool(8)
        #p.starmap(save_distances_subject, [(base_name, features_path, feature_names, estimators, files_name) for base_name in base_names])
        #p.close()
        #p.join()
        for base_name in tqdm(base_names):
            full_path = os.path.join(features_path, base_name)
            features = pickle.load(open(full_path, 'rb'))
            # Initialize distances dictionary
            distances_name = base_name.replace('.pkl', files_name+'_distances.pkl')
            distances_full_path = os.path.join(features_path, distances_name)
            distances = dict()
            for patch_key, value in features.items():
                distances[patch_key] = dict()
                for feature_name in feature_names:
                    feature_value = value[feature_name]
                    feature_value = feature_value.flatten()
                    distances[patch_key][feature_name] = estimators[feature_name].get_mahalanobis(feature_value)
            pickle.dump(distances, open(distances_full_path, "wb"))
            