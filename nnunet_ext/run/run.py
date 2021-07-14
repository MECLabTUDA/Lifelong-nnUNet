#########################################################################################################
#----------Alternative entry points without specifying parameters in the console.-----------------------#
#----------See https://github.com/MIC-DKFZ/nnUNet and --------------------------------------------------#
#----------https://github.com/MIC-DKFZ/nnUNet/blob/master/setup.py for details -------------------------#
#########################################################################################################

import sys
import os
import nnunet.experiment_planning.nnUNet_plan_and_preprocess as plan_and_preprocess
import nnunet_ext.nnunet.inference.predict_simple as predict_simple
import nnunet.run.run_training as run_training
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.calibration.uncertainty_calc import estimate_multivariate_gaussian_save_distances
from nnunet_ext.calibration.eval.per_subject_evaluate import per_subject_eval_with_uncertainties

results_path = os.environ.get('RESULTS_FOLDER')

def set_devices(devices=[0]):
    r"""Set GPU indexes.
    :param devices: a list of CUDA devices that can be used in this run.
    """
    for idx, c in enumerate(devices):
        devices[idx] = str(c)  # Change type from int to str, otherwise join_texts_with_char will throw an error
    cuda = join_texts_with_char(devices, ',')    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

def nnUNet_plan_and_preprocess(args=None, task_id=None, 
    verify_dataset_integrity=True):
    r"""Execute the nnUNet_plan_and_preprocess command as in setup.py.
    """
    if args is None:
        args = ['']  # To simulate the path to the called file
        args += ['-t', task_id]
        if verify_dataset_integrity:
            args += ['--verify_dataset_integrity']
    sys.argv = args  # Necessary because the main files don't include args=None
    plan_and_preprocess.main()

def nnUNet_train(args=None, model_type='3d_fullres', trainer='nnUNetTrainerV2', 
    task_id=None, fold_ix=0):
    r"""Execute the nnUNet_train command as in setup.py.
    """
    if args is None:
        args = ['']
        args += [model_type, trainer, task_id, str(fold_ix)]
    sys.argv = args
    run_training.main()

def nnUNet_predict(args=None, input_path=None, output_path=None, task_id=None, 
    model_type='3d_fullres', checkpoint='model_final_checkpoint', fold_ix=0, 
    disable_tta=True, extract_outputs=False, mcdo=-1, softmaxed=True, 
    output_features_path=None, feature_paths=None):
    r"""Execute the nnUNet_predict command as in setup.py.
    """
    if args is None:
        args = ['']
        args += ['-i', input_path]
        args += ['-o', output_path]
        args += ['-t', task_id, '-f', str(fold_ix), '-m', model_type, '-chk', checkpoint]
        if disable_tta:
            args += ['--disable_tta']
        if extract_outputs or mcdo>-1:
            # Instead of predictions, outputs are extracted
            args += ['--output_probabilities']
        if not softmaxed:
            args += ['--no_softmax']
        # If mcdo > -1, Dropout is activated and output ix "mcdo" is extracted
        args += ['-mcdo', str(mcdo)]
        if output_features_path and feature_paths:
            args += ['-of', output_features_path]
            args += ['-feature_paths'] + feature_paths
    sys.argv = args
    predict_simple.main()

def nnUNet_extract_outputs(inputs_path, pred_dataset_name, task_id, 
    model_type='3d_fullres', checkpoint='model_final_checkpoint', fold_ix=0):
    r"""Extract model outputs, before converting into predictions.
    :param inputs_path: folder where images are stored
    :param pred_dataset_name: name of the dataset for which outputs are to be
        extracted (name of images stored in inputs_path)
    :param task_id: Task (dataset name) of the pre-trained model that is loaded
    :param model_type: Network type, e.g. 3d_fullres
    :param checkpoint: Checkpoint from which to restore the model state
    :param fold_ix: Fold of the model instance that is restored, often 0
    """
    extract_path = os.path.join(results_path, 'outputs', task_id, str(fold_ix), pred_dataset_name)
    nnUNet_predict(input_path=inputs_path, output_path=extract_path, task_id=task_id,
        model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix, disable_tta=True,
        extract_outputs=True, mcdo=-1, softmaxed=True)

def nnUNet_extract_non_softmaxed_outputs(inputs_path, pred_dataset_name, task_id, 
    model_type='3d_fullres', checkpoint='model_final_checkpoint', fold_ix=0):
    r"""Extract model outputs before applying the softmax operation.
    :param inputs_path: folder where images are stored
    :param pred_dataset_name: name of the dataset for which outputs are to be
        extracted (name of images stored in inputs_path)
    :param task_id: Task (dataset name) of the pre-trained model that is loaded
    :param model_type: Network type, e.g. 3d_fullres
    :param checkpoint: Checkpoint from which to restore the model state
    :param fold_ix: Fold of the model instance that is restored, often 0
    """
    extract_path = os.path.join(results_path, 'non_softmaxed_outputs', task_id, str(fold_ix), pred_dataset_name)
    nnUNet_predict(input_path=inputs_path, output_path=extract_path, task_id=task_id,
        model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix, disable_tta=True,
        extract_outputs=True, mcdo=-1, softmaxed=False)

def nnUNet_extract_MCDO_outputs(inputs_path, pred_dataset_name, mcdo_ix, task_id,
    model_type='3d_fullres', checkpoint='model_final_checkpoint', fold_ix=0):
    r"""Extract model outputs with activated MC Dropout.
    :param inputs_path: folder where images are stored
    :param pred_dataset_name: name of the dataset for which outputs are to be
        extracted (name of images stored in inputs_path)
    :param mcdo_ix: index that is given to this specific output, for extracting
        an MCDO uncertainty, thisshould be executed with several mcdo_ixs
    :param task_id: Task (dataset name) of the pre-trained model that is loaded
    :param model_type: Network type, e.g. 3d_fullres
    :param checkpoint: Checkpoint from which to restore the model state
    :param fold_ix: Fold of the model instance that is restored, often 0
    """
    extract_path = os.path.join(results_path, 'MC_outputs', task_id, str(fold_ix), pred_dataset_name)
    assert mcdo_ix > -1
    nnUNet_predict(input_path=inputs_path, output_path=extract_path, task_id=task_id,
        model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix, disable_tta=True,
        extract_outputs=True, mcdo=mcdo_ix, softmaxed=True)

def nnUNet_extract_features(inputs_path, pred_dataset_name, feature_paths, task_id, 
    model_type='3d_fullres', checkpoint='model_final_checkpoint', fold_ix=0):
    r"""Extract model activations for nn.Modules specified in 'feature_paths'.
    Please note that not the original features are stored, but rather down-pooled
    variants to reduce mainly the time needed to calculate the covariance matrix, 
    but also the storage space.
    :param inputs_path: folder where images are stored
    :param pred_dataset_name: name of the dataset for which outputs are to be
        extracted (name of images stored in inputs_path)
    :param feature_paths: paths to the feature names, e.g.
        ['conv_blocks_context.6.blocks.1.conv']
    :param task_id: Task (dataset name) of the pre-trained model that is loaded
    :param model_type: Network type, e.g. 3d_fullres
    :param checkpoint: Checkpoint from which to restore the model state
    :param fold_ix: Fold of the model instance that is restored, often 0
    """
    features_root_path = os.path.join(results_path, 'features', task_id, str(fold_ix))
    output_features_path = os.path.join(features_root_path, pred_dataset_name)
    nnUNet_predict(input_path=inputs_path, output_path=output_features_path, task_id=task_id,
                    model_type=model_type, checkpoint=checkpoint, fold_ix=fold_ix, disable_tta=True,
                    extract_outputs=True, mcdo=-1, softmaxed=True, 
                    output_features_path=output_features_path, feature_paths=feature_paths)

def nnUNet_estimate_gaussian(task_id, fold_ix, train_ds_names, store_ds_names,
    feature_paths=None):
    r"""Estimate a multivariate Gaussian distribution and save distances to that
    distribution.
    :param task_id: Task (dataset name) of the pre-trained model that is loaded
    :param fold_ix: Fold of the model instance that is restored, often 0
    :param train_ds_names: A list of dataset names (for which features were 
        previously extracted and the name was stated in 'pred_dataset_name') that
        form in-distribution train data, for estimating the mean and covariance
        matrix.
    :param store_ds_names: A list of other dataset names for which the distances 
        to 'train_ds_names' in the feature space will be stored.
    :param feature_paths: Features that are considered for the distance.
    """
    features_root_path = os.path.join(results_path, 'features', task_id, str(fold_ix))
    estimate_multivariate_gaussian_save_distances(features_root_path, 
        train_ds_names=train_ds_names, store_ds_names=store_ds_names, 
        feature_names=feature_paths)

def nnUNet_extract_uncertainties(pred_dataset_name, task_id, fold_ix, 
    feature_paths, targets_path=None, label=1, nr_labels=2, temperatures=[10]):
    r"""Extract uncertainty values with several methods.
    :param pred_dataset_name: name of the dataset for which outputs are to be
        extracted (name of images stored in inputs_path)
    :param task_id: Task (dataset name) of the pre-trained model that is loaded
    :param fold_ix: Fold of the model instance that is restored, often 0
    :param feature_paths: paths to the feature names, e.g.
        ['conv_blocks_context.6.blocks.1.conv']
    :param targets_path: The directory where targets are stored for 
        'pred_dataset_name',until 'labelsTr'  
    :param label: label for the class of interest
    :param nr_labels: number of classes in the segmentation masks (for 
        one foreground and background nr_labels == 2)
    temperatures: temperatures for which temperature scaling is calculated
    """
    if targets_path is None:
        targets_path = os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data', pred_dataset_name, 'labelsTr')
    predictions_path = os.path.join(results_path, 'predictions', task_id, str(fold_ix), pred_dataset_name)
    outputs_path = os.path.join(results_path, 'outputs', task_id, str(fold_ix), pred_dataset_name)
    non_softmaxed_outputs_path = os.path.join(results_path, 'non_softmaxed_outputs', task_id, str(fold_ix), pred_dataset_name)
    MC_outputs_path = os.path.join(results_path, 'MC_outputs', task_id, str(fold_ix), pred_dataset_name)
    features_root_path = os.path.join(results_path, 'features', task_id, str(fold_ix))
    output_features_path = os.path.join(features_root_path, pred_dataset_name)
    eval_storage_path = os.path.join(os.environ.get('EVALUATION_FOLDER'), task_id, str(fold_ix))
    eval_path = os.path.join(eval_storage_path, pred_dataset_name)
    eval_path = os.path.join(os.environ.get('EVALUATION_FOLDER'), task_id, str(fold_ix), pred_dataset_name)
    base_names = [f[:-7] for f in os.listdir(targets_path) if '.nii.gz' in f]
    df = per_subject_eval_with_uncertainties(eval_path, base_names, 
        predictions_path=predictions_path, targets_path=targets_path, outputs_path=outputs_path, 
        non_softmaxed_outputs_path=non_softmaxed_outputs_path, MC_outputs_path=MC_outputs_path, 
        features_path=output_features_path, feature_key=feature_paths[0],
        label=label, nr_labels=nr_labels, part=fold_ix, temperatures=temperatures)
    df.to_csv(os.path.join(eval_path, 'df.csv'), sep='\t')





