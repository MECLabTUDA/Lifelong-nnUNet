
# Lifelong-nnUNet: Out-of-distribution detection

Lifelong-nnUNet can only be used after it has been succesfully installed *-- including all dependencies --* following [these instructions](../README.md#installation). Further, all relevant paths have to be set so that the right directories for training, preprocessing, storing etc. can be extracted. This process is described [here](setting_up_paths.md).


## Implemented Methods

This project currently includes the following methods for detecting OOD samples:
* [Maximum Softmax](https://arxiv.org/pdf/1610.02136.pdf) consists of taking the maximum softmax output
* [Temperature Scaling](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf) performs temperature scaling on the outputs before applying the softmax operation
* [KL from Uniform](https://arxiv.org/pdf/1906.12340.pdf) computes the KL divergence of the outputs from an uniform distribution
* [Monte Carlo Dropout](http://proceedings.mlr.press/v48/gal16.pdf) consists of doing several forward passes whilst activating the Dropout layers that would usually be dormant during inference
* [Mahalanobis](https://arxiv.org/pdf/2107.05975.pdf) (modifying [this method](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf) for segmentation) consists of 1. extracting low-dimensional features from the in-distribution train data, 2. estimating a multi-variate Gaussian distribution and 3. during test time, considering the Mahalanobis distance of the input to the learned distribution as an uncertainty metric.

All methods output masks with voxel-wise uncertainties (or *confidences* in the case of the first three methods). For OOD detection, these values are averaged, and inverted if necessary, to obtain subject-level novelty scores.

## nnUNet model

The OOD detection methods can be applied to newly-trained full-resolution nnUNet models as well as existing models trained with the [original nnUNet framework](https://github.com/MIC-DKFZ/nnUNet). Until now, we have not had any dependency errors for models trained with different versions, though if a new model needs to be trained we recommend using the nnUNet version specified in the `requirements.txt` file.

You may notice that we adapted the `nnUNetTrainerV2` and `nnUNetTrainer` classes. The `model_restore` file was also adapted so that models can be restored which were trained with the original `nnUNetTrainerV2` trainer. However, please note that, at the time, we only support (by which we mean that we have tested) `3d_fullres` models trained with the `nnUNetTrainerV2` class. The project will soon be updated with support for other models.

Please note that the model must be stored at `RESULTS_FOLDER -> nnUNet`, following the regular nnUNet structure.

If you prefer to work from a different Python module instead of the console, you can train a model with the `nnUNet_train` method in `nnunet_ext.run.run`, which simply mimics the regular `nnUNet_train` command. GPUs can be set with the `set_devices` method, also in `nnunet_ext.run.run`.


## Extraction of Outputs and Features

Before OOD detection methods can be evaluated, output masks and/or features must be extracted from the model. Please note that storing these files may require significant memory depending on the data, so we recommend to first make the extraction for one characteristic subject and then ensure that enough memory is available. By default, all outputs are stored in the `RESULTS_FOLDER` path.

You can extract these outputs with the methods specified below within `nnunet_ext.run.run`. Arguments `input_path` (folder where new images are located) and `pred_dataset_name` (name of the dataset you give to the images in `input_path`) are always required. Arguments `task_id`, `model_type`, `checkpoint` and `fold_ix` are used to specify the model instance from which outputs are extracted.

Depending on the method, following outputs are necessary:

* **Maximum Softmax and KL from Uniform**: For both these methods, simple output masks need to be extracted, so execute `nnUNet_extract_outputs` which sets `extract_outputs = True`.

* **Temperature Scaling**: For this method, it's important that network outputs have not yet been softmaxed, so execute `nnUNet_extract_non_softmaxed_outputs` which sets `extract_outputs = True` and `softmaxed = False`.

* **Monte Carlo Dropout**: For this method, different outputs must be extracted while Dropout is activated, so execute `nnUNet_extract_MCDO_outputs`. This operation should be repeated for N indexes `mcdo > -1` (if `mcdo == -1`, no Dropout is activated). For determining the uncertainty, the standard deviation between all extracted outputs is assessed.

* **Mahalanobis**: For this method, several steps must be carried out:
    1. Extract features for all datasets with `nnUNet_extract_features`, with a `feature_paths` argument specifying for which layers feature maps should be stored. For instance, you may set `feature_paths = ['conv_blocks_context.4.blocks.1.conv']` as all `3d_fullres` models we encountered had at least four blocks.
    2. Estimate the multi-variate Gaussian and save distances to this distribution with `nnUNet_estimate_gaussian`, specifying the name of the in-distribution training dataset(s) (`train_ds_names`) as well as all datasets for which the distance must be extracted (`store_ds_names`). This method uses the stored features to calculate distances to the in-distribution training data, and saves these distances.

In addition, regular predictions should be extracted so that the performance of the model on the new data can be assessed.

## Extraction of Uncertainties

After all extractions specified in the previous step (for the chosen methods) are finished, subject-level uncertainty scores can be extracted. The method `nnUNet_extract_uncertainties` also in `nnunet_ext.run.run` extracts uncertainties for all implemented methods. For each OOD detection method, it extracts a `.json` file with the uncertainty value for each subject. In addition, it extracts Dice and IoU scores for all subjects, and outputs a `df.csv` file which summarizes all uncertainties.

For a more fine-grained extraction of uncertainties and performance values, the behavior of `per_subject_eval_with_uncertainties` in `nnunet_ext.calibration.eval.per_subject_evaluate` should be modified.

## OOD evaluation

Finally, the OOD detection methods can be evaluated. This can be done with the `nnunet_ext.calibration.eval.eval_ood_detection` module, which uses the previously extracted uncertainty and performance values.

The `evaluate_uncertainty_method` method calculates common OOD detection metrics such as FPR, Detection Error and Calibration Error (ECE). For this, the user must specify the names of `id_test` and `ood` datasets for which the evaluation should be performed. 

A plot showing the performance (Dice and IoU) of the model for each dataset can be extracted with the `plot_dataset_performance_boxplot` method.