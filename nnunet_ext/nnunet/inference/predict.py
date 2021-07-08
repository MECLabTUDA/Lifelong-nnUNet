#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Taken from commit specified in requirements.txt from https://github.com/MIC-DKFZ/nnUNet.
# Changes include:
# - Removing parser logic, as in our case only called through nnunet_ext.nnunet.inference.predict_simple
# - Leaving only the default 'normal' mode, as it is stated in predict_simple to use this mode
# - Adding predict_from_folder arguments output_probabilities, uncertainty_tta and mcdo

import argparse
from copy import deepcopy
from typing import Tuple, Union, List

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet_ext.nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Process, Queue
import torch
import SimpleITK as sitk
import shutil
from multiprocessing import Pool
from nnunet.postprocessing.connected_components import load_remove_save, load_postprocessing
from nnunet_ext.nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet_ext.nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.one_hot_encoding import to_one_hot

def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes,
                             transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            d, _, dct = preprocess_fn(l)
            # print(output_file, dct)
            if segs_from_prev_stage[i] is not None:
                assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                    ".nii.gz"), "segs_from_prev_stage" \
                                " must point to a " \
                                "segmentation file"
                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                # check to see if shapes match
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                seg_prev = seg_prev.transpose(transpose_forward)
                seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1, cval=0)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            """There is a problem with python process communication that prevents us from communicating obejcts 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")
    # restore output
    # sys.stdout = sys.__stdout__


def preprocess_multithreaded(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            output_files[i::num_processes],
                                                            segs_from_prev_stage[i::num_processes],
                                                            classes, trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # this should not happen but better safe than sorry right
            p.join()

        q.close()


def predict_cases(model, list_of_lists, output_filenames, folds, save_npz, num_threads_preprocessing,
                  num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, mixed_precision=True, 
                  overwrite_existing=False,
                  all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                  segmentation_export_kwargs: dict = None, disable_postprocessing: bool = False,
                  output_probabilities: bool = False, tta: int = -1, mcdo: int = -1, no_softmax=False,
                  features_dirs=None, feature_paths=None):
    """
    :param segmentation_export_kwargs:
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param list_of_lists: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
    :param output_filenames: [output_file_case0.nii.gz, output_file_case1.nii.gz, ...]
    :param folds: default: (0, 1, 2, 3, 4) (but can also be 'all' or a subset of the five folds, for example use (0, )
    for using only fold_0
    :param save_npz: default: False
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param segs_from_prev_stage:
    :param do_tta: default: True, can be set to False for a 8x speedup at the cost of a reduced segmentation quality
    :param overwrite_existing: default: True
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        # if save_npz=True then we should also check for missing npz files
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not isfile(j)) or (save_npz and not isfile(j[:-7] + '.npz'))]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name, mcdo=mcdo)
    # If required, disable 'self.network.inference_apply_nonlin = softmax_helper',
    # set during the trainer initialization
    if no_softmax:
        trainer.network.inference_apply_nonlin = lambda x: x

    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)
    print("starting prediction...")
    all_output_files = []
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        all_output_files.append(all_output_files)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data
        
        if features_dirs:  # Feature extraction
            _, filename = os.path.split(output_filename)
            filename = filename.split('.')[0]
            features_dir = [d for d in features_dirs if filename in os.path.split(d)[1]]
            assert len(features_dir) == 1
            features_dir = features_dir[0]
            for p in params:
                trainer.load_checkpoint_ram(p, False)
                trainer.save_features(
                    d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                    step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                    mixed_precision=mixed_precision, tta=tta, mcdo=mcdo, 
                    features_dir=features_dir, feature_paths=feature_paths)[1][None]
        else:  # Save predictions or network outputs
            # If file exists, do not export
            out_fname = output_filename
            if output_probabilities and tta != -1:
                part = tta
            elif output_probabilities and mcdo != -1:
                part = mcdo
            elif output_probabilities:
                part = folds[0]
            else:
                part = -1
            if part != -1:
                out_fname = out_fname[:-7] + "_" + str(0) + "_part_" + str(part) + ".nii.gz"
            else:
                out_fname = out_fname[:-7] + "_" + str(0) + ".nii.gz"

            # This has changed somewhat from the original nnUNet_uncertainty code due to changes in the nnUNet code
            if os.path.isfile(out_fname):
                print('File found, not predicting')
            else:
                print("predicting", output_filename)
                trainer.load_checkpoint_ram(params[0], False)
                softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                    d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                    step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                    mixed_precision=mixed_precision, tta=tta, mcdo=mcdo)[1]  # Added tta=tta, mcdo=mcdo

                for p in params[1:]:
                    trainer.load_checkpoint_ram(p, False)
                    softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                        d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                        step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                        mixed_precision=mixed_precision, tta=tta, mcdo=mcdo)[1]  # Added tta=tta, mcdo=mcdo

                if len(params) > 1:
                    softmax /= len(params)

                transpose_forward = trainer.plans.get('transpose_forward')
                if transpose_forward is not None:
                    transpose_backward = trainer.plans.get('transpose_backward')
                    softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

                if save_npz:  # No
                    npz_file = output_filename[:-7] + ".npz"
                else:
                    npz_file = None

                if hasattr(trainer, 'regions_class_order'):
                    region_class_order = trainer.regions_class_order  # Is None
                else:
                    region_class_order = None

                """There is a problem with python process communication that prevents us from communicating obejcts 
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                filename or np.ndarray and will handle this automatically"""
                bytes_per_voxel = 4
                if all_in_gpu:
                    bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
                if np.prod(softmax.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
                    print(
                        "This output is too large for python process-process communication. Saving output temporarily to disk")
                    np.save(output_filename[:-7] + ".npy", softmax)
                    softmax = output_filename[:-7] + ".npy"

                # TODO: is this necessary to repeat?
                if output_probabilities and tta != -1:
                    part = tta
                elif output_probabilities and mcdo != -1:
                    part = mcdo
                elif output_probabilities:
                    part = folds[0]
                else:
                    part = -1

                results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                ((softmax, output_filename, dct, interpolation_order, region_class_order,
                                                    None, None,
                                                    npz_file, None, force_separate_z, interpolation_order_z, True, output_probabilities, part),)
                                                )) # Added True, output_probabilities, part

            print("inference done. Now waiting for the segmentation export to finish...")
            _ = [i.get() for i in results]
            # now apply postprocessing
            # first load the postprocessing properties if they are present. Else raise a well visible warning
            if not disable_postprocessing:
                results = []
                pp_file = join(model, "postprocessing.json")
                if isfile(pp_file):
                    print("postprocessing...")
                    shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
                    # for_which_classes stores for which of the classes everything but the largest connected component needs to be
                    # removed
                    for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
                    results.append(pool.starmap_async(load_remove_save,
                                                    zip(output_filenames, output_filenames,
                                                        [for_which_classes] * len(output_filenames),
                                                        [min_valid_obj_size] * len(output_filenames))))
                    _ = [i.get() for i in results]
                else:
                    print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                        "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                        "%s" % model)

    pool.close()
    pool.join()


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def predict_from_folder(model: str, input_folder: str, output_folder: str, folds: Union[Tuple[int], List[int]],
                        save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
                        lowres_segmentations: Union[str, None],
                        part_id: int, num_parts: int, tta: bool, mixed_precision: bool = True,
                        overwrite_existing: bool = True, mode: str = 'normal', overwrite_all_in_gpu: bool = None,
                        step_size: float = 0.5, checkpoint_name: str = "model_final_checkpoint",
                        segmentation_export_kwargs: dict = None, disable_postprocessing: bool = False,
                        output_probabilities: bool = False, uncertainty_tta: int = -1, mcdo: int = -1, no_softmax = False,
                        features_folder=None, feature_paths=None):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases

    :param model:
    :param input_folder:
    :param output_folder:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :param mixed_precision:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there. None is default (no overwrite)
    :return:
    """
    maybe_mkdir_p(output_folder)
    shutil.copy(join(model, 'plans.pkl'), output_folder)

    assert isfile(join(model, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))['num_modalities']

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]

    if lowres_segmentations is not None:
        assert isdir(lowres_segmentations), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [join(lowres_segmentations, i + ".nii.gz") for i in case_ids]
        assert all([isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present. " \
                                                               "(I was searching for case_id.nii.gz in that folder)"
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None

    if features_folder is not None:
        if not os.path.isdir(features_folder):
            os.makedirs(features_folder)
        features_dirs = [join(features_folder, i) for i in case_ids]
        features_dirs = features_dirs[part_id::num_parts]
    else:
        features_dirs = None

    if mode == "normal":
        if overwrite_all_in_gpu is None:
            all_in_gpu = False
        else:
            all_in_gpu = overwrite_all_in_gpu


        return predict_cases(model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds,
                             save_npz, num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations, tta,
                             mixed_precision=mixed_precision, overwrite_existing=overwrite_existing,
                             all_in_gpu=all_in_gpu,
                             step_size=step_size, checkpoint_name=checkpoint_name,
                             segmentation_export_kwargs=segmentation_export_kwargs,
                             disable_postprocessing=disable_postprocessing,
                             output_probabilities=output_probabilities, tta=uncertainty_tta, mcdo=mcdo, no_softmax=no_softmax,
                             features_dirs=features_dirs, feature_paths=feature_paths)
    else:
        raise ValueError("unrecognized mode. Must be normal.")