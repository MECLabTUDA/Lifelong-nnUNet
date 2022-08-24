import numpy as np
import shutil, os, pickle, json
import SimpleITK as sitk
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data
from nnunet.preprocessing.cropping import get_case_identifier, load_case_from_list_of_files
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p

def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split(os.sep)[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
            # -- Add label of fixed and moving image during registration -- #
            cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split(os.sep)[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}

def no_crop(task_string, override=False, *args):
    r"""Same as crop, but just copies the data and makes no cropping --> used for registration only.
    """
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)
    nr_dims = _copy(lists, overwrite_existing=override, output_folder=cropped_out_dir)
    
    shutil.copy(join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)
    return nr_dims
    
    
def _copy(list_of_files, overwrite_existing=False, output_folder=None):
    r"""Replaces the actual function in the nnUNet that would perform the cropping of the images
    """
    # -- Copy segmentations first -- #
    output_folder_gt = os.path.join(output_folder, "gt_segmentations")
    maybe_mkdir_p(output_folder_gt)
    for case in list_of_files:
        case_identifier = get_case_identifier(case)
        print(case_identifier)
        labels = [x for x in case if 'labels' in x]
        imgs = [x for x in case if 'labels' not in x]
        for lab in labels:
            shutil.copy(lab, output_folder_gt)
        # -- Copy the data as well in one file: F - F - M - M (img - seg; F=fixed, M=moving) -- #
        try:
            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(output_folder, "%s.npz" % case_identifier))
                        or not os.path.isfile(os.path.join(output_folder, "%s.pkl" % case_identifier))):
                # -- No cropping here! -- #
                data_f, seg_f, properties = load_case_from_list_of_files([imgs[0]], [labels[0]])
                data_m, seg_m, _ = load_case_from_list_of_files([imgs[1]], [labels[1]])
                if data_f.shape[:2] == (1, 1):
                    data_f = data_f[0]
                if seg_f.shape[:2] == (1, 1):
                    seg_f = seg_f[0]
                if data_m.shape[:2] == (1, 1):
                    data_m = data_m[0]
                if seg_m.shape[:2] == (1, 1):
                    seg_m = seg_m[0]
                    
                properties["size_after_cropping"] = data_f[0].shape
                properties["classes"] = np.unique(seg_f)
                all_data = np.vstack((data_f, data_m, seg_m, seg_f))
                # all_data = np.vstack((data_f, seg_f, data_m, seg_m))
                np.savez_compressed(os.path.join(output_folder, "%s.npz" % case_identifier), data=all_data)
                with open(os.path.join(output_folder, "%s.pkl" % case_identifier), 'wb') as f:
                    pickle.dump(properties, f)
                    
                # -- Save labels with correct dimensions (B, C, D, H, W) in gt_segmentation folder -- #
                # sitk.WriteImage(sitk.GetImageFromArray(seg_f), os.path.join(output_folder_gt, labels[0].split(os.sep)[-1]))
                # sitk.WriteImage(sitk.GetImageFromArray(seg_m), os.path.join(output_folder_gt, labels[1].split(os.sep)[-1]))
                
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e
        
    return all_data.shape[0]