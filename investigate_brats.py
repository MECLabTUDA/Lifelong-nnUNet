import os
from batchgenerators.utilities.file_and_folder_operations import load_json
import SimpleITK as sitk
import numpy as np
import tqdm
import pandas as pd
from skimage.measure import label, regionprops
import skimage.measure

ROOT = "/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/nnUNet_raw_data/"

def compute_num_tumors(seg_mask, class_idx: int):
    binary_mask = (seg_mask == class_idx).astype(int)
    labeled_mask = skimage.measure.label(binary_mask)
    regions = regionprops(labeled_mask)

    tumor_count = 0
    min_tumor_area = 100 
    for region in regions:
        if region.area > min_tumor_area:
            tumor_count += 1
    return tumor_count

per_task = []
labels = load_json(os.path.join(ROOT, "Task204_BraTS4", "dataset.json"))['labels']
for task in ["Task306_BraTS6", "Task313_BraTS13", "Task316_BraTS16", "Task320_BraTS20", "Task321_BraTS21"]:
    rel_amount = []
    for segmentation in tqdm.tqdm(os.listdir(os.path.join(ROOT, task, "labelsTr"))):
        seg_mask = sitk.ReadImage(os.path.join(ROOT, task, "labelsTr", segmentation))
        seg_mask = sitk.GetArrayFromImage(seg_mask).astype(int)

        t1_img = sitk.ReadImage(os.path.join(ROOT, task, "imagesTr", segmentation[:-len(".nii.gz")] + "_0000.nii.gz"))
        t1_img = sitk.GetArrayFromImage(t1_img)

        num_voxels = np.prod(seg_mask.shape)
        num_brain_voxels = np.count_nonzero(t1_img)
        assert num_brain_voxels <= num_voxels

        seg_labels, seg_counts = np.unique(seg_mask, return_counts=True)
        seg_count_dict = dict(zip(seg_labels, seg_counts))

        rel_amount_per_label = []
        for i, label in enumerate(labels):
            #rel_amount_per_label.append(seg_count_dict.get(i, 0) / num_voxels)
            #rel_amount_per_label.append(compute_num_tumors(seg_mask, i))
            rel_amount_per_label.append(seg_count_dict.get(i, 0) / num_brain_voxels)

        rel_amount.append(rel_amount_per_label)


    rel_amount = np.array(rel_amount)
    rel_amount = rel_amount.mean(axis=0)
    d = {'Task': task}
    for i, label in enumerate(labels):
        d[labels[label]] = rel_amount[i]
    per_task.append(d)



def my_format(x):
    if isinstance(x, float):
        return round(x*100, 3)
    else:
        return x



df = pd.DataFrame.from_records(per_task)
print(df)
df = df.applymap(my_format)
print(df.to_latex(index=False))