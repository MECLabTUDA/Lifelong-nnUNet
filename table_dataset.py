import os
from batchgenerators.utilities.file_and_folder_operations import load_json
import SimpleITK as sitk
import numpy as np
import tqdm
import pandas as pd
from skimage.measure import label

ROOT = "/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/nnUNet_raw_data/"


table = []
#labels = load_json(os.path.join(ROOT, "Task204_BraTS4", "dataset.json"))['labels']

for domain in ["Cardiac", "Hippocampus", "Prostate", "Brain"]:
    domain_tasks = {
        "Brain": ["Task306_BraTS6", "Task313_BraTS13", "Task316_BraTS16", "Task320_BraTS20", "Task321_BraTS21"],
        "Hippocampus": ["Task197_DecathHip", "Task198_Dryad", "Task199_HarP"],
        "Prostate": ["Task111_Prostate-BIDMC", "Task112_Prostate-I2CVB", "Task113_Prostate-HK", "Task115_Prostate-UCL", "Task116_Prostate-RUNMC"],
        "Cardiac": ["Task008_mHeartA", "Task009_mHeartB"]
    }
    all_shapes_per_domain = []
    all_num_samples_per_domain = []
    for task in domain_tasks[domain]:
        all_shapes_per_task = []
        subject_ids = os.listdir(os.path.join(ROOT, task, "labelsTr"))
        for segmentation in tqdm.tqdm(subject_ids):
            seg_mask = sitk.ReadImage(os.path.join(ROOT, task, "labelsTr", segmentation))
            #seg_mask = sitk.GetArrayFromImage(seg_mask).astype(int)

            #t1_img = sitk.ReadImage(os.path.join(ROOT, task, "imagesTr", segmentation[:-len(".nii.gz")] + "_0000.nii.gz"))
            #t1_img = sitk.GetArrayFromImage(t1_img)
            all_shapes_per_task.append(sitk.GetArrayFromImage(seg_mask).astype(int).shape)
        all_shapes_per_task = np.array(all_shapes_per_task)

        table.append({
            "Task": task,
            "Nr. subjects": len(subject_ids),
            "Median shape": np.median(all_shapes_per_task, axis=0).astype(int),
        })
        all_shapes_per_domain.extend(all_shapes_per_task)
        all_num_samples_per_domain.append(len(subject_ids))

    #table.append({
    #    "Task": domain,
    #    "Nr. subjects": sum(all_num_samples_per_domain),
    #    "Median shape": np.median(all_shapes_per_task, axis=0).astype(int),
    #})
        

    d = [{
        "temp": np.median(all_shapes_per_task, axis=0).astype(int)
        }]

    all_shapes_per_task_median_str = pd.DataFrame.from_records(d)["temp"].to_string(header=False, index=False)
    table.append({
        "Task": r"\textbf{" + domain + r"}",
        "Nr. subjects": r"\textbf{" + str(sum(all_num_samples_per_domain)) + r"}",
        "Median shape": r"\textbf{" + all_shapes_per_task_median_str + r"}",
    })
        



def rename_tasks(task_str: str):
    if task_str == "Task306_BraTS6":
        return r"\emph{Site6}"
    elif task_str=="Task313_BraTS13":
        return r"\emph{Site13}"
    elif task_str=="Task316_BraTS16":
        return r"\emph{Site16}"
    elif task_str=="Task320_BraTS20":
        return r"\emph{Site20}"
    elif task_str=="Task321_BraTS21":
        return r"\emph{Site21}"
    elif task_str=="Task197_DecathHip":
        return r"\emph{DecathHip}"
    elif task_str=="Task198_Dryad":
        return r"\emph{Dryad}"
    elif task_str=="Task199_HarP":
        return r"\emph{HarP}"
    elif task_str=="Task111_Prostate-BIDMC":
        return r"\emph{BIDMC}"
    elif task_str=="Task112_Prostate-I2CVB":
        return r"\emph{I2CVB}"
    elif task_str=="Task113_Prostate-HK":
        return r"\emph{HK}"
    elif task_str=="Task115_Prostate-UCL":
        return r"\emph{UCL}"
    elif task_str=="Task116_Prostate-RUNMC":
        return r"\emph{RUNMC}"
    elif task_str=="Task008_mHeartA":
        return r"\emph{Siemens}"
    elif task_str=="Task009_mHeartB":
        return r"\emph{Philips}"
    else:
        return task_str
    



df = pd.DataFrame.from_records(table)
df['Task'] = df['Task'].apply(rename_tasks)
print(df)
print(df.to_latex(index=False, escape=False))