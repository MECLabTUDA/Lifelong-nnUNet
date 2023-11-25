import os, shutil
from nnunet_ext.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier, nnUNet_raw_data, base
import pandas as pd
from nnunet.dataset_conversion.utils import generate_dataset_json

SITE_ID = 18

modalities = ["t1n", "t1c", "t2w", "t2f"]
modalities_full_names =  ["T1 native", "post-contrast T1-weighted (T1Gd)", 
                          "T2-weighted", "T2 Fluid Attenuated Inversion Recovery (T2-FLAIR)"]

ROOT = os.path.join(base, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")

TARGET = os.path.join(nnUNet_raw_data, f"Task{200 + SITE_ID}_BraTS{SITE_ID}")

# create the target directories
os.makedirs(os.path.join(TARGET, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(TARGET, "imagesTs"), exist_ok=True)
os.makedirs(os.path.join(TARGET, "labelsTr"), exist_ok=True)

# load meta data from csv file
df = pd.read_csv(os.path.join(ROOT, "BraTS2023_2017_GLI_Mapping.csv"), sep=";")
df = df[df["Site No (represents the originating institution)"] == SITE_ID]

print(df)
for _, row in df.iterrows():
    if isinstance(row["BraTS2023"], float):
        print("skip:", row["BraTS2023"])
        continue
    # copy images
    for modality_i, modality in enumerate(modalities):
        src = os.path.join(ROOT, row["BraTS2023"] , row["BraTS2023"] +  f"-{modality}.nii.gz")
        dst = os.path.join(TARGET, "imagesTr", f"{row['BraTS2023']}_{modality_i:04d}.nii.gz")
        shutil.copy(src, dst)
    
    # copy labels
    src = os.path.join(ROOT, row["BraTS2023"], row["BraTS2023"]+ "-seg.nii.gz")
    dst = os.path.join(TARGET, "labelsTr", f"{row['BraTS2023']}.nii.gz")
    shutil.copy(src, dst)

generate_dataset_json(os.path.join(TARGET, "dataset.json"), os.path.join(TARGET, "imagesTr"), 
                      os.path.join(TARGET, "imagesTs"), modalities_full_names, 
                      {0: "background", 1: "NCR (necrotic tumor core)", 2: "ED (peritumoral edematous/invaded tissue)", 3: "ET (GD-enhancing tumor)"}, 
                      f"ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData-Site{SITE_ID}", 
                      dataset_description=f"ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData-Site{SITE_ID}", 
                      dataset_reference="https://www.synapse.org/#!Synapse:syn51156910/wiki/621282", 
                      dataset_release="0.0")