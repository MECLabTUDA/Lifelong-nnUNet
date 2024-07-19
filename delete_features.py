import glob
import os
from itertools import chain

start = "/local/scratch/clmn1/master_thesis/results_folder"

for a in ["distilled_output", "ground_truth", "retrained/distilled_output", "retrained/ground_truth"]:
    print(a)
    for file in glob.glob(os.path.join(start, a, "*/*/nnUNet_ext/3d_fullres/*/metadata/Generic_UNet/SEQ/feature_rehearsal2/extracted_features/*/*.pkl")):
        os.remove(file)
    for file in glob.glob(os.path.join(start, a, "*/*/nnUNet_ext/3d_fullres/*/metadata/Generic_UNet/SEQ/feature_rehearsal2/extracted_features/*/*.npy")):
        os.remove(file)

start2 = "/local/scratch/clmn1/master_thesis"

files_iter = chain(
(glob.glob(os.path.join(start2, "**/extracted_features_tr/*/*/*.pkl"   ), recursive=True)),
(glob.glob(os.path.join(start2, "**/extracted_features_val/*/*/*.pkl" ) , recursive=True)),
(glob.glob(os.path.join(start2, "**/generated_features_tr/*/*/*.pkl"   ), recursive=True)),
(glob.glob(os.path.join(start2, "**/extracted_features_tr/*/*/*.npy"   ), recursive=True)),
(glob.glob(os.path.join(start2, "**/extracted_features_val/*/*/*.npy" ) , recursive=True)),
(glob.glob(os.path.join(start2, "**/generated_features_tr/*/*/*.npy"   ), recursive=True)),
(glob.glob(os.path.join(start2, "**/generated_features_tr/*/meta.pkl"), recursive=True)),
)


for file in files_iter:
    print(file)
    os.remove(file)