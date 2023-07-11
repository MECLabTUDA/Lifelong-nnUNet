import glob
import os

start = "/local/scratch/clmn1/master_thesis/results_folder"

for a in ["distilled_output", "ground_truth", "retrained/distilled_output", "retrained/ground_truth"]:
    print(a)
    for file in glob.glob(os.path.join(start, a, "*/*/nnUNet_ext/3d_fullres/*/metadata/Generic_UNet/SEQ/feature_rehearsal2/extracted_features/*/*.pkl")):
        os.remove(file)
    for file in glob.glob(os.path.join(start, a, "*/*/nnUNet_ext/3d_fullres/*/metadata/Generic_UNet/SEQ/feature_rehearsal2/extracted_features/*/*.npy")):
        os.remove(file)