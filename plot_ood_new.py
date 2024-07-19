import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd
import os

plt.rcParams['text.usetex'] = True


def rename_tasks(task_name: str):
    if task_name.endswith("DecathHip"):
        return "DecathHip"
    elif task_name.endswith("Dryad"):
        return "Dryad"
    elif task_name.endswith("HarP"):
        return "HarP"
    elif task_name.endswith("mHeartA"):
        return "Siemens"
    elif task_name.endswith("mHeartB"):
        return "Philips"
    elif task_name.endswith("Prostate-BIDMC"):
        return "BIDMC"
    elif task_name.endswith("Prostate-I2CVB"):
        return "I2CVB"
    elif task_name.endswith("Prostate-HK"):
        return "HK"
    elif task_name.endswith("Prostate-UCL"):
        return "UCL"
    elif task_name.endswith("Prostate-RUNMC"):
        return "RUNMC"
    return "unknown task"

def task_color(task_name: str):
    if task_name.endswith("DecathHip"):
        return "red"
    elif task_name.endswith("Dryad"):
        return "green"
    elif task_name.endswith("HarP"):
        return "blue"
    elif task_name.endswith("mHeartA"):
        return "purple"
    elif task_name.endswith("mHeartB"):
        return "orange"
    elif task_name.endswith("Prostate-BIDMC"):
        return "brown"
    elif task_name.endswith("Prostate-I2CVB"):
        return "pink"
    elif task_name.endswith("Prostate-HK"):
        return "gray"
    elif task_name.endswith("Prostate-UCL"):
        return "olive"
    elif task_name.endswith("Prostate-RUNMC"):
        return "cyan"
    return "black"

def rename_val(val: str):
    if val == "val":
        return "test"
    elif val == "train":
        return "train"
    elif val == "test":
        return "val"
    return "unknown"

def hippocampus_vae_mse_0():
    root_path = "/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task097_DecathHip_Task098_Dryad_Task099_HarP"
    trained_on = ["Task097_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkips"
    file = "ood_scores_vae_reconstruction.csv"

    threshold_on_95_train = 0.0055873438483104

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"MSE of reconstruction and original > tau \implies OOD", threshold_on_95_train

def hippocampus_uncertainty_0():
    root_path = "/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task097_DecathHip_Task098_Dryad_Task099_HarP"
    trained_on = ["Task097_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkips"
    file = "ood_scores_uncertainty.csv"

    threshold_on_95_train = 0.0025233626365661

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"Uncertainty > \tau \implies OOD", threshold_on_95_train


# [x] threshold = 0.0171928624132354 (estimated on train and val)
# [ ] threshold = 0.0055873438483104 (estimated on train only)
def hippocampus_uncertainty_mse_temperature_0():
    root_path = "/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task097_DecathHip_Task098_Dryad_Task099_HarP"
    trained_on = ["Task097_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkips"
    file = "ood_scores_uncertainty_mse_temperature.csv"

    threshold_on_95_train = 0.0021607652306556

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"Scaled Uncertainty > \tau \implies OOD", threshold_on_95_train


def hippocampus_uncertainty_3_split_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_uncertainty.csv"

    threshold_on_95_validation = 0.002795398235321

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"Uncertainty > \tau \implies OOD", threshold_on_95_validation



def hippocampus_vae_mse_3_split_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_vae_reconstruction.csv"

    threshold_on_95_validation = 0.0221546945561255

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, "VAE reconstruction", threshold_on_95_validation


#temperature: 0.0221546945561255
def hippocampus_uncertainty_mse_temperature_3_split_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_uncertainty_mse_temperature.csv"

    threshold_on_95_validation = 0.0022281791482652

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"Scaled Uncertainty > \tau \implies OOD", threshold_on_95_validation






def prostate_vae_mse_3_split_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC"
    trained_on = ["Task111_Prostate-BIDMC"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_vae_reconstruction.csv"

    threshold_on_95_validation = 0

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, "VAE reconstruction", threshold_on_95_validation

##################################################################

def seeded_hippocampus_sequential_softmax_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerSequential"
    file = "ood_scores_uncertainty.csv"

    threshold_on_95_test = 0.0025082319974899

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"1-Softmax Sequential trained on \emph{DecathHip}", threshold_on_95_test

def seeded_hippocampus_sequential_softmax_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerSequential"
    file = "ood_scores_uncertainty.csv"

    threshold_on_95_test = 0.0025082319974899

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"1-Softmax Sequential trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test


def seeded_hippocampus_rehearsal_softmax_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerRehearsal"
    file = "ood_scores_uncertainty.csv"

    threshold_on_95_test = 0.0025082319974899

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"1-Softmax Rehearsal trained on \emph{DecathHip}", threshold_on_95_test

def seeded_hippocampus_rehearsal_softmax_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerRehearsal"
    file = "ood_scores_uncertainty.csv"

    threshold_on_95_test = 0.0025082319974899

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"1-Softmax Rehearsal trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test


def seeded_hippocampus_seg_dist_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerSegDist"
    file = "ood_scores_segmentation_distortion.csv"

    threshold_on_95_test = 0.0010104899249999998

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"Segmentation Distortion trained on \emph{DecathHip}", threshold_on_95_test

def seeded_hippocampus_seg_dist_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerSegDist"
    file = "ood_scores_segmentation_distortion.csv"

    threshold_on_95_test = 0.0048686641

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"Segmentation Distortion trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test


def seeded_hippocampus_seg_dist_pool_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerSegDist"
    file = "ood_scores_segmentation_distortion.csv"

    threshold_on_95_test = 0.0010104899249999998

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"Segmentation Distortion + Model Pool trained on \emph{DecathHip}", threshold_on_95_test


def seeded_hippocampus_seg_dist_pool_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerSegDist"
    file = "ood_scores_segmentation_distortion.csv"

    threshold_on_95_test = max(0.0010104899249999998, 0.0048686641)

    csv_path0 = os.path.join(root_path, all_tasks, "Task197_DecathHip",f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    csv_path1 = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return [csv_path0, csv_path1], trained_on, r"Segmentation Distortion + Model Pool trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test


def seeded_hippocampus_cvae_seg_dist_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkips"
    file = "ood_scores_segmentation_distortion.csv"

    threshold_on_95_test = 0.00336454252

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"cVAE Segmentation Distortion trained on \emph{DecathHip}", threshold_on_95_test

def seeded_hippocampus_cvae_seg_dist_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerVAERehearsalNoSkips"
    file = "ood_scores_segmentation_distortion.csv"

    threshold_on_95_test = max(0.00336454252, 0.0094586729)

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"cVAE Segmentation Distortion trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test


def seeded_hippocampus_ccvae_seg_dist_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_segmentation_distortion.csv"

    threshold_on_95_test = 0.0033189527749999987

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"ccVAE Segmentation Distortion trained on \emph{DecathHip}", threshold_on_95_test

def seeded_hippocampus_ccvae_seg_dist_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_segmentation_distortion.csv"

    threshold_on_95_test = max(0.0033189527749999987, 0.00992243975)

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"ccVAE Segmentation Distortion trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test


def seeded_hippocampus_cvae_reconstruction_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkips"
    file = "ood_scores_vae_reconstruction.csv"

    threshold_on_95_test = 0.018849379789815938

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"cVAE Reconstruction trained on \emph{DecathHip}", threshold_on_95_test

def seeded_hippocampus_cvae_reconstruction_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerVAERehearsalNoSkips"
    file = "ood_scores_vae_reconstruction.csv"

    threshold_on_95_test = max(0.018849379789815938, 0.019073720282499856)

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"cVAE Reconstruction trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test


def seeded_hippocampus_ccvae_reconstruction_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_vae_reconstruction.csv"

    threshold_on_95_test = 0.022144482741647323

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"ccVAE Reconstruction trained on \emph{DecathHip}", threshold_on_95_test

def seeded_hippocampus_ccvae_reconstruction_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_vae_reconstruction.csv"

    threshold_on_95_test = max(0.022144482741647323, 0.023007950314624513)

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"ccVAE Reconstruction trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test


def seeded_hippocampus_cvae_scaled_softmax_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkips"
    file = "ood_scores_uncertainty_mse_temperature_threshold_0.018849379789815938.csv"

    threshold_on_95_test = 0.0023098286805730197

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"cVAE scaled Softmax trained on \emph{DecathHip}", threshold_on_95_test

def seeded_hippocampus_cvae_scaled_softmax_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerVAERehearsalNoSkips"
    file = "ood_scores_uncertainty_mse_temperature_threshold_0.018849379789815938_0.019073720282499856.csv"

    threshold_on_95_test = max(0.0023098286805730197, 0.001559910923242545)

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"cVAE scaled Softmax trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test


def seeded_hippocampus_ccvae_scaled_softmax_0():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_uncertainty_mse_temperature_threshold_0.022144482741647323.csv"

    threshold_on_95_test = 0.00222699210305745

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"ccVAE Reconstruction trained on \emph{DecathHip}", threshold_on_95_test

def seeded_hippocampus_ccvae_scaled_softmax_1():
    root_path = "/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/"
    all_tasks = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
    trained_on = ["Task197_DecathHip", "Task198_Dryad"]
    trainer = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth"
    file = "ood_scores_uncertainty_mse_temperature_threshold_0.022144482741647323_0.023007950314624513.csv"

    threshold_on_95_test = max(0.00222699210305745, 0.0014233030275337948)

    csv_path = os.path.join(root_path, all_tasks, '_'.join(trained_on),f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", file)
    return csv_path, trained_on, r"ccVAE Reconstruction trained on \emph{DecathHip}, \emph{Dryad}", threshold_on_95_test




if __name__ == "__main__":

    configuration = seeded_hippocampus_ccvae_scaled_softmax_1

    csv_path, trained_on, title, threshold = configuration()
    if isinstance(csv_path, list):
        df = pd.concat([pd.read_csv(p, sep="\t") for p in csv_path])
    else:
        df = pd.read_csv(csv_path, sep="\t")

    if 'assumed task_idx' in df.columns:
        df = df.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')


    tasks = list(set(df.loc[:,"Task"]))
    tasks.sort()

    min_ood_score = 0
    max_ood_score = max(df['ood_score'])
    for task in tasks:
        xs = np.linspace(min_ood_score, max_ood_score, 1000)


        #split_data = task in trained_on
        if task == trained_on[-1]:
            arr = ['train', 'test', 'val',]
        elif task in trained_on:
            arr = ['val']
        else:
            arr = [None]
            
        for val in arr:
            subset_df = df[df['Task'] == task]
            if arr[0] is not None:
                subset_df = subset_df[subset_df['split'] == val] #only validation data

            y = []
            for x in xs:
                #count amount of values in subset_df where values at column 'ood_score' is less than x
                y.append(len([v for v in subset_df['ood_score'] if v > x]) / len(subset_df))
            assert len(y) == len(xs)
            if task in trained_on:
                if val=='train':
                    linestyle = 'solid'
                elif val == 'val':
                    linestyle = 'dashed'
                elif val == 'test':
                    linestyle = 'dotted'
                sns.lineplot(x=xs, y=y, label=f"{rename_tasks(task)}, {rename_val(val)}", linestyle=linestyle, color=task_color(task))
            else:
                sns.lineplot(x=xs, y=y, label=rename_tasks(task), linestyle='dashed', color=task_color(task))
    if threshold is not None:
        plt.axvline(x=threshold, color='black', linestyle='dashed')#, label="95% threshold")
    plt.xlabel(r"Threshold $\tau$")
    plt.ylabel("Amount of samples classified as OOD")
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"plots/{configuration.__name__}.pdf", bbox_inches='tight')