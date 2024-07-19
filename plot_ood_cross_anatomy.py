import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_ood_new import rename_val, task_color
from plot_utils import rename_tasks, convert_epoch_string_to_int
plt.rcParams['text.usetex'] = True

ROOT = "/local/scratch/clmn1/master_thesis/seeded/evaluation3/nnUNet_ext/2d/"

def task_color2(task_name: str):
    if task_name.endswith("DecathHip"):
        return "red"
    elif task_name.endswith("BraTS6"):
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


def get_static_threshold(_trainer: dict):
    try:
        trained_on = _trainer['trained_on']
        trainer = _trainer['trainer']
        method = _trainer['method']
        df = pd.read_csv(os.path.join(ROOT, '_'.join(trained_on), trained_on[0], f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", trained_on[0], method), 
                         sep='\t')
        df = df[df['split'] == 'test']

        assert np.all(df['Task'] == trained_on[0]), df
        assert df['case'].is_unique, df
        # choose threshold such that 95% of the values in df['ood_score'] are below the threshold
        threshold = np.percentile(df['ood_score'], 95)
    
    except Exception as e:
        print(e)
        print(_trainer)
        return None
    return threshold


tasks_list_hippocampus = ['Task197_DecathHip', 'Task198_Dryad', 'Task199_HarP']
tasks_list_prostate = ['Task111_Prostate-BIDMC', 'Task112_Prostate-I2CVB', 'Task113_Prostate-HK', 'Task115_Prostate-UCL', 'Task116_Prostate-RUNMC']
tasks_list_brats = ['Task306_BraTS6', 'Task313_BraTS13', 'Task316_BraTS16', 'Task320_BraTS20', 'Task321_BraTS21']

unet_SegDist_hippocampus = {
            'trained_on': tasks_list_hippocampus,
            'trainer': 'nnUNetTrainerSegDist',
            'method': 'ood_scores_segmentation_distortion_normalized.csv'
        }
unet_SegDist_hippocampus['threshold'] = get_static_threshold(unet_SegDist_hippocampus)

unet_softmax_hippocampus = {
    'trained_on': tasks_list_hippocampus,
    'trainer': 'nnUNetTrainerSequential',
    'method': 'ood_scores_uncertainty.csv'
}
unet_softmax_hippocampus['threshold'] = get_static_threshold(unet_softmax_hippocampus)

ccvae_reconstruction_hippocampus = {
            'trained_on': tasks_list_hippocampus,
            'trainer': 'nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth',
            'method': 'ood_scores_vae_reconstruction.csv'
        }
ccvae_reconstruction_hippocampus['threshold'] = get_static_threshold(ccvae_reconstruction_hippocampus)


unet_SegDist_prostate = {
            'trained_on': tasks_list_prostate,
            'trainer': 'nnUNetTrainerSegDist',
            'method': 'ood_scores_segmentation_distortion_normalized.csv'
        }
unet_SegDist_prostate['threshold'] = get_static_threshold(unet_SegDist_prostate)

unet_softmax_prostate = {
    'trained_on': tasks_list_prostate,
    'trainer': 'nnUNetTrainerSequential',
    'method': 'ood_scores_uncertainty.csv'
}
unet_softmax_prostate['threshold'] = get_static_threshold(unet_softmax_prostate)

ccvae_reconstruction_prostate = {
            'trained_on': tasks_list_prostate,
            'trainer': 'nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth',
            'method': 'ood_scores_vae_reconstruction.csv'
        }
ccvae_reconstruction_prostate['threshold'] = get_static_threshold(ccvae_reconstruction_prostate)


unet_SegDist_brats = {
            'trained_on': tasks_list_brats,
            'trainer': 'nnUNetTrainerSegDist',
            'method': 'ood_scores_segmentation_distortion_normalized.csv'
        }
unet_SegDist_brats['threshold'] = get_static_threshold(unet_SegDist_brats)

unet_softmax_brats = {
    'trained_on': tasks_list_brats,
    'trainer': 'nnUNetTrainerSequential',
    'method': 'ood_scores_uncertainty.csv'
}
unet_softmax_brats['threshold'] = get_static_threshold(unet_softmax_brats)

ccvae_reconstruction_brats = {
            'trained_on': tasks_list_brats,
            'trainer': 'nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth',
            'method': 'ood_scores_vae_reconstruction.csv'
        }
ccvae_reconstruction_brats['threshold'] = get_static_threshold(ccvae_reconstruction_brats)


all_trainers = [unet_SegDist_hippocampus, unet_softmax_hippocampus, ccvae_reconstruction_hippocampus,
                unet_SegDist_prostate, unet_softmax_prostate, ccvae_reconstruction_prostate,
                unet_SegDist_brats, unet_softmax_brats, ccvae_reconstruction_brats]


trainer_str = "unet_SegDist_hippocampus"
trainer = eval(trainer_str)

min_ood_score = np.inf
max_ood_score = 0
for evaluate_on in ["Task197_DecathHip", "Task111_Prostate-BIDMC", "Task306_BraTS6", "Task008_mHeartA", "Task009_mHeartB"]:
    ood_scores = pd.read_csv(os.path.join(ROOT, '_'.join(trainer['trained_on']), trainer['trained_on'][0], f"{trainer['trainer']}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", evaluate_on, trainer['method']),
                             sep='\t')
    max_ood_score = max(max_ood_score, max(ood_scores['ood_score']))
    min_ood_score = min(min_ood_score, min(ood_scores['ood_score']))
    
range = max_ood_score - min_ood_score
min_ood_score -= range * 0.02
max_ood_score += range * 0.02

evaluate_on_all_tasks = ["Task197_DecathHip", "Task111_Prostate-BIDMC", "Task306_BraTS6", "Task008_mHeartA", "Task009_mHeartB"]
evaluate_on_all_tasks.remove(trainer['trained_on'][0])
evaluate_on_all_tasks.insert(0, trainer['trained_on'][0])

for evaluate_on in evaluate_on_all_tasks:
    ood_scores = pd.read_csv(os.path.join(ROOT, '_'.join(trainer['trained_on']), trainer['trained_on'][0], f"{trainer['trainer']}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", evaluate_on, trainer['method']),
                             sep='\t')
    # remove duplicates
    ood_scores = ood_scores.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')

    linestyle = 'solid'
    label = f"{rename_tasks(evaluate_on)}"
    if evaluate_on in trainer['trained_on']:
        ood_scores = ood_scores[ood_scores['split'] == 'val']#<- this is our test set
        linestyle = 'dashed'
        label = f"{rename_tasks(evaluate_on)} (ID)"
    
    y = []
    xs = np.linspace(min_ood_score, max_ood_score, 10000)
    for x in xs:
        #count amount of values in subset_df where values at column 'ood_score' is less than x
        y.append(len([v for v in ood_scores['ood_score'] if v > x]) / len(ood_scores))

    sns.lineplot(x=xs, y=y, label=label, linestyle=linestyle, color=task_color2(evaluate_on))

if trainer_str == "unet_SegDist_hippocampus":
    plt.legend(loc='upper right')

plt.grid()
plt.axvline(x=trainer['threshold'], color='black', linestyle='dashed')
plt.xlabel(r"Threshold $\tau$")
plt.ylabel("Amount of samples classified as OOD")
plt.savefig(f"plots/ood_detection_cross_anatomy/{trainer_str}.pdf", bbox_inches='tight')
print(f"Saved plot to plots/ood_detection_cross_anatomy/{trainer_str}.pdf")