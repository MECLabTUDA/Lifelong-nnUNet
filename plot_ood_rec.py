import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_ood_new import rename_tasks, task_color, rename_val

plt.rcParams['text.usetex'] = True

def get_y(df, xs):
    y = []
    for x in xs:
        y.append(len([v for v in df['ood_score'] if v > x]) / len(df))
    return y


config = 3


if config == 0:
    df = pd.read_csv('/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP/Task197_DecathHip_Task198_Dryad/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/ood_scores_vae_reconstruction.csv', sep='\t')
    THRESHOLD_1 = 0.018849379789815938
    THRESHOLD_2 = 0.019073720282499856
    TEXT = "cVAE Reconstruction"
elif config == 1:
    df = pd.read_csv('/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP/Task197_DecathHip_Task198_Dryad/nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/ood_scores_vae_reconstruction.csv', sep='\t')
    THRESHOLD_1 = 0.022144482741647323
    THRESHOLD_2 = 0.023007950314624513
    TEXT = "ccVAE Reconstruction"
elif config == 2:
    df = pd.read_csv('/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP/Task197_DecathHip_Task198_Dryad/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/ood_scores_segmentation_distortion.csv', sep='\t')
    THRESHOLD_1 = 0.00336454252
    THRESHOLD_2 = 0.0094586729
    TEXT = "cVAE Segmentation Distortion"
elif config == 3:
    df = pd.read_csv('/local/scratch/clmn1/master_thesis/seeded/evaluation/nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP/Task197_DecathHip_Task198_Dryad/nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/ood_scores_segmentation_distortion.csv', sep='\t')
    THRESHOLD_1 = 0.0033189527749999987
    THRESHOLD_2 = 0.00992243975
    TEXT = "ccVAE Segmentation Distortion"







min_ood_score = 0
max_ood_score = max(df['ood_score'])


df_0 = df[df['assumed task_idx'] == 0]
df_1 = df[df['assumed task_idx'] == 1]
xs = np.linspace(min_ood_score, max_ood_score, 1000)


fig, ax = plt.subplots(2, 1)

####################################################
####################################################
####################################################
#fig.add_subplot(2, 1, 1)
ax[0].grid()
ax[0].set_title(r"Conditional $t=0$")
ax[0].set_xticklabels([])




df_0_DecathHip = df_0[df_0['Task'] == "Task197_DecathHip"]
df_0_DecathHip = df_0_DecathHip[df_0_DecathHip['split'] == "val"]
sns.lineplot(x=xs, y=get_y(df_0_DecathHip, xs), label="DecathHip, test", color=task_color("DecathHip"), linestyle="dashed", ax=ax[0])

df_0_Dryad = df_0[df_0['Task'] == "Task198_Dryad"]
df_0_Dryad_val = df_0_Dryad[df_0_Dryad['split'] == "val"]
df_0_Dryad_test = df_0_Dryad[df_0_Dryad['split'] == "test"]
df_0_Dryad_train = df_0_Dryad[df_0_Dryad['split'] == "train"]
sns.lineplot(x=xs, y=get_y(df_0_Dryad_val, xs), label="Dryad, test", color=task_color("Dryad"), linestyle="dashed", ax=ax[0])
sns.lineplot(x=xs, y=get_y(df_0_Dryad_test, xs), label="Dryad, val", color=task_color("Dryad"), linestyle="dotted", ax=ax[0])
sns.lineplot(x=xs, y=get_y(df_0_Dryad_train, xs), label="Dryad, train", color=task_color("Dryad"), linestyle="solid", ax=ax[0])


df_0_HarP = df_0[df_0['Task'] == "Task199_HarP"]
sns.lineplot(x=xs, y=get_y(df_0_HarP, xs), label="HarP", linestyle="dashed", ax=ax[0])

####################################################
####################################################
####################################################
ax[1].grid()
ax[1].set_title(r"Conditional $t=1$")

df_1_DecathHip = df_1[df_1['Task'] == "Task197_DecathHip"]
df_1_DecathHip = df_1_DecathHip[df_1_DecathHip['split'] == "val"]
sns.lineplot(x=xs, y=get_y(df_1_DecathHip, xs), label="DecathHip, test", color=task_color("DecathHip"), linestyle="dashed", ax=ax[1])

df_1_Dryad = df_1[df_1['Task'] == "Task198_Dryad"]
df_1_Dryad_val = df_1_Dryad[df_1_Dryad['split'] == "val"]
df_1_Dryad_test = df_1_Dryad[df_1_Dryad['split'] == "test"]
df_1_Dryad_train = df_1_Dryad[df_1_Dryad['split'] == "train"]
sns.lineplot(x=xs, y=get_y(df_1_Dryad_val, xs), label="Dryad, test", color=task_color("Dryad"), linestyle="dashed", ax=ax[1])
sns.lineplot(x=xs, y=get_y(df_1_Dryad_test, xs), label="Dryad, val", color=task_color("Dryad"), linestyle="dotted", ax=ax[1])
sns.lineplot(x=xs, y=get_y(df_1_Dryad_train, xs), label="Dryad, train", color=task_color("Dryad"), linestyle="solid", ax=ax[1])

df_1_HarP = df_1[df_1['Task'] == "Task199_HarP"]
sns.lineplot(x=xs, y=get_y(df_1_HarP, xs), label="HarP", linestyle="dashed", ax=ax[1])

ax[0].axvline(x=THRESHOLD_1, linestyle='dotted',color="black")
ax[1].axvline(x=THRESHOLD_2, linestyle='dotted',color="black")

ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel(r"Threshold $\tau$")

fig.text(0.04, 0.5, 'Amount of samples classified as OOD', va='center', rotation='vertical')

fig.suptitle(TEXT + r" trained on \emph{DecathHip}, \emph{Dryad}")
plt.savefig(f"plot_ood_rec.png", bbox_inches='tight')