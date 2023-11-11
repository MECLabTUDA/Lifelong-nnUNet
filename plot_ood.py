import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd


def rename_tasks(task_name: str):
    if task_name == "Task097_DecathHip":
        return "DecathHip"
    elif task_name == "Task098_Dryad":
        return "Dryad"
    elif task_name == "Task099_HarP":
        return "HarP"
    elif task_name == "Task008_mHeartA":
        return "Siemens"
    elif task_name == "Task009_mHeartB":
        return "Philips"
    elif task_name == "Task011_Prostate-BIDMC":
        return "BIDMC"
    elif task_name == "Task012_Prostate-I2CVB":
        return "I2CVB"
    elif task_name == "Task013_Prostate-HK":
        return "HK"
    elif task_name == "Task015_Prostate-UCL":
        return "UCL"
    elif task_name == "Task016_Prostate-RUNMC":
        return "RUNMC"
    return "unknown task"



df = pd.read_csv("/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/ood_scores_vae_reconstruction.csv", sep="\t")
trained_on = ["Task097_DecathHip"]

tasks = list(set(df.loc[:,"Task"]))

min_ood_score = 0
max_ood_score = max(df['ood_score'])

for task in tasks:
    xs = np.linspace(min_ood_score, max_ood_score, 1000)

    subset_df = df[df['Task'] == task]

    #subset_df = subset_df[subset_df['is_val'] == True] #only validation data

    y = []
    for x in xs:
        #count amount of values in subset_df where values at column 'ood_score' is less than x
        y.append(len([v for v in subset_df['ood_score'] if v > x]) / len(subset_df))
    assert len(y) == len(xs)
    if task in trained_on:
        sns.lineplot(x=xs, y=y, label=rename_tasks(task))
    else:
        sns.lineplot(x=xs, y=y, label=rename_tasks(task), linestyle='dashed')

plt.xlabel(r"Threshold \tau")
plt.ylabel("Percentage of samples classified as OOD")
plt.title("MSE of reconstruction and original > tau \implies OOD")
plt.savefig("plot_ood.png", bbox_inches='tight')