import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import rename_tasks

ROOT = "/local/scratch/clmn1/master_thesis/tests/evaluation/nnUNet_ext/2d"
MASK = "mask_3"
METRIC = "Dice"

all_tasks = list(sorted(filter(lambda x: "BraTS" in x, os.listdir(ROOT))))
print(all_tasks)

confusion_matrix = np.zeros((len(all_tasks), len(all_tasks)))

for task in all_tasks:
    csv_file = os.path.join(ROOT, task, task, "nnUNetTrainerSequential__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/val_metrics_eval.csv")
    df = pd.read_csv(csv_file, sep="\t")
    df = df[df["metric"] == METRIC]
    df = df[df["seg_mask"] == MASK]
    for eval_task in all_tasks:
        sub_df = df[df["Task"] == eval_task]
        mean = np.mean(sub_df["value"])
        confusion_matrix[all_tasks.index(task), all_tasks.index(eval_task)] = mean * 100



task_names = [rename_tasks(task) for task in all_tasks]
df_cm = pd.DataFrame(confusion_matrix, index=task_names, columns=task_names)
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.5)
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel("Evaluated Task")
plt.ylabel("Training Task")
#plt.title(f"Metric: {METRIC}, Segmentation: {MASK}")
plt.savefig("confusion_matrix.pdf")

