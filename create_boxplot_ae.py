import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

from create_confusion import rename_model
from create_confusion_ae import AE_METHODS_SORTED

#EVAL_PATH = "/local/scratch/clmn1/what_is_wrong/evaluation/nnUNet_ext/expert_gate/"
EVAL_PATH = "/local/scratch/clmn1/what_is_wrong/evaluation_test/nnUNet_ext/expert_gate/"
MIN_VALUE = 0

ALL_TASKS = "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
USE_MASK = "mask_1"

#ALL_TASKS = "Task008_mHeartA_Task009_mHeartB"
#USE_MASK = "mask_3"

#ALL_TASKS = "Task097_DecathHip_Task098_Dryad_Task099_HarP"
#USE_MASK = "mask_1"





f, ax = plt.subplots(figsize=(5, 2.5)) #(7,6)
ax.set_xscale("log")

data = pd.read_csv(os.path.join(EVAL_PATH, ALL_TASKS, "agnostic", "agnostic_evaluation.csv"), sep="\t")
data['method'] = "Task Identity"
data = data.drop(data[data['metric'] != 'Dice'].index)
data = data.drop(data[data['seg_mask'] != USE_MASK].index)

input_path = os.path.join(EVAL_PATH, ALL_TASKS, ALL_TASKS)  #go to evaluation with all models directly
for method in AE_METHODS_SORTED:
    frame = pd.read_csv(os.path.join(input_path, method, "fold_0", "expert_gate_evaluation.csv"), sep="\t")
    frame['method'] = rename_model(method)
    frame = frame.drop(frame[frame['metric'] != 'Dice'].index)
    frame = frame.drop(frame[frame['seg_mask'] != USE_MASK].index)
    
    data = pd.concat([data, frame])



data['value'] = data["value"].apply(lambda x: 1-x)
data = data.reset_index()
print(data)

plt.gca().invert_xaxis()

# https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_palette = {method: "#B1D7D0" if method == "Task Identity" else "#BCC6DE" for method in data.method.unique()}




# Plot the orbital period with horizontal boxes
#sns.boxplot(x="distance", y="method", data=planets,
#            whis=[0, 100], width=.6, palette="vlag")
b = sns.boxplot(x="value", y="method", data=data, showfliers=False, palette=my_palette)
#b.set_xticks([0,0.2,0.4,0.6,0.8,1])
#plt.xlim(MIN_VALUE,1)

# Add in points to show each observation
s = sns.stripplot(x="value", y="method", data=data,
             size=4, color=".3", linewidth=0)

## Hippocampus
#s.set_xticks([0.05, 0.2,0.4,0.6,0.8,1])
#plt.gca().set_xticklabels([95, 80, 60, 40, 20, 0])


## Prostate
s.set_xticks([0.04, 0.2,0.4,0.6,0.8,1])
plt.gca().set_xticklabels([96, 80, 60, 40, 20, 0])

## Cardiac
#s.set_xticks([0.02, 0.2,0.4,0.6,0.8,1])
#plt.gca().set_xticklabels([98, 80, 60, 40, 20, 0])

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
ax.set(xlabel="Dice")
sns.despine(trim=True, left=True)
plt.savefig(os.path.join(EVAL_PATH, ALL_TASKS, "boxplot.svg"), bbox_inches='tight')
print(os.path.join(EVAL_PATH, ALL_TASKS, "boxplot.svg"))