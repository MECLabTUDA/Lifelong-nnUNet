import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from create_confusion import rename_model, rename_method_base


def join_texts_with_char(texts, combine_with):
    r"""This function takes a list of strings and joins them together with the combine_with between each text from texts.
        :param texts: List of strings
        :param combine_with: A char or string with which each element in the list will be connected with
        :return: A String consisting of every string from the list combined using combine_with
    """
    assert isinstance(combine_with, str), "The character to combine the string series from the list with needs to be from type <string>."
    # -- Combine the string series from the list with the combine_with -- #
    return combine_with.join(texts)
    

BEST_EXPERT_METHOD = "nnUNetTrainerExpertGateUNet"

list_of_tasks = ["Task011_Prostate-BIDMC", "Task012_Prostate-I2CVB", "Task013_Prostate-HK", "Task015_Prostate-UCL", "Task016_Prostate-RUNMC"]
JOINED_TASK = "Task032_Prostate_joined"
USE_MASK = "mask_1"


#list_of_tasks = ["Task008_mHeartA", "Task009_mHeartB"]
#JOINED_TASK = "Task031_Cardiac_joined"
#USE_MASK = "mask_3"

#list_of_tasks = ["Task097_DecathHip", "Task098_Dryad", "Task099_HarP"]
#JOINED_TASK = "Task033_Hippocampus_joined"
#USE_MASK = "mask_1"




EVAL_PATH = "/local/scratch/clmn1/what_is_wrong/evaluation_test/nnUNet_ext/3d_fullres"

input_path = os.path.join(EVAL_PATH, 
    join_texts_with_char(list_of_tasks, '_'), 
    join_texts_with_char(list_of_tasks, '_'))
EVAL_PATH_EXPERT_GATE = os.path.join("/local/scratch/clmn1/what_is_wrong/evaluation/nnUNet_ext/expert_gate/", 
    join_texts_with_char(list_of_tasks, '_'), 
    join_texts_with_char(list_of_tasks, '_'))




PATH_EXTENSION = "Generic_UNet/SEQ/last_head/fold_0"

f, ax = plt.subplots(figsize=(5, 2.5))
ax.set_xscale("log")


data = []
frame = pd.read_csv(os.path.join(EVAL_PATH, JOINED_TASK, JOINED_TASK, "nnUNetTrainerSequential__nnUNetPlansv2.1", PATH_EXTENSION,"val_metrics_eval.csv"), sep="\t")
frame['method'] = "Joint training"
frame = frame.drop(frame[frame['metric'] != 'Dice'].index)
frame = frame.drop(frame[frame['seg_mask'] != USE_MASK].index)
data.append(frame)

for method in os.listdir(input_path):
    frame = pd.read_csv(os.path.join(input_path, method, PATH_EXTENSION,"val_metrics_eval.csv"), sep="\t")
    frame['method'] = rename_method_base(method.split('_')[0])
    frame = frame.drop(frame[frame['metric'] != 'Dice'].index)
    frame = frame.drop(frame[frame['seg_mask'] != USE_MASK].index)
    data.append(frame)
    
    
frame = pd.read_csv(os.path.join(EVAL_PATH_EXPERT_GATE, BEST_EXPERT_METHOD, "fold_0", "expert_gate_evaluation.csv"), sep="\t")
frame['method'] = rename_model(BEST_EXPERT_METHOD) + " EG"
frame = frame.drop(frame[frame['metric'] != 'Dice'].index)
frame = frame.drop(frame[frame['seg_mask'] != USE_MASK].index)
data.append(frame)


data = pd.concat(data)
data['value'] = data["value"].apply(lambda x: 1-x)

data = data.reset_index()
#planets = sns.load_dataset("planets")
print(data)

plt.gca().invert_xaxis()

# https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_palette = {method: "#B1D7D0" if method == "Joint training" else "#BCC6DE" for method in data.method.unique()}

# Plot the orbital period with horizontal boxes
#sns.boxplot(x="distance", y="method", data=planets,
#            whis=[0, 100], width=.6, palette="vlag")
b = sns.boxplot(x="value", y="method", data=data, showfliers=False, palette=my_palette, order=["Joint training", 
                                                                                                "Sequential",
                                                                                                "EWC",
                                                                                                "LwF",
                                                                                                "MiB",
                                                                                                rename_model(BEST_EXPERT_METHOD) + " EG"])
#b.set_xticks([0,0.2,0.4,0.6,0.8,1])
#plt.xlim(0.00001,1)

# Add in points to show each observation
s = sns.stripplot(x="value", y="method", data=data,
             size=4, color=".3", linewidth=0)

#s.set_xticks([0.05, 0.2,0.4,0.6,0.8,1])
#plt.gca().set_xticklabels([95, 80, 60, 40, 20, 0])


#s.set_xticks([0.04, 0.2,0.4,0.6,0.8,1])
#plt.gca().set_xticklabels([96, 80, 60, 40, 20, 0])


s.set_xticks([0.02, 0.2,0.4,0.6,0.8,1])
plt.gca().set_xticklabels([98, 80, 60, 40, 20, 0])




# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
ax.set(xlabel="Dice")
sns.despine(trim=True, left=True)
plt.savefig(os.path.join(EVAL_PATH, join_texts_with_char(list_of_tasks, '_'), "boxplot_Dice_last.svg"), bbox_inches='tight')
print(os.path.join(EVAL_PATH, join_texts_with_char(list_of_tasks, '_'), "boxplot_Dice_last.svg"))