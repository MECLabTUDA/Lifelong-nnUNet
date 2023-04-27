from nnunet_ext.paths import evaluation_output_dir
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base = os.path.join(evaluation_output_dir, "3d_fullres")
end = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/val_metrics_eval.csv"
combinations = ["Task011_Prostate-BIDMC", 
                "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB"]


rehearsal = {'trained_on': "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
 'trainer': "nnUNetTrainerRehearsal"}
feature_rehearsal2 = {'trained_on': "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
 'trainer': "nnUNetTrainerFeatureRehearsal2"}
dummy_sequential = {'trained_on': "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
 'trainer': "nnUNetTrainerSequential"}


trainers = [rehearsal, feature_rehearsal2, dummy_sequential]

data = []

## data needs to have ["case_name", "last task trained", "task the case belongs to", "value"]
for i, task in enumerate(combinations):
    for trainer in trainers:
        frame = pd.read_csv(os.path.join(base,trainer['trained_on'], task, trainer['trainer'] + end), sep="\t")
        frame = frame.drop(frame[frame['metric'] != 'Dice'].index)
        frame = frame.drop(frame[frame['seg_mask'] != "mask_1"].index)
        frame['timepoint'] = i
        frame['Trainer'] = trainer['trainer']
        data.append(frame)
        print(frame)

data = pd.concat(data)



sns.lineplot(x="timepoint", y="value",
             hue="Task", style="Trainer",
             data=data, errorbar=None)

plt.savefig("plot.png")
