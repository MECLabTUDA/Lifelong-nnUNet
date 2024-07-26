
import os
import pandas as pd
import numpy as np

EVAL_PATH = "/local/scratch/clmn1/what_is_wrong/evaluation/nnUNet_ext/expert_gate/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"

EVAL_PATH_EXPERT_GATE = "/local/scratch/clmn1/what_is_wrong/evaluation/nnUNet_ext/expert_gate/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"

PATH_EXTENSION = "Generic_UNet/SEQ/last_head/fold_0"


final = []

for method in os.listdir(EVAL_PATH_EXPERT_GATE):
    if method == "agnostic":
        continue
    frame = pd.read_csv(os.path.join(EVAL_PATH_EXPERT_GATE, method, "summarized_expert_gate_evaluation.csv"), sep="\t")
    frame = frame.drop(frame[frame['metric'] != 'Dice'].index)
    frame = frame['mean +/- std'].str.split(' ').tolist()
    frame = np.array(frame)
    frame = frame[:,0]
    frame = frame.astype(float)
    
    np.std(frame)
    
    data = {'method': [method], 'mean': [np.mean(frame)], 'std': [np.std(frame)]}
    
    df = pd.DataFrame(data=data)
    final.append(df)
    
final = pd.concat(final)

print(final)