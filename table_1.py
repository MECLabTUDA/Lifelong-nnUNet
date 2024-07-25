import eval_helper
import numpy as np
import pandas as pd
import scipy.stats, os
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

from table_utils import insert_best_sbest, rename_trainer

def subtract_dicts(d1, d2):
    assert d1.keys() == d2.keys()
    return {k: d1[k] - d2[k] for k in d1.keys()}



TRAINERS = [
            ("nnUNetTrainerSequential", "uncertainty"),
            ("nnUNetTrainerEWC", "uncertainty"),
            ("nnUNetTrainerMiB", "uncertainty"),
            ("nnUNetTrainerSegDist", "segmentation_distortion_normalized_pool"),
            ("nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth", "vae_reconstruction"),
            ]





ROOT = os.environ['RESULTS_FOLDER'] # or specify the path directly
anatomy = "hippocampus"
assert anatomy in ["hippocampus", "prostate"]

if anatomy == "hippocampus":
    ### Hippocampus
    TRAINED_ON = [197, 198, 199]
elif anatomy == "prostate": 
    ### Prostate
    TRAINED_ON = [111, 112, 113, 115, 116]


sequential_performance_list = []
for i in range(len(TRAINED_ON)-1):
    use_model = TRAINED_ON[:i+1]
    df = eval_helper.get_df("nnUNetTrainerSequential", "2d", 
                            TRAINED_ON, use_model, TRAINED_ON, 
                            evaluation_folder=ROOT, 
                            append_ood=True, ood_method="uncertainty")
    assert np.all(df['split'] == 'val')
    df = df[df["metric"] == "Dice"]
    df = df[df["seg_mask"] == "mask_1"]
    temp = df[df['Task'] == convert_id_to_task_name(TRAINED_ON[i])]
    sequential_performance_list.append(dict())
    for subject_id, value in zip(temp['subject_id'], temp['value']):
        sequential_performance_list[i][subject_id] = value




table = []
for trainer, ood_method in TRAINERS:

    row = {'Trainer': rename_trainer(trainer, ood_method)}

    confusion_matrix_list = []
    for i in range(len(TRAINED_ON)-1):
        confusion_matrix_list.append([])
        for j in range(len(TRAINED_ON)-1):
            confusion_matrix_list[-1].append(dict())


    for i in range(len(TRAINED_ON)-1):
        use_model = TRAINED_ON[:i+1]
        if ood_method.endswith("pool"):
            df = eval_helper.get_df_model_pool(trainer, "2d", TRAINED_ON, use_model, TRAINED_ON, 
                                    evaluation_folder=ROOT, 
                                    append_ood=True, ood_method="segmentation_distortion_normalized")
        else:
            df = eval_helper.get_df(trainer, "2d", TRAINED_ON, use_model, TRAINED_ON, 
                                    evaluation_folder=ROOT, 
                                    append_ood=True, ood_method=ood_method)
            
        assert np.all(df['split'] == 'val')
        df = df[df["metric"] == "Dice"]
        df = df[df["seg_mask"] == "mask_1"]
    

        for k in range(len(TRAINED_ON)-1):
            temp = df[df['Task'] == convert_id_to_task_name(TRAINED_ON[k])]
            if not temp['subject_id'].is_unique:
                temp = temp.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['subject_id','Task'], keep='first')
            assert temp['subject_id'].is_unique, f"{trainer}, {ood_method}, {i}, {k}"
            #trained on task i, eval on task k
            for subject_id, value in zip(temp['subject_id'], temp['value']):
                confusion_matrix_list[i][k][subject_id] = value
    

    temp = []
    for i in range(len(TRAINED_ON)-1):
        temp.extend(np.array(confusion_matrix_list[-1][i].values()).tolist())
    row['Dice'] = f"{np.round(np.mean(temp) * 100, 1)}#pm{np.round(np.std(temp) * 100, 1)}"



    temp = []
    for i in range(len(TRAINED_ON)-2):
        #Lifelong nnU-Net:
        assert len(confusion_matrix_list[-1][i]) == len(confusion_matrix_list[i][i])
        temp.extend(subtract_dicts(confusion_matrix_list[-1][i], confusion_matrix_list[i][i]).values())
    row['BWT'] = f"{np.round(np.mean(temp) * 100, 1)}#pm{np.round(np.std(temp) * 100, 1)}"
    
    temp = []
    for i in range(1, len(TRAINED_ON)-1):
        # Lifelong nnU-Net
        #temp.append((confusion_matrix[i, i] - sequential_performance[i]) / sequential_performance[i])
        temp.extend(subtract_dicts(confusion_matrix_list[i][i], sequential_performance_list[i]).values())
    row['FWT'] = f"{np.round(np.mean(temp) * 100, 1)}#pm{np.round(np.std(temp) * 100, 1)}"
    temp.sort()
    
    table.append(row)

table = pd.DataFrame(table)
insert_best_sbest(table, f'Dice', True)
insert_best_sbest(table, f'BWT', True)
insert_best_sbest(table, f'FWT', True)
with pd.option_context("max_colwidth", 1000):
    print(table.to_latex(index=False).replace("\\{", "{").replace("\\}", "}").replace("\#", '\\'))