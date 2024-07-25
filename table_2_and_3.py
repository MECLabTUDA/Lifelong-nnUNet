import eval_helper
import numpy as np
import pandas as pd
import scipy.stats, os
from table_utils import rename_trainer, round_to_2, insert_best_sbest, expected_calibration_error


ROOT = os.environ['RESULTS_FOLDER'] # or specify the path directly

table = 2
anatomy = "prostate"
assert table in [2, 3]
assert anatomy in ["hippocampus", "prostate"]


if table == 2:
    #### Table 2
    TRAINERS = [
                ("nnUNetTrainerSequential", "uncertainty"),
                ("nnUNetTrainerEWC", "uncertainty"),
                ("nnUNetTrainerMiB", "uncertainty"),
                ("nnUNetTrainerSegDist", "segmentation_distortion_normalized_pool"),
                ("nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth", "vae_reconstruction"),
                ]
    if anatomy == "hippocampus":
        ### Hippocampus
        TRAINED_ON = [197, 198, 199]
        EVAL_ON = [197, 198, 199, 400, 401, 402]
    elif anatomy == "prostate":
        ### Prostate
        TRAINED_ON = [111, 112, 113, 115, 116]
        EVAL_ON = [111, 112, 113, 115, 116, 403, 404, 405, 406, 407]
elif table == 3:
    #### Table 3
    TRAINERS = [
                ("nnUNetTrainerSegDist", "segmentation_distortion_normalized_pool"),
                ("nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth", "vae_mahalanobis"),
                ("nnUNetTrainerVAERehearsalNoSkips", "vae_reconstruction"),
                ("nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth", "vae_reconstruction"),
                ]
    if anatomy == "hippocampus":
        ### Hippocampus
        TRAINED_ON = [197, 198, 199]
        EVAL_ON = [197, 198, 199]
    elif anatomy == "prostate":
        ### Prostate
        TRAINED_ON = [111, 112, 113, 115, 116]
        EVAL_ON = [111, 112, 113, 115, 116]





PERFORM_OOD_DETECTION = True

ONLY_ID_DATA = False

table = []
for trainer, ood_method in TRAINERS:

    row = {'Trainer': rename_trainer(trainer, ood_method)}
    for i in range(len(TRAINED_ON)-1):
        use_model = TRAINED_ON[:i+1]
        if ood_method.endswith("pool"):
            df = eval_helper.get_df_model_pool(trainer, "2d", TRAINED_ON, use_model, EVAL_ON, 
                                    evaluation_folder=ROOT, 
                                    append_ood=True, ood_method="segmentation_distortion_normalized")
            thresholds = eval_helper.get_all_ood_thresholds(trainer, "2d", TRAINED_ON, use_model, 
                                                        evaluation_folder=ROOT,
                                                        ood_method="segmentation_distortion_normalized")
        else:
            df = eval_helper.get_df(trainer, "2d", TRAINED_ON, use_model, EVAL_ON, 
                                    evaluation_folder=ROOT, 
                                    append_ood=True, ood_method=ood_method)
            thresholds = eval_helper.get_all_ood_thresholds(trainer, "2d", TRAINED_ON, use_model, 
                                                        evaluation_folder=ROOT,
                                                        ood_method=ood_method)
        

        df = df[df["metric"] == "Dice"]
        df = df[df["seg_mask"] == "mask_1"]
        df = df.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['subject_id','Task'], keep='first')

        if ONLY_ID_DATA:
            df = df[df['is_ood'] == False]

        correlation = scipy.stats.spearmanr(df['ood_score'], df['value'])
        ece = expected_calibration_error(df.copy())

        all = len(df)
        positive = len(df[df['value'] < 0.5])
        negative = all - positive

        print(thresholds)
        print(df)

        if PERFORM_OOD_DETECTION:
            max_threshold = max(thresholds.values())
            df = df[df['ood_score'] < max_threshold]
            

        dice = np.mean(df["value"])

        row[f'Dice{i}'] = f"{round_to_2(dice)}#pm{round_to_2(np.std(df['value']), 0)}"

        row[f'ECE{i}'] = ece

    table.append(row)
table = pd.DataFrame(table)


for i in range(len(TRAINED_ON)-1):
    insert_best_sbest(table, f'Dice{i}', True)
    insert_best_sbest(table, f'ECE{i}', False)

table = table.applymap(round_to_2)
print(table)
with pd.option_context("max_colwidth", 1000):
    print(table.to_latex(index=False).replace("\\{", "{").replace("\\}", "}").replace("\#", '\\'))