import eval_helper
import numpy as np
import pandas as pd
import scipy.stats


def expected_calibration_error(df, num_bins: int = 15):
    num_cases = len(df)

    #min max normalize ood_scores['ood_score'] to [0,1]
    df['ood_score'] = (df['ood_score'] - min(df['ood_score'])) / (max(df['ood_score']) - min(df['ood_score']))
    df['ood_score'] = 1 - df['ood_score'] #convert to confidence

    #min max normalize ood_scores['value'] to [0,1]
    df['value'] = (df['value'] - min(df['value'])) / (max(df['value']) - min(df['value']))

    min_ood_score = df['ood_score'].min()
    max_ood_score = df['ood_score'].max()

    bins = np.linspace(min_ood_score, max_ood_score, num_bins)
    df['bin'] = np.digitize(df['ood_score'], bins)
    ece = []
    
    for bin in range(1, len(bins)):
        bin_df = df[df['bin'] == bin]
        if len(bin_df) == 0:
            continue
        bin_acc = np.mean(bin_df['value'])
        bin_conf = np.mean(bin_df['ood_score'])
        bin_size = len(bin_df)
        ece.append(bin_size * np.abs(bin_acc - bin_conf))
    ece = np.sum(ece) / num_cases
    return ece

def rename_trainer(trainer: str, ood_method: str):
    if trainer == "nnUNetTrainerSequential":
        return r"Seq., SM #cite{hendrycks2016baseline}"
    elif trainer == "nnUNetTrainerEWC":
        return r"EWC #cite{kirkpatrick2017overcoming}, SM #cite{hendrycks2016baseline}"
    elif trainer == "nnUNetTrainerMiB":
        return r"MiB #cite{cermelli2020modeling}, SM #cite{hendrycks2016baseline}"
    elif trainer == "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth":
        if ood_method == "vae_mahalanobis":
            return r"ccVAE, Mah."
        elif ood_method == "vae_reconstruction":
            return r"#textbf{ccVAE (ours)}"
    elif trainer == "nnUNetTrainerSegDist":
        if ood_method == "segmentation_distortion_normalized":
            return r"Seq., Seg. Dist. #cite{lennartz2023segmentation}"
        elif ood_method == "segmentation_distortion_normalized_pool":
            return r"MPool #cite{gonzalez2022task} SD #cite{lennartz2023segmentation}"
    elif trainer == "nnUNetTrainerVAERehearsalNoSkips":
        return r"cVAE, Rec."
    elif trainer == "nnUNetTrainerRehearsal":
        return r"Rehearsal"
    assert False

def round_to_2(x, digits=1):
    if isinstance(x, str):
        return x
    return f"{round(x * 100, 1):2.{digits}f}"

def insert_best_sbest(df, column, _max):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(column, str)
    assert isinstance(_max, bool)
    
    if _max:
        best_value =  np.sort(df[column].values)[-1]
        sbest_value = np.sort(df[column].values)[-2]
    else:
        best_value =  np.sort(df[column].values)[0]
        sbest_value = np.sort(df[column].values)[1]

    for row in range(len(df)):
        if df[column][row] == best_value:
            df[column][row] = f"#best{round_to_2(df[column][row])}"
        elif df[column][row] == sbest_value:
            df[column][row] = f"#sbest{round_to_2(df[column][row])}"

TRAINERS = [
            ("nnUNetTrainerSegDist", "segmentation_distortion_normalized"),
            #("nnUNetTrainerSequential", "uncertainty"),
            #("nnUNetTrainerEWC", "uncertainty"),
            #("nnUNetTrainerMiB", "uncertainty"),
            #("nnUNetTrainerSegDist", "segmentation_distortion_normalized_pool"),
            #("nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth", "vae_mahalanobis"),
            #("nnUNetTrainerVAERehearsalNoSkips", "vae_reconstruction"),
            #("nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth", "vae_reconstruction"),
            ]

#TRAINED_ON = [197, 198, 199]
#EVAL_ON = [197, 198]                                               # NO LAST + NO AUG
#EVAL_ON = [197, 198, 199]                                          # LAST + NO AUG
#EVAL_ON = [197, 198, 199, 400, 401, 402]                           # LAST + AUG
#EVAL_ON = [197, 198, 400, 401]                                     # NO LAST + AUG


TRAINED_ON = [111, 112, 113, 115, 116]
#EVAL_ON = [111, 112, 113, 115, 116, 403, 404, 405, 406, 407]       # LAST + AUG
#EVAL_ON = [111, 112, 113, 115, 116]                                # LAST + NO AUG
EVAL_ON = [111, 112, 113, 115]                                      # NO LAST + NO AUG
#EVAL_ON = [111, 112, 113, 115, 403, 404, 405, 406]                 # NO LAST + AUG
#EVAL_ON = [403, 404, 405, 406, 407]


ROOT = "/local/scratch/clmn1/master_thesis/seeded/evaluation_augmentations"
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

        #print("\n")
        #print(thresholds)
        #temp = df
        ##temp = temp[temp['ood_score'] < max_threshold]
        ##temp = temp[temp['value'] < 0.4]
        #print(temp[temp['ood_score'] < max_threshold])
        #print(temp[temp['ood_score'] > max_threshold])

        correlation = scipy.stats.spearmanr(df['ood_score'], df['value'])
        ece = expected_calibration_error(df.copy())

        all = len(df)
        positive = len(df[df['value'] < 0.5])
        negative = all - positive

        print(thresholds)
        print(df)

        if PERFORM_OOD_DETECTION:
            THRESHOLD_TYPE = "max" # "advanced" or "ideal" or "max
            if THRESHOLD_TYPE == "advanced":
                classified_as_id = []
                for i, threshold in enumerate(thresholds.values()):
                    temp = df
                    temp = temp[temp['assumed task_idx'] == i]
                    temp = temp[temp['ood_score'] < threshold]
                    classified_as_id.append(temp)
                df = pd.concat(classified_as_id)
            elif THRESHOLD_TYPE == "max":
                max_threshold = max(thresholds.values())
                df = df[df['ood_score'] < max_threshold]
            elif THRESHOLD_TYPE == "ideal":
                #DATA LEAKAGE!!!
                best_score = 0
                best_threshold = None
                for possible_threshold in df['ood_score']:
                    possible_frame = df[df['ood_score'] < possible_threshold]
                    possible_dice = np.mean(possible_frame['value'])
                    if possible_dice > best_score:
                        best_score = possible_dice
                        best_threshold = possible_threshold
                df = df[df['ood_score'] < best_threshold]
            else:
                assert False

        fp = len(df[df['value'] < 0.5])
        tn = len(df) - fp
        tp = positive - fp
        fn = negative - tn

        #f1 = 2 * tp / (2 * tp + fp + fn)


        dice = np.mean(df["value"])

        #row[f'Dice{i}'] = dice
        row[f'Dice{i}'] = f"{round_to_2(dice)}#pm{round_to_2(np.std(df['value']), 0)}"

        #row[f'ECE{i}'] = ece
        #row[f'Correlation{i}'] = -correlation.correlation

        row[f'Coverage{i}'] = len(df) / all

        #row[f'f1{i}'] = f1
        #row[f'ID{i}'] = (tp+fp) / all
        #row[f'ID{i}'] = len(df) / all
        #row[f'sum{i}'] = (tp+fp) / all + dice

    table.append(row)
table = pd.DataFrame(table)


for i in range(len(TRAINED_ON)-1):
    insert_best_sbest(table, f'Dice{i}', True)
    insert_best_sbest(table, f'Coverage{i}', True)
    #insert_best_sbest(table, f'ECE{i}', False)
    #insert_best_sbest(table, f'Correlation{i}', True)
    #insert_best_sbest(table, f'ID{i}', True)
    pass

#table['Trainer'] = table['Trainer'].apply(rename_trainer)
table = table.applymap(round_to_2)
print(table)
with pd.option_context("max_colwidth", 1000):
    print(table.to_latex(index=False).replace("\\{", "{").replace("\\}", "}").replace("\#", '\\'))