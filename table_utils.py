import numpy as np
import pandas as pd
import scipy.stats, os
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
            return r"MPool #cite{gonzalez2022task}, SD #cite{lennartz2023segmentation}"
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