import os
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
import pandas as pd
import numpy as np


def get_df(trainer: str, resolution: str, trained_on: list[int], use_model: list[int], evaluate_on: list[int],
           eval_only: bool = True,
           evaluation_folder: str = os.environ['EVALUATION_FOLDER'],
           append_ood: bool = False, ood_method: str = None):
    trained_on = [convert_id_to_task_name(i) for i in trained_on]
    use_model = [convert_id_to_task_name(i) for i in use_model]
    evaluate_on = [convert_id_to_task_name(i) for i in evaluate_on]

    eval_path = os.path.join(evaluation_folder, "nnUNet_ext", resolution, join_texts_with_char(trained_on, "_"), join_texts_with_char(use_model, "_"),
                 f"{trainer}__nnUNetPlansv2.1", "Generic_UNet", "SEQ", "head_None", "fold_0")
    
    dfs = []
    for eval_task in evaluate_on:
        if eval_only:
            seg_df = pd.read_csv(os.path.join(eval_path, eval_task, "val_metrics_eval.csv"), sep="\t")
        else:
            seg_df = pd.read_csv(os.path.join(eval_path, eval_task, "val_metrics_all.csv"), sep="\t")

        if append_ood:
            ood_df = pd.read_csv(os.path.join(eval_path, eval_task, f"ood_scores_{ood_method}.csv"), sep="\t")
            #rename case -> subject_id
            ood_df = ood_df.rename(columns={"case": "subject_id"})
            #join dataframes on subject_id and Task
            seg_df = seg_df.merge(ood_df, on=["subject_id", "Task"])
        dfs.append(seg_df)
    return pd.concat(dfs)

def get_df_model_pool(trainer: str, resolution: str, trained_on: list[int], use_model: list[int], evaluate_on: list[int],
           eval_only: bool = True,
           evaluation_folder: str = os.environ['EVALUATION_FOLDER'],
           append_ood: bool = False, ood_method: str = None):
    assert append_ood, "Only append_ood is supported for model pool"
    assert ood_method == "segmentation_distortion_normalized", "Only segmentation distortion is supported for model pool"
    trained_on = [convert_id_to_task_name(i) for i in trained_on]
    use_model = [convert_id_to_task_name(i) for i in use_model]
    evaluate_on = [convert_id_to_task_name(i) for i in evaluate_on]

    
    seg_dfs = []
    for i, _ in enumerate(use_model):
        eval_path = os.path.join(evaluation_folder, "nnUNet_ext", resolution, join_texts_with_char(trained_on, "_"), join_texts_with_char(use_model[:i+1], "_"),
                    "nnUNetTrainerSequential__nnUNetPlansv2.1", "Generic_UNet", "SEQ", "head_None", "fold_0")
        for eval_task in evaluate_on:
            if eval_only:
                seg_df = pd.read_csv(os.path.join(eval_path, eval_task, "val_metrics_eval.csv"), sep="\t")
            else:
                seg_df = pd.read_csv(os.path.join(eval_path, eval_task, "val_metrics_all.csv"), sep="\t")
            seg_df['assumed task_idx'] = i
            seg_dfs.append(seg_df)

    ood_dfs = []
    for i, _ in enumerate(use_model):
        eval_path = os.path.join(evaluation_folder, "nnUNet_ext", resolution, join_texts_with_char(trained_on, "_"), join_texts_with_char(use_model[:i+1], "_"),
                    f"{trainer}__nnUNetPlansv2.1", "Generic_UNet", "SEQ", "head_None", "fold_0")
        for eval_task in evaluate_on:
            if append_ood:
                ood_df = pd.read_csv(os.path.join(eval_path, eval_task, f"ood_scores_{ood_method}.csv"), sep="\t")
                #rename case -> subject_id
                ood_df = ood_df.rename(columns={"case": "subject_id"})
                if eval_only:
                    ood_df = ood_df[ood_df["split"] == "val"]
                ood_df = ood_df[ood_df['assumed task_idx'] == i]
            ood_dfs.append(ood_df)
    
    seg_dfs = pd.concat(seg_dfs)
    ood_dfs = pd.concat(ood_dfs)

    #print(len(seg_dfs))
    #print(len(ood_dfs))
    
    #print(seg_dfs)
    #print(ood_dfs.to_string())

    #join dataframes on subject_id, Task and assumed task_idx
    seg_dfs = seg_dfs.merge(ood_dfs, on=["subject_id", "Task", "assumed task_idx"])

    return seg_dfs

def get_all_ood_thresholds(trainer: str, resolution: str, trained_on: list[int], use_model: list[int],
           evaluation_folder: str = os.environ['EVALUATION_FOLDER'],
           split: str = 'test', 
           ood_method: str = None):
    trained_on = [convert_id_to_task_name(i) for i in trained_on]
    use_model = [convert_id_to_task_name(i) for i in use_model]

    eval_path = os.path.join(evaluation_folder, "nnUNet_ext", resolution, join_texts_with_char(trained_on, "_"))
    
    thresholds = {}
    for i, task in enumerate(use_model):
        ood_df = pd.read_csv(os.path.join(eval_path,
                                          join_texts_with_char(use_model[:i+1], "_"),
                                          f"{trainer}__nnUNetPlansv2.1", "Generic_UNet", "SEQ", "head_None", "fold_0",
                                          task, f"ood_scores_{ood_method}.csv"), sep="\t")
        ood_df = ood_df[ood_df["split"] == split]
        ood_df = ood_df.sort_values(by=['ood_score'], ascending=True).drop_duplicates(subset=['case'], keep='first')
        threshold = np.percentile(ood_df['ood_score'], 95)
        thresholds[task] = threshold
    return thresholds


if __name__ == '__main__':
    evaluation_folder = "/local/scratch/clmn1/master_thesis/seeded/evaluation_augmentations"
    df = get_df("nnUNetTrainerSequential", "2d", [197, 198, 199], [197], [197, 198, 199, 400, 401, 402], 
                evaluation_folder=evaluation_folder, 
                append_ood=True, ood_method="uncertainty")
    thresholds = get_all_ood_thresholds("nnUNetTrainerSequential", "2d", [197, 198, 199], [197], 
                                      evaluation_folder=evaluation_folder,
                                      ood_method="uncertainty")
    print(df)
    print(thresholds)