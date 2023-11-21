import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd


def find_threshold(df, trained_on):
    subset_df = df[df['Task'] == trained_on[0]]
    #subset_df = subset_df[subset_df['is_val'] == False] #only validation data
    min_ood_score = 0
    max_ood_score = max(df['ood_score'])
    xs = np.linspace(min_ood_score, max_ood_score, 1000)
    xs = subset_df['ood_score'].sort_values()
    for x in xs[::-1]:
        #collect the samples classified as in-distribution, i.e. where uncertainty is lower than x
        tpr = len([v for v in subset_df['ood_score'] if v < x]) / len(subset_df)
        if tpr <= 0.95:
            return x
        
def compute_fpr(df, threshold, task, is_ood):
    subset_df = df
    #subset_df = df[df['is_val'] == True]
    subset_df = subset_df[subset_df['Task'] == task]
    if is_ood:
        #collect the samples classified as in-distribution, i.e. where uncertainty is lower than x
        #but they are OOD actually
        return len([v for v in subset_df['ood_score'] if v < threshold]) / len(subset_df)
    else:
        #collect the samples classified as OOD, i.e. where uncertainty is greater than x
        #but they are in-distribution actually
        return len([v for v in subset_df['ood_score'] if v >= threshold]) / len(subset_df)
    



df_ood = pd.read_csv("data/ood_scores_vae_reconstruction.csv", sep="\t")
df_segmentation = pd.read_csv("data/val_metrics_all.csv", sep="\t")
# join dataframes where df_ood['case'] == df_segmentation['subject_id']
df = pd.merge(df_ood, df_segmentation, left_on='case', right_on='subject_id')

trained_on = ["Task097_DecathHip"]

threshold = find_threshold(df_ood, trained_on)