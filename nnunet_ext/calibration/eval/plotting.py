# ------------------------------------------------------------------------------
# Plotting with seaborn.
# ------------------------------------------------------------------------------

import os
import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd

def plot_uncertainty_performance(df, metric, hue='Dist', style='Dist', save_name='uncertainty_vs_dice', boundary=0.5, save_path=None, figsize=(4, 2), x_norm=False, y_norm=True, ending='.png'):

    if hue == 'Dataset':
        my_palette=['#264653','#e76f51', '#2a9d8f', '#e9c46a']
    else:
        my_palette=['#262626','#f4a261']

    plt.figure(figsize=figsize)
    with sns.axes_style("whitegrid"):
        ax = sns.scatterplot(x='NormedUncertainty',
            y=metric,
            hue=hue,
            style=hue,
            alpha=1.,
            s=15,
            palette=my_palette,
            markers={"ID": "o", "OOD": "^"},
            edgecolor='gray',
            data=df)
    # Format titles
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc=2)
    ax.set_xlabel('Uncertainty')
    # normalize if specified
    if x_norm:
        ax.set(xlim=(0, 1))
    if y_norm:
        ax.set(ylim=(0, 1))
    if boundary is not None:
        plt.plot([boundary, boundary], [0, 1], color='gray')
    # Save plot
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        file_name = save_name+ending
        # Transparent background and grid
        plt.savefig(os.path.join(save_path, file_name), transparent=True, bbox_inches="tight", dpi = 300)
    else:
        plt.show()

def boxplot(df, x, y, hue, save_path, file_name='boxplot.png', figsize=(7, 2)):

    if hue == 'Dataset':
        my_palette=['#264653','#e76f51', '#2a9d8f', '#e9c46a']
    else:
        my_palette=['#2e5d7d','#f4a261']

    plt.figure(figsize=figsize)

    with sns.axes_style("whitegrid"):
        ax = sns.boxplot(x=x,
            y=y,
            hue=hue,
            data=df,
            palette=my_palette)
        ax.set(xlabel='')

    ax.set(ylim=(-0.1, 1))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc=2)
    if y == 'NormedUncertainty':
        ax.set_ylabel('Uncertainty')
    else:
        ax.set_ylabel('')

    plt.savefig(os.path.join(save_path, file_name), bbox_inches="tight", dpi = 300)
