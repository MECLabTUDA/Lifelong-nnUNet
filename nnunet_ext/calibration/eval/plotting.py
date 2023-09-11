# ------------------------------------------------------------------------------
# Plotting with seaborn.
# ------------------------------------------------------------------------------

import os
import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd

def plot_uncertainty_performance(df, metric, hue='Dist', style='Dist', save_name='uncertainty_vs_dice', boundary=0.5, save_path=None, figsize=(4, 2), x_norm=True, y_norm=True, ending='.png', normalize=False):

    if hue == 'Dataset':
        # Green
        #my_palette={'Original': '#262626', 'Strong': "#005F73", 'Medium': "#3BE4E7", 'Weak': "#94D2BD"}
        # Orange
        my_palette={'Original': '#262626', 'Strong': "#9B2226", 'Medium': "#CA6702", 'Weak': "#EE9B00"}
        markers={'Original': "o", 'Strong': "P", 'Medium': "^", 'Weak': "s"}
        # Green and lilac
        my_palette={'ID Covid-19': '#262626', 'In-house Covid-19': "#8e3b99", 'In-house non-Covid': "#e17a99"}
        markers={'ID Covid-19': "o", 'In-house Covid-19': "^", 'In-house non-Covid': "^"}

    else:
        my_palette=['#262626','#f4a261']
        markers={"ID": "o", "OOD": "^"}

    if normalize:
        x_item = 'NormedUncertainty'
    else:
        x_item = 'Uncertainty'

    plt.figure(figsize=figsize)
    with sns.axes_style("whitegrid"):
        ax = sns.scatterplot(x=x_item,
            y=metric,
            hue=hue,
            style=hue,
            alpha=1.,
            s=15,
            palette=my_palette,
            markers=markers,
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
        # Transparent background and grid
        plt.savefig(os.path.join(save_path, save_name + '.png'), transparent=True, bbox_inches="tight", dpi = 300)
        plt.savefig(os.path.join(save_path, save_name + '.svg'), transparent=True, bbox_inches="tight", dpi = 300)
    else:
        plt.show()

def boxplot(df, x, y, save_path, hue=None, file_name='boxplot', color_palette=None, figsize=(7, 2)):

    if hue == 'Dataset':
        my_palette=['#264653','#e76f51', '#2a9d8f', '#e9c46a']
    else:
        my_palette=color_palette

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

    plt.savefig(os.path.join(save_path, file_name + '.png'), bbox_inches="tight", dpi = 300)
    plt.savefig(os.path.join(save_path, file_name + '.svg'), bbox_inches="tight", dpi = 300)


def violinplot(df, x, y, hue, save_path, file_name='violinplot', figsize=(6, 2)):

    #my_palette={'Original': '#262626', 'Strong artifacts': "#005F73", 'Medium artifacts': "#3BE4E7", 'Weak artifacts': "#94D2BD", 'Strong affine tr.': "#9B2226", 'Medium affine tr.': "#CA6702", 'Weak affine tr.': "#EE9B00"}
    my_palette = {'ID Covid-19': '#262626', 'In-house Covid-19': "#8e3b99", 'In-house non-Covid': "#e17a99"}

    plt.figure(figsize=figsize)

    with sns.axes_style("whitegrid"):
        ax = sns.violinplot(x=x,
            y=y,
            data=df,
            palette=my_palette)
        ax.set(xlabel='')

    #ax.set(ylim=(-0.1, 1.2))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc=2)
    if y == 'NormedUncertainty':
        ax.set_ylabel('Uncertainty')
    else:
        ax.set_ylabel('')

    plt.savefig(os.path.join(save_path, file_name + '.png'), bbox_inches="tight", dpi = 300)
    plt.savefig(os.path.join(save_path, file_name + '.svg'), bbox_inches="tight", dpi = 300)