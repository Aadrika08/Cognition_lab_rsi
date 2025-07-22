#!/usr/bin/env python3
"""
BCI Trial-Wise Heatmap Generator (Final Professional Version - Corrected)

This script loads all pre-analyzed `_ANALYSIS.csv` files and creates a
publication-quality, multi-panel heatmap figure. This version includes a
fix for the y-axis tick label error.

Usage: python plot_heatmaps_final.py
"""
import pandas as pd
import os
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- PROFESSIONAL COLOR THEME & CONFIGURATION ---
DATA_FOLDER = 'dataa/'
FEATURE_COLUMNS = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']
HEATMAP_PALETTE = "Blues"
CORRECT_COLOR = '#55A868'
INCORRECT_COLOR = '#C44E52'

# --- Apply the theme ---
try:
    plt.rcParams['font.family'] = 'Helvetica'
except: print("Warning: Helvetica font not found.")
sns.set_theme(style="white", context="paper", font_scale=1.2)

def create_trial_heatmaps():
    """
    Finds all _ANALYSIS.csv files and generates the final heatmap figure.
    """
    print("--- Generating Final Trial-Wise Heatmaps ---")
    
    search_path = os.path.join(DATA_FOLDER, '*_ANALYSIS.csv')
    analysis_files = glob.glob(search_path)

    if not analysis_files:
        print(f"Error: No `_ANALYSIS.csv` files found in '{DATA_FOLDER}'.")
        return

    print(f"Found {len(analysis_files)} analysis files to plot.")

    # --- Pre-computation Step: Find the global color scale ---
    global_min = float('inf')
    global_max = float('-inf')
    all_dfs = [pd.read_csv(f) for f in analysis_files]
    
    for df in all_dfs:
        min_val = df[FEATURE_COLUMNS].min().min()
        max_val = df[FEATURE_COLUMNS].max().max()
        if min_val < global_min: global_min = min_val
        if max_val > global_max: global_max = max_val
    print(f"Global Alpha Power Range calculated: {global_min:.2f} to {global_max:.2f}")

    # --- Create the Figure ---
    fig, axes = plt.subplots(len(analysis_files), 1, figsize=(20, 5 * len(analysis_files)), 
                             sharex=True, gridspec_kw={'hspace': 0.6})
    fig.suptitle('Trial-Wise Alpha Power Dynamics and Model Performance', fontsize=22, fontweight='bold')

    if len(analysis_files) == 1:
        axes = [axes]

    annotation_cmap = ListedColormap([INCORRECT_COLOR, CORRECT_COLOR])

    for i, df in enumerate(all_dfs):
        ax = axes[i]
        
        heatmap_data = df[FEATURE_COLUMNS].transpose()
        
        sns.heatmap(heatmap_data, ax=ax, cmap=HEATMAP_PALETTE, 
                    vmin=global_min, vmax=global_max,
                    cbar_kws={'label': 'Alpha Power'})
        
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(ax.get_xticks())
        ax_top.xaxis.set_ticks_position('top')
        
        for spine in ax_top.spines.values():
            spine.set_visible(False)
        ax_top.tick_params(axis='both', length=0, labelsize=0)
        
        correctness_bar = df['prediction_is_correct'].astype(int).values.reshape(1, -1)
        ax_top.imshow(correctness_bar, aspect='auto', extent=(0, len(df), 0, 1), cmap=annotation_cmap)
        
        ax_top.set_yticks([])
        ax_top.set_ylabel('Performance', rotation=0, ha='right', va='center', fontweight='bold', labelpad=30)
        
        participant_name = os.path.splitext(os.path.basename(analysis_files[i]))[0].replace('_ANALYSIS', '')
        ax.set_title(f'Participant: {participant_name}', fontweight='bold')
        ax.set_xlabel('Trial Number', fontweight='bold')
        ax.set_ylabel('EEG Channel', fontweight='bold')
        
        # --- THE FIX for Y-AXIS LABELS ---
        # Step 1: Explicitly tell matplotlib WHERE to put the ticks (in the middle of each row)
        ax.set_yticks(np.arange(len(FEATURE_COLUMNS)) + 0.5)
        # Step 2: Now that the ticks exist, we can set their labels.
        ax.set_yticklabels(FEATURE_COLUMNS, rotation=0, ha='right')
        # --------------------------------

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print("\nDisplaying final figure...")
    plt.show()

if __name__ == "__main__":
    create_trial_heatmaps()