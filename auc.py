#!/usr/bin/env python3
"""
BCI Final Report Generator (v12 - Per-Fold Accuracy Reporting)

This single script performs all necessary analysis on a specific list of BCI
data files. It PRINTS the 5-fold accuracy scores for each model and participant,
and generates ALL FIVE final figures for the research paper.

Usage: python generate_full_report.py
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- PROFESSIONAL THEME CONFIGURATION ---
BAR_BLUE = '#3E6FED'
BAR_RED = '#D43F3A'
ROC_COLORS = ['#0072BD', '#D95319', '#2ca02c', '#9467bd']
DIST_LEFT_COLOR = '#d62728'
DIST_RIGHT_COLOR = '#1f77b4'
SOFT_GRAY = '#7f7f7f'
BACKGROUND_WHITE = '#FFFFFF'
GRID_GRAY = '#e5e5e5'
TEXT_BLACK = '#000000'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'DejaVu Sans',
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.facecolor': BACKGROUND_WHITE,
    'axes.facecolor': BACKGROUND_WHITE,
    'axes.edgecolor': SOFT_GRAY,
    'xtick.color': TEXT_BLACK,
    'ytick.color': TEXT_BLACK,
    'axes.labelcolor': TEXT_BLACK,
    'text.color': TEXT_BLACK,
    'grid.color': GRID_GRAY,
})
# ---------------------------------------------------------

# --- ANALYSIS CONFIGURATION ---
SPECIFIC_FILE_PATHS = [
    '/home/aadrika/Documents/Project_attention/dataa/file01_ANALYSIS.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file02_ANALYSIS.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file03_ANALYSIS.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file04_ANALYSIS.csv'
]
ASYMMETRY_THRESHOLD = 0.3
FEATURE_COLUMNS = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']
LABEL_COLUMN = 'target_side'
NUM_SPLITS = 5
RANDOM_STATE = 42
CONFIDENCE_QUANTILE = 0.25
# --------------------------------

def generate_final_report():
    """
    Finds all files, runs analyses, prints accuracies, and generates all plots.
    """
    print("--- BCI Final Report Generator ---")
    
    all_results = []
    
    for i, filepath in enumerate(SPECIFIC_FILE_PATHS):
        participant_name = f"Participant {i + 1}"
        print(f"\nProcessing data for {participant_name} ({os.path.basename(filepath)})...")
        
        if not os.path.exists(filepath):
            print(f"  - WARNING: File not found. Skipping.")
            continue
        
        df = pd.read_csv(filepath)

        # The helper function now returns the full list of scores
        scores_all, pred_all, prob_all = get_cv_metrics(df, filter_attention=False)
        if scores_all is None: continue
        
        print("\n--- Model 1: Unfiltered (All Trials) ---")
        print(f"  - 5-Fold Accuracy Scores: {[f'{s*100:.2f}%' for s in scores_all]}")
        print(f"  - Mean Accuracy: {np.mean(scores_all) * 100:.2f}%")
        
        df_filtered = df[df['alpha_asymmetry'].abs() > ASYMMETRY_THRESHOLD].copy()
        scores_filtered, pred_filtered, prob_filtered = get_cv_metrics(df_filtered, filter_attention=True)
        
        print("\n--- Model 2: High-Attention Trials ---")
        if scores_filtered is None:
            print(f"  - Skipping High-Attention model for {participant_name}, not enough trials.")
            pred_filtered, prob_filtered = None, None
        else:
             print(f"  - Filtered to {len(df_filtered)} trials.")
             print(f"  - 5-Fold Accuracy Scores: {[f'{s*100:.2f}%' for s in scores_filtered]}")
             print(f"  - Mean Accuracy: {np.mean(scores_filtered) * 100:.2f}%")

        all_results.append({
            'participant': participant_name,
            'dataframe': df,
            'dataframe_filtered': df_filtered,
            'accuracy_all': np.mean(scores_all) if scores_all is not None else None,
            'predictions_all': pred_all,
            'probabilities_all': prob_all,
            'accuracy_filtered': np.mean(scores_filtered) if scores_filtered is not None else None,
            'predictions_filtered': pred_filtered,
            'probabilities_filtered': prob_filtered,
        })

    if not all_results:
        print("\nNo data was successfully processed. Exiting.")
        return

    # --- Generate all FIVE final plots ---
    plot_roc_curves(all_results, model_type='all')
    plot_roc_curves(all_results, model_type='filtered')
    plot_dual_axis_summary(all_results)
    plot_asymmetry_distributions(all_results, CONFIDENCE_QUANTILE)
    plot_confidence_vs_asymmetry(all_results)
    plot_confusion_matrices(all_results)

    print("\n--- Analysis Complete ---")
    plt.show()

def get_cv_metrics(df, filter_attention=False):
    """
    Helper function to get CV scores, predictions, and probabilities.
    NOW RETURNS THE FULL LIST OF SCORES.
    """
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].map({'left': 0, 'right': 1})
    
    if np.min(np.bincount(y)) < NUM_SPLITS:
        print(f"  - SKIPPING CV: Not enough samples in one class for {NUM_SPLITS}-fold split.")
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    cv = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # Calculate the 5 accuracy scores
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    # Get per-trial predictions and probabilities
    predictions = cross_val_predict(model, X_scaled, y, cv=cv)
    probabilities = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')
    
    # Return the full list of scores, along with predictions and probabilities
    return scores, predictions, probabilities


def plot_roc_curves(results, model_type):
    """Generates the ROC curve plots for a specific model type."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    title = 'ROC Curves (All Trials Model)'
    if model_type == 'filtered':
        title = 'ROC Curves (High-Attention Model)'

    for i, res in enumerate(results):
        participant_name = res['participant']
        if model_type == 'all':
            y_mapped = res['dataframe'][LABEL_COLUMN].map({'left': 0, 'right': 1})
            y_score = res['probabilities_all'][:, 1]
            acc = res['accuracy_all']
        else:
            if res['probabilities_filtered'] is None: continue
            y_mapped = res['dataframe_filtered'][LABEL_COLUMN].map({'left': 0, 'right': 1})
            y_score = res['probabilities_filtered'][:, 1]
            acc = res['accuracy_filtered']
            
        fpr, tpr, _ = roc_curve(y_mapped, y_score)
        auc = roc_auc_score(y_mapped, y_score)
        ax.plot(fpr, tpr, color=ROC_COLORS[i], lw=2.5, label=f'{participant_name} (AUC={auc:.2f}, Acc={acc*100:.1f}%)')

    style_roc_plot(ax, title)


def plot_dual_axis_summary(results):
    """Creates the dual-axis bar chart for Accuracy and AUC."""
    plot_data = []
    for r in results:
        if r['probabilities_filtered'] is not None:
            y_filtered_mapped = r['dataframe_filtered'][LABEL_COLUMN].map({'left': 0, 'right': 1})
            auc_filtered = roc_auc_score(y_filtered_mapped, r['probabilities_filtered'][:, 1])
            plot_data.append({'participant': r['participant'], 'accuracy': r['accuracy_filtered'], 'auc': auc_filtered})

    df_plot = pd.DataFrame(plot_data)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    x = np.arange(len(df_plot['participant']))
    bar_width = 0.35
    
    acc_bars = ax1.bar(x - bar_width/2, df_plot['accuracy'] * 100, width=bar_width, color=BAR_BLUE, label='Accuracy (%)')
    ax1.set_ylabel('Accuracy (%)', color=BAR_BLUE)
    ax1.tick_params(axis='y', colors=BAR_BLUE)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    auc_bars = ax2.bar(x + bar_width/2, df_plot['auc'], width=bar_width, color=BAR_RED, label='ROC AUC')
    ax2.set_ylabel('ROC AUC Score', color=BAR_RED)
    ax2.tick_params(axis='y', colors=BAR_RED)
    ax2.set_ylim(0, 1.0)

    ax1.set_xlabel('Participant')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_plot['participant'])
    ax1.set_title('Per-Participant Performance (High-Attention Model)', pad=20)

    for bar in acc_bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', va='bottom', ha='center')
    for bar in auc_bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', va='bottom', ha='center')
        
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.tight_layout()


def plot_asymmetry_distributions(results, confidence_quantile):
    """Creates a side-by-side KDE plot comparing asymmetry distributions."""
    df_all = results[0]['dataframe']
    participant_name = results[0]['participant']
    
    df_all['asymmetry_magnitude'] = df_all['alpha_asymmetry'].abs()
    attention_threshold = df_all['asymmetry_magnitude'].quantile(confidence_quantile)
    df_filtered = df_all[df_all['asymmetry_magnitude'] >= attention_threshold].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    fig.suptitle(f'Signal Distribution Comparison ({participant_name})', fontsize=22)
    
    sns.kdeplot(data=df_all, x='alpha_asymmetry', hue='target_side', 
                fill=True, alpha=0.7, ax=axes[0],
                palette={'left': BOLD_RED, 'right': RICH_BLUE}, hue_order=['left', 'right'])
    
    axes[0].set_title('(A) All Trials')
    axes[0].axvline(0, color=SOFT_GRAY, linestyle='--')
    axes[0].set_xlabel("Alpha Asymmetry (log O1 - log O2)")
    axes[0].set_ylabel("Density")
    axes[0].legend(title='Cue', labels=['Left Cue', 'Right Cue'])

    sns.kdeplot(data=df_filtered, x='alpha_asymmetry', hue='target_side', 
                fill=True, alpha=0.7, ax=axes[1],
                palette={'left': BOLD_RED, 'right': RICH_BLUE}, hue_order=['left', 'right'])
    
    axes[1].set_title('(B) High-Attention Trials Only')
    axes[1].axvline(0, color=SOFT_GRAY, linestyle='--')
    axes[1].set_xlabel("Alpha Asymmetry (log O1 - log O2)")
    axes[1].set_ylabel("") 
    axes[1].legend(title='Cue', labels=['Left Cue', 'Right Cue'])
    
    sns.despine(fig=fig)
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_confidence_vs_asymmetry(results):
    """Creates separate scatter plots of Model Confidence vs. Signal Strength."""
    print("\nGenerating separate confidence vs. signal strength plots...")
    
    for i, res in enumerate(results):
        fig, ax = plt.subplots(figsize=(10, 7))
        
        df = res['dataframe'].copy()
        participant_name = res['participant']
        probabilities = res['probabilities_all']
        
        df['model_confidence'] = np.max(probabilities, axis=1)
        df['asymmetry_magnitude'] = df['alpha_asymmetry'].abs()

        sns.regplot(
            data=df, x='asymmetry_magnitude', y='model_confidence', ax=ax,
            scatter_kws={'color': BAR_BLUE, 'alpha': 0.3, 'edgecolor': 'w', 's': 60},
            line_kws={'color': BAR_RED, 'linewidth': 3}
        )
        
        ax.set_title(f'Model Confidence vs. Signal Strength\n(Participant: {participant_name})')
        ax.set_xlim(left=0)
        ax.set_ylim(0.45, 1.05)
        ax.set_ylabel('Model Prediction Confidence')
        ax.set_xlabel('Signal Strength (|Alpha Asymmetry|)')
        
        sns.despine(ax=ax)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()


def plot_confusion_matrices(results):
    """
    Creates a side-by-side comparison of the aggregated confusion matrices.
    """
    y_true_all = pd.concat([res['dataframe'][LABEL_COLUMN].map({'left': 0, 'right': 1}) for res in results])
    preds_all = np.concatenate([res['predictions_all'] for res in results])
    
    y_true_filtered_list = [res['dataframe_filtered'][LABEL_COLUMN].map({'left': 0, 'right': 1}) for res in results if res['predictions_filtered'] is not None]
    if not y_true_filtered_list:
        print("No high-attention data to create a confusion matrix for.")
        return
        
    y_true_filtered = pd.concat(y_true_filtered_list)
    preds_filtered = np.concatenate([res['predictions_filtered'] for res in results if res['predictions_filtered'] is not None])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Aggregated Confusion Matrices Across All Sessions', fontsize=22)
    
    acc_all = accuracy_score(y_true_all, preds_all)
    cm_all = confusion_matrix(y_true_all, preds_all, labels=[0, 1])
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted Left', 'Predicted Right'], 
                yticklabels=['Actual Left', 'Actual Right'],
                annot_kws={"size": 20})
    axes[0].set_title(f'(A) All Trials\nOverall Accuracy: {acc_all*100:.2f}%')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    acc_filtered = accuracy_score(y_true_filtered, preds_filtered)
    cm_filtered = confusion_matrix(y_true_filtered, preds_filtered, labels=[0, 1])
    sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                xticklabels=['Predicted Left', 'Predicted Right'], 
                yticklabels=['Actual Left', 'Actual Right'],
                annot_kws={"size": 20})
    axes[1].set_title(f'(B) High-Attention Trials\nOverall Accuracy: {acc_filtered*100:.2f}%')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('')

    sns.despine(fig=fig)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

def style_roc_plot(ax, title):
    """Applies the professional theme to an ROC plot."""
    ax.plot([0, 1], [0, 1], color=SOFT_GRAY, lw=2.5, linestyle='--', label='Chance Level (AUC = 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', color=GRID_GRAY)
    sns.despine(ax=ax)
    fig = ax.get_figure()
    fig.tight_layout()

if __name__ == "__main__":
    generate_final_report()