#!/usr/bin/env python3
"""
BCI Individual Confidence Plotter — ✨ Polished Version ✨

This script loads raw EEG data files and generates publication-ready scatter plots
for each participant, showing how model confidence varies with alpha asymmetry strength.

Each plot = one participant
X-axis = Absolute Alpha Asymmetry (signal strength)
Y-axis = Logistic Regression Confidence
"""

import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- FANCY COLOR PALETTE ---
SLATE_BLUE = "#001C8C"
CRIMSON_RED = "#D50007"
PALE_BG = "#FFFFFF"
GRAY = '#999999'
DARK_TEXT = "#000000"

# --- Plot Theme Setup ---
try:
    plt.rcParams['font.family'] = 'Helvetica'
except:
    print("⚠️ Helvetica font not found. Using default font.")

plt.rcParams.update({
    'figure.facecolor': PALE_BG,
    'axes.facecolor': PALE_BG,
    'axes.edgecolor': GRAY,
    'text.color': DARK_TEXT,
    'axes.labelcolor': DARK_TEXT,
    'xtick.color': DARK_TEXT,
    'ytick.color': DARK_TEXT,
    'grid.color': GRAY,
})

sns.set_context("talk", font_scale=1.4)
sns.set_style("whitegrid")

# --- CONFIG ---
DATA_FOLDER = 'data/'
FEATURE_COLUMNS = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']
TARGET_COLUMN = 'target_side'

def create_individual_confidence_plots():
    """
    Creates scatter+regression plots showing relationship between alpha asymmetry
    and model confidence for each participant file.
    """
    print("📊 Generating Individual Confidence vs. Signal Strength Plots")

    search_path = os.path.join(DATA_FOLDER, '*.csv')
    csv_files = sorted([
        f for f in glob.glob(search_path)
        if not any(s in os.path.basename(f) for s in ['_RESULTS', '_SIMULATED', '_FORGED', '_corrected'])
    ])

    if not csv_files:
        print(f"❌ No valid CSV files found in '{DATA_FOLDER}'")
        return

    print(f"✅ Found {len(csv_files)} participant files")

    for i, filepath in enumerate(csv_files):
        df = pd.read_csv(filepath)

        # Basic cleanup
        df = df.dropna(subset=FEATURE_COLUMNS + ['asymmetry'])

        # Feature matrix + label
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN].map({'left': 0, 'right': 1})

        # Standardize + train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_scaled, y)

        # Predict probs
        df['model_confidence'] = np.max(model.predict_proba(X_scaled), axis=1)
        df['asymmetry_magnitude'] = df['asymmetry'].abs()

        # --- Plot Setup ---
        fig, ax = plt.subplots(figsize=(10, 7))

        sns.regplot(
            data=df,
            x='asymmetry_magnitude',
            y='model_confidence',
            ax=ax,
            scatter_kws={'color': SLATE_BLUE, 'alpha': 0.5, 's': 60, 'edgecolor': 'white'},
            line_kws={'color': CRIMSON_RED, 'linewidth': 2.5},
            ci=95
        )

        # --- Pretty Labels ---
        participant_id = os.path.splitext(os.path.basename(filepath))[0]
        ax.set_title(f'Model Confidence vs. Alpha Asymmetry\nParticipant: {participant_id}',
                     fontweight='bold', fontsize=18, pad=20)

        ax.set_xlabel('Alpha Asymmetry Magnitude (|O1-O2|)', fontsize=15, labelpad=10)
        ax.set_ylabel('Model Prediction Confidence (P̂)', fontsize=15, labelpad=10)

        ax.set_xlim(left=0)
        ax.set_ylim(0.45, 1.02)

        ax.tick_params(axis='both', which='major', labelsize=12)
        sns.despine()
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Optional: Save the figure (uncomment to activate)
        # output_dir = 'plots'
        # os.makedirs(output_dir, exist_ok=True)
        # plt.savefig(os.path.join(output_dir, f"{participant_id}_confidence_plot.png"), dpi=300)

        plt.tight_layout()

    print(f"\n✨ Done! Displaying {len(csv_files)} plots now...")
    plt.show()


if __name__ == "__main__":
    create_individual_confidence_plots()
