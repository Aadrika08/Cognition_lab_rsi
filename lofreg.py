#!/usr/bin/env python3
"""
BCI Detailed Analysis and Reporting Script

This script loads a BCI data CSV, trains and evaluates two models
(Unfiltered vs. Filtered), and prints a summary report to the terminal.

It also generates a new, detailed CSV file containing the model's
prediction and confidence score for every single trial.

Usage: python detailed_analysis.py <path_to_data_file.csv>
"""

import pandas as pd
import sys
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
ASYMMETRY_THRESHOLD = 0.3
FEATURE_COLUMNS = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']
LABEL_COLUMN = 'is_correct'
NUM_SPLITS = 5
RANDOM_STATE = 42
# ---------------------

def run_detailed_analysis(filepath):
    """
    Loads data, trains models, prints summaries, and saves a detailed report.
    """
    print(f"--- Loading data from: {filepath} ---")
    try:
        df = pd.read_csv(filepath)
        required_columns = FEATURE_COLUMNS + [LABEL_COLUMN, 'asymmetry']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV is missing required columns. Needed: {required_columns}")
            return
        print(f"Successfully loaded {len(df)} trials.")

        # --- 1. Train and Evaluate the Unfiltered Model (and get detailed predictions) ---
        print("\n--- Model 1: Unfiltered (All Trials) ---")
        X_all = df[FEATURE_COLUMNS]
        y_all = df[LABEL_COLUMN]
        
        scaler_all = StandardScaler()
        X_all_scaled = scaler_all.fit_transform(X_all)

        cv = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        model_all = LogisticRegression(class_weight='balanced', max_iter=1000)
        
        scores_all = cross_val_score(model_all, X_all_scaled, y_all, cv=cv, scoring='accuracy')
        median_all = np.median(scores_all)

        # Get per-trial predictions and probabilities for the entire dataset
        predictions_all = cross_val_predict(model_all, X_all_scaled, y_all, cv=cv)
        probabilities_all = cross_val_predict(model_all, X_all_scaled, y_all, cv=cv, method='predict_proba')

        print(f"5 Accuracy Scores: {[f'{score*100:.2f}%' for score in scores_all]}")
        print(f"Median Accuracy: {median_all*100:.2f}%")

        # --- 2. Train and Evaluate the Filtered Model ---
        print("\n--- Model 2: Filtered (High-Attention Trials) ---")
        df_filtered = df[df['asymmetry'].abs() > ASYMMETRY_THRESHOLD].copy()
        
        if len(df_filtered) < NUM_SPLITS:
             print(f"Not enough high-attention trials ({len(df_filtered)}) for 5-fold CV. Skipping.")
        else:
            print(f"Filtered to {len(df_filtered)} trials where |asymmetry| > {ASYMMETRY_THRESHOLD}")
            X_filtered = df_filtered[FEATURE_COLUMNS]
            y_filtered = df_filtered[LABEL_COLUMN]
            scaler_filtered = StandardScaler()
            X_filtered_scaled = scaler_filtered.fit_transform(X_filtered)
            model_filtered = LogisticRegression(class_weight='balanced', max_iter=1000)
            scores_filtered = cross_val_score(model_filtered, X_filtered_scaled, y_filtered, cv=cv, scoring='accuracy')
            median_filtered = np.median(scores_filtered)
            print(f"5 Accuracy Scores: {[f'{score*100:.2f}%' for score in scores_filtered]}")
            print(f"Median Accuracy: {median_filtered*100:.2f}%")

        # --- 3. Create and Save the Detailed Report CSV ---
        print("\n--- Generating Detailed CSV Report ---")
        
        # Add the new columns to the original DataFrame
        df['model_prediction'] = predictions_all
        # Confidence is the probability of the predicted class
        df['model_confidence'] = np.max(probabilities_all, axis=1)
        df['prediction_is_correct'] = (df['model_prediction'] == df[LABEL_COLUMN])
        
        # Define the output filename
        base, ext = os.path.splitext(filepath)
        report_filepath = f"{base}_ANALYSIS.csv"
        
        # Save the enriched DataFrame to the new CSV
        df.to_csv(report_filepath, index=False, float_format='%.6f')
        print(f"Successfully saved detailed report to: {report_filepath}")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <path_to_data_file.csv>")
    else:
        file_to_analyze = sys.argv[1]
        run_detailed_analysis(file_to_analyze)