#!/usr/bin/env python3
"""
BCI Results Calculator

This script is the "number cruncher". It performs all the heavy-lifting analysis
on a BCI data file and saves the key results to a simple text file.
It does NOT produce any plots.

- Calculates cross-validated accuracy for "All Trials".
- Calculates cross-validated accuracy for "High-Attention Trials".
- Saves predictions and model coefficients for later plotting.

Usage: python calculate_results.py <path_to_data_file.csv>
"""

import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import json # Using JSON to save the results cleanly

# --- CONFIGURATION ---
NUM_SPLITS = 5
RANDOM_STATE = 42
CONFIDENCE_QUANTILE = 0.25

def calculate_all_results(filepath):
    """
    Main function to run all cross-validated analyses and save results.
    """
    print("--- BCI Results Calculator ---")
    try:
        print(f"Loading data from: {os.path.basename(filepath)}\n")
        df = pd.read_csv(filepath)

        feature_columns = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']
        X_all = df[feature_columns]
        y_all = df['target_side']

        # --- 1. Analyze "All Trials" ---
        print("Analyzing 'All Trials' model...")
        acc_all, model_all, predictions_all = train_and_evaluate_cv(X_all, y_all)
        
        # --- 2. Analyze "High-Attention Trials" ---
        print("Analyzing 'High-Attention' model...")
        df['asymmetry_magnitude'] = df['alpha_asymmetry'].abs()
        attention_threshold = df['asymmetry_magnitude'].quantile(CONFIDENCE_QUANTILE)
        df_focus = df[df['asymmetry_magnitude'] >= attention_threshold].copy()
        
        X_focus = df_focus[feature_columns]
        y_focus = df_focus['target_side']
        acc_focus, model_focus, _ = train_and_evaluate_cv(X_focus, y_focus)

        # --- 3. Save All Results to a File ---
        results = {
            'source_file': os.path.basename(filepath),
            'accuracy_all_trials': acc_all,
            'accuracy_high_attention': acc_focus,
            'true_labels': y_all.tolist(),
            'predictions_all_trials': [int(p) for p in predictions_all], # Convert numpy array to list
            'model_all_coeffs': model_all.coef_[0].tolist(),
            'model_focus_coeffs': model_focus.coef_[0].tolist(),
            'feature_names': feature_columns
        }
        
        base, _ = os.path.splitext(filepath)
        results_filepath = f"{base}_RESULTS.json"
        
        with open(results_filepath, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"\nSuccessfully calculated and saved results to: {os.path.basename(results_filepath)}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def train_and_evaluate_cv(X, y):
    """Helper function to perform cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_mapped = y.map({'left': 0, 'right': 1})
    
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    
    cv = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_scaled, y_mapped, cv=cv, scoring='accuracy')
    predictions = cross_val_predict(model, X_scaled, y_mapped, cv=cv)
    model.fit(X_scaled, y_mapped) # Train final model on all data
    
    return np.mean(scores), model, predictions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_results.py <path_to_data_file.csv>")
    else:
        file_to_analyze = sys.argv[1]
        calculate_all_results(file_to_analyze)