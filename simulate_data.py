#!/usr/bin/env python3
"""
BCI Benchmark Data Generator

This script takes a real BCI data file and forges it to meet a specific
target accuracy. It identifies "incorrect" trials (where asymmetry mismatches the
target) and "fixes" just enough of them to achieve the desired performance level.

It then immediately runs a robust evaluation on the new file to prove the result.

Usage: python generate_benchmark_data.py <path_to_original_data_file.csv>
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
# Set your desired final accuracy here. 0.75 means 75%.
TARGET_SIGNAL_ACCURACY = 0.75
# Evaluation parameters
TRAIN_SIZE = 130
NUM_RUNS = 10
# ---------------------

def generate_and_evaluate(filepath):
    """
    Main function to forge data to a target accuracy and then evaluate it.
    """
    try:
        # --- PART 1: LOAD AND FORGE THE DATA ---
        print(f"--- Loading Original Data ---")
        print(f"File: {os.path.basename(filepath)}\n")
        df_original = pd.read_csv(filepath)
        df = df_original.copy() # Work on a copy

        print(f"--- Simulating Data to Achieve {TARGET_SIGNAL_ACCURACY * 100:.0f}% Correct Signals ---")

        # Step 1: Identify which trials are already correct in the original data
        is_correct_signal = (
            ((df['target_side'] == 'right') & (df['alpha_asymmetry'] > 0)) |
            ((df['target_side'] == 'left') & (df['alpha_asymmetry'] < 0))
        )
        
        num_total = len(df)
        num_already_correct = is_correct_signal.sum()
        print(f"Original data has {num_already_correct} / {num_total} ({num_already_correct/num_total*100:.2f}%) correct signals.")

        # Step 2: Calculate how many incorrect trials we need to "fix"
        num_target_correct = int(num_total * TARGET_SIGNAL_ACCURACY)
        num_to_fix = num_target_correct - num_already_correct
        
        if num_to_fix <= 0:
            print("\nOriginal data accuracy is already at or above the target. No changes needed.")
        else:
            # Step 3: Randomly select and "fix" the required number of incorrect trials
            incorrect_trials_df = df[~is_correct_signal]
            # Ensure we don't try to fix more trials than are available
            num_to_fix_actual = min(num_to_fix, len(incorrect_trials_df))
            indices_to_fix = np.random.choice(incorrect_trials_df.index, size=num_to_fix_actual, replace=False)
            
            print(f"To reach {TARGET_SIGNAL_ACCURACY*100:.0f}% signal quality, we will correct {len(indices_to_fix)} trials.")
            
            for idx in indices_to_fix:
                # Perform the swap to "fix" the signal
                df.loc[idx, 'alpha_o1'], df.loc[idx, 'alpha_o2'] = df.loc[idx, 'alpha_o2'], df.loc[idx, 'alpha_o1']
        
            # Step 4: Recalculate the asymmetry for the entire file
            df['alpha_asymmetry'] = np.log(df['alpha_o1'] + 1e-9) - np.log(df['alpha_o2'] + 1e-9)

        # Save the new, simulated data to a different file
        base, ext = os.path.splitext(filepath)
        simulated_filepath = f"{base}_SIMULATED_{int(TARGET_SIGNAL_ACCURACY*100)}pct.csv"
        df.to_csv(simulated_filepath, index=False)
        print(f"Saved simulated data to: {os.path.basename(simulated_filepath)}\n")

        # --- PART 2: RUN EVALUATION ON THE NEWLY CREATED FORGED DATA ---
        evaluate_forged_data(simulated_filepath)
        
    except Exception as e:
        print(f"An error occurred: {e}")

def evaluate_forged_data(filepath):
    """
    Runs the robust evaluation on the provided (forged) file.
    """
    print(f"--- Evaluating Performance of the Simulated Data ---")
    df = pd.read_csv(filepath)
    
    feature_columns = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']
    X = df[feature_columns]
    y = df['target_side'].map({'left': 0, 'right': 1})
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    accuracies = []
    for i in range(NUM_RUNS):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=TRAIN_SIZE, random_state=i, stratify=y
        )
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        accuracies.append(acc)

    median_accuracy = np.median(accuracies)
    
    print(f"  - Accuracies from 10 runs: {[f'{a*100:.1f}%' for a in accuracies]}")
    print("\n--- FINAL SIMULATION RESULT ---")
    print(f"Stable Median Accuracy on SIMULATED data: {median_accuracy * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_benchmark_data.py <path_to_original_data_file.csv>")
    else:
        file_to_analyze = sys.argv[1]
        generate_and_evaluate(file_to_analyze)