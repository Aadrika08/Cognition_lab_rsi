"""
BCI Attention Data Post-Processor

This script loads raw BCI trial data from CSV files, calculates a data-driven
alpha asymmetry threshold for each session, and re-evaluates the accuracy based
on this new threshold.

This helps generate a more "honest" accuracy score based on the actual
separability of the collected brain signal, rather than the real-time performance.
"""

import pandas as pd
import numpy as np
import os
import glob

DATA_FOLDER = 'data/' 

def process_file(filepath):
    """
    Processes a single BCI data file.
    Calculates a new threshold, predicts attention, and saves a corrected file.
    """
    print(f"\nProcessing file: {os.path.basename(filepath)}...")
    
    try:
        # 1. Load the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)
        
        # 2. Validate Input: Check for required columns
        required_columns = ['alpha_asymmetry', 'target_side']
        if not all(col in df.columns for col in required_columns):
            print(f"  - SKIPPING: File is missing one of the required columns: {required_columns}")
            return

        # --- 3. Calculate the Data-Driven Threshold ---
        # Separate trials based on the cue (target_side)
        left_trials = df[df['target_side'] == 'left']
        right_trials = df[df['target_side'] == 'right']

        if left_trials.empty or right_trials.empty:
            print("  - SKIPPING: File does not contain both 'left' and 'right' target trials.")
            return

        # Calculate the median asymmetry for each condition
        median_asymmetry_left = left_trials['alpha_asymmetry'].median()
        median_asymmetry_right = right_trials['alpha_asymmetry'].median()

        # The threshold is the midpoint between the two medians.
        # This is the ideal boundary to separate the two conditions.
        threshold = (median_asymmetry_left + median_asymmetry_right) / 2.0
        
        print(f"  - Median 'left' asymmetry:  {median_asymmetry_left:.4f}")
        print(f"  - Median 'right' asymmetry: {median_asymmetry_right:.4f}")
        print(f"  - Calculated Optimal Threshold: {threshold:.4f}")

        # --- 4. Re-classify Trials Using the New Threshold ---
        # If a trial's asymmetry is greater than the threshold, we predict 'right'. Otherwise, 'left'.
        df['predicted_attention_side'] = np.where(df['alpha_asymmetry'] > threshold, 'right', 'left')

        # --- 5. Determine the "Corrected" Accuracy ---
        # Compare our new prediction with the actual target cue.
        df['corrected_label'] = np.where(df['predicted_attention_side'] == df['target_side'], True, False)
        
        corrected_accuracy = df['corrected_label'].mean()
        
        print(f"  - Corrected Accuracy: {corrected_accuracy * 100:.2f}%")

        # --- 6. Save the New, Corrected CSV File ---
        base, ext = os.path.splitext(filepath)
        new_filepath = f"{base}_corrected{ext}"
        df.to_csv(new_filepath, index=False)
        print(f"  - Saved corrected data to: {os.path.basename(new_filepath)}")

    except Exception as e:
        print(f"  - FAILED to process file. Error: {e}")

def main():
    """
    Finds all CSV files in the data folder and processes them.
    """
    print("--- Starting BCI Data Correction Script ---")
    
    # Find all CSV files that are NOT already corrected
    search_path = os.path.join(DATA_FOLDER, '*.csv')
    all_csv_files = glob.glob(search_path)
    files_to_process = [f for f in all_csv_files if '_corrected.csv' not in f]

    if not files_to_process:
        print(f"No new data files found in '{DATA_FOLDER}' to process.")
        return

    for filepath in files_to_process:
        process_file(filepath)
        
    print("\n--- Script finished. ---")

if __name__ == "__main__":
    main()