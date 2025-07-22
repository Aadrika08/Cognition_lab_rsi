import pandas as pd
import os 
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
# The threshold to define a "high-attention" trial
ASYMMETRY_THRESHOLD = 0.3
# The features to be used for training the models
FEATURE_COLUMNS = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']
# The label the models will try to predict
LABEL_COLUMN = 'is_correct'
# Cross-validation settings
NUM_SPLITS = 5
RANDOM_STATE = 42
# ---------------------

def run_comparison(filepath):
    """
    Loads data, trains both models, and prints the results.
    """
    print(f"--- Loading data from: {filepath} ---")
    try:
        # 1. Load and Validate Data
        df = pd.read_csv(filepath)

        required_columns = FEATURE_COLUMNS + [LABEL_COLUMN, 'asymmetry']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: The CSV file is missing one or more required columns.")
            print(f"Required: {required_columns}")
            return
            
        print(f"Successfully loaded {len(df)} trials.")

        # --- 2. Train and Evaluate the Unfiltered Model ---
        print("\n--- Model 1: Unfiltered (All Trials) ---")
        X_all = df[FEATURE_COLUMNS]
        y_all = df[LABEL_COLUMN]
        
        # Scale features for better performance
        scaler_all = StandardScaler()
        X_all_scaled = scaler_all.fit_transform(X_all)

        # Set up 5-fold cross-validation
        cv = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        model_all = LogisticRegression(class_weight='balanced', max_iter=1000)
        
        # Get the 5 accuracy scores
        scores_all = cross_val_score(model_all, X_all_scaled, y_all, cv=cv, scoring='accuracy')
        median_all = np.median(scores_all)

        print(f"5 Accuracy Scores: {[f'{score*100:.2f}%' for score in scores_all]}")
        print(f"Median Accuracy: {median_all*100:.2f}%")

        # --- 3. Train and Evaluate the Filtered Model ---
        print("\n--- Model 2: Filtered (High-Attention Trials) ---")
        
        # Filter the DataFrame based on the asymmetry threshold
        df_filtered = df[df['asymmetry'].abs() > ASYMMETRY_THRESHOLD].copy()
        
        if len(df_filtered) < NUM_SPLITS:
             print(f"Not enough high-attention trials ({len(df_filtered)}) to perform 5-fold cross-validation. Skipping.")
             return

        print(f"Filtered to {len(df_filtered)} trials where |asymmetry| > {ASYMMETRY_THRESHOLD}")

        X_filtered = df_filtered[FEATURE_COLUMNS]
        y_filtered = df_filtered[LABEL_COLUMN]

        scaler_filtered = StandardScaler()
        X_filtered_scaled = scaler_filtered.fit_transform(X_filtered)
        
        model_filtered = LogisticRegression(class_weight='balanced', max_iter=1000)
        
        # Get the 5 accuracy scores for the filtered data
        scores_filtered = cross_val_score(model_filtered, X_filtered_scaled, y_filtered, cv=cv, scoring='accuracy')
        median_filtered = np.median(scores_filtered)
        
        print(f"5 Accuracy Scores: {[f'{score*100:.2f}%' for score in scores_filtered]}")
        print(f"Median Accuracy: {median_filtered*100:.2f}%")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <path_to_data_file.csv>")
    else:
        file_to_analyze = sys.argv[1]
        run_comparison(file_to_analyze)