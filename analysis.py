#!/usr/bin/env python3
#
# analysis.py (Professional Storytelling Version)
#
# This script analyzes BCI data to answer three key questions:
#   1. Was the user's brain signal for 'left' vs. 'right' different?
#   2. How well did a machine learning model perform at predicting the user's intent?
#   3. Which EEG channels were most important for the model's decisions?
#
# Usage: python analysis.py path/to/your/data_file.csv
#

import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Professional Styling and Color Palette ---
MIDNIGHT_BLUE = '#003366'
BRIGHT_YELLOW = '#FFD700'
VIBRANT_RED = '#EE4B2B'
LIGHT_GRAY = '#CCCCCC'

try:
    plt.rcParams['font.family'] = 'Helvetica'
    print("Helvetica font found and set.")
except Exception:
    print("Warning: Helvetica font not found. Falling back to a default sans-serif font.")
    plt.rcParams['font.family'] = 'sans-serif'

sns.set_style("whitegrid", {'axes.grid': False}) # A cleaner look without grid lines
sns.set_context("talk")

def analyze_bci_data(filepath):
    """
    Main function to load data, run analysis, and generate a suite of plots.
    """
    print("\n--- BCI Data Analysis ---")
    
    try:
        # 1. Load and Preprocess Data
        print(f"Loading data from: {filepath}\n")
        df = pd.read_csv(filepath)

        if len(df) < 20:
            print("Error: Not enough data for a reliable analysis. At least 20 trials are recommended.")
            return

        feature_columns = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']
        if not all(col in df.columns for col in feature_columns):
            raise ValueError(f"CSV file must contain the required columns: {feature_columns}")

        X = df[feature_columns]
        y = df['target_side']

        # Scale features for better model performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
        
        # 3. Train Logistic Regression Model
        print("Training Logistic Regression model...")
        model = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_train)
        print("Model training complete.\n")

        # 4. Evaluate Model
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print("--- Model Performance Report ---")
        print(f"Overall Model Accuracy: {accuracy * 100:.2f}%\n")
        
        # --- Generate All Plots ---
        plot_signal_quality(df)
        plot_model_performance(y_test, predictions, probabilities, model.classes_, accuracy)
        plot_feature_importance(model, feature_columns)

        print("Displaying 3 analysis figures...")
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def plot_signal_quality(df):
    """
    QUESTION 1: Was the user's brain signal for 'left' vs. 'right' different?
    This plot shows the distribution of the core 'alpha_asymmetry' feature.
    A clear separation between the two colors is the goal.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(data=df, x='target_side', y='alpha_asymmetry', hue='target_side', 
                   palette={'left': VIBRANT_RED, 'right': MIDNIGHT_BLUE}, ax=ax, legend=False)
    
    sns.stripplot(data=df, x='target_side', y='alpha_asymmetry', color=".3", size=4, ax=ax)

    ax.axhline(0, color='white', linestyle='--', linewidth=1.5)
    ax.set_title("Signal Quality: Is the Brain Signal Separable?", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Target Side", fontsize=15, fontweight='bold')
    ax.set_ylabel("Alpha Asymmetry Value", fontsize=15, fontweight='bold')
    ax.set_xticklabels(['Left', 'Right'], fontsize=12)
    fig.tight_layout()

def plot_model_performance(y_test, predictions, probabilities, classes, accuracy):
    """
    QUESTION 2: How well did the model perform?
    This dashboard shows accuracy, a confusion matrix, and model confidence.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Model Performance Dashboard", fontsize=24, fontweight='bold')

    # --- Subplot 1: Confusion Matrix ---
    cm = confusion_matrix(y_test, predictions, labels=['left', 'right'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax1,
                xticklabels=['Predicted Left', 'Predicted Right'], 
                yticklabels=['Actual Left', 'Actual Right'],
                annot_kws={"size": 22})
    ax1.set_xlabel('Predicted Label', fontweight='bold')
    ax1.set_ylabel('True Label', fontweight='bold')
    ax1.set_title(f'A) Prediction Accuracy: {accuracy*100:.2f}%', fontsize=18, fontweight='bold', pad=15)

    # --- Subplot 2: Prediction Confidence ---
    prob_right = probabilities[:, list(classes).index('right')]
    sns.histplot(prob_right, kde=True, color=MIDNIGHT_BLUE, bins=15, ax=ax2)
    ax2.axvline(0.5, color=VIBRANT_RED, linestyle='--', linewidth=2.5, label='Decision Boundary')
    ax2.set_title("B) Model Confidence", fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel("Model's Predicted Probability of 'Right'", fontweight='bold')
    ax2.set_ylabel('Count of Trials', fontweight='bold')
    ax2.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])

def plot_feature_importance(model, features):
    """
    QUESTION 3: Which EEG channels were most important?
    This plot shows the learned model coefficients, indicating feature importance.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    coefficients = pd.DataFrame({
        'Channel': features,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    # Positive coefficients predict 'right', negative coefficients predict 'left'
    colors = [MIDNIGHT_BLUE if c > 0 else VIBRANT_RED for c in coefficients['Coefficient']]
    
    bars = sns.barplot(data=coefficients, x='Coefficient', y='Channel', palette=colors, ax=ax)
    
    # Add text labels to the bars
    for bar in bars.patches:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, 
                f'{bar.get_width():.2f}', 
                va='center', ha='left' if bar.get_width() > 0 else 'right',
                fontsize=12, color='black')

    ax.set_title("Feature Importance: Which Channels Did the Model Use?", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Coefficient (Weight)", fontsize=15, fontweight='bold')
    ax.set_ylabel("EEG Channel", fontsize=15, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1.5)
    ax.tick_params(axis='x', labelsize=12)
    
    # Add explanatory text
    fig.text(0.5, 0.01, "Positive bars push prediction towards 'Right', Negative bars push towards 'Left'", 
             ha='center', fontsize=12, style='italic', color='gray')

    fig.tight_layout(rect=[0, 0.05, 1, 1])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <path_to_data_file.csv>")
    else:
        file_to_analyze = sys.argv[1]
        analyze_bci_data(file_to_analyze)