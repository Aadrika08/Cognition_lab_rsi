import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import os

# Define paths to participant files
file_paths = [
    '/home/aadrika/Documents/Project_attention/dataa/file01.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file02.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file03.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file04.csv'
]

# Define training sizes (percentages)
train_sizes = np.linspace(0.1, 1.0, 10)

# Store learning curves for each participant
participant_results = {}

# Process each participant file
for file_path in file_paths:
    df = pd.read_csv(file_path)
    pid = os.path.splitext(os.path.basename(file_path))[0]  # e.g. 'file01'

    # Extract features and label
    X = df[['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']].values
    y = df['is_correct'].values  # binary label: 1 = correct, 0 = incorrect

    # Shuffle all data
    X, y = shuffle(X, y, random_state=42)

    # High-attention filter: top 75% by abs(confidence)
    df['abs_conf'] = df['confidence'].abs()
    threshold = df['abs_conf'].quantile(0.25)  # Keep top 75%
    df_high = df[df['abs_conf'] >= threshold]

    X_high = df_high[['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']].values
    y_high = df_high['is_correct'].values

    # Shuffle high-attention data
    X_high, y_high = shuffle(X_high, y_high, random_state=42)

    acc_all = []
    acc_high = []

    for size in train_sizes:
        n_all = int(len(X) * size)
        n_high = int(len(X_high) * size)

        # Train model on all trials
        if n_all >= 10:
            model = LogisticRegression(solver='liblinear')
            score = cross_val_score(model, X[:n_all], y[:n_all], cv=5).mean()
            acc_all.append(score)
        else:
            acc_all.append(np.nan)

        # Train model on high-attention trials
        if n_high >= 10:
            model = LogisticRegression(solver='liblinear')
            score = cross_val_score(model, X_high[:n_high], y_high[:n_high], cv=5).mean()
            acc_high.append(score)
        else:
            acc_high.append(np.nan)

    participant_results[pid] = {
        'train_sizes': train_sizes * 100,
        'all_acc': acc_all,
        'high_acc': acc_high
    }

# --------- Plotting Clean Subplots ---------
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.flatten()

for i, (pid, result) in enumerate(participant_results.items()):
    ax = axs[i]
    ax.plot(result['train_sizes'], result['all_acc'], '--o', color='tab:blue', label='All Trials')
    ax.plot(result['train_sizes'], result['high_acc'], '-o', color='tab:orange', label='High Attention')
    ax.set_title(f'Participant: {pid}')
    ax.set_xlabel('Training Set Size (%)')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend(loc='lower right')

# Shared title and layout
plt.suptitle("Learning Curves: All vs. High-Attention Trials", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Optional: Save figure to file
# plt.savefig("learning_curves_clean.png", dpi=300)

plt.show()
