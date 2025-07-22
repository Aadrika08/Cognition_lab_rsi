import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

# Set style
sns.set(style="whitegrid", palette="muted")

# Get all CSVs
csv_files = sorted(glob.glob("*file01_ANALYSIS.csv"))  # make sure these are named logically like participant1.csv, etc.

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, file in enumerate(csv_files):
    data = pd.read_csv(file)
    
    # Assuming your CSV has these two columns:
    y_true = data["is_correct"]
    y_pred = data["model_prediction"]
    
    # Jitter x-axis a bit to avoid overlap
    jitter = (pd.Series(y_true) + 0.05 * (np.random.rand(len(y_true)) - 0.5))

    # Scatter plot
    sns.scatterplot(
        x=jitter,
        y=y_pred,
        hue=y_true,
        palette={0: "blue", 1: "red"},
        alpha=0.6,
        ax=axs[i],
        legend=False
    )
    
    axs[i].set_title(f"Participant {i+1}")
    axs[i].set_xlabel("True Label (jittered)")
    axs[i].set_ylabel("Predicted Probability")
    axs[i].set_ylim(-0.05, 1.05)
    axs[i].set_xlim(-0.2, 1.2)
    axs[i].axhline(0.5, color="gray", linestyle="--", linewidth=1)

plt.suptitle("Model Prediction vs True Labels – All Participants", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
