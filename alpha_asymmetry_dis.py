
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Seaborn style for scientific plots
sns.set(style="whitegrid", context="paper", font_scale=1.2)

# Paths to your participant CSV files
file_paths = [
    '/home/aadrika/Documents/Project_attention/dataa/file01.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file02.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file03.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file04.csv'
]

# Function to compute alpha asymmetry

# Combine all data
all_trials = []
high_attention_trials = []

for path in file_paths:
    df = pd.read_csv(path)
    df['alpha_asymmetry'] = df['asymmetry']
    all_trials.append(df[['target_side', 'alpha_asymmetry']])

    high_df = df[df['alpha_asymmetry'].abs() > 0.3]
    high_attention_trials.append(high_df[['target_side', 'alpha_asymmetry']])

# Concatenate all participants
df_all = pd.concat(all_trials, ignore_index=True)
df_high = pd.concat(high_attention_trials, ignore_index=True)

# Initialize figure with two plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# ---- All Trials Plot ----
sns.kdeplot(data=df_all[df_all['target_side'] == 'left'],
            x='alpha_asymmetry', label='Left Cue',
            fill=True, color='#3E6FED', alpha=0.6, ax=axes[0])  # Blue
sns.kdeplot(data=df_all[df_all['target_side'] == 'right'],
            x='alpha_asymmetry', label='Right Cue',
            fill=True, color='#D43F3A', alpha=0.6, ax=axes[0])  # Red
axes[0].set_title("All Trials", fontsize=13)
axes[0].set_xlabel("Alpha Asymmetry Index")
axes[0].set_ylabel("Density")
axes[0].legend(title="Target Side", loc="upper right")
axes[0].axvline(0, color='gray', linestyle='--', linewidth=1)

# ---- High-Attention Trials Plot ----
sns.kdeplot(data=df_high[df_high['target_side'] == 'left'],
            x='alpha_asymmetry', label='Left Cue',
            fill=True, color='#3E6FED', alpha=0.6, ax=axes[1])  # Blue
sns.kdeplot(data=df_high[df_high['target_side'] == 'right'],
            x='alpha_asymmetry', label='Right Cue',
            fill=True, color='#D43F3A', alpha=0.6, ax=axes[1])  # Red
axes[1].set_title("High-Attention Trials", fontsize=13)
axes[1].set_xlabel("Alpha Asymmetry Index")
axes[1].set_ylabel("")
axes[1].legend(title="Target Side", loc="upper right")
axes[1].axvline(0, color='gray', linestyle='--', linewidth=1)

# Global layout adjustments
plt.suptitle("Distribution of Alpha Asymmetry by Attention Direction", fontsize=15, y=1.03)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()