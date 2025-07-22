import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_theme(style="darkgrid", palette="flare")  # closest to heatwave

# Paths
data_dir = "./data"
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

# Constants
CONFIDENCE_QUANTILE = 0.75
feature_columns = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']

# Storage
summary = []

def train_and_evaluate_cv(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5)
    return scores.mean(), scores

# Go through all CSVs
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)

        # --- All Trials ---
        X_all = df[feature_columns]
        y_all = df['target_side']
        acc_all, folds_all = train_and_evaluate_cv(X_all, y_all)

        # --- High Attention Trials ---
        df['asymmetry_magnitude'] = df['alpha_asymmetry'].abs()
        threshold = df['asymmetry_magnitude'].quantile(CONFIDENCE_QUANTILE)
        df_focus = df[df['asymmetry_magnitude'] >= threshold]
        X_focus = df_focus[feature_columns]
        y_focus = df_focus['target_side']

        if len(df_focus) >= 5:  # minimum for 5-fold
            acc_focus, folds_focus = train_and_evaluate_cv(X_focus, y_focus)
        else:
            acc_focus, folds_focus = np.nan, []

        summary.append({
            "File": file,
            "All Trial Acc": round(acc_all * 100, 2),
            "High Attn Acc": round(acc_focus * 100, 2) if not np.isnan(acc_focus) else None,
            "Delta": round((acc_focus - acc_all) * 100, 2) if not np.isnan(acc_focus) else None,
            "High Attn Count": len(df_focus)
        })

        # --- Plot Regression of Attention Over Time ---
        plt.figure(figsize=(8, 5))
        sns.regplot(x=np.arange(len(df)), y=df['asymmetry_magnitude'], scatter_kws={"s": 30}, line_kws={"color": "red"})
        plt.title(f"{file} | Attention Drift Over Time")
        plt.xlabel("Trial Index")
        plt.ylabel("|Alpha Asymmetry|")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file}_regplot.png"))
        plt.close()

        # --- Histogram ---
        plt.figure(figsize=(7, 4))
        sns.histplot(df['asymmetry_magnitude'], bins=20, color='orange', kde=True)
        plt.title(f"{file} | Histogram of |Alpha Asymmetry|")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file}_histogram.png"))
        plt.close()

# --- Save Summary Table ---
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)

# --- Combined Bar Plot ---
plt.figure(figsize=(10, 6))
summary_df_melt = summary_df.melt(id_vars="File", value_vars=["All Trial Acc", "High Attn Acc"], var_name="Type", value_name="Accuracy")
sns.barplot(data=summary_df_melt, x="File", y="Accuracy", hue="Type")
plt.title("Accuracy Comparison: All Trials vs High-Attention")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_accuracy_comparison.png"))
plt.close()

# --- Line Plot of Accuracy Delta ---
plt.figure(figsize=(8, 4))
sns.lineplot(data=summary_df, x="File", y="Delta", marker="o", color="crimson")
plt.axhline(0, ls="--", color="gray")
plt.title("Improvement in Accuracy (High Attention - All Trials)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "delta_accuracy_lineplot.png"))
plt.close()
