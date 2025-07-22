import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os

# === Load EEG Data ===
file_paths = [
    '/home/aadrika/Documents/Project_attention/dataa/file01.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file02.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file03.csv',
    '/home/aadrika/Documents/Project_attention/dataa/file04.csv'
]

dfs = []
for path in file_paths:
    df = pd.read_csv(path)
    df['participant'] = os.path.basename(path).split('.')[0]
    df['trial_index'] = df.index
    df['alpha_asymmetry'] = (df['alpha_o1'] - df['alpha_o2']) / (df['alpha_o1'] + df['alpha_o2'] + 1e-6)
    df['abs_asymmetry'] = df['alpha_asymmetry'].abs()
    dfs.append(df)

# Combine all participants
data = pd.concat(dfs, ignore_index=True)

# Features and label
features = ['alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4']
X = data[features].values
y = data['is_correct'].astype(int).values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 7-Fold Cross-Validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("🎯 7-Fold Accuracy Report:\n-------------------------")

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train = y[train_idx]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = (y_pred == y[test_idx]).mean()
    print(f"Fold {fold_idx + 1}: Accuracy = {acc:.2%}")

    for i, idx in enumerate(test_idx):
        confidence = float(y_prob[i])
        results.append({
            'participant': data.loc[idx, 'participant'],
            'trial_index': data.loc[idx, 'trial_index'],
            'true_label': int(y[idx]),
            'predicted_label': int(y_pred[i]),
            'model_confidence': confidence,
            'confidence_level': (
                'high' if confidence > 0.7 else
                'medium' if confidence > 0.5 else
                'low'
            ),
            'is_high_attention': confidence > 0.7,
            'alpha_o1': data.loc[idx, 'alpha_o1'],
            'alpha_o2': data.loc[idx, 'alpha_o2'],
            'alpha_p3': data.loc[idx, 'alpha_p3'],
            'alpha_p4': data.loc[idx, 'alpha_p4'],
            'alpha_asymmetry': data.loc[idx, 'alpha_asymmetry'],
            'abs_asymmetry': data.loc[idx, 'abs_asymmetry'],
            'target_side': data.loc[idx, 'target_side'] if 'target_side' in data.columns else 'unknown',
            'is_correct': data.loc[idx, 'is_correct'],
            'fold': fold_idx + 1
        })

# === Save to CSV ===
output_df = pd.DataFrame(results)
output_df.to_csv("logreg_7fold_results.csv", index=False)
print("\n✅ Saved to: logreg_7fold_results.csv")
