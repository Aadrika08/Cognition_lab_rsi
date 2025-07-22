import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
df = pd.read_csv(sys.argv[1])

# Drop rows with missing values (just in case)
df = df.dropna(subset=['asymmetry', 'is_correct'])

X = df[['asymmetry']].values  # Input: asymmetry
y = df['is_correct'].values   # Output: correct or not

accuracies = []

for i in range(10):
    # Random 130 train, 20 test
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    train_idx = indices[:130]
    test_idx = indices[130:150]  # assuming you have enough rows

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Run {i+1} ✅ Accuracy: {acc * 100:.2f}%")

# Median accuracy
median_accuracy = np.median(accuracies)
print("\n==============================")
print(f"📊 Median Accuracy over 10 runs: {median_accuracy * 100:.2f}%")
print("==============================")