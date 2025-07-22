import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def evaluate_logistic_model(file_path):
    df = pd.read_csv(file_path)

    # Strip column names in case of weird spaces
    df.columns = df.columns.str.strip()

    # Clean up
    df = df.dropna(subset=["alpha_o1", "alpha_o2", "alpha_p3", "alpha_p4", "alpha_asymmetry", "target_side"])
    df = df.reset_index(drop=True)

    # Encode target: left = 0, right = 1
    df["target_encoded"] = LabelEncoder().fit_transform(df["target_side"])

    # Feature columns
    features = ["alpha_o1", "alpha_o2", "alpha_p3", "alpha_p4", "alpha_asymmetry"]
    target = "target_encoded"

    accuracies = []
    max_start = len(df) - 150

    if max_start < 0:
        print("Not enough trials. Need at least 150 rows.")
        return None

    for start_idx in range(max_start + 1):
        train_df = df.iloc[start_idx:start_idx + 130]
        test_df = df.iloc[start_idx + 130:start_idx + 150]

        model = LogisticRegression()
        model.fit(train_df[features], train_df[target])
        
        preds = model.predict(test_df[features])
        acc = accuracy_score(test_df[target], preds)
        accuracies.append(acc)

    median_accuracy = np.median(accuracies)
    return round(median_accuracy * 100, 2)

if __name__ == "__main__":
    file = "/home/aadrika/Documents/Project_attention/data/bci_attention_data_test06_20250718_203523_corrected.csv"
    result = evaluate_logistic_model(file)
    if result is not None:
        print(f"{file} => Median Accuracy: {result}%")
