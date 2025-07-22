import pandas as pd
import matplotlib.pyplot as plt

# Data manually entered from summary (update these with real values!)
data = {
    'Participant': ['P01', 'P02', 'P03', 'P04'],
    'Unfiltered_Median': [50.00, 60.00, 43.33, 46.67],
    'Filtered_Median': [44.44, 66.67, 53.85, 61.54],
    'Median_Asymmetry': [0.354, 0.376, 0.411, 0.397]  # Replace with real per-participant medians
}

df = pd.DataFrame(data)

# Calculate change in accuracy
df['Accuracy_Change'] = df['Filtered_Median'] - df['Unfiltered_Median']

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Median_Asymmetry'], df['Accuracy_Change'], color='darkblue', s=100)

# Add participant labels
for i, row in df.iterrows():
    plt.text(row['Median_Asymmetry'] + 0.002, row['Accuracy_Change'], row['Participant'], fontsize=10)

# Plot decorations
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Accuracy Gain vs Median Alpha Asymmetry', fontsize=14, fontweight='bold')
plt.xlabel('|Median Alpha Asymmetry| (Filtered Trials)', fontsize=12)
plt.ylabel('Change in Model Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
