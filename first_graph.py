import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use a clean, publication-ready style (from your original script)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.6)

# Data (Median accuracies) (from your original script)
# Replace these with your actual values if needed
model1_medians = [50.00, 60.00, 43.33, 46.67]
model2_medians = [44.44, 66.67, 53.85, 61.54]

# Compute average of medians (from your original script)
avg_model1 = np.mean(model1_medians)
avg_model2 = np.mean(model2_medians)

# Plotting (from your original script)
models = ['Unfiltered (Model 1)', 'Filtered (Model 2)']
avg_accuracies = [avg_model1, avg_model2]

plt.figure(figsize=(8, 6))

# --- THIS IS THE ONLY LINE I HAVE CHANGED ---
bars = plt.bar(models, avg_accuracies, color=['#4C72B0', '#C44E52'])
# -------------------------------------------

plt.title('Comparison of Average Median Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

# Annotate the bars (from your original script)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/5, yval + 1, f'{yval:.2f}%', ha='center', fontsize=11)

plt.tight_layout()
plt.show()