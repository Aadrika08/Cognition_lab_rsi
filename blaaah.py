import json
import os
import glob
import matplotlib.pyplot as plt

# Automatically get all *_RESULTS.json files in the current folder
json_files = glob.glob('*_RESULTS.json')

participants = []
accuracy_all = []
accuracy_focus = []
accuracy_gain = []

for fname in json_files:
    with open(fname, 'r') as f:
        data = json.load(f)
        name = data['source_file'].split('.')[0]
        acc_all = data['accuracy_all_trials']
        acc_focus = data['accuracy_high_attention']
        gain = acc_focus - acc_all

        participants.append(name)
        accuracy_all.append(acc_all)
        accuracy_focus.append(acc_focus)
        accuracy_gain.append(gain)

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(participants, accuracy_gain, color=plt.cm.coolwarm([x / max(accuracy_gain) for x in accuracy_gain]))

for bar, gain in zip(bars, accuracy_gain):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{gain:.2%}', ha='center', fontsize=10)

plt.title('Accuracy Gain from Attention Filtering per Participant')
plt.xlabel('Participant (File)')
plt.ylabel('Accuracy Gain (%)')
plt.ylim(0, max(accuracy_gain) + 0.05)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
