import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("final_data_with_prognosis.csv")
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Symptom list provided
symptom_pool = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", 
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue",
    "muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue",
    "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss"
]

# Define symptom groups
symptom_groups = [
    ["itching", "skin_rash", "nodal_skin_eruptions"],
    ["shivering", "chills", "joint_pain"],
    ["vomiting", "fatigue", "acidity"],
    ["anxiety", "cold_hands_and_feets", "mood_swings"]
]

# Range of tree sizes
tree_range = list(range(10,800 , 50))
group_confidences = []

# Main loop
for symptoms in symptom_groups:
    confidences = []
    input_vector = [1 if col in symptoms else 0 for col in X.columns]

    for n in tree_range:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        probas = model.predict_proba([input_vector])[0]
        confidence = np.max(probas)
        confidences.append(confidence)

    group_confidences.append(confidences)

# Convert to numpy array for mean computation
group_confidences = np.array(group_confidences)
avg_confidence_per_tree = np.mean(group_confidences, axis=0)

# Slope calculation
slopes = np.gradient(avg_confidence_per_tree)
abs_slopes = np.abs(slopes)
min_slope_idx = np.argmin(abs_slopes)
stable_tree_count = tree_range[min_slope_idx]

# Plotting
plt.figure(figsize=(12, 6))
for i, group in enumerate(symptom_groups):
    plt.plot(tree_range, group_confidences[i], label=f"Group {i+1}: {', '.join(group)}", alpha=0.6, linestyle='--')

plt.plot(tree_range, avg_confidence_per_tree, color='black', marker='o', linewidth=3, label="Average Confidence")
plt.axvline(x=stable_tree_count, color='green', linestyle=':', label=f"Stable Point @ {stable_tree_count} Trees")
plt.xlabel("Number of Trees")
plt.ylabel("Confidence Score")
plt.title("Confidence Variation vs Tree Count Across Symptom Groups")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Report result
print(f"\nStable confidence (minimal slope) = {avg_confidence_per_tree[min_slope_idx]:.4f} at {stable_tree_count} trees.")
