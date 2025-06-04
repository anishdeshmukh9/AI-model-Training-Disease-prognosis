import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("final_data_with_prognosis.csv")

# Separate features and target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with 300 trees
clf = RandomForestClassifier(n_estimators=360, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
accuracy = clf.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model saved as model.pkl")
