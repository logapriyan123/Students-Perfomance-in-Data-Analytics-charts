import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("StudentsPerformance.csv")

# Display first 5 rows
print(dataset.head())

# Convert categorical columns into numerical values
label_encoder = LabelEncoder()

for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column] = label_encoder.fit_transform(dataset[column])

# Input and output variables
X = dataset.drop('math score', axis=1)
y = dataset['math score']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Calculate accuracy (R² score)
accuracy = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nAccuracy (R² Score):", accuracy)
print("Mean Absolute Error:", mae)

# -------- Chart 1: Actual vs Predicted --------
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual vs Predicted")
plt.show()

# -------- Chart 2: Feature Importance --------
importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,5))
plt.barh(feature_names, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# -------- Chart 3: Distribution of Math Scores --------
plt.figure(figsize=(6,5))
plt.hist(dataset['math score'], bins=10)
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.title("Math Score Distribution")
plt.show()

# -------- Chart 4: Reading Score vs Math Score --------
plt.figure(figsize=(6,5))
plt.scatter(dataset['reading score'], dataset['math score'])
plt.xlabel("Reading Score")
plt.ylabel("Math Score")
plt.title("Reading Score vs Math Score")
plt.show()

# -------- Chart 5: Writing Score vs Math Score --------
plt.figure(figsize=(6,5))
plt.scatter(dataset['writing score'], dataset['math score'])
plt.xlabel("Writing Score")
plt.ylabel("Math Score")
plt.title("Writing Score vs Math Score")
plt.show()
