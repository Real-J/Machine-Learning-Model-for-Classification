import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import joblib  # For saving the best model

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load Dataset
dataset_path = "diabetesData.csv"  # Change this to the actual file path
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

data = pd.read_csv(dataset_path)

# Display dataset info
print("Dataset Preview:\n", data.head())
print("\nDataset Info:\n")
print(data.info())

# Handle Missing Values (if any)
data.fillna(data.median(), inplace=True)  # Replace NaN with median values

# Define features and target
target_column = 'Outcome'  # Change this if your target column has a different name
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset")

X = data.drop(columns=[target_column])
y = data[target_column]

# Handle Imbalanced Data using SMOTE
print("\nOriginal Class Distribution:", np.bincount(y))
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print("Resampled Class Distribution:", np.bincount(y))

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Evaluate Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel="poly", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15)
}

results = {}
trained_models = {}

for model_name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        trained_models[model_name] = model
        print(f"Accuracy with {model_name}: {accuracy:.4f}")
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Hyperparameter Tuning for Random Forest
print("\nTuning Random Forest Hyperparameters...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
tuned_accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\nBest parameters for Random Forest:", grid_search.best_params_)
print("Tuned Random Forest Accuracy:", tuned_accuracy_rf)

# Feature Importance Analysis (Random Forest)
importances = best_rf.feature_importances_
feature_names = data.drop(columns=[target_column]).columns

plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.show()

# Save the Best Model
joblib.dump(best_rf, "best_diabetes_model.pkl")
print("\nBest model saved as 'best_diabetes_model.pkl'")

# Load Model (for Future Predictions)
loaded_model = joblib.load("best_diabetes_model.pkl")
y_pred_loaded = loaded_model.predict(X_test)
print("\nLoaded Model Accuracy:", accuracy_score(y_test, y_pred_loaded))

# Plot Confusion Matrix for the Best Model
conf_matrix = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["No Diabetes", "Diabetes"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix: Random Forest (Tuned)")
plt.show()
