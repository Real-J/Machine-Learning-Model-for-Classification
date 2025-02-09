# Machine Learning Model for Classification

This repository contains a **machine learning pipeline** that processes a dataset, trains multiple models, tunes hyperparameters, and evaluates performance using accuracy and confusion matrices.

The pipeline supports **handling imbalanced data using SMOTE** and includes **Random Forest hyperparameter tuning** using `GridSearchCV`.

## 🚀 Features
- **Preprocessing**: Encodes categorical data, normalizes numerical features, and handles missing values.
- **Handles Imbalanced Data**: Uses **SMOTE (Synthetic Minority Over-sampling Technique)**.
- **Trains multiple models**:
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - k-Nearest Neighbors (KNN)
  - (Optional) LightGBM (if installed)
- **Hyperparameter Tuning**: Optimizes **Random Forest** with `GridSearchCV`.
- **Feature Importance Analysis**: Displays feature contributions for model decisions.
- **Model Saving and Loading**: Saves the best-trained model for future predictions.

## 📂 Dataset Information
The dataset should be a **CSV file** with relevant attributes for classification tasks. The dataset used in this project focuses on medical diagnostics, predicting **diabetes based on health indicators**.

### **Dataset Columns**
- `Pregnancies`: Number of times the person has been pregnant
- `Glucose`: Plasma glucose concentration after a 2-hour oral glucose tolerance test
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Genetic predisposition score
- `Age`: Age in years
- `Outcome`: Target variable (0 = No Diabetes, 1 = Diabetes)

The dataset is cleaned by handling missing values using median imputation.

## 🏆 Machine Learning Models
The pipeline implements various models to classify individuals based on their health attributes.

### **Model Overview**
1. **Decision Tree Classifier**
   - A simple tree-based model that learns decision rules.
   - Good for interpretability but prone to overfitting.
2. **Random Forest Classifier**
   - An ensemble of decision trees, reducing variance and improving generalization.
   - Tuned using `GridSearchCV` to find the best hyperparameters.
3. **Support Vector Machine (SVM)**
   - Uses a polynomial kernel to separate classes.
   - Works well for small and medium-sized datasets.
4. **k-Nearest Neighbors (KNN)**
   - Classifies based on the majority vote of k nearest neighbors.
   - Performs best with well-distributed data.
5. **LightGBM (Optional)**
   - Gradient boosting framework for high-performance classification.
   - Used if installed; skipped otherwise.

## 🛠 Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/Real-J/ml-classification.git
   cd ml-classification
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   If using Anaconda, install missing dependencies:
   ```sh
   conda install -c conda-forge imbalanced-learn lightgbm
   ```

## 🚀 Usage
1. **Place the dataset** (`your_dataset.csv`) in the project folder.
2. **Run the script**:
   ```sh
   python ml_pipeline.py
   ```
3. The script will:
   - Preprocess the dataset
   - Handle imbalanced data using SMOTE
   - Train multiple models
   - Tune Random Forest hyperparameters
   - Display feature importance
   - Plot the confusion matrix
   - Save the best model as `best_model.pkl`

## 📊 Model Performance Example
```
Original class distribution: [500 268]
Resampled class distribution: [500 500]
Accuracy with Decision Tree: 0.7450
Accuracy with Random Forest: 0.7900
Accuracy with SVM: 0.7100
Accuracy with KNN: 0.7050
Accuracy with LightGBM: 0.7800
Tuned Random Forest Accuracy: 0.81
```

## 📈 Feature Importance
The script generates a **feature importance plot**, helping to identify the most influential features in classification.

## 🔍 Troubleshooting
- If you get **ModuleNotFoundError**, install missing libraries:
  ```sh
  pip install numpy pandas scikit-learn seaborn matplotlib imbalanced-learn lightgbm
  ```
- If LightGBM is not installed, it will **skip training** that model without affecting the rest.

## 🤖 Future Improvements
- Support for deep learning models (e.g., TensorFlow/Keras, PyTorch)
- AutoML integration for hyperparameter tuning
- Web API for real-time model inference

## 📜 License
This project is licensed under the **MIT License**.


