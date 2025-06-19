import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib # For saving/loading models
import matplotlib.pyplot as plt
import seaborn as sns

# Comments: This script trains an XGBoost classifier on the preprocessed data
# to predict equipment failure. It includes steps for data scaling, model training,
# hyperparameter tuning (using GridSearchCV), and comprehensive evaluation.

# Load the processed data
try:
    train_df = pd.read_csv('processed_train_data.csv')
    test_df = pd.read_csv('processed_test_data.csv')
    print("Processed training and testing data loaded successfully.")
except FileNotFoundError:
    print("Error: Processed data not found. Please run Step 2 (Data Preprocessing) first.")
    exit()

# Define features and target (ensure consistency with preprocessing step)
features = [col for col in train_df.columns if col not in ['equipment_id', 'timestamp', 'is_failure', 'failure_type', 'time_to_failure_minutes', 'is_failing_soon', 'days_since_last_failure']]
target = 'is_failing_soon'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# --- Data Scaling ---
# It's good practice to scale features for tree-based models, although less critical than for
# distance-based models. It can sometimes help with convergence and feature importance interpretation.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for better readability with column names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)

print("\nFeatures scaled.")

# --- XGBoost Model Training ---

# Initialize XGBoost Classifier
# Use `use_label_encoder=False` and `eval_metric='logloss'` to avoid deprecation warnings
# For imbalanced datasets (which failure prediction often is), `scale_pos_weight` can be helpful.
# It's calculated as (number of negative samples) / (number of positive samples).
scale_pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale Pos Weight for imbalanced data: {scale_pos_weight_value:.2f}")

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic', # For binary classification
    eval_metric='logloss',       # Evaluation metric
    use_label_encoder=False,     # To suppress deprecation warning
    n_estimators=100,            # Number of boosting rounds (trees)
    learning_rate=0.1,           # Step size shrinkage
    max_depth=5,                 # Maximum depth of a tree
    subsample=0.8,               # Subsample ratio of the training instance
    colsample_bytree=0.8,        # Subsample ratio of columns when constructing each tree
    random_state=42,             # For reproducibility
    scale_pos_weight=scale_pos_weight_value # Handle class imbalance
)

print("\nTraining initial XGBoost model...")
xgb_clf.fit(X_train_scaled_df, y_train)
print("Initial model training complete.")

# --- Model Evaluation (Initial) ---
y_pred_initial = xgb_clf.predict(X_test_scaled_df)
y_prob_initial = xgb_clf.predict_proba(X_test_scaled_df)[:, 1] # Probability of positive class

print("\n--- Initial Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_initial):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_initial):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_initial):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_initial):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_initial):.4f}")

print("\nClassification Report (Initial Model):\n", classification_report(y_test, y_pred_initial))
print("\nConfusion Matrix (Initial Model):\n", confusion_matrix(y_test, y_pred_initial))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred_initial)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Failing Soon', 'Failing Soon'],
            yticklabels=['Not Failing Soon', 'Failing Soon'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Initial XGBoost Model)')
plt.show()


# --- Hyperparameter Optimization using GridSearchCV ---
# This step can be time-consuming. For Azure ML, you'd use HyperDrive for this.
# We'll define a smaller parameter grid for a local demonstration.

print("\nStarting Hyperparameter Optimization using GridSearchCV (this may take a while)...")
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9]
}

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='f1', # F1-score is often good for imbalanced classification
    cv=3,         # 3-fold cross-validation
    n_jobs=-1,    # Use all available CPU cores
    verbose=1
)

grid_search.fit(X_train_scaled_df, y_train)

print("\nBest hyperparameters found by GridSearchCV:")
print(grid_search.best_params_)

best_xgb_clf = grid_search.best_estimator_
print("\nBest model trained.")

# --- Model Evaluation (Optimized) ---
y_pred_optimized = best_xgb_clf.predict(X_test_scaled_df)
y_prob_optimized = best_xgb_clf.predict_proba(X_test_scaled_df)[:, 1]

print("\n--- Optimized Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_optimized):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_optimized):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_optimized):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_optimized):.4f}")

print("\nClassification Report (Optimized Model):\n", classification_report(y_test, y_pred_optimized))
print("\nConfusion Matrix (Optimized Model):\n", confusion_matrix(y_test, y_pred_optimized))

# Plot Confusion Matrix for optimized model
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Failing Soon', 'Failing Soon'],
            yticklabels=['Not Failing Soon', 'Failing Soon'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Optimized XGBoost Model)')
plt.show()

# --- Save the trained model and scaler ---
model_filename = 'xgboost_predictive_maintenance_model.joblib'
scaler_filename = 'scaler_for_predictive_maintenance.joblib'

joblib.dump(best_xgb_clf, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"\nOptimized XGBoost model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")

# --- Feature Importance Visualization ---
# Get feature importances from the best model
feature_importances = best_xgb_clf.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15)) # Top 15 features
plt.title('Top 15 Feature Importances from XGBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

