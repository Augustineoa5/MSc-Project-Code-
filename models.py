
**Block 1 - Load and Preprocess Dataset**
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

try:
    # Load the dataset and combine Date and Time into Datetime
    data = pd.read_csv('household_consumption.csv')
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
    data.set_index('Datetime', inplace=True)
    data.drop(['Date', 'Time'], axis=1, inplace=True)

    # Prepare features (using Global_active_power as consumption, adding synthetic day_of_week and temperature)
    data['consumption'] = pd.to_numeric(data['Global_active_power'], errors='coerce')  # Convert to numeric, coerce errors to NaN
    data['day_of_week'] = data.index.dayofweek  # 0 = Monday, 6 = Sunday
    data['temperature'] = np.random.normal(20, 5, data.shape[0]).clip(10, 30)  # Synthetic temperature (N(20, 5), 10-30°C)

    features = ['consumption', 'day_of_week', 'temperature', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    X = data[features]

    # Define anomaly flag based on consumption deviation
    data['consumption'] = data['consumption'].ffill()  # Forward fill NaN values
    y = (np.abs(data['consumption'] - data['consumption'].rolling(window=24, min_periods=1).mean()) > 2 * data['consumption'].rolling(window=24, min_periods=1).std()) | data['consumption'].isna()

    # Split into train and test sets (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print("Data loaded and split successfully.")
except FileNotFoundError:
    print("Error: 'household_consumption.csv' not found. Please check the file path.")
except KeyError as e:
    print(f"Error: Missing column {e}. Please verify the dataset columns.")
except ValueError as e:
    print(f"Error: Date parsing or calculation issue - {e}. Check datetime format or data consistency.")
except Exception as e:
    print(f"Unexpected error: {e}. Please review the code or dataset.")

"""**Train and Evaluate Decision Tree Model with Pruning**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset and inspect the index
data = pd.read_csv('household_consumption.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
data.set_index('Datetime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
print("First few index values:", data.index[:5])

# Clean non-numeric values in ALL numeric columns - replace '?' with NaN
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

print("Cleaning numeric columns...")
for col in numeric_columns:
    if col in data.columns:
        question_marks = (data[col] == '?').sum()
        if question_marks > 0:
            print(f"Cleaning {col}: Found {question_marks} '?' values")
        data[col] = pd.to_numeric(data[col].replace('?', np.nan), errors='coerce')

# Prepare features and target
data['consumption'] = data['Global_active_power']
data['day_of_week'] = data.index.dayofweek
data['hour'] = data.index.hour
data['temperature'] = np.random.normal(20, 5, data.shape[0]).clip(10, 30)

# Forward fill NaN values for all numeric columns
for col in numeric_columns:
    if col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"Forward filling {nan_count} NaN values in {col}")
        data[col] = data[col].ffill()

# Also forward fill consumption
data['consumption'] = data['consumption'].ffill()

# Define all available features
all_features = ['consumption', 'hour', 'day_of_week', 'temperature', 'Global_reactive_power',
               'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

available_features = [f for f in all_features if f in data.columns]
print("Available features:", available_features)

# Prepare feature matrix and target - Improved anomaly detection logic
X = data[available_features]

# Create a proper anomaly detection target based on statistical outliers
consumption_rolling_mean = data['consumption'].rolling(window=96, min_periods=1).mean()
consumption_rolling_std = data['consumption'].rolling(window=96, min_periods=1).std()

# Define anomalies as points that deviate significantly from rolling statistics
y = (np.abs(data['consumption'] - consumption_rolling_mean) >
     2 * consumption_rolling_std) & (~data['consumption'].isna())

# Remove rows with NaN values
mask = ~(X.isna().any(axis=1) | consumption_rolling_std.isna())
X = X[mask]
y = y[mask]

print("Dataset shape after cleaning:", X.shape)
print("Target distribution:\n", y.value_counts())

# Ensure we have enough anomalies for meaningful analysis
if y.sum() < 10:
    print("Warning: Very few anomalies detected. Consider adjusting the threshold.")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature selection analysis - try different feature subsets
feature_subsets = {
    'Core Features': ['consumption', 'hour'],
    'Time Features': ['consumption', 'hour', 'day_of_week'],
    'Environmental': ['consumption', 'hour', 'day_of_week', 'temperature'],
    'All Available': available_features
}

# Define parameter grid for optimization
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced']
}

# Train models with different feature subsets
models = {}
results = {}

for subset_name, features in feature_subsets.items():
    if all(f in available_features for f in features):
        print(f"\nTraining Decision Tree with {subset_name}: {features}")

        X_train_subset = X_train[features]
        X_test_subset = X_test[features]

        # Scale features for better performance (optional for Decision Trees but good practice)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)

        # Initialize Decision Tree
        dt_base = DecisionTreeClassifier(random_state=42)

        # Perform GridSearchCV
        grid_search = GridSearchCV(
            estimator=dt_base,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        # Fit on training data (using original features as DT handles different scales well)
        grid_search.fit(X_train_subset, y_train)
        dt_model = grid_search.best_estimator_

        # Predict on test set
        y_pred = dt_model.predict(X_test_subset)

        # Get prediction probabilities for ROC curve
        y_pred_proba = dt_model.predict_proba(X_test_subset)[:, 1]

        # Store model and results
        models[subset_name] = {'model': dt_model, 'scaler': scaler}
        results[subset_name] = {
            'features': features,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'X_test': X_test_subset,
            'best_params': grid_search.best_params_
        }

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {results[subset_name]['accuracy']:.4f}")

# Check if we have any results
if not results:
    print("No models were successfully trained. Check your data and feature availability.")
    exit()

# Select best performing model for detailed analysis
best_subset = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\nBest performing model: {best_subset} (Accuracy: {results[best_subset]['accuracy']:.4f})")

best_model_info = models[best_subset]
best_model = best_model_info['model']
best_scaler = best_model_info['scaler']
best_features = results[best_subset]['features']
best_y_pred = results[best_subset]['y_pred']
best_y_pred_proba = results[best_subset]['y_pred_proba']
best_X_test = results[best_subset]['X_test']

# Convert boolean y_test to binary for consistency
y_test_binary = y_test.astype(int)

# Detailed evaluation of best model
print(f"\nDetailed Results for {best_subset}:")
print(f"Accuracy: {results[best_subset]['accuracy']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, best_y_pred, target_names=['Normal', 'Anomaly']))

# Cross-validation score
cv_scores = cross_val_score(best_model, best_X_test, y_test_binary, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 24))

# 1. Confusion Matrix for Best Model
plt.subplot(5, 3, 1)
cm = confusion_matrix(y_test_binary, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Confusion Matrix - {best_subset}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. Normalized Confusion Matrix
plt.subplot(5, 3, 2)
if cm.sum() > 0:
    cm_normalized = confusion_matrix(y_test_binary, best_y_pred, normalize='true')
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# 3. Feature Importance (Built-in)
plt.subplot(5, 3, 3)
feature_importance_df = pd.DataFrame({
    'feature': best_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
plt.xlabel('Feature Importance (Gini/Entropy)')
plt.title(f'Feature Importance - {best_subset}', fontsize=14, fontweight='bold')

# 4. Model Comparison Across Feature Subsets
plt.subplot(5, 3, 4)
subset_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in subset_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(subset_names)))

bars = plt.bar(range(len(subset_names)), accuracies, color=colors)
plt.xticks(range(len(subset_names)), subset_names, rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Model Performance by Feature Subset', fontsize=14, fontweight='bold')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 5. ROC Curve
plt.subplot(5, 3, 5)
try:
    fpr, tpr, thresholds = roc_curve(y_test_binary, best_y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
except Exception as e:
    plt.text(0.5, 0.5, f'ROC curve\ncalculation failed:\n{str(e)}',
             ha='center', va='center', transform=plt.gca().transAxes)
    roc_auc = 0.5

# 6. Prediction Probability Distribution
plt.subplot(5, 3, 6)
if len(best_y_pred_proba) > 0:
    normal_proba = best_y_pred_proba[y_test_binary == 0]
    anomaly_proba = best_y_pred_proba[y_test_binary == 1]

    if len(normal_proba) > 0:
        plt.hist(normal_proba, bins=min(30, len(normal_proba)), alpha=0.7,
                label='Normal (True)', color='blue', density=True)
    if len(anomaly_proba) > 0:
        plt.hist(anomaly_proba, bins=min(30, len(anomaly_proba)), alpha=0.7,
                label='Anomaly (True)', color='red', density=True)

    plt.xlabel('Anomaly Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend()

# 7. Cross-Validation Scores
plt.subplot(5, 3, 7)
plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='lightcoral', alpha=0.8)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--',
            label=f'Mean: {cv_scores.mean():.3f}')
plt.xlabel('CV Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores', fontsize=14, fontweight='bold')
plt.legend()
plt.ylim(0, 1)

# 8. Feature Correlation Matrix
plt.subplot(5, 3, 8)
correlation_matrix = best_X_test.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# 9. Decision Tree Visualization (simplified)
plt.subplot(5, 3, 9)
# Plot a simplified version of the tree (first 3 levels only)
if len(best_features) <= 6:  # Only plot if not too many features
    try:
        plot_tree(best_model, max_depth=3, feature_names=best_features,
                 class_names=['Normal', 'Anomaly'], filled=True, fontsize=8)
        plt.title('Decision Tree Structure (Top 3 levels)', fontsize=14, fontweight='bold')
    except Exception as e:
        plt.text(0.5, 0.5, f'Tree visualization\nnot available:\n{str(e)[:50]}...',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Decision Tree Structure - Error', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Too many features\nfor tree visualization',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Decision Tree Structure - Too Complex', fontsize=14, fontweight='bold')

# 10. Class Distribution Comparison
plt.subplot(5, 3, 10)
true_counts = y_test_binary.value_counts()
pred_counts = pd.Series(best_y_pred).value_counts()

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, [true_counts.get(0, 0), true_counts.get(1, 0)], width,
        label='True', color='lightblue', alpha=0.8)
plt.bar(x + width/2, [pred_counts.get(0, 0), pred_counts.get(1, 0)], width,
        label='Predicted', color='lightcoral', alpha=0.8)

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('True vs Predicted Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(x, ['Normal', 'Anomaly'])
plt.legend()

# 11. Learning Curve
plt.subplot(5, 3, 11)
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
train_accuracies = []
for size in train_sizes:
    n_samples = int(size * len(X_train))
    temp_model = DecisionTreeClassifier(**results[best_subset]['best_params'], random_state=42)
    temp_model.fit(X_train[best_features].iloc[:n_samples], y_train.iloc[:n_samples])
    pred = temp_model.predict(best_X_test)
    train_accuracies.append(accuracy_score(y_test_binary, pred))

plt.plot(train_sizes, train_accuracies, 'o-', color='blue', label='Training Accuracy')
plt.xlabel('Training Set Size (fraction)')
plt.ylabel('Accuracy')
plt.title('Learning Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 12. Tree Depth vs Accuracy Analysis
plt.subplot(5, 3, 12)
depth_range = range(1, 21)
depth_accuracies = []

for depth in depth_range:
    temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    temp_model.fit(best_X_test, y_test_binary)
    pred = temp_model.predict(best_X_test)
    depth_accuracies.append(accuracy_score(y_test_binary, pred))

plt.plot(depth_range, depth_accuracies, 'o-', color='green')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Tree Depth', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add text annotation for best depth instead of problematic axvline
best_depth = results[best_subset]['best_params'].get('max_depth', None)
if best_depth is not None:
    plt.text(0.7, 0.95, f'Best model depth: {best_depth}',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
else:
    plt.text(0.7, 0.95, 'Best model depth: Unlimited',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# 13. Permutation Feature Importance
plt.subplot(5, 3, 13)
try:
    perm_importance = permutation_importance(best_model, best_X_test, y_test_binary,
                                           n_repeats=5, random_state=42)
    perm_feature_importance_df = pd.DataFrame({
        'feature': best_features,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=True)

    plt.barh(range(len(perm_feature_importance_df)), perm_feature_importance_df['importance'],
             xerr=perm_feature_importance_df['std'])
    plt.yticks(range(len(perm_feature_importance_df)), perm_feature_importance_df['feature'])
    plt.xlabel('Permutation Importance')
    plt.title('Permutation Feature Importance', fontsize=14, fontweight='bold')
except Exception as e:
    plt.text(0.5, 0.5, f'Permutation importance\ncalculation failed:\n{str(e)[:50]}...',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Permutation Importance - Error', fontsize=14, fontweight='bold')

# 14. Model Complexity Analysis
plt.subplot(5, 3, 14)
complexity_metrics = {
    'Tree Depth': best_model.get_depth(),
    'Leaf Nodes': best_model.get_n_leaves(),
    'Total Nodes': best_model.tree_.node_count,
    'Features Used': len(best_features)
}

bars = plt.bar(range(len(complexity_metrics)), list(complexity_metrics.values()),
               color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
plt.xticks(range(len(complexity_metrics)), list(complexity_metrics.keys()), rotation=45, ha='right')
plt.ylabel('Count')
plt.title('Model Complexity Metrics', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, value in zip(bars, complexity_metrics.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{value}', ha='center', va='bottom', fontweight='bold')

# 15. Model Parameters Summary - FIXED VERSION
plt.subplot(5, 3, 15)
plt.axis('off')
param_text = f"Best Model: {best_subset}\n\n"
param_text += f"Parameters:\n"
for key, value in results[best_subset]['best_params'].items():
    if key == 'max_depth' and value is None:
        param_text += f"• {key}: Unlimited\n"
    else:
        param_text += f"• {key}: {value}\n"
param_text += f"\nFeatures: {len(best_features)}\n"
param_text += f"Accuracy: {results[best_subset]['accuracy']:.4f}\n"
param_text += f"ROC AUC: {roc_auc:.4f}\n"
param_text += f"Tree Depth: {best_model.get_depth()}\n"
param_text += f"Leaf Nodes: {best_model.get_n_leaves()}"

plt.text(0.1, 0.9, param_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
plt.title('Model Summary', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n" + "="*70)
print("DETAILED DECISION TREE ANALYSIS")
print("="*70)

if cm.size > 0:
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (Correctly predicted Normal): {tn}")
    print(f"False Positives (Incorrectly predicted Anomaly): {fp}")
    print(f"False Negatives (Missed Anomalies): {fn}")
    print(f"True Positives (Correctly predicted Anomaly): {tp}")

    print(f"\nPerformance Metrics:")
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
    f1_score = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0

    print(f"Precision (Anomaly): {precision:.4f}")
    print(f"Recall (Anomaly): {recall:.4f}")
    print(f"Specificity (Normal): {specificity:.4f}")
    print(f"F1-Score (Anomaly): {f1_score:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

print(f"\nBest Model Configuration ({best_subset}):")
print(f"Best parameters: {results[best_subset]['best_params']}")
print(f"Features used: {best_features}")
print(f"Feature count: {len(best_features)}")

print(f"\nModel Complexity:")
print(f"Tree depth: {best_model.get_depth()}")
print(f"Number of leaf nodes: {best_model.get_n_leaves()}")
print(f"Total nodes: {best_model.tree_.node_count}")

print(f"\nBuilt-in Feature Importance Ranking:")
for i, (feature, importance) in enumerate(zip(feature_importance_df['feature'],
                                            feature_importance_df['importance']), 1):
    print(f"{i}. {feature}: {importance:.4f}")

print(f"\nModel Comparison Summary:")
for subset_name, result in results.items():
    print(f"{subset_name}: {result['accuracy']:.4f} (Features: {len(result['features'])})")

print("\nDecision Tree model trained, evaluated, and visualized successfully.")
print("Note: Decision Tree provides interpretable rules and built-in feature importance.")

"""**Optimised Decision Tree**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset and inspect the index
data = pd.read_csv('household_consumption.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
data.set_index('Datetime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
print("First few index values:", data.index[:5])  # Check index format

# Clean non-numeric values in ALL numeric columns - replace '?' with NaN
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

print("Cleaning numeric columns...")
for col in numeric_columns:
    if col in data.columns:
        question_marks = (data[col] == '?').sum()
        if question_marks > 0:
            print(f"Cleaning {col}: Found {question_marks} '?' values")
        data[col] = pd.to_numeric(data[col].replace('?', np.nan), errors='coerce')

# Prepare features and target
data['consumption'] = data['Global_active_power']
data['day_of_week'] = data.index.dayofweek  # 0 = Monday, 6 = Sunday
data['temperature'] = np.random.normal(20, 5, data.shape[0]).clip(10, 30)  # Synthetic temperature (N(20, 5), 10-30°C)

# Forward fill NaN values for all numeric columns
print("Forward filling NaN values...")
for col in numeric_columns:
    if col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"Forward filling {nan_count} NaN values in {col}")
            data[col] = data[col].ffill()

# Also forward fill consumption
data['consumption'] = data['consumption'].ffill()

# Define features and check availability
features = ['consumption', 'day_of_week', 'temperature', 'Global_reactive_power',
           'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# Verify all features exist and are clean
available_features = []
for feature in features:
    if feature in data.columns:
        available_features.append(feature)
    else:
        print(f"Warning: Feature '{feature}' not found in dataset")

print("Available features:", available_features)

X = data[available_features]
y = (np.abs(data['consumption'] - data['consumption'].rolling(window=24, min_periods=1).mean()) >
     2 * data['consumption'].rolling(window=24, min_periods=1).std()) | data['consumption'].isna()

# Final data validation and cleaning
print("Dataset shape before final cleaning:", data.shape)
print("Any remaining NaN values:\n", X.isna().sum())

# Remove any remaining rows with NaN values
initial_shape = X.shape[0]
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]
removed_rows = initial_shape - X.shape[0]
if removed_rows > 0:
    print(f"Removed {removed_rows} rows with NaN values")

print("Dataset shape after cleaning:", X.shape)
print("Target distribution:\n", y.value_counts())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree with pruning (max depth = 5) and class weights
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
dt_model.fit(X_train, y_train)

# Predict on test set
y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 16))

# 1. Confusion Matrix
plt.subplot(3, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. Confusion Matrix (Normalized)
plt.subplot(3, 3, 2)
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 3. Feature Importance
plt.subplot(3, 3, 3)
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()

# 4. Class Distribution
plt.subplot(3, 3, 4)
class_counts = y.value_counts()
plt.pie(class_counts.values, labels=['Normal', 'Anomaly'], autopct='%1.1f%%',
        colors=['lightblue', 'lightcoral'])
plt.title('Class Distribution', fontsize=14, fontweight='bold')

# 5. Prediction Confidence Distribution
plt.subplot(3, 3, 5)
confidence_scores = np.max(y_pred_proba, axis=1)
plt.hist(confidence_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')

# 6. ROC-like curve (Prediction probabilities by class)
plt.subplot(3, 3, 6)
normal_proba = y_pred_proba[y_test == False, 0]  # Probability of normal class for normal samples
anomaly_proba = y_pred_proba[y_test == True, 1]   # Probability of anomaly class for anomaly samples

plt.hist(normal_proba, bins=30, alpha=0.7, label='Normal (True Normal)', color='blue')
plt.hist(anomaly_proba, bins=30, alpha=0.7, label='Anomaly (True Anomaly)', color='red')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.title('Prediction Probability by True Class', fontsize=14, fontweight='bold')
plt.legend()

# 7. Decision Tree Visualization (simplified)
plt.subplot(3, 3, 7)
plot_tree(dt_model, max_depth=3, feature_names=available_features,
          class_names=['Normal', 'Anomaly'], filled=True, fontsize=8)
plt.title('Decision Tree (Depth=3)', fontsize=14, fontweight='bold')

# 8. Training vs Test Performance Comparison
plt.subplot(3, 3, 8)
y_train_pred = dt_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

performance_data = ['Training', 'Testing']
accuracies = [train_accuracy, test_accuracy]
colors = ['lightgreen', 'lightcoral']

bars = plt.bar(performance_data, accuracies, color=colors)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Training vs Testing Accuracy', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 9. Prediction Distribution
plt.subplot(3, 3, 9)
pred_counts = pd.Series(y_pred).value_counts()
plt.bar(['Normal', 'Anomaly'], [pred_counts[False], pred_counts[True]],
        color=['lightblue', 'lightcoral'])
plt.ylabel('Count')
plt.title('Prediction Distribution', fontsize=14, fontweight='bold')

# Add value labels
for i, v in enumerate([pred_counts[False], pred_counts[True]]):
    plt.text(i, v + max(pred_counts) * 0.01, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Print detailed confusion matrix analysis
print("\n" + "="*60)
print("DETAILED CONFUSION MATRIX ANALYSIS")
print("="*60)

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Correctly predicted Normal): {tn}")
print(f"False Positives (Incorrectly predicted Anomaly): {fp}")
print(f"False Negatives (Missed Anomalies): {fn}")
print(f"True Positives (Correctly predicted Anomaly): {tp}")

print(f"\nPrecision (Anomaly): {tp/(tp+fp):.3f}")
print(f"Recall (Anomaly): {tp/(tp+fn):.3f}")
print(f"Specificity (Normal): {tn/(tn+fp):.3f}")
print(f"F1-Score (Anomaly): {2*tp/(2*tp+fp+fn):.3f}")

print("\nDecision Tree model trained and evaluated successfully.")
print("Note: The model shows low recall for anomalies (True class) due to class imbalance.")
print("Consider using oversampling (e.g., SMOTE) or adjusting the anomaly threshold for better performance.")

"""**Train the Isolation Forests Model**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset and prepare data (same as Decision Tree preprocessing)
data = pd.read_csv('household_consumption.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
data.set_index('Datetime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
print("First few index values:", data.index[:5])

# Clean non-numeric values in ALL numeric columns - replace '?' with NaN
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

print("Cleaning numeric columns...")
for col in numeric_columns:
    if col in data.columns:
        question_marks = (data[col] == '?').sum()
        if question_marks > 0:
            print(f"Cleaning {col}: Found {question_marks} '?' values")
        data[col] = pd.to_numeric(data[col].replace('?', np.nan), errors='coerce')

# Prepare features and target
data['consumption'] = data['Global_active_power']
data['day_of_week'] = data.index.dayofweek
data['temperature'] = np.random.normal(20, 5, data.shape[0]).clip(10, 30)

# Forward fill NaN values
for col in numeric_columns:
    if col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"Forward filling {nan_count} NaN values in {col}")
            data[col] = data[col].ffill()

data['consumption'] = data['consumption'].ffill()

# Define features
features = ['consumption', 'day_of_week', 'temperature', 'Global_reactive_power',
           'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

available_features = [f for f in features if f in data.columns]
print("Available features:", available_features)

X = data[available_features]
y = (np.abs(data['consumption'] - data['consumption'].rolling(window=24, min_periods=1).mean()) >
     2 * data['consumption'].rolling(window=24, min_periods=1).std()) | data['consumption'].isna()

# Remove rows with NaN values
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

print("Dataset shape after cleaning:", X.shape)
print("Target distribution:\n", y.value_counts())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features for better Isolation Forest performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate contamination rate based on actual data
contamination_rate = y_train.sum() / len(y_train)
print(f"Calculated contamination rate: {contamination_rate:.4f}")

# Initialize and train the Isolation Forest model
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=max(0.01, min(0.5, contamination_rate)),  # Use data-based contamination
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_scaled)

# Predict on test set (1 for normal, -1 for anomaly)
y_pred_iso = iso_forest.predict(X_test_scaled)
y_pred_binary = [0 if x == -1 else 1 for x in y_pred_iso]  # Convert: -1 (anomaly) to 0, 1 (normal) to 1

# Get anomaly scores
anomaly_scores = iso_forest.score_samples(X_test_scaled)
decision_scores = iso_forest.decision_function(X_test_scaled)

# Convert boolean y_test to binary for consistency
y_test_binary = y_test.astype(int)

# Evaluate model
accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"\nIsolation Forest Results:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary, target_names=['Normal', 'Anomaly']))

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 16))

# 1. Confusion Matrix
plt.subplot(3, 3, 1)
cm = confusion_matrix(y_test_binary, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix - Isolation Forest', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. Normalized Confusion Matrix
plt.subplot(3, 3, 2)
cm_normalized = confusion_matrix(y_test_binary, y_pred_binary, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn_r',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 3. Anomaly Scores Distribution
plt.subplot(3, 3, 3)
normal_scores = anomaly_scores[y_test_binary == 1]
anomaly_scores_true = anomaly_scores[y_test_binary == 0]

plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal (True)', color='blue', density=True)
plt.hist(anomaly_scores_true, bins=50, alpha=0.7, label='Anomaly (True)', color='red', density=True)
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.title('Anomaly Scores Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Decision Boundary')

# 4. ROC Curve
plt.subplot(3, 3, 4)
# Convert anomaly scores to probabilities (higher score = more normal)
y_scores = -anomaly_scores  # Flip scores so higher = more anomalous
fpr, tpr, thresholds = roc_curve(y_test_binary, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")

# 5. Feature Correlation with Anomaly Scores
plt.subplot(3, 3, 5)
correlations = []
for i, feature in enumerate(available_features):
    corr = np.corrcoef(X_test.iloc[:, i], anomaly_scores)[0, 1]
    correlations.append(abs(corr))

feature_corr_df = pd.DataFrame({
    'feature': available_features,
    'correlation': correlations
}).sort_values('correlation', ascending=True)

plt.barh(range(len(feature_corr_df)), feature_corr_df['correlation'])
plt.yticks(range(len(feature_corr_df)), feature_corr_df['feature'])
plt.xlabel('Absolute Correlation with Anomaly Score')
plt.title('Feature-Anomaly Score Correlation', fontsize=14, fontweight='bold')

# 6. Decision Scores vs True Labels
plt.subplot(3, 3, 6)
plt.scatter(range(len(decision_scores)), decision_scores,
           c=['red' if label == 0 else 'blue' for label in y_test_binary],
           alpha=0.6, s=1)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Decision Score')
plt.title('Decision Scores by True Class', fontsize=14, fontweight='bold')
plt.legend(['Decision Boundary', 'Normal', 'Anomaly'])

# 7. Prediction Confidence Analysis
plt.subplot(3, 3, 7)
confidence_scores = np.abs(decision_scores)
plt.hist(confidence_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Prediction Confidence (|Decision Score|)')
plt.ylabel('Frequency')
plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')

# 8. Class Distribution Comparison
plt.subplot(3, 3, 8)
true_counts = y_test_binary.value_counts()
pred_counts = pd.Series(y_pred_binary).value_counts()

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, [true_counts[1], true_counts[0]], width,
        label='True', color='lightblue', alpha=0.8)
plt.bar(x + width/2, [pred_counts[1], pred_counts[0]], width,
        label='Predicted', color='lightcoral', alpha=0.8)

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('True vs Predicted Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(x, ['Normal', 'Anomaly'])
plt.legend()

# 9. Threshold Analysis
plt.subplot(3, 3, 9)
thresholds_range = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 100)
accuracies = []
precisions = []
recalls = []

for threshold in thresholds_range:
    y_pred_thresh = (anomaly_scores < threshold).astype(int)
    acc = accuracy_score(y_test_binary, y_pred_thresh)
    accuracies.append(acc)

    # Calculate precision and recall manually to avoid warnings
    tp = ((y_pred_thresh == 0) & (y_test_binary == 0)).sum()
    fp = ((y_pred_thresh == 0) & (y_test_binary == 1)).sum()
    fn = ((y_pred_thresh == 1) & (y_test_binary == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)

plt.plot(thresholds_range, accuracies, label='Accuracy', color='blue')
plt.plot(thresholds_range, precisions, label='Precision', color='red')
plt.plot(thresholds_range, recalls, label='Recall', color='green')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Default Threshold')
plt.xlabel('Anomaly Score Threshold')
plt.ylabel('Score')
plt.title('Performance vs Threshold', fontsize=14, fontweight='bold')
plt.legend()

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n" + "="*60)
print("DETAILED ISOLATION FOREST ANALYSIS")
print("="*60)

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Correctly predicted Normal): {tn}")
print(f"False Positives (Incorrectly predicted Anomaly): {fp}")
print(f"False Negatives (Missed Anomalies): {fn}")
print(f"True Positives (Correctly predicted Anomaly): {tp}")

print(f"\nModel Parameters:")
print(f"Number of estimators: {iso_forest.n_estimators}")
print(f"Contamination rate used: {iso_forest.contamination}")
print(f"Actual anomaly rate in test set: {y_test_binary.sum()/len(y_test_binary):.4f}")

print(f"\nPerformance Metrics:")
precision = tp/(tp+fp) if (tp+fp) > 0 else 0
recall = tp/(tp+fn) if (tp+fn) > 0 else 0
specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
f1_score = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0

print(f"Precision (Anomaly): {precision:.4f}")
print(f"Recall (Anomaly): {recall:.4f}")
print(f"Specificity (Normal): {specificity:.4f}")
print(f"F1-Score (Anomaly): {f1_score:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print(f"\nAnomaly Score Statistics:")
print(f"Min anomaly score: {anomaly_scores.min():.4f}")
print(f"Max anomaly score: {anomaly_scores.max():.4f}")
print(f"Mean anomaly score: {anomaly_scores.mean():.4f}")
print(f"Std anomaly score: {anomaly_scores.std():.4f}")

print("\nIsolation Forest model trained, evaluated, and visualized successfully.")
print("Note: Isolation Forest is unsupervised, so performance depends heavily on the contamination parameter.")
print("Consider adjusting the contamination rate based on domain knowledge for better results.")

"""**Adjusted Isolation Forest**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset and prepare data
data = pd.read_csv('household_consumption.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
data.set_index('Datetime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
print("First few index values:", data.index[:5])

# Clean non-numeric values in ALL numeric columns - replace '?' with NaN
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

print("Cleaning numeric columns...")
for col in numeric_columns:
    if col in data.columns:
        question_marks = (data[col] == '?').sum()
        if question_marks > 0:
            print(f"Cleaning {col}: Found {question_marks} '?' values")
        data[col] = pd.to_numeric(data[col].replace('?', np.nan), errors='coerce')

# Prepare features and target
data['consumption'] = data['Global_active_power']
data['day_of_week'] = data.index.dayofweek
data['hour'] = data.index.hour
data['temperature'] = np.random.normal(20, 5, data.shape[0]).clip(10, 30)

# Forward fill NaN values
for col in numeric_columns:
    if col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"Forward filling {nan_count} NaN values in {col}")
            data[col] = data[col].ffill()

data['consumption'] = data['consumption'].ffill()

# Define all available features
all_features = ['consumption', 'hour', 'day_of_week', 'temperature', 'Global_reactive_power',
               'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

available_features = [f for f in all_features if f in data.columns]
print("Available features:", available_features)

# Prepare feature matrix and target - Fixed anomaly detection logic
X = data[available_features]

# Create a proper anomaly detection target based on statistical outliers
consumption_rolling_mean = data['consumption'].rolling(window=96, min_periods=1).mean()
consumption_rolling_std = data['consumption'].rolling(window=96, min_periods=1).std()

# Define anomalies as points that deviate significantly from rolling statistics
y = (np.abs(data['consumption'] - consumption_rolling_mean) >
     2 * consumption_rolling_std) & (~data['consumption'].isna())

# Remove rows with NaN values
mask = ~(X.isna().any(axis=1) | consumption_rolling_std.isna())
X = X[mask]
y = y[mask]

print("Dataset shape after cleaning:", X.shape)
print("Target distribution:\n", y.value_counts())

# Ensure we have enough anomalies for meaningful analysis
if y.sum() < 10:
    print("Warning: Very few anomalies detected. Consider adjusting the threshold.")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature selection analysis - try different feature subsets
feature_subsets = {
    'Core Features': ['consumption', 'hour'],
    'Time Features': ['consumption', 'hour', 'day_of_week'],
    'Environmental': ['consumption', 'hour', 'day_of_week', 'temperature'],
    'All Available': available_features
}

# Define parameter grid for optimization - Fixed contamination parameter
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.5, 0.7, 1.0],
    'contamination': ['auto']  # Use 'auto' or calculate based on actual data
}

# Custom scorer for Isolation Forest that handles the -1/1 output properly
def isolation_forest_scorer(estimator, X, y):
    """Custom scorer that converts IF predictions to match binary labels"""
    predictions = estimator.predict(X)
    # Convert: -1 (anomaly) -> 1, 1 (inlier) -> 0 to match y
    predictions_binary = np.where(predictions == -1, 1, 0)
    return accuracy_score(y, predictions_binary)

# Train models with different feature subsets
models = {}
results = {}

for subset_name, features in feature_subsets.items():
    if all(f in available_features for f in features):
        print(f"\nTraining Isolation Forest with {subset_name}: {features}")

        X_train_subset = X_train[features]
        X_test_subset = X_test[features]

        # Scale features for better performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)

        # Calculate contamination based on actual training data
        contamination_rate = max(0.01, min(0.5, y_train.mean()))  # Ensure reasonable range

        # Initialize Isolation Forest
        if_base = IsolationForest(random_state=42, contamination=contamination_rate)

        # Modified parameter grid for this subset
        current_param_grid = {k: v for k, v in param_grid.items() if k != 'contamination'}
        current_param_grid['contamination'] = [contamination_rate]

        # Perform GridSearchCV with custom scorer
        grid_search = GridSearchCV(
            estimator=if_base,
            param_grid=current_param_grid,
            cv=3,  # Reduced CV folds due to potential class imbalance
            scoring=isolation_forest_scorer,
            n_jobs=-1
        )

        # Fit on scaled training data
        grid_search.fit(X_train_scaled, y_train)
        if_model = grid_search.best_estimator_

        # Predict on test set: convert -1/1 to 1/0 for anomaly/normal
        y_pred = if_model.predict(X_test_scaled)
        y_pred = np.where(y_pred == -1, 1, 0)

        # Get decision function scores for ROC curve
        y_pred_proba = -if_model.decision_function(X_test_scaled)
        # Normalize scores to [0,1] for ROC curve
        if y_pred_proba.max() != y_pred_proba.min():
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        else:
            y_pred_proba = np.zeros_like(y_pred_proba)

        # Store model and results
        models[subset_name] = {'model': if_model, 'scaler': scaler}
        results[subset_name] = {
            'features': features,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'X_test': X_test_scaled,
            'best_params': grid_search.best_params_
        }

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {results[subset_name]['accuracy']:.4f}")

# Check if we have any results
if not results:
    print("No models were successfully trained. Check your data and feature availability.")
    exit()

# Select best performing model for detailed analysis
best_subset = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\nBest performing model: {best_subset} (Accuracy: {results[best_subset]['accuracy']:.4f})")

best_model_info = models[best_subset]
best_model = best_model_info['model']
best_scaler = best_model_info['scaler']
best_features = results[best_subset]['features']
best_y_pred = results[best_subset]['y_pred']
best_y_pred_proba = results[best_subset]['y_pred_proba']
best_X_test = results[best_subset]['X_test']

# Convert boolean y_test to binary for consistency
y_test_binary = y_test.astype(int)

# Detailed evaluation of best model
print(f"\nDetailed Results for {best_subset}:")
print(f"Accuracy: {results[best_subset]['accuracy']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, best_y_pred, target_names=['Normal', 'Anomaly']))

# Cross-validation score (using custom scorer and proper data)
X_scaled_for_cv = best_scaler.transform(X[best_features])
cv_scores = cross_val_score(
    best_model, X_scaled_for_cv, y.astype(int),
    scoring=isolation_forest_scorer, cv=3
)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 20))

# 1. Confusion Matrix for Best Model
plt.subplot(4, 3, 1)
cm = confusion_matrix(y_test_binary, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Confusion Matrix - {best_subset}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. Normalized Confusion Matrix
plt.subplot(4, 3, 2)
if cm.sum() > 0:
    cm_normalized = confusion_matrix(y_test_binary, best_y_pred, normalize='true')
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# 3. Feature Importance
plt.subplot(4, 3, 3)
try:
    perm_importance = permutation_importance(
        best_model, best_X_test, y_test_binary,
        n_repeats=5, random_state=42, scoring=isolation_forest_scorer
    )
    feature_importance_df = pd.DataFrame({
        'feature': best_features,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=True)

    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance - {best_subset}', fontsize=14, fontweight='bold')
except Exception as e:
    plt.text(0.5, 0.5, f'Feature importance\ncalculation failed:\n{str(e)}',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Feature Importance - Error', fontsize=14, fontweight='bold')

# 4. Model Comparison Across Feature Subsets
plt.subplot(4, 3, 4)
subset_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in subset_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(subset_names)))

bars = plt.bar(range(len(subset_names)), accuracies, color=colors)
plt.xticks(range(len(subset_names)), subset_names, rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Model Performance by Feature Subset', fontsize=14, fontweight='bold')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 5. ROC Curve
plt.subplot(4, 3, 5)
try:
    fpr, tpr, thresholds = roc_curve(y_test_binary, best_y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
except Exception as e:
    plt.text(0.5, 0.5, f'ROC curve\ncalculation failed:\n{str(e)}',
             ha='center', va='center', transform=plt.gca().transAxes)
    roc_auc = 0.5

# 6. Prediction Probability Distribution
plt.subplot(4, 3, 6)
if len(best_y_pred_proba) > 0:
    normal_proba = best_y_pred_proba[y_test_binary == 0]
    anomaly_proba = best_y_pred_proba[y_test_binary == 1]

    if len(normal_proba) > 0:
        plt.hist(normal_proba, bins=min(30, len(normal_proba)), alpha=0.7,
                label='Normal (True)', color='blue', density=True)
    if len(anomaly_proba) > 0:
        plt.hist(anomaly_proba, bins=min(30, len(anomaly_proba)), alpha=0.7,
                label='Anomaly (True)', color='red', density=True)

    plt.xlabel('Anomaly Score (Normalized)')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    plt.legend()

# 7. Cross-Validation Scores
plt.subplot(4, 3, 7)
plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='lightcoral', alpha=0.8)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--',
            label=f'Mean: {cv_scores.mean():.3f}')
plt.xlabel('CV Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores', fontsize=14, fontweight='bold')
plt.legend()
plt.ylim(0, 1)

# 8. Feature Correlation Matrix (using original unscaled data for interpretability)
plt.subplot(4, 3, 8)
X_test_original = X_test[best_features]
correlation_matrix = X_test_original.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# 9. Class Distribution Comparison
plt.subplot(4, 3, 9)
true_counts = y_test_binary.value_counts()
pred_counts = pd.Series(best_y_pred).value_counts()

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, [true_counts.get(0, 0), true_counts.get(1, 0)], width,
        label='True', color='lightblue', alpha=0.8)
plt.bar(x + width/2, [pred_counts.get(0, 0), pred_counts.get(1, 0)], width,
        label='Predicted', color='lightcoral', alpha=0.8)

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('True vs Predicted Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(x, ['Normal', 'Anomaly'])
plt.legend()

# 10. Contamination Rate Analysis
plt.subplot(4, 3, 10)
contamination_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
contamination_accuracies = []

for cont_rate in contamination_rates:
    temp_model = IsolationForest(contamination=cont_rate, random_state=42)
    temp_model.fit(best_X_test)
    pred = temp_model.predict(best_X_test)
    pred = np.where(pred == -1, 1, 0)
    contamination_accuracies.append(accuracy_score(y_test_binary, pred))

plt.plot(contamination_rates, contamination_accuracies, 'o-', color='green')
plt.xlabel('Contamination Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Contamination Rate', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 11. Anomaly Detection Timeline (if datetime index available)
plt.subplot(4, 3, 11)
try:
    # Create a timeline plot showing detected anomalies
    test_indices = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
    anomaly_indices = [i for i, pred in enumerate(best_y_pred) if pred == 1]

    plt.scatter(range(len(best_y_pred)), best_y_pred, c=best_y_pred,
               cmap='RdYlBu', alpha=0.6, s=10)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Prediction (0=Normal, 1=Anomaly)')
    plt.title('Anomaly Detection Timeline', fontsize=14, fontweight='bold')
    plt.colorbar(label='Prediction')
except Exception as e:
    plt.text(0.5, 0.5, 'Timeline plot\nnot available',
             ha='center', va='center', transform=plt.gca().transAxes)

# 12. Model Parameters Summary
plt.subplot(4, 3, 12)
plt.axis('off')
param_text = f"Best Model: {best_subset}\n\n"
param_text += f"Parameters:\n"
for key, value in results[best_subset]['best_params'].items():
    param_text += f"• {key}: {value}\n"
param_text += f"\nFeatures: {len(best_features)}\n"
param_text += f"Accuracy: {results[best_subset]['accuracy']:.4f}\n"
param_text += f"ROC AUC: {roc_auc:.4f}"

plt.text(0.1, 0.9, param_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
plt.title('Model Summary', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n" + "="*70)
print("DETAILED ISOLATION FOREST ANALYSIS")
print("="*70)

if cm.size > 0:
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (Correctly predicted Normal): {tn}")
    print(f"False Positives (Incorrectly predicted Anomaly): {fp}")
    print(f"False Negatives (Missed Anomalies): {fn}")
    print(f"True Positives (Correctly predicted Anomaly): {tp}")

    print(f"\nPerformance Metrics:")
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
    f1_score = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0

    print(f"Precision (Anomaly): {precision:.4f}")
    print(f"Recall (Anomaly): {recall:.4f}")
    print(f"Specificity (Normal): {specificity:.4f}")
    print(f"F1-Score (Anomaly): {f1_score:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

print(f"\nBest Model Configuration ({best_subset}):")
print(f"Best parameters: {results[best_subset]['best_params']}")
print(f"Features used: {best_features}")
print(f"Feature count: {len(best_features)}")

if 'feature_importance_df' in locals():
    print(f"\nFeature Importance Ranking:")
    for i, (feature, importance) in enumerate(zip(feature_importance_df['feature'],
                                                feature_importance_df['importance']), 1):
        print(f"{i}. {feature}: {importance:.4f}")

print(f"\nModel Comparison Summary:")
for subset_name, result in results.items():
    print(f"{subset_name}: {result['accuracy']:.4f} (Features: {len(result['features'])})")

print("\nIsolation Forest model trained, evaluated, and visualized successfully.")
print("Note: The model uses proper contamination parameter and feature scaling for better performance.")

"""**Random Forests with Feature Subsetting**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset and prepare data (same as previous models)
data = pd.read_csv('household_consumption.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
data.set_index('Datetime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
print("First few index values:", data.index[:5])

# Clean non-numeric values in ALL numeric columns - replace '?' with NaN
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

print("Cleaning numeric columns...")
for col in numeric_columns:
    if col in data.columns:
        question_marks = (data[col] == '?').sum()
        if question_marks > 0:
            print(f"Cleaning {col}: Found {question_marks} '?' values")
        data[col] = pd.to_numeric(data[col].replace('?', np.nan), errors='coerce')

# Prepare features and target
data['consumption'] = data['Global_active_power']
data['day_of_week'] = data.index.dayofweek
data['hour'] = data.index.hour  # Extract hour from datetime index
data['temperature'] = np.random.normal(20, 5, data.shape[0]).clip(10, 30)

# Forward fill NaN values
for col in numeric_columns:
    if col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"Forward filling {nan_count} NaN values in {col}")
            data[col] = data[col].ffill()

data['consumption'] = data['consumption'].ffill()

# Define all available features
all_features = ['consumption', 'hour', 'day_of_week', 'temperature', 'Global_reactive_power',
               'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

available_features = [f for f in all_features if f in data.columns]
print("Available features:", available_features)

# Prepare feature matrix and target
X = data[available_features]
y = (np.abs(data['consumption'] - data['consumption'].rolling(window=96, min_periods=1).mean()) >
     2 * data['consumption'].rolling(window=96, min_periods=1).std()) | data['consumption'].isna()

# Remove rows with NaN values
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

print("Dataset shape after cleaning:", X.shape)
print("Target distribution:\n", y.value_counts())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature selection analysis - try different feature subsets
feature_subsets = {
    'Core Features': ['consumption', 'hour'],
    'Time Features': ['consumption', 'hour', 'day_of_week'],
    'Environmental': ['consumption', 'hour', 'day_of_week', 'temperature'],
    'All Available': available_features
}

# Train models with different feature subsets
models = {}
results = {}

for subset_name, features in feature_subsets.items():
    if all(f in available_features for f in features):
        print(f"\nTraining Random Forest with {subset_name}: {features}")

        X_train_subset = X_train[features]
        X_test_subset = X_test[features]

        # Initialize and train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,  # Increased for better performance
            max_depth=15,      # Increased depth
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train_subset, y_train)
        y_pred = rf_model.predict(X_test_subset)
        y_pred_proba = rf_model.predict_proba(X_test_subset)

        # Store model and results
        models[subset_name] = rf_model
        results[subset_name] = {
            'features': features,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'X_test': X_test_subset
        }

        print(f"Accuracy: {results[subset_name]['accuracy']:.4f}")

# Select best performing model for detailed analysis
best_subset = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\nBest performing model: {best_subset} (Accuracy: {results[best_subset]['accuracy']:.4f})")

best_model = models[best_subset]
best_features = results[best_subset]['features']
best_y_pred = results[best_subset]['y_pred']
best_y_pred_proba = results[best_subset]['y_pred_proba']
best_X_test = results[best_subset]['X_test']

# Convert boolean y_test to binary for consistency
y_test_binary = y_test.astype(int)

# Detailed evaluation of best model
print(f"\nDetailed Results for {best_subset}:")
print(f"Accuracy: {results[best_subset]['accuracy']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, best_y_pred, target_names=['Normal', 'Anomaly']))

# Cross-validation score
cv_scores = cross_val_score(best_model, best_X_test, y_test_binary, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 20))

# 1. Confusion Matrix for Best Model
plt.subplot(4, 3, 1)
cm = confusion_matrix(y_test_binary, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Confusion Matrix - {best_subset}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. Normalized Confusion Matrix
plt.subplot(4, 3, 2)
cm_normalized = confusion_matrix(y_test_binary, best_y_pred, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn_r',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 3. Feature Importance for Best Model
plt.subplot(4, 3, 3)
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': best_features,
    'importance': importances
}).sort_values('importance', ascending=True)

plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
plt.xlabel('Feature Importance')
plt.title(f'Feature Importance - {best_subset}', fontsize=14, fontweight='bold')

# 4. Model Comparison Across Feature Subsets
plt.subplot(4, 3, 4)
subset_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in subset_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(subset_names)))

bars = plt.bar(range(len(subset_names)), accuracies, color=colors)
plt.xticks(range(len(subset_names)), subset_names, rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Model Performance by Feature Subset', fontsize=14, fontweight='bold')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 5. ROC Curve
plt.subplot(4, 3, 5)
fpr, tpr, thresholds = roc_curve(y_test_binary, best_y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")

# 6. Prediction Probability Distribution
plt.subplot(4, 3, 6)
normal_proba = best_y_pred_proba[y_test_binary == 1, 1]  # P(anomaly) for normal samples
anomaly_proba = best_y_pred_proba[y_test_binary == 0, 1]  # P(anomaly) for anomaly samples

plt.hist(normal_proba, bins=30, alpha=0.7, label='Normal (True)', color='blue', density=True)
plt.hist(anomaly_proba, bins=30, alpha=0.7, label='Anomaly (True)', color='red', density=True)
plt.xlabel('Predicted Probability (Anomaly)')
plt.ylabel('Density')
plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
plt.legend()

# 7. Tree Depth Analysis (for trees in forest)
plt.subplot(4, 3, 7)
tree_depths = [tree.tree_.max_depth for tree in best_model.estimators_]
plt.hist(tree_depths, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Tree Depth')
plt.ylabel('Number of Trees')
plt.title('Distribution of Tree Depths in Forest', fontsize=14, fontweight='bold')

# 8. Cross-Validation Scores
plt.subplot(4, 3, 8)
cv_fold_scores = cross_val_score(best_model, best_X_test, y_test_binary, cv=5)
plt.bar(range(1, 6), cv_fold_scores, color='lightcoral', alpha=0.8)
plt.axhline(y=cv_fold_scores.mean(), color='red', linestyle='--',
            label=f'Mean: {cv_fold_scores.mean():.3f}')
plt.xlabel('CV Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores', fontsize=14, fontweight='bold')
plt.legend()
plt.ylim(0, 1)

# 9. Feature Correlation Matrix
plt.subplot(4, 3, 9)
correlation_matrix = best_X_test.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# 10. Learning Curve
plt.subplot(4, 3, 10)
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
train_accuracies = []
for size in train_sizes:
    n_samples = int(size * len(X_train))
    temp_model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)
    temp_model.fit(best_X_test.iloc[:n_samples], y_test_binary.iloc[:n_samples])
    pred = temp_model.predict(best_X_test)
    train_accuracies.append(accuracy_score(y_test_binary, pred))

plt.plot(train_sizes, train_accuracies, 'o-', color='blue', label='Training Accuracy')
plt.xlabel('Training Set Size (fraction)')
plt.ylabel('Accuracy')
plt.title('Learning Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 11. Class Distribution Comparison
plt.subplot(4, 3, 11)
true_counts = y_test_binary.value_counts()
pred_counts = pd.Series(best_y_pred).value_counts()

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, [true_counts[1], true_counts[0]], width,
        label='True', color='lightblue', alpha=0.8)
plt.bar(x + width/2, [pred_counts[1], pred_counts[0]], width,
        label='Predicted', color='lightcoral', alpha=0.8)

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('True vs Predicted Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(x, ['Normal', 'Anomaly'])
plt.legend()

# 12. Out-of-Bag Score Evolution
plt.subplot(4, 3, 12)
oob_scores = []
n_estimators_range = range(10, 101, 10)
for n_est in n_estimators_range:
    temp_model = RandomForestClassifier(n_estimators=n_est, oob_score=True, random_state=42)
    temp_model.fit(best_X_test, y_test_binary)
    oob_scores.append(temp_model.oob_score_)

plt.plot(n_estimators_range, oob_scores, 'o-', color='green')
plt.xlabel('Number of Estimators')
plt.ylabel('OOB Score')
plt.title('Out-of-Bag Score vs Number of Trees', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n" + "="*70)
print("DETAILED RANDOM FOREST ANALYSIS")
print("="*70)

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Correctly predicted Normal): {tn}")
print(f"False Positives (Incorrectly predicted Anomaly): {fp}")
print(f"False Negatives (Missed Anomalies): {fn}")
print(f"True Positives (Correctly predicted Anomaly): {tp}")

print(f"\nBest Model Configuration ({best_subset}):")
print(f"Number of estimators: {best_model.n_estimators}")
print(f"Max depth: {best_model.max_depth}")
print(f"Features used: {best_features}")
print(f"Feature count: {len(best_features)}")

print(f"\nPerformance Metrics:")
precision = tp/(tp+fp) if (tp+fp) > 0 else 0
recall = tp/(tp+fn) if (tp+fn) > 0 else 0
specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
f1_score = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0

print(f"Precision (Anomaly): {precision:.4f}")
print(f"Recall (Anomaly): {recall:.4f}")
print(f"Specificity (Normal): {specificity:.4f}")
print(f"F1-Score (Anomaly): {f1_score:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print(f"\nFeature Importance Ranking:")
for i, (feature, importance) in enumerate(zip(feature_importance_df['feature'],
                                            feature_importance_df['importance']), 1):
    print(f"{i}. {feature}: {importance:.4f}")

print(f"\nModel Comparison Summary:")
for subset_name, result in results.items():
    print(f"{subset_name}: {result['accuracy']:.4f} (Features: {len(result['features'])})")

print("\nRandom Forest model trained, evaluated, and visualized successfully.")
print("Note: Random Forest with balanced class weights helps handle class imbalance.")
print("The model comparison shows how different feature subsets affect performance.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset and prepare data
data = pd.read_csv('household_consumption.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
data.set_index('Datetime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
print("First few index values:", data.index[:5])

# Clean non-numeric values in ALL numeric columns - replace '?' with NaN
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

print("Cleaning numeric columns...")
for col in numeric_columns:
    if col in data.columns:
        question_marks = (data[col] == '?').sum()
        if question_marks > 0:
            print(f"Cleaning {col}: Found {question_marks} '?' values")
        data[col] = pd.to_numeric(data[col].replace('?', np.nan), errors='coerce')

# Prepare features and target
data['consumption'] = data['Global_active_power']
data['day_of_week'] = data.index.dayofweek
data['hour'] = data.index.hour
data['temperature'] = np.random.normal(20, 5, data.shape[0]).clip(10, 30)

# Forward fill NaN values
for col in numeric_columns:
    if col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"Forward filling {nan_count} NaN values in {col}")
            data[col] = data[col].ffill()

data['consumption'] = data['consumption'].ffill()

# Define all available features
all_features = ['consumption', 'hour', 'day_of_week', 'temperature', 'Global_reactive_power',
               'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

available_features = [f for f in all_features if f in data.columns]
print("Available features:", available_features)

# Prepare feature matrix and target
X = data[available_features]
y = (np.abs(data['consumption'] - data['consumption'].rolling(window=96, min_periods=1).mean()) >
     2 * data['consumption'].rolling(window=96, min_periods=1).std()) | data['consumption'].isna()

# Remove rows with NaN values
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

print("Dataset shape after cleaning:", X.shape)
print("Target distribution:\n", y.value_counts())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature selection analysis - try different feature subsets
feature_subsets = {
    'Core Features': ['consumption', 'hour'],
    'Time Features': ['consumption', 'hour', 'day_of_week'],
    'Environmental': ['consumption', 'hour', 'day_of_week', 'temperature'],
    'All Available': available_features
}

# Define parameter grid for optimization (reduced for faster execution)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

print(f"Starting hyperparameter optimization...")
print(f"Total combinations per subset: {np.prod([len(v) for v in param_grid.values()])}")

# Train models with different feature subsets
models = {}
results = {}

for subset_name, features in feature_subsets.items():
    if all(f in available_features for f in features):
        print(f"\nTraining Optimized Random Forest with {subset_name}: {features}")

        X_train_subset = X_train[features]
        X_test_subset = X_test[features]

        # Initialize Random Forest for GridSearch
        rf_base = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # Perform GridSearchCV with reduced CV for speed
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=3,  # Reduced from 5 for faster execution
            scoring='f1',  # Use F1 instead of accuracy for imbalanced data
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_subset, y_train)
        rf_model = grid_search.best_estimator_

        y_pred = rf_model.predict(X_test_subset)
        y_pred_proba = rf_model.predict_proba(X_test_subset)

        # Store model and results
        models[subset_name] = rf_model
        results[subset_name] = {
            'features': features,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'X_test': X_test_subset,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
        print(f"Test Accuracy: {results[subset_name]['accuracy']:.4f}")

# Select best performing model for detailed analysis
best_subset = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\nBest performing model: {best_subset} (Accuracy: {results[best_subset]['accuracy']:.4f})")

best_model = models[best_subset]
best_features = results[best_subset]['features']
best_y_pred = results[best_subset]['y_pred']
best_y_pred_proba = results[best_subset]['y_pred_proba']
best_X_test = results[best_subset]['X_test']

# Convert boolean y_test to binary for consistency
y_test_binary = y_test.astype(int)

# Detailed evaluation of best model
print(f"\nDetailed Results for {best_subset}:")
print(f"Accuracy: {results[best_subset]['accuracy']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, best_y_pred, target_names=['Normal', 'Anomaly']))

# Cross-validation score with best model
cv_scores = cross_val_score(best_model, best_X_test, y_test_binary, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 20))

# 1. Confusion Matrix for Best Model
plt.subplot(4, 3, 1)
cm = confusion_matrix(y_test_binary, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Confusion Matrix - {best_subset}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. Normalized Confusion Matrix
plt.subplot(4, 3, 2)
cm_normalized = confusion_matrix(y_test_binary, best_y_pred, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn_r',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 3. Feature Importance for Best Model
plt.subplot(4, 3, 3)
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': best_features,
    'importance': importances
}).sort_values('importance', ascending=True)

plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
plt.xlabel('Feature Importance')
plt.title(f'Feature Importance - {best_subset}', fontsize=14, fontweight='bold')

# 4. Model Comparison Across Feature Subsets
plt.subplot(4, 3, 4)
subset_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in subset_names]
cv_scores_all = [results[name]['best_score'] for name in subset_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(subset_names)))

x = np.arange(len(subset_names))
width = 0.35

bars1 = plt.bar(x - width/2, accuracies, width, label='Test Accuracy', color=colors, alpha=0.8)
bars2 = plt.bar(x + width/2, cv_scores_all, width, label='CV F1-Score', color=colors, alpha=0.6)

plt.xticks(x, subset_names, rotation=45, ha='right')
plt.ylabel('Score')
plt.title('Model Performance by Feature Subset', fontsize=14, fontweight='bold')
plt.legend()
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars1, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

# 5. ROC Curve
plt.subplot(4, 3, 5)
fpr, tpr, thresholds = roc_curve(y_test_binary, best_y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")

# 6. Prediction Probability Distribution
plt.subplot(4, 3, 6)
if len(y_test_binary[y_test_binary == 1]) > 0 and len(y_test_binary[y_test_binary == 0]) > 0:
    normal_proba = best_y_pred_proba[y_test_binary == 1, 1]
    anomaly_proba = best_y_pred_proba[y_test_binary == 0, 1]

    plt.hist(normal_proba, bins=30, alpha=0.7, label='Normal (True)', color='blue', density=True)
    plt.hist(anomaly_proba, bins=30, alpha=0.7, label='Anomaly (True)', color='red', density=True)
    plt.xlabel('Predicted Probability (Anomaly)')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Insufficient data for both classes', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')

# 7. Tree Depth Analysis
plt.subplot(4, 3, 7)
tree_depths = [tree.tree_.max_depth for tree in best_model.estimators_]
plt.hist(tree_depths, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Tree Depth')
plt.ylabel('Number of Trees')
plt.title('Distribution of Tree Depths in Forest', fontsize=14, fontweight='bold')

# 8. Cross-Validation Scores
plt.subplot(4, 3, 8)
cv_fold_scores = cross_val_score(best_model, best_X_test, y_test_binary, cv=5)
plt.bar(range(1, 6), cv_fold_scores, color='lightcoral', alpha=0.8)
plt.axhline(y=cv_fold_scores.mean(), color='red', linestyle='--',
            label=f'Mean: {cv_fold_scores.mean():.3f}')
plt.xlabel('CV Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores', fontsize=14, fontweight='bold')
plt.legend()
plt.ylim(0, 1)

# 9. Feature Correlation Matrix
plt.subplot(4, 3, 9)
if len(best_features) > 1:
    correlation_matrix = best_X_test.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Single feature - no correlation matrix', ha='center', va='center',
             transform=plt.gca().transAxes)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# 10. Learning Curve (Fixed)
plt.subplot(4, 3, 10)
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
train_accuracies = []
val_accuracies = []

for size in train_sizes:
    n_samples = int(size * len(X_train))
    if n_samples > 10:  # Ensure minimum sample size
        temp_model = RandomForestClassifier(**results[best_subset]['best_params'],
                                          class_weight='balanced', random_state=42)

        # Use the training data for learning curve
        X_train_sample = X_train[best_features].iloc[:n_samples]
        y_train_sample = y_train.iloc[:n_samples]

        temp_model.fit(X_train_sample, y_train_sample)

        # Evaluate on training sample and test set
        train_pred = temp_model.predict(X_train_sample)
        test_pred = temp_model.predict(best_X_test)

        train_accuracies.append(accuracy_score(y_train_sample, train_pred))
        val_accuracies.append(accuracy_score(y_test_binary, test_pred))

plt.plot(train_sizes[:len(train_accuracies)], train_accuracies, 'o-', color='blue', label='Training Accuracy')
plt.plot(train_sizes[:len(val_accuracies)], val_accuracies, 'o-', color='red', label='Validation Accuracy')
plt.xlabel('Training Set Size (fraction)')
plt.ylabel('Accuracy')
plt.title('Learning Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 11. Class Distribution Comparison
plt.subplot(4, 3, 11)
true_counts = y_test_binary.value_counts()
pred_counts = pd.Series(best_y_pred).value_counts()

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, [true_counts.get(1, 0), true_counts.get(0, 0)], width,
        label='True', color='lightblue', alpha=0.8)
plt.bar(x + width/2, [pred_counts.get(1, 0), pred_counts.get(0, 0)], width,
        label='Predicted', color='lightcoral', alpha=0.8)

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('True vs Predicted Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(x, ['Normal', 'Anomaly'])
plt.legend()

# 12. Hyperparameter Impact Analysis
plt.subplot(4, 3, 12)
param_names = list(results[best_subset]['best_params'].keys())
param_values = list(results[best_subset]['best_params'].values())

# Convert to string for display
param_display = [f"{name}:\n{value}" for name, value in zip(param_names, param_values)]
y_pos = np.arange(len(param_names))

plt.barh(y_pos, [1] * len(param_names), color='skyblue', alpha=0.7)
plt.yticks(y_pos, param_display)
plt.xlabel('Optimal Parameter')
plt.title('Best Hyperparameters', fontsize=14, fontweight='bold')
plt.xlim(0, 1.2)

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n" + "="*70)
print("DETAILED OPTIMIZED RANDOM FOREST ANALYSIS")
print("="*70)

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Correctly predicted Normal): {tn}")
print(f"False Positives (Incorrectly predicted Anomaly): {fp}")
print(f"False Negatives (Missed Anomalies): {fn}")
print(f"True Positives (Correctly predicted Anomaly): {tp}")

print(f"\nBest Model Configuration ({best_subset}):")
print(f"Best parameters: {results[best_subset]['best_params']}")
print(f"Best CV F1-score: {results[best_subset]['best_score']:.4f}")
print(f"Features used: {best_features}")
print(f"Feature count: {len(best_features)}")

print(f"\nPerformance Metrics:")
precision = tp/(tp+fp) if (tp+fp) > 0 else 0
recall = tp/(tp+fn) if (tp+fn) > 0 else 0
specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
f1_score = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0

print(f"Precision (Anomaly): {precision:.4f}")
print(f"Recall (Anomaly): {recall:.4f}")
print(f"Specificity (Normal): {specificity:.4f}")
print(f"F1-Score (Anomaly): {f1_score:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print(f"\nFeature Importance Ranking:")
for i, (feature, importance) in enumerate(zip(feature_importance_df['feature'],
                                            feature_importance_df['importance']), 1):
    print(f"{i}. {feature}: {importance:.4f}")

print(f"\nModel Comparison Summary:")
for subset_name, result in results.items():
    print(f"{subset_name}: Accuracy={result['accuracy']:.4f}, CV F1={result['best_score']:.4f} (Features: {len(result['features'])})")

print("\nOptimized Random Forest model trained, evaluated, and visualized successfully.")
print("Note: Used F1-score for hyperparameter optimization to better handle class imbalance.")
print("GridSearchCV ensures optimal hyperparameters for each feature subset.")

"""**XGBoost train and evaluated**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset and prepare data
data = pd.read_csv('household_consumption.csv')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
data.set_index('Datetime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
print("First few index values:", data.index[:5])

# Clean non-numeric values in ALL numeric columns - replace '?' with NaN
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

print("Cleaning numeric columns...")
for col in numeric_columns:
    if col in data.columns:
        question_marks = (data[col] == '?').sum()
        if question_marks > 0:
            print(f"Cleaning {col}: Found {question_marks} '?' values")
        data[col] = pd.to_numeric(data[col].replace('?', np.nan), errors='coerce')

# Prepare features and target
data['consumption'] = data['Global_active_power']
data['day_of_week'] = data.index.dayofweek
data['hour'] = data.index.hour
data['temperature'] = np.random.normal(20, 5, data.shape[0]).clip(10, 30)

# Forward fill NaN values
for col in numeric_columns:
    if col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"Forward filling {nan_count} NaN values in {col}")
            data[col] = data[col].ffill()

data['consumption'] = data['consumption'].ffill()

# Define all available features
all_features = ['consumption', 'hour', 'day_of_week', 'temperature', 'Global_reactive_power',
               'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

available_features = [f for f in all_features if f in data.columns]
print("Available features:", available_features)

# Prepare feature matrix and target
X = data[available_features]
y = (np.abs(data['consumption'] - data['consumption'].rolling(window=96, min_periods=1).mean()) >
     2 * data['consumption'].rolling(window=96, min_periods=1).std()) | data['consumption'].isna()

# Remove rows with NaN values
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

print("Dataset shape after cleaning:", X.shape)
print("Target distribution:\n", y.value_counts())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature selection analysis - try different feature subsets
feature_subsets = {
    'Core Features': ['consumption', 'hour'],
    'Time Features': ['consumption', 'hour', 'day_of_week'],
    'Environmental': ['consumption', 'hour', 'day_of_week', 'temperature'],
    'All Available': available_features
}

# Train models with different feature subsets
models = {}
results = {}

for subset_name, features in feature_subsets.items():
    if all(f in available_features for f in features):
        print(f"\nTraining XGBoost with {subset_name}: {features}")

        X_train_subset = X_train[features]
        X_test_subset = X_test[features]

        # Initialize and train XGBoost
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=15,
            min_child_weight=2,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        xgb_model.fit(X_train_subset, y_train)
        y_pred = xgb_model.predict(X_test_subset)
        y_pred_proba = xgb_model.predict_proba(X_test_subset)

        # Store model and results
        models[subset_name] = xgb_model
        results[subset_name] = {
            'features': features,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'X_test': X_test_subset
        }

        print(f"Accuracy: {results[subset_name]['accuracy']:.4f}")

# Select best performing model for detailed analysis
best_subset = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\nBest performing model: {best_subset} (Accuracy: {results[best_subset]['accuracy']:.4f})")

best_model = models[best_subset]
best_features = results[best_subset]['features']
best_y_pred = results[best_subset]['y_pred']
best_y_pred_proba = results[best_subset]['y_pred_proba']
best_X_test = results[best_subset]['X_test']

# Convert boolean y_test to binary for consistency
y_test_binary = y_test.astype(int)

# Detailed evaluation of best model
print(f"\nDetailed Results for {best_subset}:")
print(f"Accuracy: {results[best_subset]['accuracy']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, best_y_pred, target_names=['Normal', 'Anomaly']))

# Cross-validation score
cv_scores = cross_val_score(best_model, best_X_test, y_test_binary, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 20))

# 1. Confusion Matrix for Best Model
plt.subplot(4, 3, 1)
cm = confusion_matrix(y_test_binary, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title(f'Confusion Matrix - {best_subset}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. Normalized Confusion Matrix
plt.subplot(4, 3, 2)
cm_normalized = confusion_matrix(y_test_binary, best_y_pred, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn_r',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 3. Feature Importance for Best Model
plt.subplot(4, 3, 3)
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': best_features,
    'importance': importances
}).sort_values('importance', ascending=True)

plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
plt.xlabel('Feature Importance')
plt.title(f'Feature Importance - {best_subset}', fontsize=14, fontweight='bold')

# 4. Model Comparison Across Feature Subsets
plt.subplot(4, 3, 4)
subset_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in subset_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(subset_names)))

bars = plt.bar(range(len(subset_names)), accuracies, color=colors)
plt.xticks(range(len(subset_names)), subset_names, rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Model Performance by Feature Subset', fontsize=14, fontweight='bold')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 5. ROC Curve
plt.subplot(4, 3, 5)
fpr, tpr, thresholds = roc_curve(y_test_binary, best_y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")

# 6. Prediction Probability Distribution
plt.subplot(4, 3, 6)
normal_proba = best_y_pred_proba[y_test_binary == 1, 1]
anomaly_proba = best_y_pred_proba[y_test_binary == 0, 1]

plt.hist(normal_proba, bins=30, alpha=0.7, label='Normal (True)', color='blue', density=True)
plt.hist(anomaly_proba, bins=30, alpha=0.7, label='Anomaly (True)', color='red', density=True)
plt.xlabel('Predicted Probability (Anomaly)')
plt.ylabel('Density')
plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
plt.legend()

# 7. Learning Curve
plt.subplot(4, 3, 7)
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
train_accuracies = []
for size in train_sizes:
    n_samples = int(size * len(X_train))
    temp_model = XGBClassifier(n_estimators=50, max_depth=15, random_state=42, use_label_encoder=False, eval_metric='logloss')
    temp_model.fit(X_train[best_features].iloc[:n_samples], y_train.iloc[:n_samples])
    pred = temp_model.predict(best_X_test)
    train_accuracies.append(accuracy_score(y_test_binary, pred))

plt.plot(train_sizes, train_accuracies, 'o-', color='blue', label='Training Accuracy')
plt.xlabel('Training Set Size (fraction)')
plt.ylabel('Accuracy')
plt.title('Learning Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Cross-Validation Scores
plt.subplot(4, 3, 8)
cv_fold_scores = cross_val_score(best_model, best_X_test, y_test_binary, cv=5)
plt.bar(range(1, 6), cv_fold_scores, color='lightcoral', alpha=0.8)
plt.axhline(y=cv_fold_scores.mean(), color='red', linestyle='--',
            label=f'Mean: {cv_fold_scores.mean():.3f}')
plt.xlabel('CV Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores', fontsize=14, fontweight='bold')
plt.legend()
plt.ylim(0, 1)

# 9. Feature Correlation Matrix
plt.subplot(4, 3, 9)
correlation_matrix = best_X_test.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# 10. Class Distribution Comparison
plt.subplot(4, 3, 10)
true_counts = y_test_binary.value_counts()
pred_counts = pd.Series(best_y_pred).value_counts()

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, [true_counts.get(1, 0), true_counts.get(0, 0)], width,
        label='True', color='lightblue', alpha=0.8)
plt.bar(x + width/2, [pred_counts.get(1, 0), pred_counts.get(0, 0)], width,
        label='Predicted', color='lightcoral', alpha=0.8)

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('True vs Predicted Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(x, ['Normal', 'Anomaly'])
plt.legend()

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n" + "="*70)
print("DETAILED XGBOOST ANALYSIS")
print("="*70)

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (Correctly predicted Normal): {tn}")
print(f"False Positives (Incorrectly predicted Anomaly): {fp}")
print(f"False Negatives (Missed Anomalies): {fn}")
print(f"True Positives (Correctly predicted Anomaly): {tp}")

print(f"\nBest Model Configuration ({best_subset}):")
print(f"Number of estimators: {best_model.n_estimators}")
print(f"Max depth: {best_model.max_depth}")
print(f"Features used: {best_features}")
print(f"Feature count: {len(best_features)}")

print(f"\nPerformance Metrics:")
precision = tp/(tp+fp) if (tp+fp) > 0 else 0
recall = tp/(tp+fn) if (tp+fn) > 0 else 0
specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
f1_score = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0

print(f"Precision (Anomaly): {precision:.4f}")
print(f"Recall (Anomaly): {recall:.4f}")
print(f"Specificity (Normal): {specificity:.4f}")
print(f"F1-Score (Anomaly): {f1_score:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print(f"\nFeature Importance Ranking:")
for i, (feature, importance) in enumerate(zip(feature_importance_df['feature'],
                                            feature_importance_df['importance']), 1):
    print(f"{i}. {feature}: {importance:.4f}")

print(f"\nModel Comparison Summary:")
for subset_name, result in results.items():
    print(f"{subset_name}: {result['accuracy']:.4f} (Features: {len(result['features'])})")

print("\nXGBoost model trained, evaluated, and visualized successfully.")
print("Note: XGBoost with scale_pos_weight helps handle class imbalance.")
print("The model comparison shows how different feature subsets affect performance.")

pip install -U notebook-as-pdf

!pyppeteer-install