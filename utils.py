import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import os
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

plt.style.use('default')
sns.set_palette("husl")

def clean_data(df):
    try:
        logger.info("Starting data cleaning...")
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"Input columns: {list(df.columns)}")
        
        # Validate required columns
        required_columns = ['Date', 'Time', 'Global_active_power']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}. Available columns: {', '.join(df.columns)}")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Validate and create datetime
        try:
            logger.info("Creating datetime column")
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
            logger.info("Datetime column created successfully")
        except ValueError as e:
            logger.error(f"DateTime parsing error: {str(e)}")
            # Try alternative formats
            try:
                df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                logger.info("Datetime created with automatic format detection")
            except:
                raise ValueError("Could not parse Date/Time columns. Please ensure Date format is 'dd/mm/yyyy' and Time format is 'hh:mm:ss'")
        
        df.set_index('Datetime', inplace=True)
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        
        # Handle numeric columns
        numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                         'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        
        logger.info("Converting numeric columns")
        for col in numeric_columns:
            if col in df.columns:
                # Replace '?' with NaN and convert to numeric
                df[col] = pd.to_numeric(df[col].replace('?', np.nan), errors='coerce')
                logger.info(f"Processed column {col}, NaN count: {df[col].isna().sum()}")
        
        # Create consumption feature
        df['consumption'] = df['Global_active_power']
        
        # Create time-based features
        logger.info("Creating time-based features")
        df['day_of_week'] = df.index.dayofweek
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        
        # Add temperature feature (simulated for this example)
        np.random.seed(42)  # For reproducibility
        df['temperature'] = np.random.normal(20, 5, df.shape[0]).clip(10, 30)
        
        # Forward fill missing values
        logger.info("Handling missing values")
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()  # Forward fill then backward fill
        
        df['consumption'] = df['consumption'].ffill().bfill()
        
        # Define available features
        all_features = ['consumption', 'hour', 'day_of_week', 'month', 'temperature', 
                       'Global_reactive_power', 'Voltage', 'Global_intensity', 
                       'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        available_features = [f for f in all_features if f in df.columns and df[f].notna().sum() > 0]
        
        logger.info(f"Available features: {available_features}")
        
        # Create features matrix
        X = df[available_features].copy()
        
        # Create anomaly labels using rolling statistics
        logger.info("Creating anomaly labels")
        window_size = min(96, len(df) // 10)  # Adaptive window size
        rolling_mean = df['consumption'].rolling(window=window_size, min_periods=1, center=True).mean()
        rolling_std = df['consumption'].rolling(window=window_size, min_periods=1, center=True).std()
        
        # Define anomalies as points significantly deviating from rolling statistics
        threshold_multiplier = 2.5
        upper_bound = rolling_mean + threshold_multiplier * rolling_std
        lower_bound = rolling_mean - threshold_multiplier * rolling_std
        
        y = ((df['consumption'] > upper_bound) | 
             (df['consumption'] < lower_bound) | 
             df['consumption'].isna()).astype(int)
        
        # Remove rows with any missing values
        logger.info("Removing rows with missing values")
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Ensure we have some anomalies
        anomaly_rate = y.sum() / len(y)
        logger.info(f"Anomaly rate: {anomaly_rate:.3f}")
        
        if anomaly_rate < 0.01:  # Less than 1% anomalies
            logger.warning("Very low anomaly rate detected, adjusting threshold")
            # Use a more sensitive threshold
            threshold_multiplier = 2.0
            upper_bound = rolling_mean + threshold_multiplier * rolling_std
            lower_bound = rolling_mean - threshold_multiplier * rolling_std
            y = ((df['consumption'] > upper_bound) | 
                 (df['consumption'] < lower_bound) | 
                 df['consumption'].isna()).astype(int)
            y = y[mask]
            
        final_anomaly_rate = y.sum() / len(y)
        logger.info(f"Final anomaly rate: {final_anomaly_rate:.3f}")
        logger.info(f"Final data shape: X={X.shape}, y={y.shape}")
        
        return X, y, available_features
        
    except Exception as e:
        logger.error(f"Error in clean_data: {str(e)}")
        raise Exception(f"Error cleaning data: {str(e)}")

def train_rf(X_train, y_train, features):
    logger.info("Training Random Forest...")
    
    feature_subsets = {
        'Core Features': ['consumption', 'hour'],
        'Time Features': ['consumption', 'hour', 'day_of_week'],
        'Environmental': ['consumption', 'hour', 'day_of_week', 'temperature'],
        'Extended': ['consumption', 'hour', 'day_of_week', 'temperature', 'month'],
        'All Available': features
    }
    
    # Simplified parameter grid for faster training
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    results = {}
    logger.info(f"Training RF with {len(feature_subsets)} feature subsets")
    
    for subset_name, subset_features in feature_subsets.items():
        available_subset_features = [f for f in subset_features if f in features]
        if len(available_subset_features) < 2:  # Need at least 2 features
            logger.warning(f"Skipping {subset_name}: insufficient features")
            continue
            
        try:
            logger.info(f"Training RF with {subset_name}: {available_subset_features}")
            X_train_subset = X_train[available_subset_features]
            
            rf_base = RandomForestClassifier(
                class_weight='balanced', 
                random_state=42, 
                n_jobs=-1
            )
            
            # Use smaller CV for faster training
            grid_search = GridSearchCV(
                estimator=rf_base, 
                param_grid=param_grid, 
                cv=3, 
                scoring='f1',  # Better for imbalanced data
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_subset, y_train)
            model = grid_search.best_estimator_
            
            # Calculate training score
            train_score = model.score(X_train_subset, y_train)
            
            results[subset_name] = {
                'model': model,
                'features': available_subset_features,
                'best_params': grid_search.best_params_,
                'train_score': train_score
            }
            
            logger.info(f"RF {subset_name} completed. Score: {train_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training RF with {subset_name}: {str(e)}")
            continue
    
    if not results:
        raise Exception("No valid feature subsets found for Random Forest")
    
    # Select best model based on training score
    best_subset = max(results, key=lambda k: results[k]['train_score'])
    logger.info(f"Best RF subset: {best_subset} with score: {results[best_subset]['train_score']:.3f}")
    
    return results[best_subset]['model'], results[best_subset]['features'], feature_subsets

def train_if(X_train, y_train, features):
    logger.info("Training Isolation Forest...")
    
    feature_subsets = {
        'Core Features': ['consumption', 'hour'],
        'Time Features': ['consumption', 'hour', 'day_of_week'],
        'Environmental': ['consumption', 'hour', 'day_of_week', 'temperature'],
        'Extended': ['consumption', 'hour', 'day_of_week', 'temperature', 'month'],
        'All Available': features
    }
    
    # Simplified parameters for faster training
    param_options = [
        {'n_estimators': 100, 'max_samples': 0.6, 'contamination': 0.1},
        {'n_estimators': 100, 'max_samples': 0.8, 'contamination': 0.1},
        {'n_estimators': 50, 'max_samples': 0.6, 'contamination': 0.05}
    ]
    
    results = {}
    logger.info(f"Training IF with {len(feature_subsets)} feature subsets")
    
    for subset_name, subset_features in feature_subsets.items():
        available_subset_features = [f for f in subset_features if f in features]
        if len(available_subset_features) < 2:
            logger.warning(f"Skipping {subset_name}: insufficient features")
            continue
            
        try:
            logger.info(f"Training IF with {subset_name}: {available_subset_features}")
            X_train_subset = X_train[available_subset_features]
            
            best_model = None
            best_score = -np.inf
            best_params = None
            
            for params in param_options:
                try:
                    model = IsolationForest(random_state=42, **params)
                    model.fit(X_train_subset)
                    
                    y_pred = model.predict(X_train_subset)
                    y_pred_binary = np.where(y_pred == -1, 1, 0)
                    
                    # Use F1 score for better evaluation of imbalanced data
                    from sklearn.metrics import f1_score
                    score = f1_score(y_train, y_pred_binary)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                        
                except Exception as e:
                    logger.warning(f"Error with params {params}: {str(e)}")
                    continue
            
            if best_model is not None:
                results[subset_name] = {
                    'model': best_model,
                    'features': available_subset_features,
                    'best_params': best_params,
                    'score': best_score
                }
                logger.info(f"IF {subset_name} completed. Score: {best_score:.3f}")
                
        except Exception as e:
            logger.error(f"Error training IF with {subset_name}: {str(e)}")
            continue
    
    if not results:
        raise Exception("No valid feature subsets found for Isolation Forest")
    
    best_subset = max(results, key=lambda k: results[k]['score'])
    logger.info(f"Best IF subset: {best_subset} with score: {results[best_subset]['score']:.3f}")
    
    return results[best_subset]['model'], results[best_subset]['features'], feature_subsets

def train_dt(X_train, y_train, features):
    logger.info("Training Decision Tree...")
    
    feature_subsets = {
        'Core Features': ['consumption', 'hour'],
        'Time Features': ['consumption', 'hour', 'day_of_week'],
        'Environmental': ['consumption', 'hour', 'day_of_week', 'temperature'],
        'Extended': ['consumption', 'hour', 'day_of_week', 'temperature', 'month'],
        'All Available': features
    }
    
    # Simplified parameter grid
    param_grid = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    
    results = {}
    logger.info(f"Training DT with {len(feature_subsets)} feature subsets")
    
    for subset_name, subset_features in feature_subsets.items():
        available_subset_features = [f for f in subset_features if f in features]
        if len(available_subset_features) < 2:
            logger.warning(f"Skipping {subset_name}: insufficient features")
            continue
            
        try:
            logger.info(f"Training DT with {subset_name}: {available_subset_features}")
            X_train_subset = X_train[available_subset_features]
            
            dt_base = DecisionTreeClassifier(
                class_weight='balanced', 
                random_state=42
            )
            
            grid_search = GridSearchCV(
                estimator=dt_base, 
                param_grid=param_grid, 
                cv=3, 
                scoring='f1', 
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_subset, y_train)
            model = grid_search.best_estimator_
            
            train_score = model.score(X_train_subset, y_train)
            
            results[subset_name] = {
                'model': model,
                'features': available_subset_features,
                'best_params': grid_search.best_params_,
                'train_score': train_score
            }
            
            logger.info(f"DT {subset_name} completed. Score: {train_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training DT with {subset_name}: {str(e)}")
            continue
    
    if not results:
        raise Exception("No valid feature subsets found for Decision Tree")
    
    best_subset = max(results, key=lambda k: results[k]['train_score'])
    logger.info(f"Best DT subset: {best_subset} with score: {results[best_subset]['train_score']:.3f}")
    
    return results[best_subset]['model'], results[best_subset]['features'], feature_subsets

def train_xgb(X_train, y_train, features):
    logger.info("Training XGBoost...")
    
    feature_subsets = {
        'Core Features': ['consumption', 'hour'],
        'Time Features': ['consumption', 'hour', 'day_of_week'],
        'Environmental': ['consumption', 'hour', 'day_of_week', 'temperature'],
        'Extended': ['consumption', 'hour', 'day_of_week', 'temperature', 'month'],
        'All Available': features
    }
    
    # Simplified parameter grid
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [50, 100],
        'subsample': [0.8, 1.0]
    }
    
    results = {}
    logger.info(f"Training XGB with {len(feature_subsets)} feature subsets")
    
    for subset_name, subset_features in feature_subsets.items():
        available_subset_features = [f for f in subset_features if f in features]
        if len(available_subset_features) < 2:
            logger.warning(f"Skipping {subset_name}: insufficient features")
            continue
            
        try:
            logger.info(f"Training XGB with {subset_name}: {available_subset_features}")
            X_train_subset = X_train[available_subset_features]
            
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            xgb_base = XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                random_state=42
            )
            
            grid_search = GridSearchCV(
                estimator=xgb_base, 
                param_grid=param_grid, 
                cv=3, 
                scoring='f1', 
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_subset, y_train)
            model = grid_search.best_estimator_
            
            train_score = model.score(X_train_subset, y_train)
            
            results[subset_name] = {
                'model': model,
                'features': available_subset_features,
                'best_params': grid_search.best_params_,
                'train_score': train_score
            }
            
            logger.info(f"XGB {subset_name} completed. Score: {train_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training XGB with {subset_name}: {str(e)}")
            continue
    
    if not results:
        raise Exception("No valid feature subsets found for XGBoost")
    
    best_subset = max(results, key=lambda k: results[k]['train_score'])
    logger.info(f"Best XGB subset: {best_subset} with score: {results[best_subset]['train_score']:.3f}")
    
    return results[best_subset]['model'], results[best_subset]['features'], feature_subsets

def evaluate_model(model, X_test, y_test, features, model_type, static_folder):
    logger.info(f"Evaluating {model_type} model...")
    
    try:
        # Ensure we have the required features
        available_features = [f for f in features if f in X_test.columns]
        if not available_features:
            raise Exception(f"No required features found in test data. Required: {features}, Available: {list(X_test.columns)}")
        
        logger.info(f"Using features for evaluation: {available_features}")
        X_test_subset = X_test[available_features]
        
        # Make predictions
        y_pred = model.predict(X_test_subset)
        
        # Handle Isolation Forest output
        if model_type == 'if':
            y_pred = np.where(y_pred == -1, 1, 0)
            # Get decision scores for ROC curve
            y_pred_proba = -model.decision_function(X_test_subset)
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        else:
            try:
                y_pred_proba = model.predict_proba(X_test_subset)[:, 1]
            except:
                # Fallback if predict_proba not available
                y_pred_proba = y_pred.astype(float)
        
        # Convert to binary if needed
        y_test_binary = y_test.astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_binary, y_pred)
        report = classification_report(y_test_binary, y_pred, target_names=['Normal', 'Anomaly'], output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test_binary, y_pred)
        
        # ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
            roc_auc = auc(fpr, tpr)
        except:
            logger.warning("Could not compute ROC curve, using dummy values")
            fpr, tpr = [0, 1], [0, 1]
            roc_auc = 0.5
        
        # Create plots directory
        plots_dir = os.path.join(static_folder, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot confusion matrix
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'Anomaly'], 
                        yticklabels=['Normal', 'Anomaly'])
            plt.title(f'Confusion Matrix - {model_type.upper()}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_plot_path = os.path.join(plots_dir, 'cm.png')
            plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Confusion matrix plot saved")
        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {str(e)}")
        
        # Plot ROC curve
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_type.upper()}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            roc_plot_path = os.path.join(plots_dir, 'roc.png')
            plt.savefig(roc_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("ROC curve plot saved")
        except Exception as e:
            logger.error(f"Error creating ROC plot: {str(e)}")
        
        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.3f}, ROC AUC: {roc_auc:.3f}")
        
        return {
            'accuracy': accuracy,
            'report': report,
            'cm_plot': 'cm.png',
            'roc_plot': 'roc.png',
            'roc_auc': roc_auc,
            'features_used': available_features
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        raise Exception(f"Error evaluating model: {str(e)}")